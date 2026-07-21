import asyncio
import atexit
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import mlflow
from confluent_kafka import Producer as KafkaProducer
from fastapi import BackgroundTasks, Depends, FastAPI, File, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from langgraph.checkpoint.redis import AsyncRedisSaver
from opentelemetry import trace
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel
from pythonjsonlogger import jsonlogger

from src.serving.agent_api.cache import SemanticCache
from src.serving.agent_api.cost_guard import check_budget
from src.serving.agent_api.graph import _is_kb_query, build_graph, run_graph_stream
from src.serving.agent_api.guardrails import Guardrails, redact_pii
from src.serving.agent_api.indexer import KBIndexer
from src.serving.agent_api.ingestion.pipeline import run_ingestion
from src.serving.agent_api.ingestion.s3_loader import S3KBLoader
from src.serving.agent_api.memory import MemoryStore
from src.serving.agent_api.policy_engine import PolicyEngine
from src.serving.agent_api.rag import RAGPipeline
from src.serving.agent_api.security import check_ip_rate_limit, require_admin_api_key
from src.serving.agent_api.tools import set_kb_indexer
from src.serving.agent_api.tracing import (
    chat_cache_hit_total,
    guardrail_block_total,
    search_cache_hit_total,
    setup_tracing,
)


def _setup_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(
        jsonlogger.JsonFormatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    )
    logging.root.handlers = [handler]
    logging.root.setLevel(os.getenv("LOG_LEVEL", "INFO"))


_setup_logging()
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Global singletons
# ------------------------------------------------------------------

rag_pipeline: RAGPipeline | None = None
memory_store: MemoryStore | None = None
guardrails: Guardrails | None = None
semantic_cache: SemanticCache | None = None
chat_semantic_cache: SemanticCache | None = None
agent_graph = None  # compiled LangGraph ReAct agent
policy_engine: PolicyEngine | None = None
kb_indexer: KBIndexer | None = None

# Singleton — trước đây /feedback tạo 1 Producer mới mỗi request (mỗi request
# phải mở connection + handshake + fetch metadata cluster từ đầu). Tạo 1 lần
# ở module-level, dùng chung cho mọi request, chỉ flush() lúc process tắt.
_feedback_producer = KafkaProducer(
    {"bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "broker:29092")}
)


def _flush_feedback_producer_on_exit() -> None:
    remaining = _feedback_producer.flush(timeout=10)
    if remaining > 0:
        logger.warning("feedback_producer_flush_incomplete remaining=%d", remaining)


atexit.register(_flush_feedback_producer_on_exit)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline, memory_store, guardrails, semantic_cache, chat_semantic_cache, agent_graph, policy_engine, kb_indexer

    logger.info("agent_api_startup")
    setup_tracing()
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=32))

    rag_pipeline = RAGPipeline()
    memory_store = MemoryStore()
    guardrails = Guardrails()
    semantic_cache = SemanticCache(
        embedding_model=rag_pipeline._model,
        redis_client=memory_store._r,
    )
    chat_semantic_cache = SemanticCache(
        embedding_model=rag_pipeline._model,
        redis_client=memory_store._r,
        namespace="chat",
    )
    # LangGraph checkpointer — persistence native và BẮT BUỘC db=0 (RedisSaver không tạo được
    # index trên db khác). Key có prefix "checkpoint*" nên không đụng semantic
    # cache vốn cũng ở db=0.
    try:
        redis_url = (
            f"redis://:{os.getenv('REDIS_PASSWORD', '123')}@"
            f"{os.getenv('REDIS_HOST', 'redis')}:{os.getenv('REDIS_PORT', '6379')}/0"
        )
        checkpointer_cm = AsyncRedisSaver.from_conn_string(redis_url)
        checkpointer = await checkpointer_cm.__aenter__()
        await checkpointer.asetup()
        logger.info("langgraph_checkpointer_ready backend=redis")
    except Exception as exc:
        # Không có checkpointer thì agent vẫn trả lời được, chỉ mất trí nhớ giữa
        # các lượt — tốt hơn là không khởi động nổi API.
        logger.error("langgraph_checkpointer_failed, chạy stateless: %s", exc)
        checkpointer_cm = None
        checkpointer = None

    agent_graph = build_graph(rag_pipeline=rag_pipeline, checkpointer=checkpointer)
    policy_engine = PolicyEngine()

    try:
        kb_indexer = KBIndexer.load()
        logger.info("kb_indexer_loaded size=%d", kb_indexer.size)
    except Exception:
        logger.info("kb_indexer_no_index_found, starting empty")
        kb_indexer = KBIndexer()

    if kb_indexer.size == 0:
        logger.info("kb_indexer_empty, running initial ingestion")
        try:

            kb_indexer = run_ingestion()
            logger.info("kb_indexer_initial_ingestion_done size=%d", kb_indexer.size)
        except Exception as exc:
            logger.warning("kb_indexer_initial_ingestion_failed: %s", exc)

    set_kb_indexer(kb_indexer)

    try:
        version = os.getenv("AGENT_PROMPT_VERSION", "v1")
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        with mlflow.start_run(run_name="agent-api-startup"):
            mlflow.log_param("prompt_version", version)
            mlflow.log_param("llm_backend", os.getenv("AGENT_LLM_BACKEND", "ollama"))
    except Exception as exc:
        logger.warning("mlflow_prompt_log_failed", extra={"error": str(exc)})

    logger.info("agent_api_ready")
    yield

    logger.info("agent_api_shutdown")
    try:
        tracer_provider = trace.get_tracer_provider()
        if hasattr(tracer_provider, "shutdown"):
            tracer_provider.shutdown()
    except Exception as exc:
        logger.warning("otel_shutdown_failed", extra={"error": str(exc)})

    if checkpointer_cm is not None:
        try:
            await checkpointer_cm.__aexit__(None, None, None)
        except Exception as exc:
            logger.warning("checkpointer_shutdown_failed", extra={"error": str(exc)})


app = FastAPI(
    title="E-commerce Shopping Agent API",
    version="2.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))
    request.state.trace_id = trace_id
    t0 = time.time()
    response = await call_next(request)
    logger.info(
        "request_completed",
        extra={
            "trace_id": trace_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": round((time.time() - t0) * 1000),
        },
    )
    response.headers["X-Trace-ID"] = trace_id
    return response


# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------


class ChatRequest(BaseModel):
    customer_id: str
    message: str


class SearchResult(BaseModel):
    product_id: str
    product_name: str
    price: float
    category_name: str
    brand_name: str
    avg_sentiment: float
    score: float


class FeedbackRequest(BaseModel):
    customer_id: str
    product_id: str
    action: str  # "click" | "purchase" | "ignore"
    session_id: str | None = None
    source: str | None = None


# ------------------------------------------------------------------
# Chat
# ------------------------------------------------------------------


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, bg: BackgroundTasks, http_request: Request):
    """Single streaming endpoint. LLM decides which tools to call."""
    if not check_ip_rate_limit(http_request):
        guardrail_block_total.labels(reason="ip_rate_limit").inc()

        async def _ip_limited():
            yield f'data: {json.dumps({"type": "blocked", "message": "Quá nhiều yêu cầu, vui lòng thử lại sau."}, ensure_ascii=False)}\n\n'

        return StreamingResponse(_ip_limited(), media_type="text/event-stream")

    # Rate limit chặn số REQUEST, không chặn CHI PHÍ: 1 request context dài +
    # nhiều vòng tool-call tốn gấp hàng chục lần request thường, nên 60 req/phút
    # vẫn đủ đốt sạch budget. Kiểm tra chi phí đã tiêu hôm nay trước khi gọi LLM.
    budget = check_budget(request.customer_id)
    if not budget.allowed:
        guardrail_block_total.labels(reason=budget.reason).inc()

        async def _budget_exceeded():
            yield f'data: {json.dumps({"type": "blocked", "message": "Đã đạt giới hạn sử dụng hôm nay, vui lòng quay lại sau."}, ensure_ascii=False)}\n\n'

        return StreamingResponse(_budget_exceeded(), media_type="text/event-stream")

    policy = policy_engine.check(request.message)
    if not policy.allowed:
        guardrail_block_total.labels(reason=policy.reason).inc()

        async def _policy_blocked():
            yield f'data: {json.dumps({"type": "blocked", "message": "Yêu cầu không hợp lệ."}, ensure_ascii=False)}\n\n'

        return StreamingResponse(_policy_blocked(), media_type="text/event-stream")

    guard = guardrails.check_input(request.message, request.customer_id)
    if not guard.allowed:
        guardrail_block_total.labels(reason=guard.reason).inc()

        async def _guard_blocked():
            yield f'data: {json.dumps({"type": "blocked", "message": "Yêu cầu không hợp lệ."}, ensure_ascii=False)}\n\n'

        return StreamingResponse(_guard_blocked(), media_type="text/event-stream")

    pref_context = memory_store.build_preference_context(request.customer_id)

    # Ẩn SĐT/email/số thẻ TRƯỚC khi câu hỏi rời hệ thống sang LLM provider.
    safe_message = redact_pii(request.message)

    # Semantic cache CHỈ cho câu hỏi policy/KB (_is_kb_query) — câu trả lời
    # không phụ thuộc customer_id/lịch sử mua hàng nên dùng lại được giữa
    # các khách khác nhau, khác product_search/recommend phụ thuộc catalog/
    # lịch sử theo thời gian thực. Cache hit bỏ qua hẳn LLM generation.
    #
    # lookup_or_claim() chống thundering herd: đo thật ở 100 concurrent, hit
    # rate rơi còn 50.9% (đáng lẽ gần 100% chỉ với 4 câu hỏi) vì nhiều request
    # TRÙNG câu hỏi tới cùng lúc lúc cache còn rỗng đều tự generate riêng —
    # xem BENCHMARK_RESULTS.md mục 21. Request đầu tiên (leader) generate
    # bình thường rồi store(); các request trùng đến sau đó (follower) ĐỢI
    # thay vì tự generate riêng.
    cacheable = _is_kb_query(safe_message)
    cached = None
    is_leader = False
    loop = asyncio.get_running_loop()
    if cacheable:
        cached, is_leader = await loop.run_in_executor(
            None, chat_semantic_cache.lookup_or_claim, safe_message
        )
        if cached is None and not is_leader:
            cached = await _wait_for_chat_cache(chat_semantic_cache, safe_message)
            if cached is None:
                # Leader không xong kịp trong thời gian chờ (timeout/crash) —
                # tự generate luôn thay vì đợi mãi, không cần giành lại lock.
                is_leader = True
        chat_cache_hit_total.labels(type="hit" if cached is not None else "miss").inc()
    else:
        chat_cache_hit_total.labels(type="not_cacheable").inc()

    async def generate():
        full_response, tool_calls = "", []
        if cached is not None:
            full_response = cached.get("full_response", "")
            tool_calls = cached.get("tool_calls", [])
            event = {
                "type": "done",
                "full_response": full_response,
                "tool_calls": tool_calls,
                "cached": True,
            }
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        else:
            try:
                async for event in run_graph_stream(
                    customer_id=request.customer_id,
                    message=safe_message,
                    pref_context=pref_context,
                    rag_pipeline=rag_pipeline,
                    graph=agent_graph,
                ):
                    if event["type"] == "done":
                        full_response = event.get("full_response", "")
                        tool_calls = event.get("tool_calls", [])
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as exc:
                logger.exception(
                    "chat_stream_failed", extra={"customer_id": request.customer_id}
                )
                if cacheable and is_leader:
                    # Nhả lock sớm — follower khác không phải đợi hết TTL
                    # (30s) cho 1 request đã hỏng.
                    chat_semantic_cache.release_lock(safe_message)
                payload = {
                    "type": "error",
                    "message": "Hệ thống gặp sự cố, vui lòng thử lại.",
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                return  # không persist memory / không cache từ một lượt hỏng

        cleaned = guardrails.check_output(full_response)
        if cacheable and cached is None and cleaned:
            await loop.run_in_executor(
                None,
                chat_semantic_cache.store,
                safe_message,
                {"full_response": cleaned, "tool_calls": tool_calls},
            )
        bg.add_task(
            _persist_memory, request.customer_id, request.message, cleaned, tool_calls
        )

    return StreamingResponse(generate(), media_type="text/event-stream")


async def _wait_for_chat_cache(
    cache: SemanticCache, query: str, max_wait: float = 8.0, poll_interval: float = 0.25
):
    """Follower trong cơ chế chống thundering herd — poll cache thay vì tự
    generate ngay, tối đa max_wait giây (đủ ngắn so với latency LLM thật
    ~5-9s để không làm follower chờ lâu hơn nếu tự generate riêng).

    Dùng lookup_exact() (Redis GET thẳng theo key hash), KHÔNG dùng lookup()
    (semantic, phải tính embedding CPU-bound mỗi lần) — bug thật tự gây ra +
    tự sửa cùng phiên: nhiều follower poll bằng lookup() làm nghẽn CPU 100%,
    toàn bộ request timeout dưới tải dù server không hề deadlock (xem
    BENCHMARK_RESULTS.md mục 21b)."""
    waited = 0.0
    while waited < max_wait:
        await asyncio.sleep(poll_interval)
        waited += poll_interval
        cached = cache.lookup_exact(query)
        if cached is not None:
            return cached
    return None


async def _persist_memory(
    customer_id: str, user_msg: str, assistant_msg: str, tool_calls: list[dict]
):
    memory_store.append_turn(customer_id, "user", user_msg)
    memory_store.append_turn(customer_id, "assistant", assistant_msg)
    for call in tool_calls:
        memory_store.update_preferences(
            customer_id, call.get("tool", ""), call.get("input", {})
        )


# ------------------------------------------------------------------
# Search (direct RAG, no LLM)
# ------------------------------------------------------------------


@app.get("/search", response_model=list[SearchResult])
async def search(
    q: str,
    max_price: Optional[float] = None,
    min_price: Optional[float] = None,
    top_k: int = 10,
):
    loop = asyncio.get_running_loop()
    if max_price is None and min_price is None:
        cached = await loop.run_in_executor(None, semantic_cache.lookup, q)
        if cached:
            search_cache_hit_total.labels(type="semantic").inc()
            return cached
    search_cache_hit_total.labels(type="miss").inc()
    results = rag_pipeline.search(
        q, max_price=max_price, min_price=min_price, top_k=top_k
    )
    if max_price is None and min_price is None:
        await loop.run_in_executor(None, semantic_cache.store, q, results)
    return results


# ------------------------------------------------------------------
# Feedback
# ------------------------------------------------------------------


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    """Receive user feedback (click/purchase/ignore) and publish to Kafka."""
    event = {
        "customer_id": req.customer_id,
        "product_id": req.product_id,
        "action": req.action,
        "session_id": req.session_id,
        "source": req.source,
        "ts": int(time.time()),
    }
    _feedback_producer.produce(
        topic="agent_feedback",
        key=req.customer_id,
        value=json.dumps(event, ensure_ascii=False).encode(),
    )
    # poll(0) không chặn — chỉ flush() lúc process tắt (atexit ở trên),
    # không phải mỗi request, để giữ được batching thật của producer.
    _feedback_producer.poll(0)
    logger.info(
        "feedback_received",
        extra={"customer_id": req.customer_id, "action": req.action},
    )
    return {"status": "ok"}


# ------------------------------------------------------------------
# Health & Metrics
# ------------------------------------------------------------------


@app.get("/health")
async def health():
    checks: dict = {"status": "ok", "rag": "ok", "memory": "unknown", "kb": "empty"}
    try:
        memory_store._r.ping()
        checks["memory"] = "ok"
    except Exception:
        checks["memory"] = "unavailable"
    if kb_indexer and kb_indexer.size > 0:
        checks["kb"] = f"ok ({kb_indexer.size} chunks)"
    return checks


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ------------------------------------------------------------------
# Admin — KB management
# ------------------------------------------------------------------


@app.post("/admin/kb/upload", dependencies=[Depends(require_admin_api_key)])
async def kb_upload(bg: BackgroundTasks, file: UploadFile = File(...)):

    data = await file.read()
    s3 = S3KBLoader()
    key = s3.upload_bytes(
        data, file.filename, file.content_type or "application/octet-stream"
    )
    bg.add_task(_reindex_kb_bg)
    return {"status": "ok", "key": key, "message": "Reindex started in background"}


@app.post("/admin/kb/reindex", dependencies=[Depends(require_admin_api_key)])
async def kb_reindex(bg: BackgroundTasks):
    bg.add_task(_reindex_kb_bg)
    return {"status": "accepted", "message": "Reindex started in background"}


@app.post("/admin/kb/reindex-webhook", dependencies=[Depends(require_admin_api_key)])
async def kb_reindex_webhook(request: Request, bg: BackgroundTasks):
    body = await request.json()
    records = body.get("Records", body.get("records", []))
    put_events = [r for r in records if "ObjectCreated" in r.get("eventName", "")]
    if not put_events:
        return {"status": "skip", "reason": "no ObjectCreated event"}
    bg.add_task(_trigger_airflow_or_reindex)
    return {"status": "ok", "events": len(put_events)}


async def _trigger_airflow_or_reindex() -> None:

    airflow_url = os.getenv("AIRFLOW_API_URL", "http://airflow-webserver:8080")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{airflow_url}/api/v1/dags/kb_reindex_pipeline/dagRuns",
                auth=(
                    os.getenv("AIRFLOW_USER", "airflow"),
                    os.getenv("AIRFLOW_PASSWORD", "airflow"),
                ),
                json={"conf": {"triggered_by": "minio_webhook"}},
            )
            resp.raise_for_status()
    except Exception as exc:
        logger.warning(
            "airflow_trigger_failed (%s) — falling back to direct reindex", exc
        )
        await asyncio.to_thread(_reindex_kb_sync)


def _reindex_kb_sync() -> None:
    global kb_indexer

    try:
        new_indexer = run_ingestion()
        kb_indexer = new_indexer
        set_kb_indexer(new_indexer)
        logger.info("kb_reindex_done chunks=%d", new_indexer.size)
    except Exception as exc:
        logger.error("kb_reindex_failed: %s", exc)


async def _reindex_kb_bg() -> None:

    await asyncio.to_thread(_reindex_kb_sync)
