import asyncio
import hashlib
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial

import pandas as pd
import redis.asyncio as redis_lib
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from feast import FeatureStore
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from pythonjsonlogger import jsonlogger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.serving.recsys_api.cache import (
    cache_recommendations,
    get_cached_recommendations,
    redis_client,
)
from src.serving.recsys_api.inference_faiss import recommend
from src.serving.recsys_api.kafka_producer import (
    producer as kafka_producer,
)
from src.serving.recsys_api.kafka_producer import (
    send_prediction_event,
)
from src.serving.recsys_api.loaders import (
    ALL_ITEM_IDS,
    LOAD_ERROR,
    READY,
    device,
    item_lookup_df,
)

logger = logging.getLogger(__name__)

# =========================================================
# JSON STRUCTURED LOGGING
# =========================================================


def _setup_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        jsonlogger.JsonFormatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    )
    logging.root.handlers = [handler]
    logging.root.setLevel(logging.INFO)


_setup_logging()

# =========================================================
# FEATURE STORE
# =========================================================

fs = FeatureStore(repo_path="src/feature_store/feature_repo/")

# =========================================================
# GRACEFUL SHUTDOWN (lifespan)
# =========================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: model + artifacts already loaded at module level (loaders.py)
    # Tăng thread pool mặc định của event loop — asyncio default là
    # min(32, os.cpu_count()+4), tức chỉ 8 thread trên máy 4 vCPU (g4dn.xlarge).
    # run_in_executor (Feast fetch, recommend(), SASRec) dùng chung pool này —
    # dưới tải đồng thời cao (>32 request cùng lúc cần executor), request phải
    # xếp hàng chờ thread trống dù bản thân từng call rất nhanh (~ms), gây
    # P95 tăng vọt (đo được thật: 1400+ user P95 vượt 200ms — benchmark GPU
    # 16/7/2026). Phần lớn call là I/O-bound (Redis/Feast, không giữ CPU khi
    # chờ), nên tăng số thread vượt số core vẫn có lợi thật, không chỉ lý
    # thuyết — cần đo lại để xác nhận trước khi chốt.
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=64))
    yield
    # Shutdown: flush pending OTEL spans and Kafka messages before pod terminates
    try:
        tracer_provider = trace.get_tracer_provider()
        if hasattr(tracer_provider, "shutdown"):
            tracer_provider.shutdown()
        logger.info("otel_spans_flushed_on_shutdown")
    except Exception as exc:
        logger.warning("otel_shutdown_failed", extra={"error": str(exc)})
    try:
        kafka_producer.flush(timeout=10)
        logger.info("kafka_producer_flushed_on_shutdown")
    except Exception as exc:
        logger.warning("kafka_flush_failed", extra={"error": str(exc)})


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="E-commerce Recommendation API",
    description="API gợi ý Top K sản phẩm sử dụng mô hình Two-Tower Deep Learning",
    version="1.0.0",
    lifespan=lifespan,
)

# =========================================================
# PROMETHEUS CUSTOM METRICS
# =========================================================

RECOMMENDATION_LATENCY = Histogram(
    "recommendation_latency_seconds",
    "End-to-end recommendation latency",
    ["source"],
)
COLD_START_COUNTER = Counter(
    "cold_start_total",
    "Total cold-start recommendations served",
)

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_code"],
)

# =========================================================
# OPENTELEMETRY TRACING
# =========================================================


def _setup_tracing() -> None:
    provider = TracerProvider()
    exporter = OTLPSpanExporter(
        endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4317"),
        insecure=True,
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)


_setup_tracing()

# =========================================================
# TRACE-ID MIDDLEWARE
# =========================================================


@app.middleware("http")
async def add_trace_id(request: Request, call_next):
    trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))
    request.state.trace_id = trace_id
    t0 = time.time()
    response = await call_next(request)
    HTTP_REQUESTS_TOTAL.labels(
        method=request.method,
        path=request.url.path,
        status_code=str(response.status_code),
    ).inc()
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


# =========================================================
# AUTH + RATE LIMIT
#
# RECSYS_API_KEY không set (mặc định) = auth tắt, để dev local không cần
# đặt key vẫn chạy được như trước. Đặt biến này trong prod.env/staging.env
# để bật auth trước khi expose ra internet.
#
# Rate limit dùng lại chính pattern Redis INCR đã có trong
# agent_api/guardrails.py (key theo request, window 60s, expire tự động) —
# không bịa cơ chế khác cho nhất quán giữa các service.
# =========================================================

RECSYS_API_KEY = os.getenv("RECSYS_API_KEY")
_RATE_LIMIT_PER_MIN = int(os.getenv("RECSYS_RATE_LIMIT_PER_MIN", "60"))
_PUBLIC_PATHS = {"/health", "/ready", "/metrics"}

# db=3 — cùng "concern" rate-limit với agent_api/guardrails.py và nlp_api,
# tách khỏi db=0 (recommendation cache) theo P2-7. Rate-limit không nên
# chung namespace với cache: 2 concern khác nhau, key pattern khác nhau.
_rate_limit_redis = redis_lib.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD") or None,
    db=3,
    decode_responses=True,
    socket_connect_timeout=1,
    socket_timeout=1,
)


def _client_identity(request: Request) -> str:
    if os.getenv("TRUST_PROXY_HEADERS", "false").lower() == "true":
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.middleware("http")
async def auth_and_rate_limit(request: Request, call_next):
    if request.url.path in _PUBLIC_PATHS:
        return await call_next(request)

    if RECSYS_API_KEY:
        provided_key = request.headers.get("X-API-Key")
        if provided_key != RECSYS_API_KEY:
            return JSONResponse(
                status_code=401, content={"detail": "Missing or invalid X-API-Key"}
            )
        rate_key = f"ratelimit:apikey:{provided_key}"
    else:
        rate_key = f"ratelimit:ip:{_client_identity(request)}"

    try:
        count = await _rate_limit_redis.incr(rate_key)
        if count == 1:
            await _rate_limit_redis.expire(rate_key, 60)
        if count > _RATE_LIMIT_PER_MIN:
            return JSONResponse(
                status_code=429, content={"detail": "Rate limit exceeded"}
            )
    except Exception:
        pass  # Redis down — fail open, cùng trade-off với guardrails.py

    return await call_next(request)


# =========================================================
# A/B EXPERIMENT ROUTING
# =========================================================


def _get_experiment_group(customer_id: str) -> str:
    """Deterministic hash-based A/B split: ~10% experiment, ~90% control."""
    hash_val = int(hashlib.md5(customer_id.encode()).hexdigest(), 16)
    return "experiment" if (hash_val % 100) < 10 else "control"


# =========================================================
# FEAST FEATURE FETCH WITH RETRY
# =========================================================


@retry(
    stop=stop_after_attempt(2), wait=wait_exponential(min=0.1, max=0.5), reraise=False
)
def _fetch_feast_features(customer_id: str) -> dict:
    return fs.get_online_features(
        entity_rows=[{"customer_id": customer_id}],
        features=[
            "customer_recent_sentiment:recent_sentiment_score",
            "customer_recent_sentiment:last_commented_product_id",
        ],
    ).to_dict()


# =========================================================
# REQUEST SCHEMA
# =========================================================


class RecommendRequest(BaseModel):
    customer_id: str
    top_k: int = Field(default=10, ge=1, le=100)


# =========================================================
# ROUTES
# =========================================================


@app.post("/recommend")
async def get_recommendations(
    request: RecommendRequest, http_request: Request, background_tasks: BackgroundTasks
):
    if not READY:
        raise HTTPException(
            status_code=503,
            detail=f"Model artifacts not loaded yet: {LOAD_ERROR}",
        )
    t0 = time.time()
    customer_id = str(request.customer_id).replace(".0", "")
    top_k = request.top_k
    trace_id = getattr(http_request.state, "trace_id", str(uuid.uuid4()))

    # ─── Redis cache ──────────────────────────────────────
    cached_result = await get_cached_recommendations(customer_id)
    if cached_result:
        return {
            "status": "success",
            "source": "redis_cache",
            "recommendations": cached_result,
        }

    # ─── A/B group ───────────────────────────────────────
    experiment_group = _get_experiment_group(customer_id)
    diversity_limit = 4 if experiment_group == "experiment" else 5

    # ─── Feast online features (retry + fallback to defaults) ───
    # run_in_executor: fs.get_online_features() là call SYNC của Feast SDK
    # (không có async API ở bản 0.42.0) — gọi trực tiếp trong route async def
    # sẽ chặn nguyên event loop cho tới khi xong (bao gồm cả round-trip Redis
    # online store). Chạy trong thread pool riêng để các request khác vẫn được
    # xử lý song song trong lúc chờ I/O này.
    loop = asyncio.get_running_loop()
    recent_score = 0.0
    last_product = None
    try:
        online_features = await loop.run_in_executor(
            None, _fetch_feast_features, customer_id
        )
        recent_score = online_features.get("recent_sentiment_score", [0.0])[0] or 0.0
        last_product = online_features.get("last_commented_product_id", [None])[0]
        if last_product is not None:
            last_product = str(last_product)
    except Exception as e:
        logger.warning(
            "Feast unavailable after retries, using defaults", extra={"error": str(e)}
        )

    # ─── Inference ───────────────────────────────────────
    # recommend() gồm cả candidate_cache (Redis sync, candidate_cache.py) +
    # forward pass model (CPU/GPU) + FAISS + rerank — cùng lý do trên, chạy
    # trong thread pool để không chặn event loop dưới tải đồng thời cao.
    result = await loop.run_in_executor(
        None,
        partial(
            recommend,
            customer_id=customer_id,
            top_k=top_k,
            recent_sentiment_score=recent_score,
            last_commented_product_id=last_product,
            diversity_limit=diversity_limit,
        ),
    )

    source = result.get("source", "model")
    latency = time.time() - t0

    # ─── Prometheus metrics ───────────────────────────────
    RECOMMENDATION_LATENCY.labels(source=source).observe(latency)
    if source == "trending":
        COLD_START_COUNTER.inc()

    logger.info(
        "recommendation_served",
        extra={
            "trace_id": trace_id,
            "customer_id": customer_id,
            "source": source,
            "experiment_group": experiment_group,
            "num_recs": len(result.get("recommendations", [])),
            "latency_ms": round(latency * 1000),
        },
    )

    # ─── Background: Kafka publish + Redis cache (non-blocking) ──
    event = {
        "customer_id": customer_id,
        "timestamp": str(pd.Timestamp.utcnow()),
        "top_k": top_k,
        "recommendations": [r["product_id"] for r in result["recommendations"]],
        "scores": [float(r["predict_score"]) for r in result["recommendations"]],
        "experiment_group": experiment_group,
        "source": source,
    }
    background_tasks.add_task(send_prediction_event, event)
    background_tasks.add_task(
        cache_recommendations, customer_id, result["recommendations"]
    )

    return result


# =========================================================
# SASREC SESSION-BASED RECOMMENDATION
# Lazy-loaded at first request; skipped if model file not found.
# =========================================================

_sasrec_model = None
_sasrec_item_enc: dict | None = None  # item_id (str) → index (int)
_sasrec_item_dec: dict | None = None  # index (int) → item_id (str)
_SASREC_PATH = os.getenv("SASREC_MODEL_PATH", "/app/artifacts/recsys_models/sasrec.pt")
_SASREC_ENC_PATH = os.getenv(
    "SASREC_ENC_PATH", "/app/artifacts/recsys_models/sasrec_item_enc.json"
)
_MAX_SEQ = 50


def _load_sasrec():
    global _sasrec_model, _sasrec_item_enc, _sasrec_item_dec
    if _sasrec_model is not None:
        return True
    import json
    from pathlib import Path

    from src.ml_models.recsys.models.sasrec import SASRec

    if not Path(_SASREC_PATH).exists() or not Path(_SASREC_ENC_PATH).exists():
        logger.warning(
            "SASRec model not found at %s — /recommend/session unavailable",
            _SASREC_PATH,
        )
        return False
    try:
        _sasrec_item_enc = json.loads(Path(_SASREC_ENC_PATH).read_text())
        _sasrec_item_dec = {v: k for k, v in _sasrec_item_enc.items()}
        n_items = len(_sasrec_item_enc)
        model = SASRec(n_items=n_items, max_seq_len=_MAX_SEQ).to(device)
        import torch

        model.load_state_dict(torch.load(_SASREC_PATH, map_location=device))
        model.eval()
        _sasrec_model = model
        logger.info("SASRec loaded: %d items", n_items)
        return True
    except Exception as exc:
        logger.error("SASRec load failed: %s", exc)
        return False


class SessionRecommendRequest(BaseModel):
    session_items: list[str] = Field(
        ..., description="Ordered list of item_ids in current session (oldest → newest)"
    )
    top_k: int = Field(default=10, ge=1, le=100)


def _run_sasrec_inference(session_items: list[str], top_k: int) -> list[dict]:
    """Phần compute thuần (torch forward pass + numpy sort + pandas lookup) —
    tách riêng để chạy qua run_in_executor, không chặn event loop."""
    import numpy as np
    import torch

    # Encode session items (unknown items mapped to 0 = padding)
    item_indices = [_sasrec_item_enc.get(iid, 0) for iid in session_items]
    item_indices = item_indices[-_MAX_SEQ:]
    padded = [0] * (_MAX_SEQ - len(item_indices)) + item_indices

    seq_t = torch.tensor([padded], dtype=torch.long, device=device)
    all_candidates = torch.arange(1, len(_sasrec_item_enc) + 1, device=device)

    with torch.no_grad():
        scores = _sasrec_model.predict_next(seq_t, all_candidates)[0].cpu().numpy()

    # Top-K by score, exclude items already in session
    session_set = set(item_indices)

    ranked = [
        (int(all_candidates[i].item()), float(scores[i]))
        for i in np.argsort(-scores)
        if int(all_candidates[i].item()) not in session_set
    ][:top_k]

    # Decode to product_ids + enrich with item_lookup
    recommendations = []
    for idx, score in ranked:
        product_id = _sasrec_item_dec.get(idx, str(idx))
        row = item_lookup_df[item_lookup_df["product_id"] == product_id]
        if row.empty:
            rec = {"product_id": product_id, "score": score}
        else:
            r = row.iloc[0]
            rec = {
                "product_id": product_id,
                "product_name": str(r.get("product_name", "")),
                "price": float(r.get("price", 0)),
                "category_name": str(r.get("category_name", "")),
                "score": score,
            }
        recommendations.append(rec)

    return recommendations


@app.post("/recommend/session")
async def recommend_session(request: SessionRecommendRequest):
    """
    Session-based recommendation using SASRec.

    Takes the user's current browsing session (ordered item IDs) and predicts
    the most likely next items. No customer_id required — cold-start friendly.
    """
    if not _load_sasrec():
        return {"error": "SASRec model not available", "recommendations": []}

    loop = asyncio.get_running_loop()
    recommendations = await loop.run_in_executor(
        None, _run_sasrec_inference, request.session_items, request.top_k
    )

    return {
        "session_length": len(request.session_items),
        "recommendations": recommendations,
        "model": "sasrec",
    }


@app.get("/health")
async def health():
    """Liveness — chỉ báo process còn chạy, không phụ thuộc Redis/Feast/artifact.
    Luôn nhẹ và nhanh để K8s không nhầm 'đang khởi động' với 'chết hẳn'."""
    return {"status": "alive"}


@app.get("/ready")
async def ready(http_response: Response):
    """Readiness — có thật sự sẵn sàng nhận traffic không. Trả 503 nếu
    artifact chưa load xong (READY=False, xem loaders.py) — khác /health ở
    chỗ đây LÀ nơi phụ thuộc bên ngoài (Redis, Feast) được kiểm tra."""
    checks: dict = {
        "artifacts": "ok" if READY else f"not_ready: {LOAD_ERROR}",
        "device": str(device),
        "num_items": len(ALL_ITEM_IDS),
    }
    try:
        await redis_client.ping()
        checks["redis"] = "ok"
    except Exception:
        checks["redis"] = "unavailable"
    try:
        _fetch_feast_features("health_check")
        checks["feast"] = "ok"
    except Exception:
        checks["feast"] = "unavailable"

    if not READY:
        http_response.status_code = 503
    return checks


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
