"""
DeepEval evaluation pipeline for the shopping agent.

Evaluates RAG answer quality using Confident AI's DeepEval framework:
  - AnswerRelevancyMetric: does the response answer the query?
  - FaithfulnessMetric: is the response grounded in retrieved context?
  - ContextualRecallMetric: does context contain the expected answer?

DeepEval vs RAGAS (ragas_eval.py):
  - Both measure similar axes but with different LLM judges
  - DeepEval has tighter Vietnamese-language support via custom model
  - Run both to triangulate score stability

IMPORTANT — this used to call `run_agent` from `agent.py`, which is dead code:
`main.py` has called `run_graph_stream` from `graph.py` (the StateGraph/Router
rewrite) since the LangGraph refactor, and no longer imports `agent.py` at
all. That meant this eval was silently scoring an implementation that isn't
the one actually running in production. Fixed to call `run_graph_stream`
through the real compiled graph instead.

Test set: product (45) + policy (30) + edge_case (15) = 90 cases from
eval_dataset.py, run through the real agent graph. adversarial (10) is
scored separately in `check_adversarial_robustness()` below — those aren't
RAG quality questions, they're a block-rate check against
Guardrails/PolicyEngine.

Usage:
    python -m src.serving.agent_api.eval.deepeval_eval
    python -m src.serving.agent_api.eval.deepeval_eval --prompt-version v2

Logs scores to MLflow experiment 'agent_eval_deepeval'.
"""

import argparse
import asyncio
import json
import logging
import os

import litellm
import mlflow
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

from src.serving.agent_api.eval.eval_dataset import (
    ADVERSARIAL_QUERIES,
    EDGE_CASE_QUERIES,
    POLICY_QUERIES,
    PRODUCT_QUERIES,
)
from src.serving.agent_api.graph import build_graph, run_graph_stream
from src.serving.agent_api.guardrails import Guardrails
from src.serving.agent_api.indexer import KBIndexer
from src.serving.agent_api.policy_engine import PolicyEngine
from src.serving.agent_api.rag import RAGPipeline
from src.serving.agent_api.tools import set_kb_indexer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Test dataset — Vietnamese shopping queries with expected answers
# ---------------------------------------------------------------------------

TEST_CASES_RAW = PRODUCT_QUERIES + POLICY_QUERIES + EDGE_CASE_QUERIES

# Phrases that indicate a graceful decline/clarification rather than a
# confident (possibly hallucinated) answer — used only for edge_case cases,
# which have no single correct retrieval to grade against.
_DECLINE_MARKERS = (
    "không tìm thấy",
    "không có thông tin",
    "không rõ",
    "bạn có thể cho biết",
    "bạn muốn hỏi về",
    "ngoài phạm vi",
    "không thể",
    "xin lỗi",
    "vui lòng cung cấp",
    "rõ hơn",
)


# ---------------------------------------------------------------------------
# LiteLLM bridge for DeepEval (uses local Ollama or Claude)
# ---------------------------------------------------------------------------


class LiteLLMJudge(DeepEvalBaseLLM):
    """Wraps litellm.completion so DeepEval uses our existing LLM setup."""

    def __init__(self):
        backend = os.getenv("AGENT_LLM_BACKEND", "ollama")
        self._api_base = None
        if backend == "claude":
            model = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
            self._model_id = f"anthropic/{model}"
        elif backend == "vllm":
            model = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
            self._api_base = os.getenv("VLLM_URL", "http://localhost:8000/v1")
            self._model_id = f"openai/{model}"
        else:
            ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
            ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434")
            os.environ["OLLAMA_API_BASE"] = ollama_url
            self._model_id = f"ollama/{ollama_model}"

    def load_model(self):
        return self._model_id

    def _build_kwargs(self, schema=None) -> dict:
        kwargs = {}
        if self._api_base:
            kwargs["api_base"] = self._api_base
            kwargs["api_key"] = os.getenv("VLLM_API_KEY", "not-needed")
            # Bug that 19/7/2026 - 3 vong lien tiep:
            # Vong 1: Qwen2.5-7B-AWQ (judge qua vLLM) thinh thoang sinh JSON
            # khong hop le ve CU PHAP. Fix: response_format json_object - het
            # loi cu phap, nhung...
            # Vong 2: JSON hop le cu phap roi nhung SAI TEN FIELD (thieu key
            # "claims") - vi base class DeepEvalBaseLLM.a_generate_with_schema()
            # goi self.a_generate(prompt, schema=schema) nhung ham cu khong
            # nhan schema, TypeError bi nuot, fallback goi khong kem schema.
            # Fix: nhan schema (Pydantic model DeepEval dinh nghia rieng cho
            # tung buoc - Claims/Truths/Verdicts/...), thu ep qua
            # extra_body={"guided_json": ...} (cu phap guided-decoding CU cua
            # vLLM), nhung...
            # Vong 3: VAN loi y het - vi vllm 0.25.1 (ban moi, khong pin
            # version) da chuyen sang chuan OpenAI Structured Outputs
            # (response_format type=json_schema) thay vi truong extra_body
            # guided_json rieng cua vLLM cu - extra_body khong con duoc
            # nhan dien nua. Fix dung: dung response_format json_schema
            # chuan OpenAI (litellm forward chac chan hon qua provider
            # "openai/", khong phu thuoc extra_body co song sot qua cac
            # lop transform hay khong).
            if schema is not None:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": schema.model_json_schema(),
                        "strict": False,
                    },
                }
            else:
                kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def generate(self, prompt: str, schema=None) -> str:
        resp = litellm.completion(
            model=self._model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
            **self._build_kwargs(schema=schema),
        )
        return resp.choices[0].message.content or ""

    async def a_generate(self, prompt: str, schema=None) -> str:
        # Bug that 19/7/2026: ban cu goi thang self.generate() (sync, blocking)
        # tu trong 1 coroutine async - chan nguyen event loop moi lan judge
        # cham diem, pha han tinh dong thoi that cua AsyncConfig(max_concurrent)
        # va gop phan vao cascade loi "Task was destroyed"/"ContextVar created
        # in a different Context" khi 1 task loi giua chung lam hong luon cac
        # task khac dang bi chan chung event loop. Dung litellm.acompletion()
        # (that su async, non-blocking) de nhieu test case chay dong thoi dung
        # nghia, khong xep hang an sau lung 1 thread.
        resp = await litellm.acompletion(
            model=self._model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
            **self._build_kwargs(schema=schema),
        )
        return resp.choices[0].message.content or ""

    def get_model_name(self) -> str:
        return self._model_id


# ---------------------------------------------------------------------------
# Run one query through the REAL compiled agent graph (graph.py, the same
# one main.py serves /chat/stream through) and collect the full response +
# whatever tool outputs were retrieved along the way.
# ---------------------------------------------------------------------------


async def _run_one_query(graph, rag_pipeline, query: str) -> tuple[str, list[str]]:
    full_response = ""
    contexts: list[str] = []
    async for event in run_graph_stream(
        customer_id="eval_user",
        message=query,
        history=[],
        pref_context="",
        rag_pipeline=rag_pipeline,
        graph=graph,
    ):
        if event["type"] == "done":
            full_response = event.get("full_response", "")
            contexts = [
                tc["output"] for tc in event.get("tool_calls", []) if tc.get("output")
            ]
    return full_response, contexts


# ---------------------------------------------------------------------------
# Build test cases (product + policy) by running the real agent on each query
# ---------------------------------------------------------------------------


async def _build_test_cases_async(
    rag_pipeline, graph, limit: int | None = None
) -> list[LLMTestCase]:
    test_cases: list[LLMTestCase] = []
    queries = PRODUCT_QUERIES + POLICY_QUERIES
    if limit is not None:
        queries = queries[:limit]
    for item in queries:
        query = item["query"]
        expected = item["expected_answer"] or item["expected_topic"] or ""

        try:
            actual_output, context = await _run_one_query(graph, rag_pipeline, query)
        except Exception as e:
            logger.warning("eval_run_failed query=%r error=%s", query, e)
            actual_output = ""
            context = []

        # DeepEval bản mới raise MissingTestCaseParamsError nếu actual_output
        # rỗng (cả khi agent lỗi lẫn khi agent trả về chuỗi rỗng thật) — lỗi
        # cứng làm hỏng CẢ BATCH chỉ vì 1 câu, thay vì chỉ chấm điểm thấp riêng
        # câu đó. Thay chuỗi rỗng bằng placeholder rõ ràng để eval chạy hết.
        if not actual_output:
            actual_output = "(agent không trả lời — lỗi hoặc response rỗng)"

        test_cases.append(
            LLMTestCase(
                input=query,
                actual_output=actual_output,
                expected_output=expected,
                retrieval_context=context or ["(không tool nào được gọi)"],
            )
        )

    return test_cases


def _build_test_cases(
    rag_pipeline, graph, limit: int | None = None
) -> list[LLMTestCase]:
    # Chạy TOÀN BỘ query trong 1 event loop duy nhất (asyncio.run() gọi 1 lần
    # cho cả batch), thay vì asyncio.run() riêng cho từng query trong vòng lặp
    # — cách cũ tạo loop mới mỗi query, nhưng graph/model dùng chung giữ một
    # tài nguyên async (client nội bộ của ChatOllama/langgraph) gắn với loop
    # của lần gọi ĐẦU TIÊN, nên mọi query sau đó lỗi "Event loop is closed"
    # ngay khi loop đầu đóng lại.
    return asyncio.run(_build_test_cases_async(rag_pipeline, graph, limit=limit))


# ---------------------------------------------------------------------------
# edge_case (15) — no single correct retrieval exists, so standard RAG
# metrics don't apply. Instead: does the agent decline/ask for clarification
# gracefully, or does it confidently hallucinate an answer?
# ---------------------------------------------------------------------------


async def _check_graceful_decline_async(rag_pipeline, graph) -> tuple[int, list[dict]]:
    declined = 0
    details = []
    for item in EDGE_CASE_QUERIES:
        query = item["query"]
        try:
            actual_output, _ = await _run_one_query(graph, rag_pipeline, query)
        except Exception as e:
            logger.warning("edge_case_run_failed query=%r error=%s", query, e)
            actual_output = ""

        is_decline = any(m in actual_output.lower() for m in _DECLINE_MARKERS)
        declined += int(is_decline)
        details.append(
            {"query": query, "declined": is_decline, "response": actual_output}
        )
    return declined, details


def check_graceful_decline(rag_pipeline, graph) -> dict:
    # Cùng lý do với _build_test_cases: 1 event loop cho cả batch, không tạo
    # loop mới mỗi query.
    declined, details = asyncio.run(_check_graceful_decline_async(rag_pipeline, graph))

    rate = round(declined / len(EDGE_CASE_QUERIES), 4) if EDGE_CASE_QUERIES else 0.0
    logger.info(
        "graceful_decline_rate=%.4f (%d/%d)", rate, declined, len(EDGE_CASE_QUERIES)
    )
    return {
        "graceful_decline_rate": rate,
        "n_edge_cases": len(EDGE_CASE_QUERIES),
        "details": details,
    }


# ---------------------------------------------------------------------------
# adversarial (10) — these should never reach the LLM at all. Scored as a
# block rate against Guardrails + PolicyEngine, not RAG quality metrics.
# ---------------------------------------------------------------------------


def check_adversarial_robustness() -> dict:
    guardrails = Guardrails()
    policy_engine = PolicyEngine()

    blocked = 0
    details = []
    for item in ADVERSARIAL_QUERIES:
        query = item["query"]
        policy_result = policy_engine.check(query)
        guard_result = (
            guardrails.check_input(query, "eval_user")
            if policy_result.allowed
            else None
        )
        was_blocked = (not policy_result.allowed) or (
            guard_result is not None and not guard_result.allowed
        )
        blocked += int(was_blocked)
        details.append(
            {
                "query": query,
                "blocked": was_blocked,
                "reason": (
                    policy_result.reason
                    if not policy_result.allowed
                    else (guard_result.reason if guard_result else "")
                ),
            }
        )

    rate = round(blocked / len(ADVERSARIAL_QUERIES), 4) if ADVERSARIAL_QUERIES else 0.0
    logger.info(
        "adversarial_block_rate=%.4f (%d/%d)", rate, blocked, len(ADVERSARIAL_QUERIES)
    )
    return {
        "adversarial_block_rate": rate,
        "n_adversarial": len(ADVERSARIAL_QUERIES),
        "details": details,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def run_eval(prompt_version: str = "v1", limit: int | None = None) -> dict:
    """Run DeepEval + graceful-decline + adversarial checks, log to MLflow."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("agent_eval_deepeval")

    rag = RAGPipeline()

    # main.py wires KB indexer vào tools._kb_indexer lúc lifespan startup
    # (main.py:93-98) TRƯỚC khi app nhận request — thiếu bước này thì
    # search_kb() luôn trả "chưa sẵn sàng" bất kể KB thật tốt hay tệ, khiến
    # toàn bộ 30 câu policy chấm sai không phải vì retrieval mà vì thiếu wiring.
    try:
        kb_indexer = KBIndexer.load()
        logger.info("kb_indexer_loaded size=%d", kb_indexer.size)
    except Exception:
        logger.info("kb_indexer_no_index_found, starting empty")
        kb_indexer = KBIndexer()
    set_kb_indexer(kb_indexer)

    graph = build_graph(rag_pipeline=rag)
    judge = LiteLLMJudge()

    metrics = [
        AnswerRelevancyMetric(threshold=0.7, model=judge),
        FaithfulnessMetric(threshold=0.7, model=judge),
        ContextualRecallMetric(threshold=0.7, model=judge),
    ]

    logger.info(
        "Running DeepEval RAG metrics on %d product+policy cases...",
        len(PRODUCT_QUERIES) + len(POLICY_QUERIES),
    )
    test_cases = _build_test_cases(rag, graph, limit=limit)
    results = evaluate(
        test_cases,
        metrics,
        async_config=AsyncConfig(
            max_concurrent=int(os.getenv("DEEPEVAL_MAX_CONCURRENT", "4"))
        ),
    )

    # Aggregate scores
    scores: dict[str, list[float]] = {
        "answer_relevancy": [],
        "faithfulness": [],
        "contextual_recall": [],
    }
    for tc_result in results.test_results:
        for metric_result in tc_result.metrics_data:
            name = metric_result.name.lower().replace(" ", "_")
            if name in scores:
                scores[name].append(metric_result.score or 0.0)

    avg_scores = {k: round(sum(v) / len(v), 4) if v else 0.0 for k, v in scores.items()}
    avg_scores["prompt_version"] = prompt_version
    avg_scores["n_test_cases"] = len(test_cases)

    logger.info(
        "Running graceful-decline check on %d edge_case queries...",
        len(EDGE_CASE_QUERIES),
    )
    decline_result = check_graceful_decline(rag, graph)
    avg_scores["graceful_decline_rate"] = decline_result["graceful_decline_rate"]

    logger.info(
        "Running adversarial robustness check on %d queries...",
        len(ADVERSARIAL_QUERIES),
    )
    adversarial_result = check_adversarial_robustness()
    avg_scores["adversarial_block_rate"] = adversarial_result["adversarial_block_rate"]

    with mlflow.start_run(run_name=f"deepeval_{prompt_version}"):
        mlflow.log_params(
            {
                "prompt_version": prompt_version,
                "n_cases": len(test_cases),
                "n_edge_cases": decline_result["n_edge_cases"],
                "n_adversarial": adversarial_result["n_adversarial"],
            }
        )
        for metric_name, score in avg_scores.items():
            if isinstance(score, (int, float)):
                mlflow.log_metric(metric_name, score)

    logger.info("DeepEval results: %s", avg_scores)
    return avg_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-version", default="v1")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Chỉ chạy N câu đầu (product+policy) — dùng để smoke-test bundle/kết nối "
        "trước khi commit chạy full 75 câu (mất 30-60+ phút), tránh lãng phí GPU quota "
        "nếu bundle/Ollama có lỗi cấu hình.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Ghi scores ra file JSON (mặc định deepeval_results_<prompt-version>.json "
        "trong thư mục hiện tại) — để không mất kết quả nếu console output bị cắt/mất "
        "(vd Kaggle notebook đóng trước khi kịp chụp lại log).",
    )
    args = parser.parse_args()
    scores = run_eval(prompt_version=args.prompt_version, limit=args.limit)
    print("\n=== DeepEval Scores ===")
    for k, v in scores.items():
        print(f"  {k}: {v}")

    out_path = args.output_json or f"deepeval_results_{args.prompt_version}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
    print(f"\nĐã ghi kết quả ra {out_path}")
