"""
RAGAS Evaluation Pipeline for Agent RAG

Runs the product (45) + policy (30) queries from eval_dataset.py — 75 total —
through the appropriate retriever (RAGPipeline for product, KBIndexer for
policy), evaluates with RAGAS metrics (faithfulness, answer_relevancy,
context_precision), and logs scores to MLflow per prompt version.

edge_case/adversarial queries are NOT included here — there is no single
"correct retrieval" to score them against with these metrics (see
eval_dataset.py docstring). They're covered by deepeval_eval.py instead.

Usage:
    python -m src.serving.agent_api.eval.ragas_eval --prompt-version 1
    RERANKER_BACKEND=neural python -m src.serving.agent_api.eval.ragas_eval
"""

import argparse
import logging
import os
import re
import sys

import mlflow
from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import HuggingfaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, faithfulness
from ragas.run_config import RunConfig

from src.serving.agent_api.eval.eval_dataset import POLICY_QUERIES, PRODUCT_QUERIES
from src.serving.agent_api.indexer import KBIndexer
from src.serving.agent_api.rag import RAGPipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RAGASEval")

TEST_QUERIES = PRODUCT_QUERIES + POLICY_QUERIES

_KB_BOILERPLATE_RE = re.compile(
    r"^#.*?\n+Nguồn:.*?\n+TIKI.*?\n+GỬI YÊU CẦU\n+Trang chủ.*?\n+.*?\n+\s*"
    r"Cập nhật lần cuối:.*?\nLượt xem:\s*\d+\s*\n+",
    re.DOTALL,
)


def _clean_kb_text(text: str) -> str:
    cleaned = _KB_BOILERPLATE_RE.sub("", text, count=1).strip()
    return cleaned or text.strip()


def build_ragas_dataset(rag, kb_indexer=None, top_k: int = 5) -> list[dict]:
    samples = []
    for item in TEST_QUERIES:
        query = item["query"]
        try:
            if item["category"] == "policy":
                if kb_indexer is None:
                    logger.warning(f"No kb_indexer provided, skipping '{query}'")
                    continue
                chunks = kb_indexer.search(query, top_k=top_k)
                context_str = (
                    "\n".join(f"- {_clean_kb_text(c['text'])}" for c in chunks)
                    if chunks
                    else "Không tìm thấy tài liệu phù hợp."
                )
                answer = (
                    _clean_kb_text(chunks[0]["text"]) if chunks else "Không tìm thấy"
                )
            else:
                products = rag.search(query, top_k=top_k)
                context_str = (
                    "\n".join(
                        f"- {p['product_name']} ({p['category_name']}) giá "
                        f"{p['price']:,.0f}đ"
                        for p in products
                    )
                    if products
                    else "Không tìm thấy sản phẩm phù hợp."
                )
                answer = (
                    f"{products[0]['product_name']} ({products[0]['category_name']}) "
                    f"giá {products[0]['price']:,.0f}đ"
                    if products
                    else "Không tìm thấy"
                )
        except Exception as e:
            logger.warning(f"Search failed for '{query}': {e}")
            context_str = "Không tìm thấy."
            answer = "Không tìm thấy"

        samples.append(
            {
                "question": query,
                "contexts": [context_str],
                "answer": answer,
                "ground_truth": item["expected_topic"] or item["expected_answer"] or "",
                "category": item["category"],
            }
        )

    return samples


def run_ragas(samples: list[dict], ragas_llm=None, ragas_embeddings=None) -> dict:
    categories = [s.pop("category") for s in samples]
    dataset = Dataset.from_list(samples)

    kwargs = {
        "dataset": dataset,
        "metrics": [faithfulness, answer_relevancy, context_precision],
        "run_config": RunConfig(
            timeout=1200, max_workers=int(os.getenv("RAGAS_MAX_WORKERS", "2"))
        ),
    }
    if ragas_llm:
        kwargs["llm"] = ragas_llm
    if ragas_embeddings:
        kwargs["embeddings"] = ragas_embeddings

    result = evaluate(**kwargs)

    df = result.to_pandas()
    df["category"] = categories
    breakdown = {}
    for cat in sorted(set(categories)):
        sub = df[df["category"] == cat]
        breakdown[cat] = {
            m: round(float(sub[m].mean()), 4)
            for m in ("faithfulness", "answer_relevancy", "context_precision")
            if m in sub.columns
        }
    logger.info("Per-category breakdown: %s", breakdown)

    return {
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "context_precision": float(result["context_precision"]),
        "breakdown": breakdown,
    }


def build_ragas_llm():
    """Build LLM judge — Claude Haiku if ANTHROPIC_API_KEY set, otherwise fall
    back to the local Ollama model (free, cùng model agent dùng ở graph.py)
    thay vì để RAGAS rơi về default LLM (thường là OpenAI, cũng cần trả phí
    và cần key khác không có sẵn) — không có judge nào coi như eval fail."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        try:
            from langchain_community.chat_models import ChatAnthropic

            llm = LangchainLLMWrapper(
                ChatAnthropic(
                    model="claude-haiku-4-5-20251001", anthropic_api_key=api_key
                )
            )
            logger.info("RAGAS LLM judge: Claude Haiku")
            return llm
        except ImportError:
            logger.warning("langchain-community not installed, falling back to Ollama")

    logger.info(
        "ANTHROPIC_API_KEY not set — RAGAS LLM judge: local Ollama (%s) thay vì "
        "rơi về default LLM (thường cần OpenAI key, cũng phải trả phí)",
        os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
    )
    from langchain_community.chat_models import ChatOllama

    llm = LangchainLLMWrapper(
        ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
            base_url=os.getenv("OLLAMA_URL", "http://ollama:11434"),
            temperature=0.0,
            num_predict=int(os.getenv("RAGAS_JUDGE_NUM_PREDICT", "256")),
        )
    )
    return llm


def build_ragas_embeddings():
    model_name = os.getenv("EMBEDDING_MODEL", "dangvantuan/vietnamese-embedding")
    logger.info("RAGAS embeddings: %s (local, free)", model_name)
    return HuggingfaceEmbeddings(model_name=model_name)


def log_to_mlflow(scores: dict, prompt_version: str, reranker_backend: str) -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("ragas_rag_eval")

    with mlflow.start_run(run_name=f"ragas_v{prompt_version}_{reranker_backend}"):
        mlflow.log_params(
            {
                "prompt_version": prompt_version,
                "reranker_backend": reranker_backend,
                "n_queries": len(TEST_QUERIES),
            }
        )
        mlflow.log_metrics({f"ragas_{k}": v for k, v in scores.items()})

    logger.info(
        f"Logged to MLflow: experiment=ragas_rag_eval run=ragas_v{prompt_version}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAGAS evaluation for Agent RAG pipeline"
    )
    parser.add_argument(
        "--prompt-version",
        default=os.getenv("AGENT_PROMPT_VERSION", "1"),
        dest="prompt_version",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Products retrieved per query"
    )
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    args = parser.parse_args()

    reranker_backend = os.getenv("RERANKER_BACKEND", "rule")
    data_path = os.getenv(
        "ITEM_LOOKUP_PATH", "/app/artifacts/recsys_models/data_menu/item_lookup.parquet"
    )

    logger.info(f"Initializing RAGPipeline from {data_path}")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))
    rag = RAGPipeline(data_path=data_path)
    try:
        kb_indexer = KBIndexer.load()
    except Exception as e:
        logger.warning(f"KBIndexer.load() failed ({e}), policy queries will be skipped")
        kb_indexer = None

    logger.info(
        f"Building RAGAS dataset ({len(PRODUCT_QUERIES)} product + "
        f"{len(POLICY_QUERIES)} policy = {len(TEST_QUERIES)} queries, "
        f"top_k={args.top_k})"
    )
    samples = build_ragas_dataset(rag, kb_indexer=kb_indexer, top_k=args.top_k)

    ragas_llm = build_ragas_llm()
    ragas_embeddings = build_ragas_embeddings()

    logger.info("Running RAGAS evaluation...")
    try:
        scores = run_ragas(
            samples, ragas_llm=ragas_llm, ragas_embeddings=ragas_embeddings
        )
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        raise

    breakdown = scores.pop("breakdown", {})

    print("\n=== RAGAS Results (tổng, product+policy gộp) ===")
    for metric, score in scores.items():
        print(f"  {metric}: {score:.4f}")
    print(f"  prompt_version: {args.prompt_version}")
    print(f"  reranker_backend: {reranker_backend}")

    print("\n=== Breakdown theo category ===")
    for cat, cat_scores in breakdown.items():
        print(f"  [{cat}]")
        for metric, score in cat_scores.items():
            print(f"    {metric}: {score:.4f}")

    if not args.no_mlflow:
        log_to_mlflow(scores, args.prompt_version, reranker_backend)
