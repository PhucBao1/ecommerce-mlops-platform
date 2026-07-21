import logging
import os
import re
import time

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.serving.recsys_api.reranker import LAPTOP_CATEGORIES, PHONE_CATEGORIES
from src.serving.recsys_api.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)

_RERANKER_LATENCY_THRESHOLD = float(os.getenv("RERANKER_LATENCY_THRESHOLD", "0.5"))

_CORE_CATEGORY_KEYWORDS: list[tuple[tuple[str, ...], set[str]]] = [
    (("điện thoại", "smartphone"), PHONE_CATEGORIES),
    (("laptop", "máy tính xách tay"), LAPTOP_CATEGORIES),
]
_ACCESSORY_QUERY_HINTS = (
    "ốp lưng",
    "ốp điện thoại",
    "bao da",
    "sạc dự phòng",
    "sạc điện thoại",
    "cáp sạc",
    "dây sạc",
    "túi đựng",
    "giá đỡ",
    "miếng dán",
    "tai nghe",
    "balo laptop",
    "túi laptop",
    "chuột",
    "bàn phím",
    "lót chuột",
)


def _detect_core_categories(query: str) -> set[str] | None:
    """Trả về set category_id cần lọc nếu query rõ ràng hỏi 1 loại sản phẩm
    cốt lõi (không phải phụ kiện), None nếu không xác định được (giữ nguyên
    hành vi search rộng như cũ)."""
    msg = query.lower()
    if any(hint in msg for hint in _ACCESSORY_QUERY_HINTS):
        return None
    for keywords, cats in _CORE_CATEGORY_KEYWORDS:
        if any(kw in msg for kw in keywords):
            return cats
    return None


_PRICE_PATTERNS = [
    (r"dưới\s*([\d,.]+)\s*triệu", "max", 1_000_000),
    (r"dưới\s*([\d,.]+)\s*tr\b", "max", 1_000_000),
    (r"dưới\s*([\d,.]+)\s*k\b", "max", 1_000),
    (r"không\s*quá\s*([\d,.]+)\s*triệu", "max", 1_000_000),
    (r"không\s*quá\s*([\d,.]+)\s*k\b", "max", 1_000),
    (r"từ\s*([\d,.]+)\s*triệu\s*đến\s*([\d,.]+)\s*triệu", "range", 1_000_000),
    (r"từ\s*([\d,.]+)\s*k\s*đến\s*([\d,.]+)\s*k\b", "range", 1_000),
    (r"khoảng\s*([\d,.]+)\s*triệu", "approx", 1_000_000),
    (r"khoảng\s*([\d,.]+)\s*k\b", "approx", 1_000),
    (r"từ\s*([\d,.]+)\s*triệu", "min", 1_000_000),
    (r"từ\s*([\d,.]+)\s*k\b", "min", 1_000),
    (r"dưới\s*([\d,.]+)\s*(?:đồng|vnd|đ)\b", "max_vnd", 1),
    (r"không\s*quá\s*([\d,.]+)\s*(?:đồng|vnd|đ)\b", "max_vnd", 1),
    (
        r"từ\s*([\d,.]+)\s*(?:(?:đồng|vnd|đ)\s*)?đến\s*([\d,.]+)\s*(?:đồng|vnd|đ)\b",
        "range_vnd",
        1,
    ),
    (r"khoảng\s*([\d,.]+)\s*(?:đồng|vnd|đ)\b", "approx_vnd", 1),
    (r"từ\s*([\d,.]+)\s*(?:đồng|vnd|đ)\b", "min_vnd", 1),
]

_DEFAULT_DATA_PATH = "/app/artifacts/recsys_models/data_menu/item_lookup.parquet"
_DEFAULT_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"


class RAGPipeline:
    def __init__(self, data_path: str | None = None, model_name: str | None = None):
        data_path = data_path or os.getenv("ITEM_LOOKUP_PATH", _DEFAULT_DATA_PATH)
        model_name = model_name or os.getenv("EMBEDDING_MODEL", _DEFAULT_MODEL)
        cache_dir = os.getenv(
            "SENTENCE_TRANSFORMERS_HOME", "/app/model_cache/sentence_transformers"
        )

        logger.info(
            "rag_init_start", extra={"data_path": data_path, "model": model_name}
        )
        self.df = pd.read_parquet(data_path)
        self.df["product_id"] = self.df["product_id"].astype(str)
        self.df = self.df.reset_index(drop=True)

        self._model = SentenceTransformer(model_name, cache_folder=cache_dir)

        self._faiss_index = None
        self._embeddings = None

        # Qdrant for server-side price/category filtering (optional)
        # Separate collection "rag_items" — different embedding space from recsys_api "items"
        self._qdrant_store: VectorStore | None = None
        if os.getenv("VECTOR_STORE_BACKEND") == "qdrant":
            try:
                self._qdrant_store = get_vector_store(
                    backend="qdrant",
                    collection=os.getenv("RAG_QDRANT_COLLECTION", "rag_items"),
                    dim=self._model.get_sentence_embedding_dimension(),
                )
                existing_count = self._qdrant_store.count()
                # Same "non-empty means already indexed" convention as KBIndexer —
                # exact equality with len(df) is too strict (hash-id collisions or a
                # handful of duplicate product_ids make off-by-a-few normal) and
                # would defeat the point of skipping. CPU embedding of ~1700+
                # products takes minutes; skip re-encoding on every container restart.
                if existing_count > 0:
                    logger.info(
                        "rag: Qdrant 'rag_items' already has %d items, skip re-embedding",
                        existing_count,
                    )
                else:
                    self._faiss_index, self._embeddings = self._build_faiss_index()
                    _payloads = [
                        {
                            "product_id": str(row["product_id"]),
                            "price": float(row.get("price", 0)),
                            "category_id": str(row.get("category_id", "")),
                        }
                        for _, row in self.df.iterrows()
                    ]
                    self._qdrant_store.upsert(
                        ids=self.df["product_id"].tolist(),
                        vectors=self._embeddings,
                        payloads=_payloads,
                    )
                    logger.info(
                        "rag: Qdrant 'rag_items' indexed %d items", len(self.df)
                    )
            except Exception as exc:
                logger.warning("rag: Qdrant unavailable (%s) — using FAISS", exc)
                self._qdrant_store = None

        if self._qdrant_store is None:
            self._faiss_index, self._embeddings = self._build_faiss_index()

        self._tfidf, self._tfidf_matrix = self._build_tfidf()

        # Neural reranker — mặc định BẬT (17/7/2026): model cũ (English-only)
        # từng khiến RERANKER_BACKEND mặc định "rule" để né dùng model không
        # phù hợp — đã đổi sang model đa ngôn ngữ (reranker.py), verify xếp
        # đúng "điện thoại" thật lên #1 thay vì lẫn với phụ kiện cùng tên,
        # latency trong ngưỡng an toàn (xem reranker.py). RERANKER_BACKEND=
        # rule vẫn có thể set thủ công nếu cần tắt (vd môi trường quá yếu).
        self._reranker = None
        if os.getenv("RERANKER_BACKEND", "neural") == "neural":
            try:
                from .reranker import CrossEncoderReranker

                self._reranker = CrossEncoderReranker()
            except Exception as e:
                logger.warning(
                    f"cross_encoder_load_failed, falling back to rule-based: {e}"
                )

        logger.info(
            "rag_init_done",
            extra={
                "n_products": len(self.df),
                "reranker": "neural" if self._reranker else "rule",
            },
        )

    # ------------------------------------------------------------------
    # Index builders
    # ------------------------------------------------------------------

    def _build_faiss_index(self):
        texts = self.df["product_name"].fillna("").tolist()
        embeddings = self._model.encode(
            texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
        ).astype(np.float32)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index, embeddings

    def _build_tfidf(self):
        texts = self.df["product_name"].fillna("").tolist()
        tfidf = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), max_features=20_000
        )
        matrix = tfidf.fit_transform(texts)
        return tfidf, matrix

    # ------------------------------------------------------------------
    # Self-querying: extract price constraints from Vietnamese text
    # ------------------------------------------------------------------

    def extract_price_filter(self, query: str) -> dict:
        q = query.lower()
        for pattern, kind, multiplier in _PRICE_PATTERNS:
            m = re.search(pattern, q)
            if m:
                strip_dot = kind.endswith("_vnd")
                base_kind = kind[: -len("_vnd")] if strip_dot else kind

                def _to_float(s: str) -> float:
                    s = s.replace(",", "")
                    if strip_dot:
                        s = s.replace(".", "")
                    return float(s)

                if base_kind == "range":
                    lo = _to_float(m.group(1)) * multiplier
                    hi = _to_float(m.group(2)) * multiplier
                    return {"min_price": lo, "max_price": hi}
                elif base_kind == "approx":
                    val = _to_float(m.group(1)) * multiplier
                    return {"min_price": val * 0.7, "max_price": val * 1.3}
                else:
                    val = _to_float(m.group(1)) * multiplier
                    return {f"{base_kind}_price": val}
        return {}

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def _dense_search(self, query: str, n: int) -> list[int]:
        q_emb = self._model.encode([query], normalize_embeddings=True).astype(
            np.float32
        )
        _, indices = self._faiss_index.search(q_emb, n)
        return indices[0].tolist()

    def _sparse_search(self, query: str, n: int) -> list[int]:
        q_vec = self._tfidf.transform([query])
        scores = cosine_similarity(q_vec, self._tfidf_matrix).flatten()
        return np.argsort(scores)[::-1][:n].tolist()

    @staticmethod
    def _rrf_fusion(dense_ranks: list, sparse_ranks: list, k: int = 60) -> list[tuple]:
        scores: dict[int, float] = {}
        for rank, idx in enumerate(dense_ranks):
            scores[idx] = scores.get(idx, 0.0) + 1 / (k + rank + 1)
        for rank, idx in enumerate(sparse_ranks):
            scores[idx] = scores.get(idx, 0.0) + 1 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        max_price: float | None = None,
        min_price: float | None = None,
        top_k: int = 10,
        n_candidates: int = 50,
    ) -> list[dict]:
        if max_price is None and min_price is None:
            extracted = self.extract_price_filter(query)
            max_price = extracted.get("max_price")
            min_price = extracted.get("min_price")

        # Qdrant path: price/category filter at DB level, then RRF with sparse
        if self._qdrant_store is not None:
            q_emb = self._model.encode([query], normalize_embeddings=True).astype(
                "float32"
            )[0]
            qdrant_hits = self._qdrant_store.search(
                query_vector=q_emb,
                top_k=n_candidates,
                price_max=max_price,
                price_min=min_price,
            )
            qdrant_pids = [pid for pid, _ in qdrant_hits]

            # Bug thật 17/7/2026: nhánh này trước đây CHỈ dense search — comment
            # nói "RRF với sparse" nhưng code không hề fusion, khác hẳn nhánh
            # FAISS bên dưới đã có sparse+RRF thật. Hệ quả: `dangvantuan/
            # vietnamese-embedding` (model chung, không fine-tune tên sản phẩm
            # TMĐT) cho similarity của "điện thoại" với chính sản phẩm
            # "Điện Thoại Samsung Galaxy A37..." (0.236) THẤP HƠN cả sản phẩm
            # không liên quan như dây mạng CAT6 (0.458, đo trực tiếp bằng
            # SentenceTransformer offline) — điện thoại thật biến mất khỏi kết
            # quả tìm kiếm dù match literal 100%. Thêm sparse (TF-IDF char
            # n-gram, `self._tfidf` đã build sẵn không phụ thuộc backend) +
            # RRF fusion bắt lại literal match mà dense bỏ lỡ — đúng thiết kế
            # ban đầu của comment, chỉ là chưa implement.
            sparse_idx = self._sparse_search(query, n_candidates)
            sparse_pids = [str(self.df.iloc[i]["product_id"]) for i in sparse_idx]

            fused = self._rrf_fusion(qdrant_pids, sparse_pids)
            hit_ids = {pid for pid, _ in fused}
            qdrant_scores = dict(fused)
            candidates = self.df[self.df["product_id"].isin(hit_ids)].copy()
            candidates["_rrf"] = candidates["product_id"].map(qdrant_scores).fillna(0.0)

            # sparse-only candidate chưa qua price filter DB-level của Qdrant
            # (chỉ áp dụng cho qdrant_hits) — áp lại ở tầng pandas cho đủ.
            if max_price is not None:
                candidates = candidates[candidates["price"] <= max_price]
            if min_price is not None:
                candidates = candidates[candidates["price"] >= min_price]
        else:
            # FAISS path: ANN first, then pandas post-filter
            dense = self._dense_search(query, n_candidates)
            sparse = self._sparse_search(query, n_candidates)
            fused = self._rrf_fusion(dense, sparse)

            candidate_idx = [idx for idx, _ in fused[: n_candidates * 2]]
            rrf_scores = {idx: score for idx, score in fused[: n_candidates * 2]}

            candidates = self.df.iloc[candidate_idx].copy()
            candidates["_rrf"] = [rrf_scores[i] for i in candidates.index]

            if max_price is not None:
                candidates = candidates[candidates["price"] <= max_price]
            if min_price is not None:
                candidates = candidates[candidates["price"] >= min_price]

        core_categories = _detect_core_categories(query)
        if core_categories is not None:
            candidates = candidates[
                candidates["category_id"].astype(str).isin(core_categories)
            ]

        if len(candidates) == 0:
            return []

        # Rule-based scoring (default fallback)
        rrf_max = candidates["_rrf"].max() or 1.0
        sentiment_max = candidates["avg_item_sentiment"].max() or 1.0
        price_max = candidates["price"].max() or 1.0
        candidates["_score"] = (
            0.6 * (candidates["_rrf"] / rrf_max)
            + 0.2 * (candidates["avg_item_sentiment"] / max(sentiment_max, 1e-9))
            + 0.2 * (1 - candidates["price"] / price_max)
        )
        candidates = candidates.sort_values("_score", ascending=False)

        top_candidates = [
            {
                "product_id": str(row["product_id"]),
                "product_name": str(row["product_name"]),
                "price": float(row["price"]),
                "list_price": float(
                    row.get("list_price", row["price"]) or row["price"]
                ),
                "discount_rate": float(row.get("discount_rate", 0) or 0),
                "category_name": str(row.get("category_name", "")),
                "brand_name": str(row.get("brand_name", "")),
                "avg_sentiment": float(row.get("avg_item_sentiment", 0)),
                "review_count": int(row.get("item_review_count", 0) or 0),
                "thumbnail_url": str(row.get("thumbnail_url", "")),
                "url": str(row.get("url", "") or ""),
                "score": float(row["_score"]),
            }
            for _, row in candidates.head(min(top_k * 2, 20)).iterrows()
        ]

        if self._reranker:
            t0 = time.time()
            reranked = self._reranker.rerank(query, top_candidates, top_k=top_k)
            elapsed = time.time() - t0
            if elapsed > _RERANKER_LATENCY_THRESHOLD:
                # Cross-encoder too slow — fall back to rule-based result
                logger.warning(
                    f"cross_encoder_slow latency={elapsed:.3f}s > {_RERANKER_LATENCY_THRESHOLD}s, using rule-based"
                )
                return top_candidates[:top_k]
            logger.debug(f"cross_encoder_rerank latency={elapsed:.3f}s")
            return reranked

        return top_candidates[:top_k]

    def get_product(self, product_id: str) -> dict | None:
        row = self.df[self.df["product_id"] == product_id]
        if row.empty:
            return None
        r = row.iloc[0]
        return {
            "product_id": str(r["product_id"]),
            "product_name": str(r["product_name"]),
            "price": float(r["price"]),
            "list_price": float(r.get("list_price", r["price"]) or r["price"]),
            "discount_rate": float(r.get("discount_rate", 0) or 0),
            "category_name": str(r.get("category_name", "")),
            "brand_name": str(r.get("brand_name", "")),
            "avg_sentiment": float(r.get("avg_item_sentiment", 0)),
            "review_count": int(r.get("item_review_count", 0) or 0),
            "thumbnail_url": str(r.get("thumbnail_url", "")),
            "url": str(r.get("url", "") or ""),
            "short_description": str(r.get("short_description", "") or ""),
            "specs_text": str(r.get("specs_text", "") or ""),
        }
