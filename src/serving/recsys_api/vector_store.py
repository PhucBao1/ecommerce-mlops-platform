"""
Abstract VectorStore interface with FAISS and Qdrant implementations.

FAISS:  in-memory, dev/fallback, no persistence, no native filtering
Qdrant: persistent, production, native payload filtering (price, category)

Switch via VECTOR_STORE_BACKEND env var:
  VECTOR_STORE_BACKEND=faiss   → FAISSVectorStore (default, dev)
  VECTOR_STORE_BACKEND=qdrant  → QdrantVectorStore (production)

Qdrant service added to docker-compose.infra.yml:
  qdrant:
    image: qdrant/qdrant:v1.12.4
    ports: ["6333:6333", "6334:6334"]
    volumes:
      - qdrant_data:/qdrant/storage
"""

import hashlib
import logging
import os
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)

_QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
_QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "items")
_VECTOR_DIM = int(os.getenv("VECTOR_DIM", "64"))

# Try importing optional backends at module level
try:
    import faiss as _faiss

    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        Range,
        VectorParams,
    )

    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False
    logger.warning("qdrant-client not installed — Qdrant backend unavailable")


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class VectorStore(ABC):
    """Abstract base for ANN vector stores used in item retrieval."""

    @abstractmethod
    def upsert(self, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None:
        """Insert or update vectors with associated metadata payloads."""

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        price_max: float | None = None,
        price_min: float | None = None,
        category_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """
        Search for nearest neighbors.

        Returns list of (item_id, score) tuples, sorted by score descending.
        Filters are applied server-side when supported (Qdrant), or post-hoc (FAISS).
        """

    @abstractmethod
    def count(self) -> int:
        """Number of vectors currently stored — used to skip redundant re-embedding."""


# ---------------------------------------------------------------------------
# FAISS implementation (in-memory, dev/fallback)
# ---------------------------------------------------------------------------


class FAISSVectorStore(VectorStore):
    """In-memory FAISS index with cosine similarity (IndexFlatIP on L2-normalized vectors)."""

    def __init__(self, dim: int = _VECTOR_DIM):
        if not _FAISS_AVAILABLE:
            raise ImportError("faiss-cpu not installed")
        self._dim = dim
        self._index = _faiss.IndexFlatIP(dim)
        self._ids: list[str] = []
        self._payloads: list[dict] = []

    def upsert(self, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None:
        vecs = vectors.astype(np.float32)
        _faiss.normalize_L2(vecs)
        self._index.add(vecs)
        self._ids.extend(ids)
        self._payloads.extend(payloads)
        logger.debug(
            "faiss_upsert: +%d vectors (total=%d)", len(ids), self._index.ntotal
        )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        price_max: float | None = None,
        price_min: float | None = None,
        category_id: str | None = None,
    ) -> list[tuple[str, float]]:
        if self._index.ntotal == 0:
            return []

        q = query_vector.astype(np.float32).reshape(1, -1)
        _faiss.normalize_L2(q)

        # Over-fetch to allow post-hoc filtering
        fetch_k = min(top_k * 10, self._index.ntotal)
        scores, indices = self._index.search(q, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._ids):
                continue
            payload = self._payloads[idx]

            if price_max is not None and payload.get("price", 0) > price_max:
                continue
            if price_min is not None and payload.get("price", 0) < price_min:
                continue
            if category_id is not None and str(payload.get("category_id", "")) != str(
                category_id
            ):
                continue

            results.append((self._ids[idx], float(score)))
            if len(results) >= top_k:
                break

        return results

    def count(self) -> int:
        return self._index.ntotal


# ---------------------------------------------------------------------------
# Qdrant implementation (persistent, production)
# ---------------------------------------------------------------------------


class QdrantVectorStore(VectorStore):
    """Qdrant vector store with native payload filtering."""

    def __init__(
        self,
        url: str = _QDRANT_URL,
        collection: str = _QDRANT_COLLECTION,
        dim: int = _VECTOR_DIM,
    ):
        if not _QDRANT_AVAILABLE:
            raise ImportError("qdrant-client not installed")
        self._client = QdrantClient(url=url)
        self._collection = collection
        self._dim = dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            )
            logger.info(
                "qdrant: created collection '%s' dim=%d", self._collection, self._dim
            )

    def upsert(self, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None:
        points_with_int_ids = [
            PointStruct(
                id=int(hashlib.md5(ids[i].encode()).hexdigest()[:8], 16),
                vector=vectors[i].tolist(),
                payload={**payloads[i], "_item_id": ids[i]},
            )
            for i in range(len(ids))
        ]
        self._client.upsert(
            collection_name=self._collection, points=points_with_int_ids
        )
        logger.debug("qdrant_upsert: +%d points to '%s'", len(ids), self._collection)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        price_max: float | None = None,
        price_min: float | None = None,
        category_id: str | None = None,
    ) -> list[tuple[str, float]]:
        # FieldCondition là Pydantic model — chỉ nhận keyword args, gọi bằng
        # positional args (key, range) như trước sẽ crash "BaseModel.__init__()
        # takes 1 positional argument but N were given".
        conditions = []
        if price_max is not None:
            conditions.append(FieldCondition(key="price", range=Range(lte=price_max)))
        if price_min is not None:
            conditions.append(FieldCondition(key="price", range=Range(gte=price_min)))
        if category_id is not None:
            conditions.append(
                FieldCondition(
                    key="category_id", match=MatchValue(value=str(category_id))
                )
            )

        query_filter = Filter(must=conditions) if conditions else None

        response = self._client.query_points(
            collection_name=self._collection,
            query=query_vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
        )
        return [
            (h.payload.get("_item_id", str(h.id)), h.score) for h in response.points
        ]

    def count(self) -> int:
        return self._client.count(collection_name=self._collection).count


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_vector_store(
    backend: str | None = None,
    collection: str | None = None,
    dim: int | None = None,
) -> VectorStore:
    """
    Return the appropriate VectorStore based on VECTOR_STORE_BACKEND env var.

    Args:
        backend:    "faiss" | "qdrant". Defaults to VECTOR_STORE_BACKEND env var.
        collection: Qdrant collection name override (e.g. "rag_items" for agent_api).
        dim:        Vector dimension override (needed when different from _VECTOR_DIM).
    """
    backend = backend or os.getenv("VECTOR_STORE_BACKEND", "faiss")

    if backend == "qdrant":
        try:
            store = QdrantVectorStore(
                collection=collection or _QDRANT_COLLECTION,
                dim=dim or _VECTOR_DIM,
            )
            logger.info(
                "vector_store: using Qdrant at %s collection=%s",
                _QDRANT_URL,
                collection or _QDRANT_COLLECTION,
            )
            return store
        except Exception as e:
            logger.warning(
                "vector_store: Qdrant unavailable (%s) — falling back to FAISS", e
            )

    logger.info("vector_store: using FAISS (in-memory) dim=%d", dim or _VECTOR_DIM)
    return FAISSVectorStore(dim=dim or _VECTOR_DIM)
