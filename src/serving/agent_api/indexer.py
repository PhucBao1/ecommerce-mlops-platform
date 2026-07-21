"""
KB Indexer — embeds document chunks and stores them in FAISS or Qdrant.

Separate from the product index in rag.py (different collection/dim).
Used for policy/FAQ/warranty KB retrieval.

Switch via VECTOR_STORE_BACKEND env var (same convention as rag.py/vector_store.py):
  VECTOR_STORE_BACKEND=faiss   → in-memory FAISS, persisted via save()/load() pickle
  VECTOR_STORE_BACKEND=qdrant  → persistent Qdrant collection "kb_docs" (survives
                                  container restarts without needing local pickle)

Usage:
    indexer = KBIndexer()
    indexer.add_chunks(chunks)            # from chunker.chunk_documents()
    results = indexer.search("đổi trả", top_k=3)

Persistence (chỉ áp dụng khi backend=faiss — Qdrant tự lưu bền vững):
    indexer.save("/app/artifacts/kb_index/")
    indexer = KBIndexer.load("/app/artifacts/kb_index/")
"""

import hashlib
import logging
import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.serving.agent_api.chunker import Chunk

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = os.getenv(
    "EMBEDDING_MODEL", "bkai-foundation-models/vietnamese-bi-encoder"
)
_DEFAULT_INDEX_PATH = os.getenv("KB_INDEX_PATH", "/app/artifacts/kb_index")
_KB_QDRANT_COLLECTION = os.getenv("KB_QDRANT_COLLECTION", "kb_docs")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False


class KBIndexer:
    """FAISS- or Qdrant-backed KB index for document chunks."""

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        cache_dir = os.getenv(
            "SENTENCE_TRANSFORMERS_HOME", "/app/model_cache/sentence_transformers"
        )
        self._model = SentenceTransformer(model_name, cache_folder=cache_dir)
        self._model.max_seq_length = (
            256  # RoBERTa max_position_embeddings=514; stay safe
        )
        self._dim = self._model.get_sentence_embedding_dimension()

        # FAISS luôn được khởi tạo (fallback + local dev path), nhưng nếu Qdrant
        # kết nối được thì mọi add_chunks/search sẽ đi qua Qdrant thay vì FAISS.
        self._index = faiss.IndexFlatIP(self._dim)
        self._chunks: list[Chunk] = []

        self._qdrant: "QdrantClient | None" = None
        if os.getenv("VECTOR_STORE_BACKEND") == "qdrant" and _QDRANT_AVAILABLE:
            try:
                self._qdrant = QdrantClient(
                    url=os.getenv("QDRANT_URL", "http://qdrant:6333")
                )
                self._ensure_qdrant_collection()
                logger.info(
                    "kb_indexer: using Qdrant collection '%s' dim=%d",
                    _KB_QDRANT_COLLECTION,
                    self._dim,
                )
            except Exception as exc:
                logger.warning(
                    "kb_indexer: Qdrant unavailable (%s) — falling back to FAISS", exc
                )
                self._qdrant = None

    def _ensure_qdrant_collection(self) -> None:
        existing = [c.name for c in self._qdrant.get_collections().collections]
        if _KB_QDRANT_COLLECTION not in existing:
            self._qdrant.create_collection(
                collection_name=_KB_QDRANT_COLLECTION,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            )
            logger.info(
                "kb_indexer: created Qdrant collection '%s' dim=%d",
                _KB_QDRANT_COLLECTION,
                self._dim,
            )

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        """Embed chunks and add to the active vector store (Qdrant or FAISS)."""
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(np.float32)

        if self._qdrant is not None:
            points = [
                PointStruct(
                    # Qdrant chỉ nhận ID dạng int/UUID — hash doc_id (source::chunk_index).
                    # Dùng md5 thay vì hash() built-in: hash() bị randomize theo
                    # PYTHONHASHSEED mỗi lần restart process, làm ID đổi liên tục và
                    # Qdrant tích lũy điểm trùng lặp thay vì overwrite đúng chunk cũ.
                    id=int(hashlib.md5(c.doc_id.encode()).hexdigest()[:8], 16),
                    vector=embeddings[i].tolist(),
                    payload={
                        "text": c.text,
                        "source": c.source,
                        "chunk_index": c.chunk_index,
                        "doc_id": c.doc_id,
                        "metadata": c.metadata,
                    },
                )
                for i, c in enumerate(chunks)
            ]
            self._qdrant.upsert(collection_name=_KB_QDRANT_COLLECTION, points=points)
            logger.info(
                "kb_indexer_added %d chunks to Qdrant '%s'",
                len(chunks),
                _KB_QDRANT_COLLECTION,
            )
        else:
            self._index.add(embeddings)
            self._chunks.extend(chunks)
            logger.info(
                "kb_indexer_added %d chunks (total: %d)", len(chunks), len(self._chunks)
            )

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search KB for most relevant chunks.

        Returns list of dicts with keys: text, source, chunk_id, score, metadata.
        """
        q_emb = self._model.encode([query], normalize_embeddings=True).astype(
            np.float32
        )

        if self._qdrant is not None:
            response = self._qdrant.query_points(
                collection_name=_KB_QDRANT_COLLECTION,
                query=q_emb[0].tolist(),
                limit=top_k,
            )
            return [
                {
                    "text": h.payload.get("text", ""),
                    "source": h.payload.get("source", ""),
                    "chunk_id": h.payload.get("doc_id", str(h.id)),
                    "score": float(h.score),
                    "metadata": h.payload.get("metadata", {}),
                }
                for h in response.points
            ]

        if self._index.ntotal == 0:
            return []

        top_k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            results.append(
                {
                    "text": chunk.text,
                    "source": chunk.source,
                    "chunk_id": chunk.doc_id,
                    "score": float(score),
                    "metadata": chunk.metadata,
                }
            )
        return results

    def save(self, dir_path: str = _DEFAULT_INDEX_PATH) -> None:
        """Persist FAISS index and chunk metadata to disk.

        No-op khi backend=qdrant — Qdrant tự lưu bền vững, không cần pickle local.
        """
        if self._qdrant is not None:
            logger.info("kb_indexer_save skipped — Qdrant handles persistence natively")
            return
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "kb.faiss"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)
        logger.info("kb_indexer_saved to %s (%d chunks)", dir_path, len(self._chunks))

    @classmethod
    def load(
        cls, dir_path: str = _DEFAULT_INDEX_PATH, model_name: str = _DEFAULT_MODEL
    ) -> "KBIndexer":
        """Load persisted index.

        backend=qdrant: chỉ cần kết nối lại, data đã nằm sẵn trong Qdrant.
        backend=faiss: đọc pickle từ disk (raise nếu chưa từng save()).
        """
        instance = cls(model_name=model_name)
        if instance._qdrant is not None:
            count = instance._qdrant.count(collection_name=_KB_QDRANT_COLLECTION).count
            logger.info(
                "kb_indexer_loaded from Qdrant collection '%s' (%d chunks)",
                _KB_QDRANT_COLLECTION,
                count,
            )
            return instance

        path = Path(dir_path)
        instance._index = faiss.read_index(str(path / "kb.faiss"))
        with open(path / "chunks.pkl", "rb") as f:
            instance._chunks = pickle.load(f)
        logger.info(
            "kb_indexer_loaded from %s (%d chunks)", dir_path, len(instance._chunks)
        )
        return instance

    @property
    def size(self) -> int:
        if self._qdrant is not None:
            return self._qdrant.count(collection_name=_KB_QDRANT_COLLECTION).count
        return len(self._chunks)
