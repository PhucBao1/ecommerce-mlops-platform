import hashlib
import json
import logging

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class SemanticCache:
    """In-process FAISS index for query-level semantic caching.

    namespace tách riêng key-space Redis + FAISS index giữa các nơi dùng
    (vd "search" cho /search, "chat" cho /chat/stream) — dùng chung 1 class
    nhưng 2 instance riêng để tránh 1 câu hỏi RAG-thuần vô tình khớp/trả về
    câu trả lời LLM đã cache (và ngược lại)."""

    def __init__(
        self,
        embedding_model,
        redis_client,
        threshold: float = 0.92,
        ttl: int = 3600,
        namespace: str = "search",
    ):
        self.model = embedding_model
        self.redis = redis_client
        self.threshold = threshold
        self.ttl = ttl
        self.namespace = namespace
        dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(dim)
        self._keys: list[str] = []

    def _encode(self, text: str) -> np.ndarray:
        return self.model.encode([text], normalize_embeddings=True).astype(np.float32)

    def lookup(self, query: str):
        if self.index.ntotal == 0:
            return None
        vec = self._encode(query)
        scores, idxs = self.index.search(vec, 1)
        if scores[0][0] >= self.threshold:
            redis_key = self._keys[idxs[0][0]]
            try:
                cached = self.redis.get(redis_key)
                if cached:
                    logger.info(
                        "cache_hit",
                        extra={
                            "namespace": self.namespace,
                            "query": query[:50],
                            "score": float(scores[0][0]),
                        },
                    )
                    return json.loads(cached)
            except Exception as exc:
                logger.warning("cache_redis_read_failed", extra={"error": str(exc)})
        return None

    def _cache_key(self, query: str) -> str:
        return f"cache_{self.namespace}:{hashlib.md5(query.encode()).hexdigest()}"

    def store(self, query: str, result) -> None:
        key = self._cache_key(query)
        try:
            self.redis.setex(key, self.ttl, json.dumps(result, ensure_ascii=False))
            self.index.add(self._encode(query))
            self._keys.append(key)
        except Exception as exc:
            logger.warning("cache_store_failed", extra={"error": str(exc)})

    def lookup_exact(self, query: str):
        """Tra thẳng Redis theo key chính xác (KHÔNG qua embedding/FAISS) —
        dùng cho follower poll lặp lại trong lúc chờ leader (xem
        lookup_or_claim/main.py._wait_for_chat_cache). Bug thật tự gây ra +
        tự sửa: poll bằng lookup() (semantic) tính lại embedding CPU-bound
        MỖI LẦN poll — 20 follower x ~32 lần poll = 640 lần encode chồng
        chất trên cùng 1 event loop, nghẽn CPU 100%, mọi request timeout
        (verify: docker stats agent-api lúc lỗi = 100%+ CPU, health check
        NỘI BỘ vẫn nhanh nhưng request thật không xong nổi). lookup_exact()
        chỉ 1 lần Redis GET — rẻ, không tính embedding — vì follower biết
        chính xác đang đợi CÙNG 1 văn bản câu hỏi (key hash, không cần tìm
        tương đồng ngữ nghĩa)."""
        try:
            cached = self.redis.get(self._cache_key(query))
            if cached:
                return json.loads(cached)
        except Exception as exc:
            logger.warning("cache_redis_read_failed", extra={"error": str(exc)})
        return None

    def _lock_key(self, query: str) -> str:
        return f"lock_{self.namespace}:{hashlib.md5(query.encode()).hexdigest()}"

    def lookup_or_claim(self, query: str, lock_ttl: int = 30):
        """Chống 'thundering herd' — nhiều request TRÙNG câu hỏi tới đồng thời
        lúc cache còn rỗng thì đều tự generate riêng thay vì đợi 1 lần
        (đo thật: hit rate rơi 50.9% ở 100 concurrent, xem BENCHMARK_RESULTS.md
        mục 21). Trả (cached, is_leader):
          - cached is not None: dùng luôn (is_leader luôn False).
          - cached is None, is_leader=True: request này THẮNG quyền generate —
            PHẢI gọi store() lúc xong, hoặc release_lock() nếu lỗi giữa chừng.
          - cached is None, is_leader=False: đã có leader khác xử lý câu này —
            caller nên đợi (poll lookup()) thay vì tự generate ngay.
        Dùng key khớp CHÍNH XÁC văn bản câu hỏi (không phải semantic) — chỉ
        chặn được trường hợp lặp lại y hệt, không chặn các cách diễn đạt khác
        nhau nhưng cùng ý (semantic thundering herd hiếm hơn, chưa xử lý).

        Thử lookup_exact() (Redis GET thẳng, không tính embedding) TRƯỚC khi
        rơi xuống lookup() (FAISS semantic, luôn phải _encode()) — 2 lý do:
        (1) rẻ hơn hẳn cho case phổ biến nhất là hỏi lại NGUYÊN VĂN câu đã
        cache, đo saturation-test cho thấy CPU-bound đúng ở bước _encode()
        này (xem BENCHMARK_RESULTS.md mục 22b); (2) khi chạy nhiều uvicorn
        worker (--workers N), FAISS index sống TRONG TỪNG PROCESS riêng
        (không share qua Redis) nên lookup() semantic chỉ thấy đúng các câu
        do CHÍNH worker đó store — worker khác sẽ generate trùng dù Redis
        đã có key. lookup_exact() luôn đúng qua mọi worker vì đọc thẳng
        Redis (shared). Đánh đổi: paraphrase khác chữ nhưng cùng ý sẽ CHỈ
        cache-hit được trên đúng worker đã từng thấy câu gốc — chấp nhận
        được vì đây vốn là phần hiếm hơn theo docstring gốc ở trên."""
        exact = self.lookup_exact(query)
        if exact is not None:
            return exact, False
        cached = self.lookup(query)
        if cached is not None:
            return cached, False
        lock_key = self._lock_key(query)
        try:
            got_lock = bool(self.redis.set(lock_key, "1", nx=True, ex=lock_ttl))
        except Exception as exc:
            logger.warning("cache_lock_failed", extra={"error": str(exc)})
            got_lock = True  # Redis lỗi — không chặn request, tự generate luôn
        return None, got_lock

    def release_lock(self, query: str) -> None:
        """Gọi khi leader lỗi giữa chừng — nhả lock sớm thay vì bắt follower
        đợi hết TTL (30s) cho 1 request đã hỏng."""
        try:
            self.redis.delete(self._lock_key(query))
        except Exception:
            pass
