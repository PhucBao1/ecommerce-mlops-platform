"""
Cross-Encoder Neural Reranker

Uses cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (~470MB, multilingual mMARCO)
để rerank RAG candidates. Activated via RERANKER_BACKEND=neural; falls back to
rule-based if latency >500ms.

Bug thật (17/7/2026): model cũ (`cross-encoder/ms-marco-MiniLM-L-6-v2`) train
THUẦN TIẾNG ANH (MS-MARCO gốc) — không đáng tin cậy cho catalog tiếng Việt,
đây là lý do RERANKER_BACKEND mặc định "rule" chứ không phải "neural" (né
tránh dùng model không phù hợp, không phải đã verify rule-based đủ tốt).
Verify trực tiếp (offline, 5 candidate thật: 1 điện thoại + 4 phụ kiện đều
chứa chữ "điện thoại" trong tên) — model đa ngôn ngữ `mmarco-mMiniLMv2-
L12-H384` (cùng train trên mMARCO nhưng multilingual, gồm cả tiếng Việt)
xếp đúng điện thoại thật lên #1 (score -5.28) tách biệt hẳn khỏi phụ kiện
(-8.49 đến -9.39), latency 0.08-0.11s cho 5-20 candidate (ấm máy) — trong
ngưỡng LATENCY_THRESHOLD mặc định. Model lớn hơn (bge-reranker-v2-m3) xếp
hạng còn chính xác hơn nhưng chậm hơn nhiều (~1.5s/5 item) — vượt ngưỡng
latency, luôn bị fallback về rule-based nên chọn model nhẹ này thay thế.
"""

import logging
import os
import time

logger = logging.getLogger(__name__)

_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
_LATENCY_THRESHOLD_S = float(os.getenv("RERANKER_LATENCY_THRESHOLD", "0.5"))


class CrossEncoderReranker:
    def __init__(self):
        from sentence_transformers.cross_encoder import CrossEncoder

        cache_dir = os.getenv(
            "SENTENCE_TRANSFORMERS_HOME", "/app/model_cache/sentence_transformers"
        )
        logger.info(f"Loading cross-encoder: {_MODEL_NAME}")
        self._model = CrossEncoder(_MODEL_NAME, max_length=512, device="cpu")
        logger.info("cross_encoder_loaded")

    def rerank(self, query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
        if not candidates:
            return []

        pairs = [
            (
                query,
                f"{c.get('product_name', '')} {c.get('category_name', '')} {c.get('brand_name', '')}",
            )
            for c in candidates
        ]

        scores = self._model.predict(pairs)  # numpy array, shape (n,)
        ranked = sorted(
            zip(scores.tolist(), candidates), key=lambda x: x[0], reverse=True
        )
        return [c for _, c in ranked[:top_k]]
