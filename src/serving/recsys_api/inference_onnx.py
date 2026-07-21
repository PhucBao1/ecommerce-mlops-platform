"""
ONNX Runtime serving layer for item tower inference.

Purpose: replace PyTorch item_tower at startup precompute step.
When ONNX_ENABLED=true, loaders.py calls precompute_item_embeddings_onnx()
instead of model.item_tower() — same result, 2-4x faster on CPU.

Falls back to PyTorch silently if .onnx file not found or onnxruntime not installed.
"""

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/artifacts/recsys_models"))
_ONNX_PATH = _MODEL_DIR / "item_tower.onnx"
_BATCH_SIZE = int(os.getenv("ONNX_BATCH_SIZE", "512"))

try:
    import onnxruntime as ort

    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False
    logger.warning("onnxruntime not installed — ONNX inference unavailable")


class ONNXItemTower:
    """
    Wraps an ONNX Runtime session for item tower inference.

    Inputs (matching export_onnx.py):
        item_id:       int64 [batch]
        item_category: int64 [batch]
        item_num:      float32 [batch, 2]

    Output:
        item_embedding: float32 [batch, embedding_dim]
    """

    def __init__(self, onnx_path: str | Path = _ONNX_PATH):
        if not _ORT_AVAILABLE:
            raise ImportError("onnxruntime not installed")
        onnx_path = str(onnx_path)
        self._session = ort.InferenceSession(
            onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        active_provider = self._session.get_providers()[0]
        logger.info(
            "ONNXItemTower loaded from %s (provider=%s)", onnx_path, active_provider
        )

    def __call__(
        self, item_ids: np.ndarray, cat_ids: np.ndarray, item_nums: np.ndarray
    ) -> np.ndarray:
        """
        Run item tower inference.

        Args:
            item_ids:  int64 [batch]
            cat_ids:   int64 [batch]
            item_nums: float32 [batch, 2]

        Returns:
            embeddings: float32 [batch, embedding_dim]
        """
        outputs = self._session.run(
            None,
            {
                "item_id": item_ids.astype(np.int64),
                "item_category": cat_ids.astype(np.int64),
                "item_num": item_nums.astype(np.float32),
            },
        )
        return outputs[0]  # [batch, embedding_dim]


def precompute_item_embeddings_onnx(
    item_ids: np.ndarray,
    cat_ids: np.ndarray,
    item_nums: np.ndarray,
    onnx_path: str | Path = _ONNX_PATH,
    batch_size: int = _BATCH_SIZE,
) -> np.ndarray:
    """
    Precompute all item embeddings using ONNX Runtime in batches.

    Drop-in replacement for:
        model.item_tower(item_id=..., item_category=..., item_num=...).cpu().numpy()

    Args:
        item_ids:  int64 [n_items]
        cat_ids:   int64 [n_items]
        item_nums: float32 [n_items, 2]
        onnx_path: path to exported item_tower.onnx
        batch_size: inference batch size (tune for memory vs speed)

    Returns:
        embeddings: float32 [n_items, embedding_dim]
    """
    tower = ONNXItemTower(onnx_path)
    n = len(item_ids)
    all_embeddings = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        emb = tower(item_ids[start:end], cat_ids[start:end], item_nums[start:end])
        all_embeddings.append(emb)

    embeddings = np.concatenate(all_embeddings, axis=0)
    logger.info("ONNX precompute done: %d items → embeddings %s", n, embeddings.shape)
    return embeddings


def is_available(onnx_path: str | Path = _ONNX_PATH) -> bool:
    """Return True if ONNX Runtime is installed and the model file exists."""
    return _ORT_AVAILABLE and Path(onnx_path).exists()
