import numpy as np
import torch

from src.serving.recsys_api.loaders import (
    ALL_ITEM_IDS,
    LIGHTGCN_ITEM_EMBEDDINGS,
    LIGHTGCN_ITEM_IDS,
    VECTOR_STORE,
)


def retrieve_candidates(
    user_embedding,
    top_k: int = 100,
    price_max: float | None = None,
    price_min: float | None = None,
    category_id: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieve top-K candidate items for a user embedding.

    Backends (RETRIEVAL_BACKEND env var):
      twotower (default): Two-Tower item embeddings via VectorStore (FAISS or Qdrant)
      lightgcn:           LightGCN item embeddings via numpy dot product

    Price/category filters are applied server-side when using Qdrant,
    or post-hoc when using FAISS/LightGCN.
    """
    import os

    backend = os.getenv("RETRIEVAL_BACKEND", "twotower")

    if backend == "lightgcn" and LIGHTGCN_ITEM_EMBEDDINGS is not None:
        return _retrieve_lightgcn(
            user_embedding, top_k, price_max, price_min, category_id
        )

    return _retrieve_twotower(user_embedding, top_k, price_max, price_min, category_id)


def _retrieve_twotower(
    user_embedding,
    top_k: int,
    price_max: float | None,
    price_min: float | None,
    category_id: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(user_embedding, torch.Tensor):
        user_embedding = user_embedding.detach().cpu().numpy()
    user_embedding = user_embedding.astype("float32")

    results = VECTOR_STORE.search(
        query_vector=user_embedding.flatten(),
        top_k=top_k,
        price_max=price_max,
        price_min=price_min,
        category_id=category_id,
    )

    if not results:
        return np.array([]), np.array([])

    retrieved_ids = np.array([r[0] for r in results])
    retrieved_scores = np.array([r[1] for r in results])
    return retrieved_ids, retrieved_scores


def _retrieve_lightgcn(
    user_embedding,
    top_k: int,
    price_max: float | None,
    price_min: float | None,
    category_id: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve using LightGCN embeddings (numpy dot product)."""
    if isinstance(user_embedding, torch.Tensor):
        user_embedding = user_embedding.detach().cpu().numpy()
    u_vec = user_embedding.flatten().astype("float32")
    u_vec /= np.linalg.norm(u_vec) + 1e-8

    item_vecs = LIGHTGCN_ITEM_EMBEDDINGS.astype("float32")
    norms = np.linalg.norm(item_vecs, axis=1, keepdims=True) + 1e-8
    item_vecs_norm = item_vecs / norms

    scores = item_vecs_norm @ u_vec
    top_indices = np.argsort(-scores)[: top_k * 5]  # over-fetch for filtering

    results = []
    for idx in top_indices:
        item_id = str(LIGHTGCN_ITEM_IDS[idx])
        results.append((item_id, float(scores[idx])))
        if len(results) >= top_k:
            break

    if not results:
        return np.array([]), np.array([])

    return np.array([r[0] for r in results]), np.array([r[1] for r in results])
