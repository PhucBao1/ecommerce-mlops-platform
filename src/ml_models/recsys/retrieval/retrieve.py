import faiss
import numpy as np
import torch

from src.serving.recsys_api.loaders import device, model

# =====================================================
# LOAD FAISS
# =====================================================

index = faiss.read_index("artifacts/recsys_models/faiss/" "faiss_index.bin")

ITEM_IDS = np.load("artifacts/recsys_models/faiss/" "item_ids.npy", allow_pickle=True)

# =====================================================
# RETRIEVE
# =====================================================


def retrieve_candidates(user_embedding, top_k=100):

    user_embedding = user_embedding.detach().cpu().numpy().astype("float32")

    faiss.normalize_L2(user_embedding)

    scores, indices = index.search(user_embedding, top_k)

    retrieved_item_ids = ITEM_IDS[indices[0]]

    return retrieved_item_ids.tolist()
