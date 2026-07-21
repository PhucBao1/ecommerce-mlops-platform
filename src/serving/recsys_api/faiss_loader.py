import faiss
import numpy as np

# =====================================================
# LOAD INDEX
# =====================================================

FAISS_INDEX = faiss.read_index("artifacts/recsys_models/faiss/faiss_index.bin")

# =====================================================
# LOAD ITEM IDS
# =====================================================

ITEM_IDS = np.load("artifacts/recsys_models/faiss/item_ids.npy", allow_pickle=True)

print("✅ FAISS loaded")
print("num_items =", len(ITEM_IDS))
