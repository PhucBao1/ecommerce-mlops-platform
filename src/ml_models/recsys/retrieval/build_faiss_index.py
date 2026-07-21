import faiss
import numpy as np

# =====================================================
# LOAD EMBEDDINGS
# =====================================================

embeddings = np.load("artifacts/recsys_models/faiss/" "item_embeddings.npy")

# embeddings = embeddings.astype("float32")

# =====================================================
# NORMALIZE
# =====================================================

# faiss.normalize_L2(embeddings)

# =====================================================
# BUILD INDEX
# =====================================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

# =====================================================
# # ADD EMBEDDINGS
# # =====================================================

index.add(embeddings)

print(f"✅ Indexed {index.ntotal} items")

# =====================================================
# SAVE
# =====================================================

faiss.write_index(index, "artifacts/recsys_models/faiss/" "faiss_index.bin")

print("✅ Saved FAISS index")
