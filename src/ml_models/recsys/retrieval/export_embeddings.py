import faiss
import numpy as np
import pandas as pd
import torch

from src.ml_models.recsys.artifacts.load_artifacts import load_artifacts
from src.ml_models.recsys.models.two_tower import TwoTowerModel

# =====================================================
# CONFIG
# =====================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "artifacts/recsys_models/model/" "best_two_tower.pt"

OUTPUT_DIR = "artifacts/recsys_models/faiss/"

# =====================================================
# LOAD ENCODERS
# =====================================================

user_encoder, item_encoder, cat_encoder, scaler = load_artifacts(
    "artifacts/recsys_models/"
)

# =====================================================
# LOAD PRODUCT DATA
# =====================================================

df_product = pd.read_parquet("product_full.parquet")

df_comment = pd.read_parquet("artifacts/recsys_models/data_menu/item_lookup.parquet")

# df_product["product_id"] = (
#   df_product["product_id"].astype(str)
# )

# df_product["category_id"] = (
#   df_product["category_id"].astype(str)
# )


# =====================================================
# CLEAN TYPES
# =====================================================
df_product["product_id"] = df_product["product_id"].astype(str)
df_product["category_id"] = df_product["category_id"].astype(str)
df_comment["product_id"] = df_comment["product_id"].astype(str)

# keep known products only
# =====================================================
# # KEEP KNOWN PRODUCTS
# # =====================================================

known_products = set(item_encoder.classes_)

df_product = df_product[df_product["product_id"].isin(known_products)]

# =====================================================
# BUILD ITEM SENTIMENT
# =====================================================
latest_sentiment = (
    df_comment.groupby("product_id")["avg_item_sentiment"].last().reset_index()
)

# =====================================================
# MERGE
# =====================================================
df_product = df_product.merge(latest_sentiment, on="product_id", how="left")

# =====================================================
# FILL NULL
# =====================================================
df_product["avg_item_sentiment"] = df_product["avg_item_sentiment"].fillna(0.0)

# =====================================================
# # BUILD MAPPINGS
# # =====================================================

item_mapping = {cls: idx for idx, cls in enumerate(item_encoder.classes_)}

cat_mapping = {cls: idx for idx, cls in enumerate(cat_encoder.classes_)}

# =====================================================
# ENCODE
# =====================================================

df_product["product_id_idx"] = df_product["product_id"].map(item_mapping)

df_product["category_id_idx"] = (
    df_product["category_id"].map(cat_mapping).fillna(0).astype(int)
)

# =====================================================
# LOAD MODEL
# =====================================================

model = TwoTowerModel(
    num_users=len(user_encoder.classes_) + 1,
    num_items=len(item_encoder.classes_) + 1,
    num_categories=len(cat_encoder.classes_) + 1,
    embedding_dim=32,
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

model.to(DEVICE)
model.eval()

# =====================================================
# BUILD ITEM EMBEDDINGS
# =====================================================

all_embeddings = []

batch_size = 1024

with torch.no_grad():

    for start in range(0, len(df_product), batch_size):

        batch_df = df_product.iloc[start : start + batch_size]

        # item_num = torch.tensor( batch_df[ [ "price", "avg_item_sentiment" ] ].values, dtype=torch.float32 ).to(DEVICE)

        # =====================================
        # # IDS
        # # =====================================

        item_ids = torch.tensor(batch_df["product_id_idx"].values, dtype=torch.long).to(
            DEVICE
        )

        category_ids = torch.tensor(
            batch_df["category_id_idx"].values, dtype=torch.long
        ).to(DEVICE)

        # =====================================
        # NUM FEATURES
        # =====================================
        # Lấy 2 cột số
        num_features = batch_df[["price", "avg_item_sentiment"]].values

        # scaler được huấn luyện trên 5 cột, nên tạo mảng dummy 5 cột
        dummy = np.zeros((len(batch_df), 5))
        dummy[:, 3] = num_features[:, 0]  # price
        dummy[:, 4] = num_features[:, 1]  # avg_item_sentiment

        dummy_df = pd.DataFrame(
            dummy,
            columns=[
                "total_reviews_so_far",
                "avg_price_preference",
                "positive_review_ratio",
                "price",
                "avg_item_sentiment",
            ],
        )

        # scale và chuyển sang tensor
        scaled = scaler.transform(dummy)
        item_num = torch.tensor(scaled[:, 3:5], dtype=torch.float32).to(DEVICE)

        # =====================================
        # # ITEM EMBEDDINGS
        # # =====================================

        item_embeddings = model.item_tower(item_ids, category_ids, item_num)

        item_embeddings = item_embeddings.cpu().numpy().astype("float32")

        all_embeddings.append(item_embeddings)

# =====================================================
# # CONCAT
# # =====================================================

all_embeddings = np.vstack(all_embeddings)

item_ids = df_product["product_id"].values

# =====================================================
# NORMALIZE EMBEDDINGS
# =====================================================
faiss.normalize_L2(all_embeddings)


# =====================================================
# # SAVE
# # =====================================================

np.save(OUTPUT_DIR + "item_embeddings.npy", all_embeddings)

np.save(OUTPUT_DIR + "item_ids.npy", item_ids)

df_product[["product_id", "category_id", "price"]].to_parquet(
    OUTPUT_DIR + "item_metadata.parquet", index=False
)


print("✅ Exported item embeddings")
print(all_embeddings.shape)
