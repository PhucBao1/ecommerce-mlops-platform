import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.ml_models.recsys.models.two_tower import TwoTowerModel
from src.serving.recsys_api.loaders import load_model_artifacts

TOP_K = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# LOAD ARTIFACTS
# =====================================================

(model, user_encoder, item_encoder, cat_encoder, scaler, item_lookup, history_df) = (
    load_model_artifacts()
)

model.eval()

# =====================================================
# LOAD DATA
# =====================================================

train_df = pd.read_parquet("data/warehouse/ml/recsys/train_df.parquet")

# =====================================================
# BUILD ITEM EMBEDDINGS
# =====================================================

print("Building item embeddings...")

all_items = item_lookup.reset_index()

item_ids = torch.tensor(
    item_encoder.transform(all_items["product_id"]), dtype=torch.long
).to(DEVICE)

category_ids = torch.tensor(
    cat_encoder.transform(all_items["category_id"].astype(str)), dtype=torch.long
).to(DEVICE)

item_num = torch.tensor(
    scaler.transform(all_items[["price", "avg_item_sentiment"]]), dtype=torch.float32
).to(DEVICE)

with torch.no_grad():

    item_embeddings = model.item_tower(
        item_id=item_ids, item_category=category_ids, item_num=item_num
    )

item_embeddings = item_embeddings.cpu()

# =====================================================
# USER FEATURES
# =====================================================

latest_user_state = train_df.sort_values("purchased_at").groupby("customer_id").tail(1)

# =====================================================
# GENERATE RECOMMENDATIONS
# =====================================================

results = []

print("Generating recommendations...")

for row in tqdm(latest_user_state.itertuples(), total=len(latest_user_state)):

    try:

        user_idx = user_encoder.transform([row.customer_id])[0]

    except:
        continue

    user_id_tensor = torch.tensor([user_idx], dtype=torch.long).to(DEVICE)

    user_num_tensor = torch.tensor(
        [
            [
                row.total_reviews_so_far,
                row.avg_price_preference,
                row.positive_review_ratio,
                row.has_history,
            ]
        ],
        dtype=torch.float32,
    ).to(DEVICE)

    # ==========================================
    # USER EMBEDDING
    # ==========================================

    with torch.no_grad():

        user_embedding = model.user_tower(
            user_id=user_id_tensor, user_num=user_num_tensor
        )

    user_embedding = user_embedding.cpu()

    # ==========================================
    # DOT PRODUCT
    # ==========================================

    scores = torch.matmul(item_embeddings, user_embedding.T).squeeze()

    topk_scores, topk_indices = torch.topk(scores, TOP_K)

    recommended_products = all_items.iloc[topk_indices.numpy()]["product_id"].tolist()

    results.append(
        {"customer_id": row.customer_id, "recommendations": recommended_products}
    )

# =====================================================
# SAVE
# =====================================================

recommendation_df = pd.DataFrame(results)

recommendation_df.to_parquet(
    "data/warehouse/gold/recsys_batch_predictions.parquet", index=False
)

print("DONE")
