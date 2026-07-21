# =========================================================
# FILE: inference.py
# =========================================================

import logging
import time

import numpy as np
import pandas as pd
import torch

from src.serving.recsys_api.loaders import *
from src.serving.recsys_api.utils import safe_transform

logger = logging.getLogger(__name__)


# =========================================================
# RECOMMEND FUNCTION
# =========================================================
def recommend(
    customer_id,
    top_k=10,
    recent_sentiment_score=0.0,
    last_commented_product_id=None,
    history_override=None,
):

    customer_id = str(customer_id).replace(".0", "")

    if history_override is not None:

        user_hist = history_override

    else:

        user_hist = USER_HISTORY_DICT.get(customer_id)

    # =====================================================
    # USER FEATURES
    # =====================================================

    if user_hist is not None and len(user_hist) > 0:
        latest_state = user_hist.iloc[-1]

        user_features = {
            "customer_id": customer_id,
            "total_reviews_so_far": latest_state["total_reviews_so_far"],
            "avg_price_preference": latest_state["avg_price_preference"],
            "positive_review_ratio": latest_state["positive_review_ratio"],
            "has_history": 1.0,
        }
        purchased_items = user_hist["product_id"].astype(str).unique()
    else:
        # Cold Start
        user_features = {
            "customer_id": customer_id,
            "total_reviews_so_far": 0.0,
            "avg_price_preference": global_avg_price,
            "positive_review_ratio": 0.5,
            "has_history": 0.0,
        }
        purchased_items = []

    # =====================================================
    # USER DF
    # =====================================================

    user_df = pd.DataFrame([user_features])

    user_df["customer_id_idx"] = safe_transform(USER_MAPPING, user_df["customer_id"])

    user_scale_df = pd.DataFrame(
        {
            "total_reviews_so_far": user_df["total_reviews_so_far"],
            "avg_price_preference": user_df["avg_price_preference"],
            "positive_review_ratio": user_df["positive_review_ratio"],
            "price": [0.0],
            "avg_item_sentiment": [0.0],
        }
    )

    scaled_user = scaler.transform(user_scale_df)

    # user_num = scaled_user[:, [0,1,2,5]]

    user_num = np.concatenate(
        [scaled_user[:, [0, 1, 2]], np.array([[user_features["has_history"]]])], axis=1
    )

    # =====================================================
    # USER VECTOR
    # =====================================================

    model.eval()

    with torch.no_grad():

        user_vector = model.user_tower(
            user_id=torch.tensor(
                user_df["customer_id_idx"].values, dtype=torch.long
            ).to(device),
            user_num=torch.tensor(user_num, dtype=torch.float32).to(device),
        )

        # ==========================================
        # DOT PRODUCT SEARCH
        # ==========================================

        scores = torch.matmul(user_vector, ALL_ITEM_VECTORS.T).squeeze(0)

        scores = scores.numpy()

    if (
        recent_sentiment_score is not None
        and recent_sentiment_score < 0
        and last_commented_product_id is not None
    ):

        mask = item_lookup_df["product_id"].astype(str) == str(
            last_commented_product_id
        )

        scores[mask.values] -= 1.0

    if (
        recent_sentiment_score is not None
        and recent_sentiment_score > 0
        and last_commented_product_id is not None
    ):

        try:

            target_cat = item_lookup_df[
                item_lookup_df["product_id"].astype(str)
                == str(last_commented_product_id)
            ]["category_id"].values[0]

            mask = item_lookup_df["category_id"] == target_cat

            scores[mask.values] += 0.15

        except:
            pass

    # =====================================================
    # TOP K
    # =====================================================
    purchased_set = set(purchased_items)

    if recent_sentiment_score < 0 and last_commented_product_id:
        purchased_set.add(str(last_commented_product_id))

    sorted_indices = np.argsort(scores)[::-1]

    top_indices = []

    for idx in sorted_indices:

        product_id = ALL_ITEM_IDS[idx]

        if product_id not in purchased_set:

            top_indices.append(idx)

        if len(top_indices) >= top_k:
            break

    top_k_df = item_lookup_df.iloc[top_indices].copy()

    top_k_df["predict_score"] = scores[top_indices]

    result = top_k_df[["product_id", "predict_score", "price", "category_id"]].to_dict(
        orient="records"
    )

    # DEBUG
    print("=" * 50)

    print("REQUEST USER:", customer_id)

    print("USER EXISTS:", user_hist is not None)

    print(user_features)

    print(user_df["customer_id_idx"])

    print(user_vector[0][:10])

    return {"status": "success", "customer_id": customer_id, "recommendations": result}
