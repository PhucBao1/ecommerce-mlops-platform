# =========================================================
# FILE: inference.py
# =========================================================

import logging
import time

import numpy as np
import pandas as pd
import torch
from prometheus_client import Histogram

from src.serving.recsys_api.candidate_cache import (
    cache_candidates,
    get_cached_candidates,
)
from src.serving.recsys_api.loaders import (
    ALL_ITEM_IDS,
    ALL_ITEM_VECTORS,
    CAT_MAPPING,
    ITEM_MAPPING,
    TRENDING_ITEM_IDS,
    USER_HISTORY_DICT,
    USER_MAPPING,
    device,
    global_avg_price,
    item_lookup_df,
    model,
    scaler,
)
from src.serving.recsys_api.reranker import rerank_candidates
from src.serving.recsys_api.retrieval import retrieve_candidates

logger = logging.getLogger(__name__)

FEATURE_PREP_LATENCY = Histogram(
    "recommend_feature_prep_latency_seconds",
    "Thời gian chuẩn bị feature (pandas/scaler) trong recommend()",
)
MODEL_INFERENCE_LATENCY = Histogram(
    "recommend_model_inference_latency_seconds",
    "Thời gian model.user_tower() forward pass (PyTorch) — ứng viên ONNX",
)
FAISS_RETRIEVAL_LATENCY = Histogram(
    "recommend_faiss_retrieval_latency_seconds",
    "Thời gian retrieve_candidates() (FAISS) khi cache miss",
)
RERANK_LATENCY = Histogram(
    "recommend_rerank_latency_seconds",
    "Thời gian rerank_candidates()",
)


# =========================================================
# RECOMMEND FUNCTION
# =========================================================
def recommend(
    customer_id,
    top_k=10,
    recent_sentiment_score=0.0,
    last_commented_product_id=None,
    history_override=None,
    diversity_limit=3,
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
        bought_cats = set(
            item_lookup_df[
                item_lookup_df["product_id"].astype(str).isin(purchased_items)
            ]["category_id"].astype(str)
        )
    else:
        # Cold start: skip Two-Tower model, return pre-computed trending items directly.
        # Popularity-based fallback is more accurate than running the model with all-default features.
        trending_df = item_lookup_df[
            item_lookup_df["product_id"].isin(TRENDING_ITEM_IDS[: top_k * 2])
        ].copy()
        trending_df["predict_score"] = 0.0
        trending_df = trending_df.head(top_k)
        trending_recs = trending_df[
            [
                c
                for c in [
                    "product_id",
                    "predict_score",
                    "price",
                    "category_id",
                    "avg_item_sentiment",
                    "product_name",
                    "thumbnail_url",
                    "category_name",
                    "brand_name",
                ]
                if c in trending_df.columns
            ]
        ].to_dict(orient="records")
        for rec in trending_recs:
            rec["explanation"] = {
                "top_reason": "Sản phẩm phổ biến nhất hiện tại",
                "factors": ["Trending"],
            }
        return {
            "status": "success",
            "customer_id": customer_id,
            "source": "trending",
            "recommendations": trending_recs,
        }

    # =====================================================
    # USER DF
    # =====================================================

    t_feature_prep_start = time.perf_counter()

    customer_id_idx = USER_MAPPING.get(customer_id, len(USER_MAPPING))

    scale_input = np.array(
        [
            [
                user_features["total_reviews_so_far"],
                user_features["avg_price_preference"],
                user_features["positive_review_ratio"],
                0.0,
                0.0,
            ]
        ],
        dtype=np.float64,
    )
    scaled_user = scaler.transform(scale_input)

    user_num = np.concatenate(
        [scaled_user[:, [0, 1, 2]], np.array([[user_features["has_history"]]])], axis=1
    )
    FEATURE_PREP_LATENCY.observe(time.perf_counter() - t_feature_prep_start)

    # =====================================================
    # USER VECTOR
    # =====================================================

    t_model_start = time.perf_counter()

    model.eval()

    with torch.no_grad():

        user_vector = model.user_tower(
            user_id=torch.tensor([customer_id_idx], dtype=torch.long).to(device),
            user_num=torch.tensor(user_num, dtype=torch.float32).to(device),
        )

    MODEL_INFERENCE_LATENCY.observe(time.perf_counter() - t_model_start)

    # ==========================================
    # FAISS RETRIEVAL
    # ==========================================

    """candidate_ids, candidate_scores = (
        retrieve_candidates(
            user_vector,
            top_k=200
        )
    )"""

    # ==========================================
    # CANDIDATE CACHE
    # ==========================================

    cached_candidates = get_cached_candidates(customer_id)

    if cached_candidates is not None:

        candidate_ids = [x["product_id"] for x in cached_candidates]

        candidate_scores = np.array([x["score"] for x in cached_candidates])

    else:

        t_faiss_start = time.perf_counter()
        candidate_ids, candidate_scores = retrieve_candidates(user_vector, top_k=200)
        FAISS_RETRIEVAL_LATENCY.observe(time.perf_counter() - t_faiss_start)

        cache_payload = [
            {"product_id": str(pid), "score": float(score)}
            for pid, score in zip(candidate_ids, candidate_scores)
        ]

        cache_candidates(customer_id, cache_payload)

    # convert thành dict để xử lý boost/rerank
    score_dict = {
        str(pid): float(score) for pid, score in zip(candidate_ids, candidate_scores)
    }

    if (
        recent_sentiment_score is not None
        and recent_sentiment_score < 0
        and last_commented_product_id is not None
    ):

        pid = str(last_commented_product_id)

        if pid in score_dict:
            score_dict[pid] -= 1.0  # boost sản phẩm gần đây có sentiment tiêu cực

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

            same_cat_items = item_lookup_df[
                item_lookup_df["category_id"] == target_cat
            ]["product_id"].astype(str)

            CATEGORY_BOOST = 0.05
            INJECT_POOL_SIZE = 8
            max_score = max(score_dict.values()) if score_dict else 0.0

            existing_same_cat = [pid for pid in same_cat_items if pid in score_dict]

            if existing_same_cat:
                for i, pid in enumerate(existing_same_cat[:INJECT_POOL_SIZE]):
                    score_dict[pid] = max_score + CATEGORY_BOOST - i * 0.01
            elif score_dict:
                inject_pool = (
                    item_lookup_df[
                        (item_lookup_df["category_id"] == target_cat)
                        & (
                            item_lookup_df["product_id"].astype(str)
                            != str(last_commented_product_id)
                        )
                    ]
                    .nlargest(INJECT_POOL_SIZE, "avg_item_sentiment")
                    .reset_index(drop=True)
                )

                for i, pid in enumerate(inject_pool["product_id"].astype(str)):
                    score_dict[pid] = max_score + CATEGORY_BOOST - i * 0.01

            bought_cats.add(str(target_cat))

        except Exception as e:
            logger.warning("Category boost failed: %s", e)

    # =====================================================
    # TOP K
    # =====================================================

    # =====================================================
    # FILTER PURCHASED
    # =====================================================

    purchased_set = set(purchased_items)

    if recent_sentiment_score < 0 and last_commented_product_id:
        purchased_set.add(str(last_commented_product_id))

    # =====================================================
    # SORT FINAL SCORES
    # =====================================================

    sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)

    final_items = []

    for pid, score in sorted_items:

        if pid not in purchased_set:

            final_items.append((pid, score))

        if len(final_items) >= top_k:
            break

    # =====================================================
    # BUILD RESULT DF
    # =====================================================

    top_product_ids = [x[0] for x in final_items]

    top_scores = [x[1] for x in final_items]

    top_k_df = item_lookup_df[
        item_lookup_df["product_id"].astype(str).isin(top_product_ids)
    ].copy()

    score_map = {pid: score for pid, score in final_items}

    top_k_df["predict_score"] = top_k_df["product_id"].astype(str).map(score_map)

    """top_k_df = top_k_df.sort_values(
        "predict_score",
        ascending=False
    )"""

    # session_items: lịch sử mua/review thật của khách, theo đúng thứ tự
    # cũ→mới (USER_HISTORY_DICT đã sort_values("purchased_at") lúc load —
    # xem loaders.py) — dùng làm input cho sasrec_score trong rerank_candidates(),
    # KHÔNG cần khách đang có 1 session duyệt web riêng, tái dùng chính lịch
    # sử mua hàng đã có sẵn.
    t_rerank_start = time.perf_counter()
    top_k_df = rerank_candidates(
        top_k_df,
        user_features,
        diversity_limit=diversity_limit,
        bought_cats=bought_cats,
        session_items=list(purchased_items),
    )
    RERANK_LATENCY.observe(time.perf_counter() - t_rerank_start)

    base_cols = [
        c
        for c in [
            "product_id",
            "predict_score",
            "price",
            "category_id",
            "avg_item_sentiment",
            "product_name",
            "thumbnail_url",
            "category_name",
            "brand_name",
        ]
        if c in top_k_df.columns
    ]
    result = top_k_df[base_cols].to_dict(orient="records")

    # Attach explanation from reranker (Task 70)
    if "_explanation" in top_k_df.columns:
        explanations = top_k_df["_explanation"].tolist()
        for rec, expl in zip(result, explanations):
            rec["explanation"] = expl

    logger.debug(
        "REQUEST USER: %s | EXISTS: %s | features: %s",
        customer_id,
        user_hist is not None,
        user_features,
    )

    return {"status": "success", "customer_id": customer_id, "recommendations": result}
