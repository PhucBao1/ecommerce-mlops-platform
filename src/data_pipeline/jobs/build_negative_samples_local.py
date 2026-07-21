"""
Chạy LOCAL sau khi tải `phase_b1_output.zip` (PhoBERT inference thật, Kaggle
GPU, xem `kaggle_phase_b1_phobert.ipynb`) về — bước negative sampling (vòng
loop O(n×m) trên ~868k dòng x 35k item) là CPU thuần, KHÔNG cần GPU, nên chạy
local (nền, bao lâu cũng được, không tốn quota GPU Kaggle).

Tái dùng nguyên hàm build_user_features/build_item_features/build_item_lookup/
generate_negative_samples/temporal_split đã có sẵn (feature_engineering.py,
negative_sampling.py, split.py) — không viết lại logic, chỉ đổi input path
(bronze thật thay vì bản demo nhỏ) + output path (không ghi đè
item_lookup.parquet/user_history.parquet đang serving, dùng hậu tố _v2 giống
Phase A).

Output (đặt trong artifacts/recsys_models/data_menu/, sẵn sàng upload lên
Kaggle cho Phase B2 — train_model.py đọc đúng 2 file train/valid_df này):
  - train_df_v2.parquet, valid_df_v2.parquet, test_df_v2.parquet
  - item_lookup_v2.parquet (ghi đè bản Phase A — giờ có sentiment PhoBERT thật)
  - user_history_v2.parquet (ghi đè bản Phase A — giờ có sentiment PhoBERT thật)

Usage:
    python -m src.data_pipeline.jobs.build_negative_samples_local \\
        --product-path /path/to/product_full.parquet \\
        --comment-path /path/to/predicted_comments.parquet
"""

import argparse
import logging
import time

import pandas as pd

from src.ml_models.recsys.data.feature_engineering import (
    build_item_features,
    build_user_features,
)
from src.ml_models.recsys.data.negative_sampling import (
    build_item_lookup,
    build_training_dataframe,
)
from src.ml_models.recsys.data.negative_sampling import (
    generate_negative_samples_fast as generate_negative_samples,
)
from src.ml_models.recsys.data.preprocessing import load_and_merge_data
from src.ml_models.recsys.data.split import temporal_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_OUT_DIR = "artifacts/recsys_models/data_menu"


def main(product_path: str, comment_path: str, n_negatives: int = 3):
    df_product, _, df = load_and_merge_data(
        product_path=product_path, comment_path=comment_path
    )
    logger.info("Merged: %d dòng (product=%d)", len(df), len(df_product))

    df["is_positive"] = df["label"].apply(lambda x: 1 if x == "Positive" else 0)

    df = df.sort_values(["customer_id", "purchased_at"]).reset_index(drop=True)
    df = build_user_features(df)
    df = build_item_features(df)  # tự sort lại theo product_id bên trong

    global_avg_price = df["price"].mean()
    df["avg_price_preference"] = df["avg_price_preference"].fillna(global_avg_price)
    df["positive_review_ratio"] = df["positive_review_ratio"].fillna(0.5)
    df["avg_item_sentiment"] = df["avg_item_sentiment"].fillna(0.0)
    df["has_history"] = (df["total_reviews_so_far"] > 0).astype(int)

    interaction_df = df[
        [
            "customer_id",
            "product_id",
            "purchased_at",
            "total_reviews_so_far",
            "avg_price_preference",
            "positive_review_ratio",
            "has_history",
            "price",
            "category_id",
            "avg_item_sentiment",
        ]
    ].copy()
    label_map = {"Positive": 1.0, "Neutral": 0.5, "Negative": 0.0}
    interaction_df["label"] = df["label"].map(label_map).fillna(1.0)

    item_lookup = build_item_lookup(df_product=df_product, df_comment=df)
    logger.info("item_lookup: %d sản phẩm", len(item_lookup))

    logger.info(
        "Bắt đầu negative sampling — CPU thuần, %d dòng interaction, có thể mất "
        "hàng chục phút tới vài giờ tuỳ máy. Không cần GPU, cứ để chạy nền.",
        len(interaction_df),
    )
    t0 = time.time()
    neg_df = generate_negative_samples(
        interaction_df=interaction_df,
        df_product=df_product,
        item_lookup=item_lookup,
        n_negatives=n_negatives,
    )
    logger.info("Negative sampling xong sau %.1f phút", (time.time() - t0) / 60)

    final_df = build_training_dataframe(interaction_df=interaction_df, neg_df=neg_df)
    train_df, valid_df, test_df = temporal_split(final_df)
    logger.info(
        "Split: train=%d valid=%d test=%d", len(train_df), len(valid_df), len(test_df)
    )

    train_df.to_parquet(f"{_OUT_DIR}/train_df_v2.parquet")
    valid_df.to_parquet(f"{_OUT_DIR}/valid_df_v2.parquet")
    test_df.to_parquet(f"{_OUT_DIR}/test_df_v2.parquet")

    item_lookup.reset_index().to_parquet(f"{_OUT_DIR}/item_lookup_v2.parquet")

    history_cols = [
        "customer_id",
        "product_id",
        "purchased_at",
        "total_reviews_so_far",
        "avg_price_preference",
        "positive_review_ratio",
        "has_history",
    ]
    df[history_cols].to_parquet(f"{_OUT_DIR}/user_history_v2.parquet")

    logger.info(
        "Xong. Upload train_df_v2/valid_df_v2/test_df_v2/user_history_v2.parquet "
        "(đổi tên bỏ hậu tố _v2 hoặc sửa path trong notebook) lên Kaggle cho Phase B2 "
        "(train_model.py đọc train_df.parquet/valid_df.parquet trực tiếp)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--product-path", required=True)
    parser.add_argument("--comment-path", required=True)
    parser.add_argument("--n-negatives", type=int, default=3)
    args = parser.parse_args()
    main(args.product_path, args.comment_path, args.n_negatives)
