"""
Phase A (local, không cần GPU) — dedup/clean bronze thật (34,987 sản phẩm/22
category, 1,058,532 comment) thành item_lookup.parquet + user_history.parquet
MỚI, tái dùng đúng logic build_user_features/build_item_features/
build_item_lookup đã có sẵn (feature_engineering.py, negative_sampling.py) để
nhất quán với pipeline train hiện tại.

Sentiment ở bước này là PROXY từ rating (1-5 -> 0-1, Positive/Neutral/Negative
theo ngưỡng rating) — KHÔNG phải PhoBERT thật. PhoBERT infer thật trên 1 triệu
comment thật là việc CPU nặng, để dành Phase B (Kaggle GPU) — chạy xong sẽ
overwrite lại avg_item_sentiment bằng số thật, không cần làm lại bước dedup/
build_item_lookup này.

KHÔNG chạy generate_negative_samples() (vòng loop O(n×m) trên 35k item x
1M+ dòng — quá chậm để chạy local, và bản thân nó chỉ cần cho việc RETRAIN
Two-Tower, không phải thứ item_lookup.parquet/user_history.parquet dùng).

Output: artifacts/recsys_models/data_menu/item_lookup_v2.parquet,
        artifacts/recsys_models/data_menu/user_history_v2.parquet
(hậu tố _v2 — không ghi đè bản đang serve, để verify trước khi swap).

Usage: python -m src.data_pipeline.jobs.build_full_catalog_local
"""

import glob
import logging

import pandas as pd

from src.ml_models.recsys.data.feature_engineering import (
    build_item_features,
    build_user_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PRODUCTS_DIR = "/tmp/claude-1000/-home-bao-BaoBao-Ecommerce/6a4157bb-9343-4c14-8cc7-c52ded5ecda3/scratchpad/bronze_products"
_COMMENTS_DIR = "/tmp/claude-1000/-home-bao-BaoBao-Ecommerce/6a4157bb-9343-4c14-8cc7-c52ded5ecda3/scratchpad/bronze_comments"
_OUT_DIR = "artifacts/recsys_models/data_menu"


def load_bronze_products() -> pd.DataFrame:
    files = glob.glob(f"{_PRODUCTS_DIR}/*.parquet")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    logger.info("Bronze products raw: %d dòng, %d category file", len(df), len(files))

    df["product_id"] = df["product_id"].astype(str)
    df["category_id"] = df["category_id"].astype(str)

    # Dedup — cùng logic deduplicate_latest (dedup.py): giữ crawl_time mới nhất
    # theo product_id (1 sản phẩm có thể bị crawl lặp giữa các category/batch).
    df = df.sort_values("crawl_time").drop_duplicates("product_id", keep="last")
    logger.info("Bronze products sau dedup: %d dòng", len(df))

    # Cùng ràng buộc DQ như expectations.py (raw_products): price > 0,
    # category_id not null.
    before = len(df)
    df = df[df["price"] > 0]
    df = df.dropna(subset=["category_id", "product_id"])
    logger.info(
        "Bronze products sau DQ filter: %d dòng (loại %d)", len(df), before - len(df)
    )

    return df


def load_bronze_comments() -> pd.DataFrame:
    files = glob.glob(f"{_COMMENTS_DIR}/*.parquet")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    logger.info("Bronze comments raw: %d dòng, %d category file", len(df), len(files))

    df["product_id"] = df["product_id"].astype(str)
    # customer_id — cùng ép kiểu + assert như preprocessing.py, tránh lặp lại
    # bug thật đã gặp (hậu tố ".0" làm USER_MAPPING trùng class trong loaders.py).
    # replace("") — vài dòng bronze thật có customer_id là chuỗi rỗng (không
    # phải NaN, fillna không bắt được), coalesce về customer_id="0" (placeholder
    # đã biết, bị loại ở train_sasrec.py/train_ltr.py) giống hệt cách NaN được xử lý.
    df["customer_id"] = (
        df["customer_id"].replace("", pd.NA).fillna(0).astype("int64").astype(str)
    )
    assert (
        not df["customer_id"].str.endswith(".0").any()
    ), "customer_id vẫn còn hậu tố '.0' — kiểm tra lại nguồn dữ liệu"

    # Dedup theo review_id, giữ crawl_time mới nhất (cùng logic dedup.py).
    df = df.sort_values("crawl_time").drop_duplicates("review_id", keep="last")
    logger.info("Bronze comments sau dedup: %d dòng", len(df))

    # DQ: rating 0-5, product_id/customer_id/review_id not null (expectations.py).
    before = len(df)
    df = df.dropna(subset=["product_id", "customer_id", "review_id"])
    df = df[(df["rating"] >= 0) & (df["rating"] <= 5)]
    logger.info(
        "Bronze comments sau DQ filter: %d dòng (loại %d)", len(df), before - len(df)
    )

    # Clean text — cùng logic comment_transform.py (strip HTML/whitespace).
    df["clean_comment"] = (
        df["comment"]
        .astype(str)
        .str.replace(r"<[^>]*>", " ", regex=True)
        .str.replace(r"\n|\t|\r", " ", regex=True)
        .str.replace(r" +", " ", regex=True)
        .str.strip()
    )

    return df


def rating_to_sentiment_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """PROXY sentiment từ rating — chờ PhoBERT thật ở Kaggle (xem docstring đầu
    file). rating>=4 -> Positive, ==3 -> Neutral, <=2 -> Negative."""
    df["sentiment_score"] = (df["rating"].clip(1, 5) - 1) / 4.0
    df["label"] = pd.cut(
        df["rating"],
        bins=[-0.1, 2.0, 3.0, 5.0],
        labels=["Negative", "Neutral", "Positive"],
    ).astype(str)
    return df


def build_item_lookup_v2(
    df_product: pd.DataFrame, df_comment: pd.DataFrame
) -> pd.DataFrame:
    """Cùng logic build_item_lookup() (negative_sampling.py) nhưng THÊM
    short_description/all_specs — 2 cột đã có sẵn trong bronze nhưng chưa từng
    được đưa vào item_lookup.parquet phục vụ, để agent trả lời được câu hỏi
    thông số/tính năng thay vì bịa hoặc từ chối trả lời."""
    latest_item_sentiment = (
        df_comment.sort_values("purchased_at")
        .groupby("product_id")["avg_item_sentiment"]
        .last()
        .fillna(0.0)
        .reset_index()
    )
    item_review_count = (
        df_comment.groupby("product_id")
        .size()
        .rename("item_review_count")
        .reset_index()
    )

    display_cols = [
        c
        for c in [
            "product_name",
            "thumbnail_url",
            "category_name",
            "brand_name",
            "short_description",
            "all_specs",
        ]
        if c in df_product.columns
    ]
    item_lookup = (
        df_product[["product_id", "price", "category_id"] + display_cols]
        .drop_duplicates("product_id")
        .merge(latest_item_sentiment, on="product_id", how="left")
        .merge(item_review_count, on="product_id", how="left")
    )
    item_lookup["avg_item_sentiment"] = item_lookup["avg_item_sentiment"].fillna(0.0)
    item_lookup["item_review_count"] = (
        item_lookup["item_review_count"].fillna(0).astype(int)
    )
    return item_lookup.reset_index(drop=True)


def main():
    df_product = load_bronze_products()
    df_comment_raw = load_bronze_comments()
    df_comment_raw = rating_to_sentiment_proxy(df_comment_raw)

    # Merge comment + product (price, category_id) — cùng preprocessing.py.
    df = df_comment_raw.merge(
        df_product[["product_id", "price", "category_id"]], on="product_id", how="left"
    )
    df = df.dropna(subset=["price", "category_id"])
    df["purchased_at"] = pd.to_datetime(df["purchased_at"])
    df["category_id"] = df["category_id"].astype(str)
    df["is_positive"] = (df["label"] == "Positive").astype(int)
    logger.info("Comment sau merge với product: %d dòng", len(df))

    # QUAN TRỌNG: build_user_features/build_item_features dùng
    # groupby(...).transform(expanding()) — cần df đã sort theo thời gian
    # TRƯỚC khi gọi, hàm gốc không tự sort theo customer_id (chỉ
    # build_item_features tự sort theo product_id). Sort đúng ở đây để
    # expanding-mean tính đúng thứ tự thời gian thật, tránh lặp lại lỗi
    # tiềm ẩn nếu input tới không theo đúng thứ tự.
    df = df.sort_values(["customer_id", "purchased_at"]).reset_index(drop=True)
    df = build_user_features(df)
    df = build_item_features(df)  # tự sort lại theo product_id bên trong

    global_avg_price = df["price"].mean()
    df["avg_price_preference"] = df["avg_price_preference"].fillna(global_avg_price)
    df["positive_review_ratio"] = df["positive_review_ratio"].fillna(0.5)
    df["avg_item_sentiment"] = df["avg_item_sentiment"].fillna(0.0)
    df["has_history"] = (df["total_reviews_so_far"] > 0).astype(int)

    item_lookup = build_item_lookup_v2(df_product, df)
    logger.info(
        "item_lookup_v2: %d sản phẩm, %d category",
        len(item_lookup),
        item_lookup["category_id"].nunique(),
    )

    user_history = df[
        [
            "customer_id",
            "product_id",
            "purchased_at",
            "total_reviews_so_far",
            "avg_price_preference",
            "positive_review_ratio",
            "has_history",
        ]
    ].copy()
    logger.info(
        "user_history_v2: %d dòng, %d khách hàng (>=2 review: %d)",
        len(user_history),
        user_history["customer_id"].nunique(),
        (user_history.groupby("customer_id").size() >= 2).sum(),
    )

    item_lookup.to_parquet(f"{_OUT_DIR}/item_lookup_v2.parquet")
    user_history.to_parquet(f"{_OUT_DIR}/user_history_v2.parquet")
    logger.info("Đã lưu %s/item_lookup_v2.parquet + user_history_v2.parquet", _OUT_DIR)


if __name__ == "__main__":
    main()
