# src/ml_models/recsys/data/build_dataset.py

from src.ml_models.recsys.data.feature_engineering import (
    build_item_features,
    build_user_features,
)
from src.ml_models.recsys.data.negative_sampling import (
    build_item_lookup,
    build_training_dataframe,
    generate_negative_samples,
)
from src.ml_models.recsys.data.preprocessing import load_and_merge_data
from src.ml_models.recsys.data.split import temporal_split


def build_dataset():

    # =====================================================
    # LOAD
    # =====================================================

    df_product, _, df = load_and_merge_data(
        product_path="product_full.parquet", comment_path="predicted_comments.parquet"
    )

    # =====================================================
    # LABEL
    # =====================================================

    df["is_positive"] = df["label"].apply(lambda x: 1 if x == "Positive" else 0)

    # =====================================================
    # USER FEATURES
    # =====================================================

    df = build_user_features(df)

    # =====================================================
    # ITEM FEATURES
    # =====================================================

    df = build_item_features(df)

    # =====================================================
    # COLD START FILL
    # =====================================================

    global_avg_price = df["price"].mean()

    df["avg_price_preference"] = df["avg_price_preference"].fillna(global_avg_price)

    df["positive_review_ratio"] = df["positive_review_ratio"].fillna(0.5)

    df["avg_item_sentiment"] = df["avg_item_sentiment"].fillna(0.0)

    df["has_history"] = (df["total_reviews_so_far"] > 0).astype(int)

    # =====================================================
    # POSITIVE INTERACTIONS
    # =====================================================

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

    # =====================================================
    # ITEM LOOKUP
    # =====================================================

    item_lookup = build_item_lookup(df_product=df_product, df_comment=df)

    # =====================================================
    # NEGATIVE SAMPLING
    # =====================================================

    neg_df = generate_negative_samples(
        interaction_df=interaction_df,
        df_product=df_product,
        item_lookup=item_lookup,
        n_negatives=3,
    )

    # =====================================================
    # FINAL DATASET
    # =====================================================

    final_df = build_training_dataframe(interaction_df=interaction_df, neg_df=neg_df)

    # =====================================================
    # SPLIT
    # =====================================================

    train_df, valid_df, test_df = temporal_split(final_df)

    # =====================================================
    # SAVE
    # =====================================================

    train_df.to_parquet("data/warehouse/ml/recsys/train_df.parquet")

    valid_df.to_parquet("data/warehouse/ml/recsys/valid_df.parquet")

    test_df.to_parquet("data/warehouse/ml/recsys/test_df.parquet")

    return (train_df, valid_df, test_df, item_lookup, df)
