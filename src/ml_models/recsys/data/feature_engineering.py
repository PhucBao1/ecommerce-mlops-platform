import numpy as np


def build_user_features(df):

    user_groups = df.groupby("customer_id")

    df["avg_price_preference"] = user_groups["price"].transform(
        lambda x: x.expanding().mean().shift(1)
    )

    df["positive_review_ratio"] = user_groups["is_positive"].transform(
        lambda x: x.expanding().mean().shift(1)
    )

    df["total_reviews_so_far"] = user_groups.cumcount()

    df["total_reviews_so_far"] = np.log1p(df["total_reviews_so_far"])

    return df


def build_item_features(df):

    df = df.sort_values(["product_id", "purchased_at"])

    df["avg_item_sentiment"] = df.groupby("product_id")["sentiment_score"].transform(
        lambda x: x.expanding().mean().shift(1)
    )

    return df
