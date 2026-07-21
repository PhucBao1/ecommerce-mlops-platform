import pandas as pd
from sklearn.preprocessing import LabelEncoder


def safe_transform(encoder, values):

    mapping = {cls: idx for idx, cls in enumerate(encoder.classes_)}

    unknown_idx = len(mapping)

    return values.map(mapping).fillna(unknown_idx).astype(int)


def fit_encoders(train_df):
    """Fit LabelEncoder cho customer_id/product_id/category_id trên train_df,
    trả về df đã có thêm cột *_idx cùng 3 encoder đã fit.
    """
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    cat_encoder = LabelEncoder()

    train_df["customer_id_idx"] = user_encoder.fit_transform(train_df["customer_id"])
    train_df["product_id_idx"] = item_encoder.fit_transform(train_df["product_id"])
    train_df["category_id_idx"] = cat_encoder.fit_transform(
        train_df["category_id"].astype(str)
    )

    return train_df, user_encoder, item_encoder, cat_encoder


def encode_datasets(train_df, valid_df, test_df):

    user_encoder = LabelEncoder()

    item_encoder = LabelEncoder()

    cat_encoder = LabelEncoder()

    # =========================
    # FIT TRAIN
    # =========================

    train_df["customer_id_idx"] = user_encoder.fit_transform(train_df["customer_id"])

    train_df["product_id_idx"] = item_encoder.fit_transform(train_df["product_id"])

    train_df["category_id_idx"] = cat_encoder.fit_transform(
        train_df["category_id"].astype(str)
    )

    # =========================
    # VALID
    # =========================

    valid_df["customer_id_idx"] = safe_transform(user_encoder, valid_df["customer_id"])

    valid_df["product_id_idx"] = safe_transform(item_encoder, valid_df["product_id"])

    valid_df["category_id_idx"] = safe_transform(
        cat_encoder, valid_df["category_id"].astype(str)
    )

    # =========================
    # TEST
    # =========================

    test_df["customer_id_idx"] = safe_transform(user_encoder, test_df["customer_id"])

    test_df["product_id_idx"] = safe_transform(item_encoder, test_df["product_id"])

    test_df["category_id_idx"] = safe_transform(
        cat_encoder, test_df["category_id"].astype(str)
    )

    return (train_df, valid_df, test_df, user_encoder, item_encoder, cat_encoder)
