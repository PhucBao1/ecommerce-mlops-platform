# preprocessing.py

import pandas as pd


def load_and_merge_data(product_path, comment_path):

    df_product = pd.read_parquet(product_path)

    df_comment = pd.read_parquet(comment_path)

    df_product["product_id"] = df_product["product_id"].astype(str)

    df_comment["product_id"] = df_comment["product_id"].astype(str)

    df_comment["customer_id"] = (
        df_comment["customer_id"].fillna(0).astype("int64").astype(str)
    )

    # Fail-fast nếu nguồn dữ liệu đổi và ép kiểu ở trên không còn đủ (vd
    # customer_id chứa giá trị không parse được thành int64) — rẻ hơn nhiều
    # so với việc lỗi lan tới encoder rồi phải debug ngược từ .pkl.
    assert not df_comment["customer_id"].str.endswith(".0").any(), (
        "customer_id vẫn còn hậu tố '.0' sau khi ép kiểu — kiểm tra lại "
        "nguồn dữ liệu, đây chính là lỗi từng gây ra USER_MAPPING trùng "
        "class trong loaders.py"
    )

    df = df_comment.merge(
        df_product[["product_id", "price", "category_id"]], on="product_id", how="left"
    )

    df = df.dropna(subset=["price", "category_id"])

    df["purchased_at"] = pd.to_datetime(df["purchased_at"])

    df["category_id"] = df["category_id"].astype(str)

    return df_product, df_comment, df
