import numpy as np
import pandas as pd


def build_item_lookup(df_product: pd.DataFrame, df_comment: pd.DataFrame):
    """
    Build latest item feature lookup table.

    Returns:
        item_lookup: indexed by product_id
    """

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
        for c in ["product_name", "thumbnail_url", "category_name", "brand_name"]
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

    item_lookup = item_lookup.set_index("product_id").sort_index()

    return item_lookup


def generate_negative_samples(
    interaction_df: pd.DataFrame,
    df_product: pd.DataFrame,
    item_lookup: pd.DataFrame,
    n_negatives: int = 3,
    random_state: int = 42,
):
    """
    Generate point-in-time safe negative samples.

    Logic:
    - only exclude products user purchased BEFORE current timestamp
    - never leak future information
    - negative items sampled randomly

    Returns:
        neg_df
    """

    np.random.seed(random_state)

    all_products = df_product["product_id"].astype(str).unique()

    negative_rows = []

    # ==========================================
    # USER HISTORY TRACKER
    # ==========================================

    user_history = {}

    # ==========================================
    # GLOBAL TIME SORT
    # ==========================================

    interaction_df = interaction_df.sort_values("purchased_at").reset_index(drop=True)

    print("Generating negative samples...")

    # ==========================================
    # MAIN LOOP
    # ==========================================

    for row in interaction_df.itertuples():

        user = str(row.customer_id)

        pos_item = str(row.product_id)

        # ======================================
        # INIT USER HISTORY
        # ======================================

        if user not in user_history:

            user_history[user] = set()

        # ======================================
        # CANDIDATE ITEMS
        # ======================================

        candidate_items = np.setdiff1d(all_products, list(user_history[user]))

        # remove positive item
        candidate_items = candidate_items[candidate_items != pos_item]

        # ======================================
        # SKIP IF EMPTY
        # ======================================

        if len(candidate_items) == 0:
            continue

        # ======================================
        # SAMPLE NEGATIVES (1 hard + n-1 random)
        # Hard negative: same category as positive item.
        # Forces the model to distinguish within a category, not just across categories.
        # ======================================

        hard_negs = []
        try:
            pos_category = str(item_lookup.loc[pos_item]["category_id"])
            same_cat = item_lookup[
                (item_lookup["category_id"] == pos_category)
                & (~item_lookup.index.isin(user_history[user]))
                & (item_lookup.index != pos_item)
            ].index.tolist()
            if same_cat:
                hard_negs = [np.random.choice(same_cat)]
        except Exception:
            pass

        n_random = max(0, min(n_negatives - len(hard_negs), len(candidate_items)))
        random_pool = candidate_items[~np.isin(candidate_items, hard_negs)]
        random_negs = (
            np.random.choice(random_pool, size=n_random, replace=False)
            if n_random > 0 and len(random_pool) > 0
            else np.array([])
        )

        sampled_negatives = np.concatenate([hard_negs, random_negs])

        # ======================================
        # BUILD NEGATIVE ROWS
        # ======================================

        for neg_item in sampled_negatives:

            try:

                item_info = item_lookup.loc[str(neg_item)]

            except Exception:
                continue

            negative_rows.append(
                {
                    # ==========================
                    # USER
                    # ==========================
                    "customer_id": user,
                    # ==========================
                    # NEGATIVE ITEM
                    # ==========================
                    "product_id": str(neg_item),
                    # ==========================
                    # TIMESTAMP
                    # ==========================
                    "purchased_at": row.purchased_at,
                    # ==========================
                    # USER PIT FEATURES
                    # ==========================
                    "total_reviews_so_far": row.total_reviews_so_far,
                    "avg_price_preference": row.avg_price_preference,
                    "positive_review_ratio": row.positive_review_ratio,
                    "has_history": row.has_history,
                    # ==========================
                    # ITEM FEATURES
                    # ==========================
                    "price": item_info["price"],
                    "category_id": str(item_info["category_id"]),
                    "avg_item_sentiment": item_info["avg_item_sentiment"],
                    # ==========================
                    # LABEL
                    # ==========================
                    "label": 0.0,
                }
            )

        # ======================================
        # UPDATE HISTORY AFTER SAMPLING
        # ======================================

        user_history[user].add(pos_item)

    # ==========================================
    # BUILD DATAFRAME
    # ==========================================

    neg_df = pd.DataFrame(negative_rows)

    print(f"Generated {len(neg_df)} " f"negative samples")

    return neg_df


def generate_negative_samples_fast(
    interaction_df: pd.DataFrame,
    df_product: pd.DataFrame,
    item_lookup: pd.DataFrame,
    n_negatives: int = 3,
    random_state: int = 42,
):
    """
    Cùng semantics với generate_negative_samples() ở trên (point-in-time safe,
    1 hard negative cùng category + (n_negatives-1) random negative, uniform
    sampling trên candidate pool hợp lệ) — chỉ đổi CÁCH tìm candidate:
    rejection-sampling (random rồi loại nếu trùng lịch sử/pos_item) thay vì
    loại-trước-rồi-random (np.setdiff1d trên toàn bộ catalog + pandas boolean
    filter trên item_lookup MỖI DÒNG, độ phức tạp O(catalog_size) ~35k mỗi
    dòng x 868k dòng = quá chậm trên máy CPU thường, ước tính nhiều giờ).

    Lịch sử mỗi user luôn nhỏ hơn nhiều so với catalog (phần lớn user chỉ
    tương tác vài sản phẩm — xem project memory: data sparsity), nên xác
    suất rejection-sampling trùng cực thấp, gần như luôn thành công ngay lần
    thử đầu. Độ phức tạp mỗi dòng giảm từ O(catalog_size) xuống O(1)
    amortized nhờ 2 cấu trúc precompute 1 lần duy nhất trước vòng loop:
    category_id -> mảng product_id cùng category (thay pandas filter), và
    product_id -> dict feature (thay Series .loc lặp lại).
    """
    rng = np.random.default_rng(random_state)

    all_products = df_product["product_id"].astype(str).unique()
    n_products = len(all_products)

    category_to_products: dict[str, np.ndarray] = {
        str(cat): grp.index.to_numpy()
        for cat, grp in item_lookup.groupby("category_id")
    }

    item_lookup_dict = item_lookup[
        ["price", "category_id", "avg_item_sentiment"]
    ].to_dict("index")

    interaction_df = interaction_df.sort_values("purchased_at").reset_index(drop=True)

    user_history: dict[str, set] = {}
    negative_rows = []

    print("Generating negative samples (fast)...")

    for row in interaction_df.itertuples():
        user = str(row.customer_id)
        pos_item = str(row.product_id)
        hist = user_history.setdefault(user, set())
        excluded = hist | {pos_item}

        sampled_negatives = []

        # ---- hard negative: cùng category, chưa mua, khác pos_item ----
        pos_category = item_lookup_dict.get(pos_item, {}).get("category_id")
        hard_neg = None
        if pos_category is not None:
            same_cat_arr = category_to_products.get(str(pos_category))
            if same_cat_arr is not None and len(same_cat_arr) > 0:
                for _ in range(10):
                    cand = same_cat_arr[rng.integers(0, len(same_cat_arr))]
                    if cand not in excluded:
                        hard_neg = cand
                        break
                else:
                    pool = [p for p in same_cat_arr if p not in excluded]
                    if pool:
                        hard_neg = rng.choice(pool)
        if hard_neg is not None:
            sampled_negatives.append(hard_neg)
            excluded = excluded | {hard_neg}

        # ---- random negatives: rejection sampling toàn catalog ----
        n_random_needed = max(0, n_negatives - len(sampled_negatives))
        if n_random_needed > 0:
            picked: set = set()
            attempts = 0
            max_attempts = n_random_needed * 20 + 50
            while len(picked) < n_random_needed and attempts < max_attempts:
                cand = all_products[rng.integers(0, n_products)]
                if cand not in excluded and cand not in picked:
                    picked.add(cand)
                attempts += 1
            if len(picked) < n_random_needed:
                # fallback hiếm gặp (user gần như đã mua hết catalog) — full setdiff
                remaining_pool = np.setdiff1d(
                    all_products, np.array(list(excluded | picked))
                )
                extra_needed = n_random_needed - len(picked)
                if len(remaining_pool) > 0:
                    extra = rng.choice(
                        remaining_pool,
                        size=min(extra_needed, len(remaining_pool)),
                        replace=False,
                    )
                    picked.update(extra.tolist())
            sampled_negatives.extend(picked)

        for neg_item in sampled_negatives:
            info = item_lookup_dict.get(str(neg_item))
            if info is None:
                continue
            negative_rows.append(
                {
                    "customer_id": user,
                    "product_id": str(neg_item),
                    "purchased_at": row.purchased_at,
                    "total_reviews_so_far": row.total_reviews_so_far,
                    "avg_price_preference": row.avg_price_preference,
                    "positive_review_ratio": row.positive_review_ratio,
                    "has_history": row.has_history,
                    "price": info["price"],
                    "category_id": str(info["category_id"]),
                    "avg_item_sentiment": info["avg_item_sentiment"],
                    "label": 0.0,
                }
            )

        hist.add(pos_item)

    neg_df = pd.DataFrame(negative_rows)
    print(f"Generated {len(neg_df)} negative samples")
    return neg_df


def build_training_dataframe(interaction_df: pd.DataFrame, neg_df: pd.DataFrame):
    """
    Combine positive + negative samples.
    """

    final_df = pd.concat([interaction_df, neg_df], ignore_index=True)

    final_df = final_df.sort_values("purchased_at").reset_index(drop=True)

    return final_df
