"""Market-basket analysis: tìm cặp category hay được mua cùng nhau (lift-based),
bổ sung cho CATEGORY_COMPLEMENTS hard-code trong reranker.py (chỉ phủ 2 ngành
hàng phone/laptop, 4/741 category — bug thật 19/7/2026, xem BENCHMARK_RESULTS.md
mục 26).

Chạy local (`python -m src.ml_models.recsys.training.build_category_complements`
từ root repo), output JSON được reranker.py load lúc startup, merge với mapping
tay (mapping tay ưu tiên nếu trùng category).
"""

import itertools
import json
from collections import Counter, defaultdict

import pandas as pd

MIN_CUSTOMERS_PER_CAT = 30  # bỏ category quá hiếm, tránh lift ảo do mẫu nhỏ
MIN_COOCCUR = 20  # số khách tối thiểu mua cả 2 category mới tính
MIN_LIFT = 1.5  # co-occur cao hơn ngẫu nhiên ít nhất 1.5 lần
TOP_N_PER_CAT = 5  # tối đa 5 complement category / category gốc

OUTPUT_PATH = "artifacts/recsys_models/category_complements.json"


def main() -> None:
    hist = pd.read_parquet("artifacts/recsys_models/data_menu/user_history.parquet")
    items = pd.read_parquet("artifacts/recsys_models/data_menu/item_lookup.parquet")

    merged = hist.merge(
        items[["product_id", "category_id"]], on="product_id", how="left"
    )
    merged = merged.dropna(subset=["category_id"])
    merged["category_id"] = merged["category_id"].astype(str)
    merged["customer_id"] = merged["customer_id"].astype(str)
    merged = merged[merged["customer_id"] != "0"]  # bỏ placeholder

    cust_cats = merged.groupby("customer_id")["category_id"].apply(set)
    n_customers = len(cust_cats)
    print(f"n_customers={n_customers}")

    cat_count: Counter = Counter()
    pair_count: Counter = Counter()
    for cats in cust_cats:
        for c in cats:
            cat_count[c] += 1
        for a, b in itertools.combinations(sorted(cats), 2):
            pair_count[(a, b)] += 1

    print(f"n_categories={len(cat_count)}, n_pairs_observed={len(pair_count)}")

    # lift(A,B) = P(A,B) / (P(A)*P(B)) — >1 nghĩa là mua cùng nhau NHIỀU HƠN
    # ngẫu nhiên, đúng định nghĩa cross-sell thật (không chỉ 2 category phổ
    # biến tình cờ cùng xuất hiện).
    complements: dict[str, list[tuple[str, float, int]]] = defaultdict(list)
    for (a, b), co in pair_count.items():
        if co < MIN_COOCCUR:
            continue
        ca, cb = cat_count[a], cat_count[b]
        if ca < MIN_CUSTOMERS_PER_CAT or cb < MIN_CUSTOMERS_PER_CAT:
            continue
        p_a, p_b, p_ab = ca / n_customers, cb / n_customers, co / n_customers
        lift = p_ab / (p_a * p_b)
        if lift < MIN_LIFT:
            continue
        complements[a].append((b, lift, co))
        complements[b].append((a, lift, co))

    result = {}
    for cat, lst in complements.items():
        lst.sort(key=lambda x: x[1], reverse=True)
        result[cat] = [c for c, _lift, _co in lst[:TOP_N_PER_CAT]]

    print(f"n_categories_with_complements={len(result)}")

    cat_name = dict(zip(items["category_id"].astype(str), items["category_name"]))
    for cat in list(result.keys())[:10]:
        names = [cat_name.get(c, c) for c in result[cat]]
        print(f"{cat_name.get(cat, cat)} ({cat}) -> {names}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
