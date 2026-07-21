"""
Generate synthetic e-commerce data cho training Two-Tower RecSys.

Chạy:
    pip install faker numpy pandas pyarrow
    python scripts/generate_fake_data.py

Output (artifacts/recsys_models/data_menu/):
    item_lookup.parquet   — item catalogue
    user_history.parquet  — user-item interactions (training data)
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker
from faker.providers import BaseProvider

fake = Faker("vi_VN")
random.seed(42)
np.random.seed(42)

# ─── Config ──────────────────────────────────────────────
N_ITEMS = 2_000
N_USERS = 5_000
N_INTERACTIONS = 15_000  # sparse: ~3 interactions per user on average
# nhưng thực tế phân phối long-tail:
# ~60% user chỉ có 1 interaction

OUT_DIR = Path("artifacts/recsys_models/data_menu")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Categories & Brands ─────────────────────────────────
CATEGORIES = {
    "dien-thoai": {"label": "Điện thoại", "price_range": (3_000_000, 35_000_000)},
    "laptop": {"label": "Laptop", "price_range": (8_000_000, 50_000_000)},
    "phu-kien": {"label": "Phụ kiện", "price_range": (50_000, 2_000_000)},
    "thoi-trang": {"label": "Thời trang", "price_range": (100_000, 3_000_000)},
    "sach": {"label": "Sách", "price_range": (30_000, 500_000)},
    "my-pham": {"label": "Mỹ phẩm", "price_range": (100_000, 5_000_000)},
    "gia-dung": {"label": "Gia dụng", "price_range": (200_000, 15_000_000)},
    "do-choi": {"label": "Đồ chơi", "price_range": (50_000, 2_000_000)},
}

BRANDS = {
    "dien-thoai": ["Apple", "Samsung", "Xiaomi", "OPPO", "Vivo"],
    "laptop": ["Dell", "HP", "Asus", "Lenovo", "Acer"],
    "phu-kien": ["Anker", "Baseus", "Ugreen", "Belkin", "Generic"],
    "thoi-trang": ["Zara", "H&M", "Canifa", "Owen", "CoolMate"],
    "sach": ["NXB Trẻ", "NXB Kim Đồng", "Alphabooks", "First News"],
    "my-pham": ["L'Oreal", "Innisfree", "The Face Shop", "Pond's", "Olay"],
    "gia-dung": ["Sunhouse", "Panasonic", "Philips", "Kangaroo", "Electrolux"],
    "do-choi": ["LEGO", "Hot Wheels", "Barbie", "Duplo", "VinaToy"],
}

# ─── 1. Generate Items ────────────────────────────────────
print("Generating items...")

items = []
cat_ids = list(CATEGORIES.keys())
# Phân phối category không đều — thực tế hơn
cat_weights = [0.20, 0.15, 0.20, 0.15, 0.08, 0.10, 0.07, 0.05]

for i in range(N_ITEMS):
    cat_id = random.choices(cat_ids, weights=cat_weights)[0]
    cat_info = CATEGORIES[cat_id]
    lo, hi = cat_info["price_range"]

    # Price phân phối log-normal (nhiều item rẻ, ít item đắt)
    price = int(np.exp(np.random.uniform(np.log(lo), np.log(hi))) / 1000) * 1000

    # Sentiment: hầu hết positive (thực tế trên Tiki bias positive)
    avg_sentiment = float(np.clip(np.random.beta(5, 2), 0, 1))  # beta(5,2) → skew high

    brand = random.choice(BRANDS[cat_id])
    product_name = f"{brand} {fake.word().capitalize()} {random.randint(1, 99)}"
    thumbnail_url = f"https://cdn.tiki.vn/media/catalog/product/{i:06d}.jpg"

    items.append(
        {
            "product_id": str(i + 1),
            "product_name": product_name,
            "category_id": cat_id,
            "category_name": cat_info["label"],
            "brand_name": brand,
            "price": float(price),
            "avg_item_sentiment": avg_sentiment,
            "thumbnail_url": thumbnail_url,
        }
    )

item_df = pd.DataFrame(items)
item_df.to_parquet(OUT_DIR / "item_lookup.parquet", index=False)
print(f"  ✓ {len(item_df)} items → item_lookup.parquet")

# ─── 2. Generate Users + Interactions ────────────────────
print("Generating user interactions...")


# Long-tail interaction count: 60% user có 1 interaction, 30% có 2-5, 10% có 5+
def sample_interaction_count():
    r = random.random()
    if r < 0.60:
        return 1
    elif r < 0.90:
        return random.randint(2, 5)
    else:
        return random.randint(6, 20)


interactions = []
user_id_counter = 10001

total_generated = 0
while total_generated < N_INTERACTIONS:
    customer_id = str(user_id_counter)
    user_id_counter += 1

    n_buys = sample_interaction_count()

    # User có price preference — mua item trong range đó
    price_pref = float(np.exp(np.random.uniform(np.log(50_000), np.log(30_000_000))))
    # Cho phép mua item giá ±50% so với preference
    price_lo = price_pref * 0.5
    price_hi = price_pref * 1.5

    # User có category preference (80% mua trong 1-2 category)
    preferred_cats = random.choices(cat_ids, weights=cat_weights, k=2)

    eligible = item_df[
        (item_df["price"] >= price_lo)
        & (item_df["price"] <= price_hi)
        & (item_df["category_id"].isin(preferred_cats))
    ]
    if len(eligible) < n_buys:
        eligible = item_df[
            (item_df["price"] >= price_lo) & (item_df["price"] <= price_hi)
        ]
    if len(eligible) == 0:
        eligible = item_df

    bought_items = eligible.sample(min(n_buys, len(eligible)), replace=False)

    # Timestamps: mua trong 90 ngày gần đây, theo thứ tự
    base_time = pd.Timestamp("2024-10-01")
    timestamps = sorted(
        [
            base_time
            + pd.Timedelta(days=random.uniform(0, 90), hours=random.randint(0, 23))
            for _ in range(len(bought_items))
        ]
    )

    pos_ratio = float(np.clip(np.random.beta(4, 2), 0, 1))

    for j, (_, item_row) in enumerate(bought_items.iterrows()):
        interactions.append(
            {
                "customer_id": customer_id,
                "product_id": item_row["product_id"],
                "purchased_at": timestamps[j],
                # Running features (snapshot at time of purchase)
                "total_reviews_so_far": j + 1,
                "avg_price_preference": price_pref,
                "positive_review_ratio": pos_ratio,
            }
        )

    total_generated += len(bought_items)

    if user_id_counter % 500 == 0:
        print(f"  {user_id_counter - 10001} users / {total_generated} interactions...")

history_df = pd.DataFrame(interactions)
history_df["purchased_at"] = pd.to_datetime(history_df["purchased_at"])
history_df.to_parquet(OUT_DIR / "user_history.parquet", index=False)

print(
    f"  ✓ {len(history_df)} interactions, {history_df['customer_id'].nunique()} users → user_history.parquet"
)

# ─── 3. Summary ──────────────────────────────────────────
counts = history_df.groupby("customer_id").size()
print("\n── Data summary ─────────────────────────────")
print(f"Items:        {len(item_df):,}")
print(f"Users:        {history_df['customer_id'].nunique():,}")
print(f"Interactions: {len(history_df):,}")
print(
    f"Sparsity:     {1 - len(history_df) / (len(item_df) * history_df['customer_id'].nunique()):.4%}"
)
print(f"\nInteractions per user:")
print(f"  1 interaction:  {(counts == 1).sum():,} users ({(counts == 1).mean():.1%})")
print(f"  2-5:            {((counts >= 2) & (counts <= 5)).sum():,} users")
print(f"  6+:             {(counts >= 6).sum():,} users")
print(f"  median:         {counts.median():.0f}")
print(f"  max:            {counts.max()}")
print(f"\nCategory distribution:")
print(item_df["category_id"].value_counts().to_string())
print("\n✅ Done. Files saved to", OUT_DIR)
