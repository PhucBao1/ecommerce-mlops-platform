# Feature Store

Feast — quản lý feature cho RecSys: offline (Parquet) + online (Redis), phục vụ real-time cho serving API.

## File interactions

```
features.py
  ├── data_sources.py         — định nghĩa FileSource + PushSource
  └── feature_store.yaml      — Feast config (registry, online/offline store)

stream.py (streaming consumer)
  └── feature_stream_processor.py
        └── Feast.push_to_online_store(df) → Redis

recsys_api/main.py
  └── FeatureStore.get_online_features(
        features=["customer_recent_sentiment:recent_sentiment_score"],
        entity_rows=[{"customer_id": "12345"}]
      ) → Redis lookup
```

## ASCII flow

```
PhoBERT inference
(streaming consumer)
        │  sentiment_score, last_product_id
        ▼
┌────────────────────────────────┐
│  feature_stream_processor.py   │
│  Feast.push_to_online_store()  │
└───────────────┬────────────────┘
                │
                ▼
       Redis online store
       Key: customer_recent_sentiment:{customer_id}
       TTL: 7 days
                │
                ▼
┌────────────────────────────────┐
│  recsys_api/main.py            │
│  fs.get_online_features()      │
│  → recent_sentiment_score      │
│  → last_commented_product_id   │
└───────────────┬────────────────┘
                │
                ▼
   Sentiment-aware scoring
   (inference_faiss.py)
```

## Cấu trúc

```
feature_store/
├── feature_repo/
│   ├── feature_store.yaml      # Feast config: registry, online store, offline store
│   ├── features.py             # Entity + FeatureView definitions
│   ├── data_sources.py         # FileSource + PushSource definitions
│   └── data/
│       ├── registry.db         # Feast registry (SQLite)
│       └── dummy_sentiment.parquet   # Offline backup cho PushSource
└── requirements.txt
```

## Entity

| Entity | Join key | Mô tả |
|---|---|---|
| `customer` | `customer_id` (str) | Thực thể khách hàng |

## Feature Views

### `customer_recent_sentiment`

**Source:** `PushSource` (`sentiment_push_source`) — streaming consumer push sau mỗi review mới.

**TTL:** 7 ngày — feature tự expire nếu khách không review trong 7 ngày.

**Online:** ✅ Bật — Redis-backed, phục vụ real-time cho RecSys API.

| Feature | Type | Mô tả | Ví dụ |
|---|---|---|---|
| `recent_sentiment_score` | Float32 | Sentiment score review gần nhất | `1.0` (Positive) / `-1.0` (Negative) / `0.0` (Neutral) |
| `last_commented_product_id` | String | ID sản phẩm khách vừa review | `"277512620"` |

**Example — feature values cho user `"7290341"`:**

```python
# Sau khi user "7290341" review sản phẩm "277512620" với rating 5 (Positive):
{
    "customer_id":               ["7290341"],
    "recent_sentiment_score":    [1.0],
    "last_commented_product_id": ["277512620"],
    "event_timestamp":           ["2025-11-01T02:31:05Z"]
}

# Sau 7 ngày không có review mới:
{
    "customer_id":               ["7290341"],
    "recent_sentiment_score":    [None],   # TTL expired
    "last_commented_product_id": [None]
}
```

## Feast commands

```bash
cd src/feature_store/feature_repo

# Apply definitions lên registry
feast apply

# Materialize offline → online store (initial load)
feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)

# Check feature views
feast feature-views list
```

## Sử dụng trong RecSys API

```python
from feast import FeatureStore

fs = FeatureStore(repo_path="src/feature_store/feature_repo/")

# Get online features cho 1 user
features = fs.get_online_features(
    features=[
        "customer_recent_sentiment:recent_sentiment_score",
        "customer_recent_sentiment:last_commented_product_id",
    ],
    entity_rows=[{"customer_id": "12345"}],
).to_dict()

recent_score = features["recent_sentiment_score"][0]   # float | None
last_product = features["last_commented_product_id"][0]  # str | None
```

RecSys API dùng tenacity retry 2 lần với exponential backoff khi Feast call fail.

## Scoring logic (feature này ảnh hưởng gợi ý như thế nào)

```python
# Trong inference_faiss.py:
if recent_score is not None and recent_score < 0 and last_product:
    # Suppress sản phẩm vừa review tiêu cực
    score[last_product_idx] -= 1.0

if recent_score is not None and recent_score > 0:
    # Boost toàn bộ sản phẩm cùng category với sản phẩm vừa review tích cực
    score[same_category_mask] += 0.15
```

## Env vars

| Var | Default | Mô tả |
|---|---|---|
| `FEAST_REDIS_HOST` | `redis` | Redis host cho online store |
| `FEAST_REDIS_PORT` | `6379` | Redis port |
| `FEAST_REDIS_PASSWORD` | — | Redis password |

## Thiết kế kỹ thuật

**Tại sao TTL = 7 ngày?**
Sentiment review phản ánh trạng thái cảm xúc gần đây của user với platform. Sau 7 ngày không tương tác, sentiment cũ không còn đủ signal để ảnh hưởng recommendation — better to return `None` và không boost/suppress gì. TTL quá ngắn (1 ngày) → nhiều cache miss, quá dài (30 ngày) → stale sentiment ảnh hưởng sai.

**Tại sao PushSource thay vì batch materialization?**
Batch materialization (mỗi giờ/ngày) → sentiment feature stale tới 24h. PushSource → streaming consumer push ngay sau khi user review → feature available trong vài giây. Quan trọng khi user review tiêu cực: không muốn tiếp tục gợi ý sản phẩm đó cho đến lần sau cùng ngày.

**Tại sao chỉ có 1 FeatureView (không có item features)?**
Item features (price, sentiment) được đọc trực tiếp từ `item_lookup.parquet` tại serving time — ít thay đổi, không cần online store real-time. User features thay đổi mỗi lần interact → cần PushSource. Feature store giữ minimal: chỉ những feature thực sự cần real-time freshness.

**Tại sao dùng `dummy_sentiment.parquet` làm offline backup?**
Feast `PushSource` bắt buộc có `batch_source` để đăng ký schema. `dummy_sentiment.parquet` có đúng schema nhưng không có data thật — chỉ là placeholder để `feast apply` không lỗi. Data thật đến từ streaming push.

## Troubleshooting

**`get_online_features` trả về `None` cho tất cả features**
TTL đã expire (user không review trong 7 ngày) hoặc feature chưa được push lần nào. RecSys API xử lý `None` bằng cách bỏ qua sentiment boost/suppress — không crash.

**`feast apply` fail: `registry.db permission denied`**
```bash
chmod 664 src/feature_store/feature_repo/data/registry.db
```

**`feast apply` fail: `FileSource path not found`**
`dummy_sentiment.parquet` phải tồn tại trước khi `feast apply`. Tạo file dummy:
```python
import pandas as pd
pd.DataFrame({
    "customer_id": ["0"],
    "recent_sentiment_score": [0.0],
    "last_commented_product_id": ["0"],
    "event_timestamp": [pd.Timestamp.now()]
}).to_parquet("src/feature_store/feature_repo/data/dummy_sentiment.parquet")
```

**Feast call timeout trong RecSys API (tenacity retry 2 lần)**
Redis quá tải hoặc offline. Kiểm tra:
```bash
redis-cli -h localhost -p 6379 ping
redis-cli -h localhost info memory | grep used_memory_human
```

**Feature values không cập nhật dù streaming consumer đang chạy**
Kiểm tra consumer đang push đúng `entity_rows` format:
```python
# Đúng format — customer_id là string
entity_df = pd.DataFrame({
    "customer_id": ["7290341"],        # phải là str, không phải int
    "recent_sentiment_score": [1.0],
    "last_commented_product_id": ["277512620"],
    "event_timestamp": [pd.Timestamp.now()]
})
fs.push("recent_sentiment_push_source", entity_df)
```
