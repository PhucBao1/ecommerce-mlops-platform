# RecSys API

FastAPI service — gợi ý sản phẩm real-time với Two-Tower + SASRec, multi-level cache, A/B routing, OTel tracing. Port **8001**.

## File interactions

```
main.py
  ├── loaders.py          — load model, encoders, scaler, data, build VECTOR_STORE
  │     ├── model.py      — TwoTowerModel definition
  │     └── vector_store.py — FAISS / Qdrant backend (via VECTOR_STORE_BACKEND env)
  ├── inference_faiss.py  — recommend() — full serving logic
  │     ├── loaders.py    — USER_HISTORY_DICT, TRENDING_ITEM_IDS, model, VECTOR_STORE
  │     ├── reranker.py   — rerank_candidates(), _build_explanation()
  │     └── candidate_cache.py — 200-item candidate cache per user
  ├── cache.py            — get/set Redis recommendation cache
  ├── kafka_producer.py   — send_prediction_event() (background task)
  └── retrieval.py        — SASRec endpoint helper (lazy load sasrec model)
```

## ASCII flow

```
POST /recommend  {"customer_id": "12345", "top_k": 5}
        │
        ├─► Redis cache hit?
        │   └── YES → return (~5ms)  [source: "redis_cache"]
        │
        ├─► A/B routing: MD5(customer_id) % 100
        │   ├── < 10  → "experiment" (diversity_limit=2)
        │   └── >= 10 → "control"    (diversity_limit=3)
        │
        ├─► Feast.get_online_features() (retry 2×, exp backoff)
        │   ├── recent_sentiment_score  (Float32 | None)
        │   └── last_commented_product_id (str | None)
        │
        ├─► has_history?
        │   └── NO → TRENDING_ITEM_IDS top-200  [source: "trending"]
        │              └── rerank → top_k → return
        │
        ├─► user_tower(user_features) → user_vector (32-dim)
        │
        ├─► candidate_cache hit?
        │   ├── YES → 200 candidates (skip FAISS)
        │   └── NO  → VECTOR_STORE.search(user_vector, top=200)
        │              └── cache 200 candidates (Redis)
        │
        ├─► Sentiment-aware re-scoring:
        │   ├── recent_sentiment < 0 → score[last_product] -= 1.0
        │   └── recent_sentiment > 0 → score[same_category] += 0.15
        │
        ├─► filter purchased items
        │
        ├─► reranker.rerank_candidates():
        │   final = predict_score
        │         + 0.05 × price_boost
        │         + 0.03 × sentiment_boost
        │         + 0.02 × quality_boost
        │
        ├─► top_k items + explanation per item
        │
        ├─► [background] kafka_producer.send_prediction_event()
        └─► [background] cache.cache_recommendations()
```

## Endpoints

### `GET /health`

```bash
curl http://localhost:8001/health
```

### `POST /recommend`

Gợi ý Top-K sản phẩm cho user đã đăng nhập.

**Request schema:**

| Field | Type | Bắt buộc | Default | Mô tả |
|---|---|---|---|---|
| `customer_id` | str | ✅ | — | ID khách hàng |
| `top_k` | int | ❌ | 10 | Số sản phẩm trả về |

```bash
curl -X POST http://localhost:8001/recommend \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "12345", "top_k": 5}'

**Request:**
```json
{"customer_id": "7290341", "top_k": 5}
```

**Response example:**
```json
{
  "recommendations": [
    {
      "product_id": "279257510",
      "product_name": "iPhone 15 Pro Max 256GB",
      "category_id": "dien-thoai",
      "price": 28990000.0,
      "thumbnail_url": "https://cdn.tiki.vn/...",
      "predict_score": 0.923,
      "final_score": 0.961,
      "explanation": {
        "top_reason": "Phù hợp tầm giá (87%)",
        "sentiment_score": 0.91,
        "factors": [
          "Phù hợp tầm giá (87%)",
          "Đánh giá tốt (4.6/5⭐)",
          "Cùng danh mục yêu thích"
        ]
      }
    }
  ],
  "source": "faiss",
  "experiment_group": "control",
  "trace_id": "3e4f2c1a..."
}
```

**Response schema:**

| Field | Type | Mô tả |
|---|---|---|
| `recommendations` | List[Item] | Danh sách sản phẩm gợi ý |
| `source` | str | `"redis_cache"` / `"faiss"` / `"qdrant"` / `"trending"` |
| `experiment_group` | str | `"control"` / `"experiment"` |
| `trace_id` | str | OpenTelemetry trace ID |

**Item schema:**

| Field | Type | Mô tả | Ví dụ |
|---|---|---|---|
| `product_id` | str | ID sản phẩm | `"279257510"` |
| `product_name` | str | Tên | `"iPhone 15 Pro Max 256GB"` |
| `category_id` | str | Danh mục | `"dien-thoai"` |
| `price` | float | Giá (VND) | `28990000.0` |
| `thumbnail_url` | str | URL ảnh | `"https://cdn.tiki.vn/..."` |
| `predict_score` | float | Raw similarity score (dot product) | `0.923` |
| `final_score` | float | Score sau rerank | `0.961` |
| `explanation.top_reason` | str | Lý do chính | `"Phù hợp tầm giá (87%)"` |
| `explanation.sentiment_score` | float | Avg sentiment item | `0.91` |
| `explanation.factors` | List[str] | Tất cả factors | `[...]` |

### `POST /recommend/session`
Gợi ý dựa trên session (không cần login).

**Request schema:**

| Field | Type | Bắt buộc | Mô tả |
|---|---|---|---|
| `session_items` | List[str] | ✅ | Danh sách product_id đã xem/mua trong session |
| `top_k` | int | ❌ | Số sản phẩm trả về (default 10) |

```bash
curl -X POST http://localhost:8001/recommend/session \
  -H "Content-Type: application/json" \
  -d '{"session_items": ["789", "456"], "top_k": 5}'
**Request:**
```json
{"session_items": ["277512620", "279257510"], "top_k": 5}
```

### `GET /metrics`

Prometheus scrape endpoint.

## Luồng serving `/recommend`

```
1. Redis cache hit? → return ngay (~5ms)
2. A/B routing: MD5(customer_id) — 10% "experiment", 90% "control"
   - control: diversity_limit=3
   - experiment: diversity_limit=2
3. Feast.get_online_features (retry 2x, exp backoff):
   - recent_sentiment_score (Float32)
   - last_commented_product_id (String)
4. Has user history? → No → trending 200 items (top purchase last 30d)
5. Two-Tower user_tower → user_vector (32-dim)
6. Candidate cache hit (200 items)? → Yes → skip FAISS
   No → FAISS/Qdrant ANN → cache 200 candidates
7. Sentiment-aware re-scoring:
   - recent_sentiment < 0 → score[last_product] -= 1.0
   - recent_sentiment > 0 → score[same_category] += 0.15
8. Filter items already purchased
9. Reranker:
   final_score = predict_score + 0.05×price_boost + 0.03×sentiment_boost + 0.02×quality_boost
10. Top-K + explanation per item
11. Background task: Kafka publish event (non-blocking)
12. Background task: Redis cache result (non-blocking)
```
## Features dùng tại serving time
| `recent_sentiment_score` | Feast online store (Redis) | Sentiment review gần nhất của user |
| `last_commented_product_id` | Feast online store (Redis) | Sản phẩm vừa review |
| `avg_price_preference` | `user_history.parquet` | Tầm giá user |
| `positive_review_ratio` | `user_history.parquet` | Tỷ lệ review tích cực |
| `total_reviews_so_far` | `user_history.parquet` | Log(số lần mua) |
| `price` | `item_lookup.parquet` | Giá item |
| `avg_item_sentiment` | `item_lookup.parquet` | Sentiment trung bình item |
| `category_id` | `item_lookup.parquet` | Category (embedding) |
| Feature | Nguồn | Type | Ví dụ |
|---|---|---|---|
| `recent_sentiment_score` | Feast / Redis | Float32 \| None | `1.0` |
| `last_commented_product_id` | Feast / Redis | str \| None | `"277512620"` |
| `avg_price_preference` | `user_history.parquet` | float | `18000000.0` |
| `positive_review_ratio` | `user_history.parquet` | float | `0.75` |
| `total_reviews_so_far` | `user_history.parquet` | float | `1.099` |
| `price` | `item_lookup.parquet` | float | `20990000.0` |
| `avg_item_sentiment` | `item_lookup.parquet` | float | `0.82` |
| `category_id` | `item_lookup.parquet` | str | `"dien-thoai"` |

## Reranker scoring

```python
# price_boost = 1.0 nếu item price trong ±20% avg_price_preference, else 0
# sentiment_boost = avg_item_sentiment của item
# quality_boost = avg_item_sentiment × log(1 + review_count)
final_score = predict_score + 0.05*price_boost + 0.03*sentiment_boost + 0.02*quality_boost
```

## Prometheus metrics

| Metric | Labels | Mô tả |
|---|---|---|

| `recommendation_latency_seconds` | `source` | Histogram — end-to-end latency |
| `cold_start_total` | — | Counter — cold start requests |

## Caching

### Recommendation cache (Redis)
- Key: `rec:{customer_id}:{top_k}`
- TTL: mặc định không expire (invalidated khi user có review mới qua streaming consumer)

### Candidate cache
- Cache 200 candidates per user (FAISS results)
- Phục vụ nhiều request liên tiếp mà không cần ANN search lại

## Graceful shutdown

Khi nhận SIGTERM:
1. Flush pending OpenTelemetry spans (timeout 10s)
2. Flush Kafka producer buffer
## Env vars

| Var | Default | Mô tả |
|---|---|---|
| `REDIS_HOST` | `redis` | Redis host |
| `REDIS_PASSWORD` | — | Redis auth |
| `VECTOR_STORE_BACKEND` | `faiss` | `"faiss"` hoặc `"qdrant"` |
| `RETRIEVAL_BACKEND` | `twotower` | `"twotower"` hoặc `"lightgcn"` |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow registry |
| `KAFKA_BOOTSTRAP_SERVERS` | `kafka:9092` | Kafka broker |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://jaeger:4317` | Jaeger endpoint |

## Chạy local

```bash
docker compose -f docker-compose.app.yml up recsys-api
# LightGCN
RETRIEVAL_BACKEND=lightgcn docker compose -f docker-compose.app.yml up recsys-api
# Qdrant
VECTOR_STORE_BACKEND=qdrant docker compose -f docker-compose.app.yml up recsys-api
```

Swagger UI: `http://localhost:8001/docs`

## Thiết kế kỹ thuật

**Tại sao 2-level cache (recommendation cache + candidate cache)?**
Recommendation cache: cache kết quả top-K cuối, serve lại không tốn compute (~5ms). Candidate cache: cache 200 candidates FAISS search — vì FAISS search là phần tốn nhất (matrix multiply), trong khi reranking nhanh hơn. Nếu user gửi nhiều request liên tiếp với `top_k` khác nhau, candidate cache tránh FAISS search mỗi lần.

**Tại sao A/B routing dùng MD5 hash thay vì random?**
Random → cùng 1 user có thể vào group khác nhau mỗi request → experiment không nhất quán, impossible to measure per-user impact. MD5 hash deterministic → cùng `customer_id` luôn vào cùng group → experiment consistent và reproducible.

**Tại sao Kafka publish và Redis cache là background tasks?**
Không nên block HTTP response vì Kafka publish hoặc Redis write. Nếu Kafka down, user vẫn nhận được recommendation ngay. Background tasks chạy sau khi response đã được trả về → p50 latency không bị ảnh hưởng bởi I/O operations không critical.

**Tại sao graceful shutdown flush Kafka trước khi tắt?**
Kafka producer buffer messages internally trước khi batch-send. Nếu process tắt đột ngột (SIGKILL), buffered events mất → không có prediction log cho analytics/monitoring. `producer.flush(timeout=10)` đảm bảo tất cả events được gửi trước khi tắt.

**Tại sao trending items là top 200 thay vì top 10?**
Trending top-10 quá hẹp: tất cả cold-start users nhận cùng 10 sản phẩm → zero diversity, poor UX. Trending top-200 → reranker có đủ candidates để apply price boost, sentiment boost, diversity limit → mỗi user nhận recommendations hơi khác nhau dù đều cold-start.

## Troubleshooting

**Startup fail: `FileNotFoundError: user_encoder.pkl`**
Training pipeline chưa chạy. Chạy generate data + training trước:
```bash
python scripts/generate_fake_data.py
python src/ml_models/recsys/train_model.py
```

**`POST /recommend` luôn trả về `source: "trending"` dù user có history**
`customer_id` không match với `USER_HISTORY_DICT` key format. Kiểm tra type — key phải là str không có `.0`:
```python
# Trong loaders.py, kiểm tra:
print(list(USER_HISTORY_DICT.keys())[:5])  # nên ra: ["10001", "10002", ...]
```

**Recommendation latency p99 > 500ms**
Candidate cache miss rate cao (FAISS search chạy mỗi request). Nguyên nhân: Redis cache bị invalidate liên tục (nhiều reviews mới). Monitor:
```promql
rate(recommendation_latency_seconds_count{source="redis_cache"}[5m])
  / rate(recommendation_latency_seconds_count[5m])
```
Nếu cache hit rate < 30%, kiểm tra Redis TTL config.

**`Feast get_online_features timeout` — tenacity retry 2 lần vẫn fail**
Redis cho Feast offline. Serving vẫn hoạt động (fallback: sentiment=None, không boost/suppress gì). Kiểm tra:
```bash
redis-cli -h redis ping
```

**`FAISS index dimension mismatch`**
Model được retrain với `embedding_dim` khác nhưng FAISS index cũ vẫn dim=32. Rebuild index:
```bash
python src/ml_models/recsys/retrieval/export_embeddings.py
python src/ml_models/recsys/retrieval/build_faiss_index.py
```

**Cold start rate > 80% (`cold_start_total` metric tăng nhanh)**
Bình thường với data nhỏ/fake. Trong production, giảm cold start bằng cách seed thêm interaction data hoặc dùng `session_items` endpoint cho new users.
