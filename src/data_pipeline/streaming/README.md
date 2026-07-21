# Streaming

Kafka consumer + producer: nhận review real-time từ `comments-raw`, chạy PhoBERT inference, push feature vào Feast online store, invalidate Redis cache.

## File interactions

```
stream.py (StreamProcessor)
  ├── sentiment_model.py          — load PhoBERT, predict_sentiment(text) → score
  ├── feature_stream_processor.py — Feast push_to_online_store(features)
  ├── metrics.py                  — Prometheus counters/gauges
  └── src/serving/recsys_api/
      └── cache.py                — invalidate_recommendation_cache(customer_id)

producer/
  ├── producer.py                 — Kafka Producer wrapper
  ├── reviews.py                  — format review event → Kafka message
  └── metrics.py                  — producer-side Prometheus metrics
```

## ASCII flow

```
Tiki API / Crawler
        │  POST review event
        ▼
┌──────────────────────────────┐
│  producer/producer.py        │
│  format → JSON message       │
│  produce → comments-raw      │
└──────────────────────────────┘
        │
        ▼  Kafka topic: comments-raw
┌──────────────────────────────┐
│  stream.py (StreamProcessor) │
│                              │
│  1. consume message          │
│  2. validate schema          │
│  3. predict_sentiment()  ────┼──► sentiment_model.py (PhoBERT)
│  4. buffer (100 msgs)        │
│  5. flush every 30s          │
│     ├── Feast push ──────────┼──► feature_stream_processor.py → Redis
│     └── invalidate cache ───┼──► cache.py → Redis DEL rec:{customer_id}
│  6. commit offset            │   (chỉ commit SAU khi flush thành công)
│  errors → recsys-dlq ────────┼──► DeadLetterQueue topic
└──────────────────────────────┘
```

## Files

| File | Mô tả |
|---|---|
| `stream.py` | Kafka consumer chính — `StreamProcessor` class |
| `feature_stream_processor.py` | Push sentiment feature vào Feast online store |
| `feedback_processor.py` | Xử lý feedback event từ RecSys API → upsample positive |
| `prediction_consumer.py` | Consume prediction events để log/monitor |
| `sentiment_model.py` | Wrapper load + inference PhoBERT (CPU, no HTTP) |
| `mock_producer.py` | Tạo fake events để test local |
| `metrics.py` | Prometheus counters/gauges cho streaming |
| `producer/` | Producer service — crawl event → Kafka topic |

## Kafka Topics

| Topic | Chiều | Schema |
|---|---|---|
| `comments-raw` | Consumer đọc vào | Review mới từ crawler/user |
| `predictions-stream` | Consumer đọc vào | Prediction events từ RecSys API |
| `features-stream` | Producer push ra | Sentiment feature updates |
| `recsys-dlq` | Dead Letter Queue | Messages lỗi không xử lý được |

## Message schema

### `comments-raw` (input)

| Field | Type | Mô tả |
|---|---|---|
| `product_id` | str | ID sản phẩm |
| `review_id` | str | ID review |
| `customer_id` | str | ID khách hàng |
| `rating` | int | Điểm 1–5 |
| `comment` | str | Nội dung review |
| `purchased_at` | str (ISO 8601) | Thời điểm mua |
| `event_timestamp` | str (ISO 8601) | Thời điểm tạo event |

**Example message:**
```json
{
  "product_id": "277512620",
  "review_id": "48392011",
  "customer_id": "7290341",
  "rating": 5,
  "comment": "Pin trâu, camera chụp siêu nét, màn hình đẹp!",
  "purchased_at": "2025-10-15T14:22:00Z",
  "event_timestamp": "2025-11-01T02:31:05Z"
}
```

### `features-stream` / Feast push (output)

Sau PhoBERT inference, processor push vào Feast online store (Redis):

| Feature | Type | Mô tả | Ví dụ |
|---|---|---|---|
| `customer_id` | str | Join key | `"7290341"` |
| `recent_sentiment_score` | Float32 | Sentiment score (-1 Negative, 0 Neutral, +1 Positive) | `1.0` |
| `last_commented_product_id` | String | ID sản phẩm vừa review | `"277512620"` |
| `event_timestamp` | datetime | Thời điểm event | `2025-11-01T02:31:05Z` |

### `recsys-dlq` (Dead Letter Queue)

```json
{
  "event": { "...original message..." },
  "error": "Feast push timeout after 5s",
  "stage": "feast_push",
  "timestamp": "2025-11-01T02:31:10Z"
}
```

## Cấu hình (env vars)

| Var | Default | Mô tả |
|---|---|---|
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka brokers |
| `BUFFER_SIZE` | `100` | Số message gom trước khi flush Feast |
| `FLUSH_INTERVAL_SEC` | `30` | Max giây giữa 2 lần flush |
| `MAX_IO_WORKERS` | `10` | Thread pool size cho Feast writes |
| `DLQ_TOPIC` | `recsys-dlq` | Dead Letter Queue topic |

## Chạy

```bash
# Consumer
python -m src.data_pipeline.streaming.stream

# Mock producer (dev/test)
python -m src.data_pipeline.streaming.mock_producer

# Metrics port :8010
curl http://localhost:8010/metrics
```

## Prometheus metrics

| Metric | Type | Mô tả |
|---|---|---|
| `stream_messages_processed_total` | Counter | Messages đã xử lý thành công |
| `stream_messages_failed_total` | Counter | Messages lỗi (→ DLQ) |
| `stream_feast_push_duration_seconds` | Histogram | Latency push vào Feast |
| `stream_buffer_size` | Gauge | Buffer size hiện tại |

## Thiết kế kỹ thuật

**Tại sao buffer 100 messages trước khi flush Feast?**
Feast `push_to_online_store` có overhead khởi tạo connection mỗi lần gọi. Flush từng message → 1 API call/message → latency tích lũy cao. Buffer 100 messages → batch push → giảm overhead 100×. Flush interval 30s đảm bảo feature không stale quá 30 giây dù traffic thấp.

**Tại sao commit offset SAU khi flush Feast thành công?**
At-least-once + idempotent push: nếu process crash sau khi flush nhưng trước khi commit offset, Kafka sẽ redeliver message — Feast push lại cùng value không thay đổi gì (idempotent). Commit offset trước flush: nếu Feast fail, mất feature update mà Kafka không redeliver.

**Tại sao invalidate Redis cache sau mỗi Feast push?**
Recommendation cache được build dựa trên `recent_sentiment_score`. Nếu user vừa review tiêu cực một sản phẩm, cache cũ vẫn đề xuất sản phẩm đó. Invalidate → request tiếp theo sẽ rebuild cache với sentiment mới. Đây là trade-off: cache miss tăng nhẹ nhưng recommendation luôn reflect sentiment mới nhất.

**Tại sao Dead Letter Queue là Kafka topic thay vì file?**
File DLQ (như crawler) phù hợp với single-process. Streaming consumer có thể chạy nhiều instance — file DLQ gây race condition khi nhiều consumer cùng ghi. Kafka DLQ topic → các instance cùng produce vào một nơi, có thể có consumer riêng để retry/monitor.

## Troubleshooting

**Consumer lag tăng dần (Kafka consumer group lag > 10000)**
Consumer xử lý chậm hơn producer. Nguyên nhân thường: Feast push timeout hoặc PhoBERT inference chậm. Kiểm tra:
```bash
# Xem consumer group lag
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --describe --group recsys-stream-group

# Xem metrics
curl http://localhost:8010/metrics | grep stream_feast_push
```
Fix: tăng `MAX_IO_WORKERS` hoặc scale thêm consumer instance.

**`Feast push timeout` — messages vào DLQ nhiều**
Redis offline hoặc quá tải. Kiểm tra:
```bash
docker compose -f docker-compose.infra.yml ps redis
redis-cli -h localhost ping
```

**`predict_sentiment` trả về sai class liên tục**
Model path sai hoặc model chưa được train. Kiểm tra `MODEL_PATH` env và chạy test nhanh:
```python
from src.data_pipeline.streaming.sentiment_model import predict_sentiment
print(predict_sentiment("Sản phẩm tệ quá"))  # Nên ra Negative
```

**DLQ topic `recsys-dlq` không tồn tại**
Kafka chưa auto-create topic. Tạo thủ công:
```bash
kafka-topics.sh --create --topic recsys-dlq \
  --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
```

**`invalidate_recommendation_cache` fail — Redis connection refused**
Streaming consumer import `cache.py` từ `recsys_api`. Nếu Redis chưa chạy, consumer sẽ log warning nhưng không crash — cache invalidation bị bỏ qua, recommendation cache có thể stale đến khi TTL expire.
