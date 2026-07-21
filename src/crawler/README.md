# Crawler

Thu thập sản phẩm và review từ Tiki API, lưu Parquet snappy vào Bronze layer (MinIO).

## File interactions

```
tiki_crawl.py
  ├── CheckpointManager   — lưu/đọc crawler_checkpoint.json
  ├── CircuitBreaker      — đếm failures, block API call khi quá ngưỡng
  ├── DeadLetterQueue     — ghi dead_letter_queue.jsonl khi batch fail
  ├── MetricsCollector    — đếm API calls, timing, bytes written
  └── AsyncLogQueue       — ghi log non-blocking (thread riêng)

schemas.py                — Pydantic validate output trước khi lưu MinIO
```

## ASCII flow

```
┌─────────────────────────────────────┐
│  crawl_product_ids()                │
│  Tiki API /v2/products?category=... │
│  → List[product_id]                 │
└──────────────┬──────────────────────┘
               │  batch 200 IDs
               ▼
┌─────────────────────────────────────┐
│  crawl_product_details()            │
│  GET /v2/products/{id}              │
│  → parse 23 fields per product      │
└──────────────┬──────────────────────┘
               │  buffer 5000 rows
               ▼
┌─────────────────────────────────────┐
│  save_to_minio()                    │
│  s3://warehouse/bronze/             │
│  products_{cat}_{ts}.parquet        │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  crawl_comments()                   │
│  GET /v2/reviews?product_id=...     │
│  → parse 11 fields per review       │
└──────────────┬──────────────────────┘
               │  buffer 20000 rows
               ▼
┌─────────────────────────────────────┐
│  save_to_minio()                    │
│  s3://warehouse/bronze/             │
│  comments_{cat}_{ts}.parquet        │
└─────────────────────────────────────┘
```

## Files

| File | Mô tả |
|---|---|
| `spiders/tiki_crawl.py` | Spider chính — `TikiCrawler` class với đầy đủ enterprise features |
| `schemas.py` | Pydantic schema `TikiProduct`, `TikiReview` — validate trước khi lưu |

## Chạy

```bash
python src/crawler/spiders/tiki_crawl.py
# Mặc định crawl 3 category: Điện thoại, Laptop, Thiết bị số
# Output: s3://warehouse/bronze/products_<cat>_<ts>.parquet
#         s3://warehouse/bronze/comments_<cat>_<ts>.parquet
```

## Output schema

### Products — `bronze/products_*.parquet`

| Field | Type | Mô tả |
|---|---|---|
| `product_id` | Int64 | ID sản phẩm Tiki |
| `sku` | Int64 | SKU sản phẩm |
| `product_name` | str | Tên sản phẩm |
| `short_description` | str | Mô tả ngắn |
| `price` | Int32 | Giá bán hiện tại (VND) |
| `list_price` | Int32 | Giá gốc trước khi giảm |
| `discount_rate` | float32 | % giảm giá |
| `rating` | float32 | Điểm đánh giá trung bình (0–5) |
| `review_count` | Int32 | Tổng số review |
| `inventory_status` | str | Trạng thái tồn kho (`"available"` / `"out_of_stock"`) |
| `stock_qty` | Int32 | Số lượng còn trong kho |
| `quantity_sold` | Int64 | Tổng số lượng đã bán |
| `brand_id` | Int64 | ID thương hiệu |
| `brand_name` | str | Tên thương hiệu |
| `category_id` | Int64 | ID danh mục |
| `category_name` | str | Tên danh mục (từ API) |
| `seller_id` | Int64 | ID nhà bán |
| `seller_name` | str | Tên nhà bán |
| `seller_logo` | str | URL logo nhà bán |
| `seller_link` | str | URL trang nhà bán |
| `url` | str | Short URL sản phẩm |
| `thumbnail_url` | str | URL ảnh đại diện |
| `all_specs` | str | Raw specifications JSON (stringify) |
| `general_category` | str | Tên danh mục tổng quát (do crawler gán) |
| `crawl_time` | timestamp | Thời điểm crawl (UTC) |
| `batch_id` | str | UUID của batch crawl |

**Example row:**

| product_id | product_name | price | list_price | discount_rate | rating | review_count | quantity_sold | brand_name | category_name | inventory_status |
|---|---|---|---|---|---|---|---|---|---|---|
| `277512620` | `"Samsung Galaxy S24 Ultra 256GB"` | `20990000` | `23490000` | `0.11` | `4.5` | `1823` | `3421` | `"Samsung"` | `"Điện Thoại & Máy Tính Bảng"` | `"available"` |

### Comments — `bronze/comments_*.parquet`

| Field | Type | Mô tả |
|---|---|---|
| `product_id` | int64 | FK → product |
| `review_id` | str | ID review |
| `comment` | str | Nội dung review (raw, có thể null nếu user chỉ rate) |
| `rating` | float32 | Điểm rating (1–5) |
| `customer_id` | str | ID khách hàng |
| `customer_name` | str | Tên hiển thị khách hàng |
| `is_buyer` | bool | Khách đã mua hàng thật (verified buyer) |
| `purchased_at` | timestamp | Ngày khách thực hiện mua hàng |
| `general_category` | str | Danh mục tổng quát |
| `crawl_time` | timestamp | Thời điểm crawl (UTC) |
| `batch_id` | str | UUID của batch crawl |

**Example rows:**

| review_id | product_id | customer_id | rating | comment | is_buyer | purchased_at |
|---|---|---|---|---|---|---|
| `"48392011"` | `277512620` | `"7290341"` | `5.0` | `"Pin trâu, camera chụp siêu nét, màn hình đẹp!"` | `True` | `2025-10-15 14:22:00` |
| `"48391050"` | `277512620` | `"8801234"` | `2.0` | `"Máy nóng, pin tụt nhanh. Thất vọng."` | `True` | `2025-10-20 09:10:00` |
| `"48390099"` | `277512620` | `"9912345"` | `4.0` | `null` | `False` | `null` |

> Dòng thứ 3: user chỉ để rating, không viết text → `comment=null`, `is_buyer=False` (chưa xác thực mua hàng).

## Enterprise features

| Feature | Class | Mô tả |
|---|---|---|
| Checkpoint/resume | `CheckpointManager` | Lưu state vào `crawler_checkpoint.json`, gián đoạn tiếp tục từ batch cuối |
| Circuit breaker | `CircuitBreaker` | Tự động dừng sau 5 failures liên tiếp, thử lại sau 300s |
| Dead Letter Queue | `DeadLetterQueue` | Ghi item fail vào `dead_letter_queue.jsonl`, retry sau |
| Metrics | `MetricsCollector` | Đếm API calls, failure rate, bytes written, timing — export `metrics.prom` |
| Exponential backoff | `@retry_with_backoff` | Retry 3 lần với delay tăng dần + jitter |
| Async logging | `AsyncLogQueue` | Log non-blocking qua thread riêng, tránh I/O blocking |
| Buffered write | `run_all_products` | Gom đủ 5000 rows mới ghi MinIO một lần, tránh small files |

## Categories mặc định

```python
categories = [
    {"name": "Điện Thoại & Máy Tính Bảng", "category_id": 1789, "url_key": "dien-thoai-may-tinh-bang"},
    {"name": "Laptop, Máy Tính & Linh Kiện", "category_id": 1846, "url_key": "laptop-may-vi-tinh-linh-kien"},
    {"name": "Thiết Bị Số - Phụ Kiện Số",   "category_id": 1815, "url_key": "thiet-bi-kts-phu-kien-so"},
]
```

## Monitor khi đang chạy

```bash
tail -f logs/tiki_crawler_*.log
watch -n 5 'cat crawler_progress.json | python -m json.tool'
cat metrics_summary.json   # sau khi hoàn thành
```

## Thiết kế kỹ thuật

**Tại sao buffer 5000 rows trước khi ghi MinIO?**
Mỗi ghi MinIO = 1 HTTP PUT request. Với hàng chục ngàn sản phẩm, ghi từng row → hàng chục ngàn requests, latency cao và tốn connection pool. Buffer 5000 rows/batch → giảm 99% số requests. File Parquet đủ lớn để Spark đọc hiệu quả (tránh small files problem).

**Tại sao Circuit Breaker threshold = 5 failures, recovery = 300s?**
Tiki API thường rate-limit theo IP sau ~5 request lỗi liên tiếp. 300s là khoảng thời gian đủ để IP được unblock. Circuit Breaker tránh tiếp tục spam request khi đã bị block — vừa bảo vệ IP crawler, vừa không waste quota.

**Tại sao Snappy compression thay vì Gzip?**
Snappy balanced giữa speed và ratio: compress nhanh (quan trọng khi crawler đang chạy real-time), decompress cũng nhanh (quan trọng khi Spark đọc). Gzip ratio tốt hơn ~20% nhưng chậm 5-10× khi compress.

**Tại sao `batch_id` UUID per batch?**
Khi cần debug hay reprocess một lần crawl cụ thể, filter theo `batch_id` để lấy đúng nhóm data mà không ảnh hưởng các batch khác. Cũng giúp trace lỗi nếu một batch có data corrupt.

**Tại sao ThreadPoolExecutor max_workers=5 (products) vs 10 (comments)?**
Product detail API nặng hơn: JSON response lớn, nhiều nested fields, parse phức tạp hơn. Comment API nhẹ hơn, có thể parallelize nhiều hơn mà không trigger rate-limit.

## Troubleshooting

**`Circuit breaker OPEN - too many failures`**
Tiki đã rate-limit IP. Chờ 300s rồi thử lại. Giảm `PRODUCT_MAX_WORKERS` xuống 2–3 nếu tiếp tục gặp:
```python
crawler = TikiCrawler()
crawler.PRODUCT_MAX_WORKERS = 2
```

**`MinIO connection refused / endpoint_url error`**
MinIO chưa chạy hoặc endpoint sai. Kiểm tra:
```bash
docker compose -f docker-compose.infra.yml ps minio
curl http://localhost:9000/minio/health/live
```

**`DataFrame validation: Empty dataframe`**
API Tiki trả về `data: []` — hết trang hoặc `url_key` sai. Kiểm tra `url_key` trong `categories` list.

**`Dead Letter Queue has N items for retry`**
Xem lỗi cụ thể và retry:
```bash
cat dead_letter_queue.jsonl | python -m json.tool | grep "error"
python -c "
from src.crawler.spiders.tiki_crawl import TikiCrawler
TikiCrawler().retry_failed_items()
"
```

**Crawler bị dừng giữa chừng**
Checkpoint tự động lưu vào `crawler_checkpoint.json`. Chạy lại cùng command — `CheckpointManager` tự skip các batch đã hoàn thành, tiếp tục từ batch chưa xong.

**`WARNING: DataFrame uses 500+MB`**
Buffer quá lớn trước khi flush. Giảm `PRODUCT_SAVE_THRESHOLD`:
```python
crawler.PRODUCT_SAVE_THRESHOLD = 2000
```

## Lưu ý

- Data Bronze là **raw, chưa clean** — duplicates, null, HTML tags vẫn còn nguyên
- Bước clean ở `data_pipeline/jobs/bronze_to_silver.py`
- `all_specs` là raw JSON stringify — cần parse thêm nếu dùng
