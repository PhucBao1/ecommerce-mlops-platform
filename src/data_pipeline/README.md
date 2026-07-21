# Data Pipeline

Spark ETL pipeline: Bronze (raw) → Silver (cleaned Iceberg) → Gold (dbt aggregated).

## File interactions

```
bronze_to_silver.py
  ├── spark/session.py            — SparkSession factory (Iceberg + S3A config)
  ├── quality/expectations.py     — Great Expectations validate trước khi ghi
  ├── transformations/
  │   ├── comment_transform.py    — strip HTML, normalize whitespace, thêm crawl_date
  │   └── product_transform.py    — normalize types, thêm crawl_date
  └── writers/iceberg_writer.py   — MERGE INTO idempotent write

inference_job.py
  ├── spark/session.py            — same SparkSession factory
  └── gọi PhoBERT (HTTP) hoặc load model trực tiếp

dbt_project/                      — chạy sau spark jobs (Gold layer)
  └── đọc Silver tables → tạo dim/fact/mart
```

## ASCII flow

```
MinIO (Bronze)
  products_*.parquet
  comments_*.parquet
        │
        ▼
┌─────────────────────────────────┐
│  bronze_to_silver.py (Spark)    │
│                                 │
│  1. Read Bronze Parquet         │
│  2. dedup by (id, crawl_time)   │
│  3. Great Expectations gate ────┼──► FAIL → job abort, alert Airflow
│  4. comment_transform.py        │
│     strip HTML, normalize space │
│  5. MERGE INTO Iceberg          │
└──────────────┬──────────────────┘
               │
               ▼
    Silver (Iceberg / MinIO)
    lakehouse.silver.cleaned_product
    lakehouse.silver.cleaned_comment
               │
               ▼
┌─────────────────────────────────┐
│  inference_job.py (Spark)       │
│  PhoBERT batch inference        │
│  → sentiment_label, scores      │
└──────────────┬──────────────────┘
               │
               ▼
    Gold (Iceberg / MinIO)
    lakehouse.gold.comment_predictions
               │
               ▼
┌─────────────────────────────────┐
│  dbt run (SQL transforms)       │
│  dim_product, fact_review       │
│  gold_brand_health_daily        │
└─────────────────────────────────┘
```

## Cấu trúc

```
data_pipeline/
├── jobs/
│   ├── bronze_to_silver.py      # Job chính Airflow gọi hàng ngày
│   ├── 02_lang_detect.py        # Language detection (filter tiếng Việt)
│   ├── inference_job.py         # Batch PhoBERT inference → Gold predictions
│   └── ab_significance.py       # A/B test statistical significance
├── quality/
│   └── expectations.py          # Great Expectations validation suite
├── spark/
│   └── session.py               # SparkSession factory
├── transformations/
│   ├── comment_transform.py     # HTML strip, normalize whitespace
│   ├── product_transform.py     # Product field normalization
│   └── dedup.py                 # Deduplication by key + timestamp
├── writers/
│   └── iceberg_writer.py        # MERGE INTO idempotent write
├── streaming/                   # → xem streaming/README.md
└── utils/
    └── table_utils.py           # Table existence, schema helpers
```

## Chạy job

```bash
# Bronze → Silver
spark-submit \
  --packages org.apache.iceberg:iceberg-spark-runtime-3.4_2.12:1.4.3 \
  src/data_pipeline/jobs/bronze_to_silver.py --date 2026-06-28

# Batch inference (PhoBERT → Gold)
spark-submit src/data_pipeline/jobs/inference_job.py \
  --input-table lakehouse.silver.cleaned_comment \
  --output-table lakehouse.gold.comment_predictions \
  --model-path artifacts/nlp_models/phobert/version_001
```

## Bronze → Silver schema

### `lakehouse.silver.cleaned_product`

Từ Bronze products, sau `deduplicate_latest(product_id, crawl_time)` + validation:

| Field | Type | Mô tả |
|---|---|---|
| `product_id` | Int64 | PK |
| `sku` | Int64 | SKU sản phẩm |
| `product_name` | str | Tên sản phẩm |
| `short_description` | str | Mô tả ngắn |
| `price` | Int32 | Giá bán (VND) |
| `list_price` | Int32 | Giá gốc |
| `discount_rate` | float32 | % giảm giá |
| `rating` | float32 | Điểm đánh giá (0–5) |
| `review_count` | Int32 | Tổng số review |
| `inventory_status` | str | Trạng thái tồn kho |
| `stock_qty` | Int32 | Số lượng kho |
| `quantity_sold` | Int64 | Tổng đã bán |
| `brand_id` | Int64 | ID thương hiệu |
| `brand_name` | str | Tên thương hiệu |
| `category_id` | Int64 | ID danh mục |
| `category_name` | str | Tên danh mục |
| `seller_id` | Int64 | ID nhà bán |
| `seller_name` | str | Tên nhà bán |
| `thumbnail_url` | str | URL ảnh đại diện |
| `url` | str | Short URL |
| `general_category` | str | Danh mục tổng quát |
| `crawl_date` | date | Partition column — ngày crawl |

### `lakehouse.silver.cleaned_comment`

Từ Bronze comments, sau dedup + `transform_comments()`:

| Field | Type | Mô tả |
|---|---|---|
| `product_id` | int64 | FK → product |
| `review_id` | str | PK |
| `customer_id` | str | ID khách hàng |
| `customer_name` | str | Tên khách |
| `rating` | float32 | Điểm 1–5 |
| `comment` | str | Text gốc (có HTML, raw) |
| `clean_comment` | str | Text đã strip HTML, normalize space |
| `is_buyer` | bool | Verified buyer |
| `purchased_at` | timestamp | Ngày mua |
| `general_category` | str | Danh mục |
| `crawl_date` | date | Partition column |
| `processed_at` | timestamp | Thời điểm Spark xử lý |

**Example rows:**

| review_id | product_id | rating | comment (raw) | clean_comment | is_buyer | crawl_date | processed_at |
|---|---|---|---|---|---|---|---|
| `"48392011"` | `277512620` | `5.0` | `"Pin trâu, <b>camera</b> chụp siêu nét"` | `"Pin trâu, camera chụp siêu nét"` | `True` | `2025-11-01` | `2025-11-01 03:15:42` |
| `"48391050"` | `277512620` | `2.0` | `"Máy nóng,\t\tpin tụt nhanh quá"` | `"Máy nóng, pin tụt nhanh quá"` | `True` | `2025-11-01` | `2025-11-01 03:15:43` |
| `"48390099"` | `277512620` | `4.0` | `null` | `null` | `False` | `2025-11-01` | `2025-11-01 03:15:44` |

### `lakehouse.gold.comment_predictions`

Sau PhoBERT batch inference:

| Field | Type | Mô tả |
|---|---|---|
| *(tất cả fields Silver)* | | Giữ nguyên từ silver.cleaned_comment |
| `sentiment_label` | str | `"Negative"` / `"Neutral"` / `"Positive"` |
| `sentiment_score` | float32 | Confidence score (0–1) của class được chọn |
| `neg_score` | float32 | P(Negative) |
| `neu_score` | float32 | P(Neutral) |
| `pos_score` | float32 | P(Positive) |
| `prediction_date` | date | Partition column |

**Example rows:**

| review_id | clean_comment | sentiment_label | sentiment_score | neg_score | neu_score | pos_score |
|---|---|---|---|---|---|---|
| `"48392011"` | `"Pin trâu, camera chụp siêu nét"` | `Positive` | `0.96` | `0.01` | `0.03` | `0.96` |
| `"48391050"` | `"Máy nóng, pin tụt nhanh quá"` | `Negative` | `0.88` | `0.88` | `0.09` | `0.03` |
| `"48390099"` | `null` | `Neutral` | `1.00` | `0.00` | `1.00` | `0.00` |

> Dòng thứ 3: `clean_comment=null` → tự động gán `Neutral` confidence 100%, không chạy model.

## Data quality (`expectations.py`)

| Check | Field | Rule |
|---|---|---|
| Not null | `comment`, `product_id`, `customer_id` | Không được null |
| Type | `rating` | Integer 1–5 |
| Non-empty | table | Row count > 0 |
| Regex | `product_id` | Match `^\d+$` |

Job **fail-fast** nếu validation không qua — data không vào Silver.

## SparkSession (`spark/session.py`)

```python
from src.data_pipeline.spark.session import create_spark_session

spark = create_spark_session(app_name="bronze_to_silver")
# Cấu hình sẵn: Iceberg catalog, S3A + MinIO, shuffle partitions
```

**Env vars cần có:** `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `CATALOG_S3_ENDPOINT`

**Spark tuning qua env (override không cần rebuild):**
```
SPARK_EXECUTOR_MEMORY=2g
SPARK_DRIVER_MEMORY=1g
SPARK_SHUFFLE_PARTITIONS=8       # dev; set 200 cho production
SPARK_BROADCAST_THRESHOLD=52428800  # 50MB
```

## Iceberg writer (`writers/iceberg_writer.py`)

Dùng `MERGE INTO` để ghi idempotent — Airflow retry không tạo duplicate:

```python
write_iceberg_table(spark, df, "lakehouse.silver.cleaned_comment", partition_col="crawl_date")
```

## Thiết kế kỹ thuật

**Tại sao dùng Apache Iceberg thay vì Parquet thuần?**
Parquet thuần không hỗ trợ ACID, không time travel, không schema evolution an toàn. Khi Airflow retry 1 job, ghi Parquet tạo duplicate file. Iceberg `MERGE INTO` idempotent — retry bao nhiêu lần ra kết quả như nhau. Iceberg hidden partitioning còn tránh partition layout sai khi query range.

**Tại sao `MERGE INTO` thay vì `INSERT OVERWRITE`?**
`INSERT OVERWRITE` xóa toàn bộ partition rồi ghi lại — nếu job fail giữa chừng mất data cả ngày. `MERGE INTO` update/insert từng record theo key (`review_id`) — safe với Airflow retry, đồng thời handle late-arriving data (review crawl lại sau khi sửa).

**Tại sao Great Expectations validate ở Silver gate?**
Fail-fast pattern: phát hiện data bad ngay tại ingestion gate, không để data corrupt vào Silver rồi mới biết model lạ. Mỗi violation được log → Airflow alert → engineer fix upstream (crawler hoặc API đổi schema). Không có GE, silent bad data sẽ ảnh hưởng training mà không có dấu hiệu nào.

**Tại sao strip HTML ở Silver chứ không ở Bronze?**
Bronze = source of truth raw data. Nếu cần reprocess với logic clean khác (regex pattern cập nhật), vẫn có raw text để replay. Silver = cleaned view. Tách rõ ràng giúp debug: nếu `clean_comment` trống bất thường, so ngay với `comment` Bronze để biết lỗi ở clean step hay crawl step.

**Tại sao `crawl_date` là partition column?**
Query pattern chính: `WHERE crawl_date = '2025-11-01'` hoặc `WHERE crawl_date >= '2025-10-01'`. Partition by date → Spark chỉ đọc file của ngày cần, không scan full table. Tương tự cho batch inference — chỉ score review mới nhất mỗi ngày.

## Troubleshooting

**Great Expectations fail: `expect_column_values_to_not_be_null` on `comment`**
Tiki cho phép user review chỉ rating mà không viết text → Bronze có NULL comments. Hai cách:
- Relax expectation: chỉ validate `product_id`, `customer_id` NOT NULL, cho `comment` nullable
- Filter NULL ra trước khi NLP inference (không ảnh hưởng RecSys training)

**`MERGE INTO` chậm trên bảng lớn (>10M rows)**
Thiếu Z-ordering, Spark phải scan nhiều file. Chạy optimize sau batch lớn:
```sql
CALL system.rewrite_data_files('lakehouse.silver.cleaned_comment',
  strategy => 'sort',
  sort_order => 'crawl_date ASC NULLS LAST');
```

**Spark OOM khi batch inference PhoBERT**
Executor memory không đủ chứa model + data. Fix:
```bash
SPARK_EXECUTOR_MEMORY=4g \
spark-submit src/data_pipeline/jobs/inference_job.py
```
Hoặc giảm `batch_size` trong `spark_batch_infer.py` (default 32).

**`S3A: Connection timeout` khi Spark đọc MinIO**
MinIO chưa sẵn sàng hoặc env sai. Kiểm tra:
```bash
curl http://localhost:9000/minio/health/live
echo "KEY=$AWS_ACCESS_KEY_ID  ENDPOINT=$CATALOG_S3_ENDPOINT"
```

**`Table lakehouse.silver.cleaned_comment does not exist`**
Chưa tạo Iceberg catalog. Chạy một lần khi khởi tạo:
```python
spark.sql("CREATE DATABASE IF NOT EXISTS lakehouse.silver")
spark.sql("CREATE DATABASE IF NOT EXISTS lakehouse.gold")
```

**Airflow DAG fail ở `dbt_run` sau khi Spark thành công**
dbt không thấy Silver table mới — dbt đọc qua Spark Thrift Server (port 10000). Kiểm tra Thrift Server đang chạy:
```bash
docker compose -f docker-compose.infra.yml ps spark-thrift
```
