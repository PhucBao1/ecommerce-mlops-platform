import argparse
import logging
import os

from pyspark.sql import functions as F

from src.data_pipeline.quality.expectations import validate_data_quality
from src.data_pipeline.spark.session import create_spark_session
from src.data_pipeline.transformations.comment_transform import transform_comments
from src.data_pipeline.transformations.dedup import deduplicate_latest
from src.data_pipeline.writers.iceberg_writer import write_iceberg_table

# ==========================================
# 1. LOGGING CONFIGURATION
# ==========================================
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("BronzeToSilverJob")

WAREHOUSE_ROOT = os.getenv("CATALOG_WAREHOUSE", "s3a://warehouse/").rstrip("/")


def process_products(spark, execution_date):

    logger.info("Processing products...")

    df_raw = spark.read.parquet(
        f"{WAREHOUSE_ROOT}/bronze/products/{execution_date}/products_*.parquet"
    )

    df_raw = deduplicate_latest(df_raw, "product_id", "crawl_time")

    validate_data_quality(df_raw, "raw_products")

    df_raw = df_raw.withColumn("crawl_date", F.to_date(F.lit(execution_date)))

    write_iceberg_table(
        spark=spark,
        df=df_raw,
        table_name="lakehouse.silver.cleaned_product",
        partition_col="crawl_date",
    )

    logger.info("✅ Đã lưu cleaned_products xuống Iceberg! Sẵn sàng cho dbt.")


def process_comments(spark, execution_date):
    logger.info("Bắt đầu xử lý Fact_Review...")
    df_raw = spark.read.parquet(
        f"{WAREHOUSE_ROOT}/bronze/comments/{execution_date}/comments_*.parquet"
    )

    df_raw = deduplicate_latest(df_raw, "review_id", "crawl_time")

    validate_data_quality(df_raw, "raw_comments")

    df_raw = df_raw.withColumn("crawl_date", F.to_date(F.lit(execution_date)))

    df_clean = transform_comments(df_raw, execution_date)

    write_iceberg_table(
        spark=spark,
        df=df_clean,
        table_name="lakehouse.silver.cleaned_comment",
        partition_col="crawl_date",
    )

    logger.info("✅ Đã lưu cleaned_comments xuống Iceberg! Sẵn sàng cho dbt.")


# ==========================================
# 5. MAIN ORCHESTRATION FUNCTION
# (CALLED BY AIRFLOW)
# ==========================================
if __name__ == "__main__":
    # Receive execution_date from Airflow
    # Example: --date 2026-04-26
    parser = argparse.ArgumentParser(description="Spark Job: Bronze to Silver")

    parser.add_argument("--date", required=False, help="Execution date từ Airflow")

    args = parser.parse_args()

    logger.info(f"🚀 Bắt đầu Job ETL Bronze to Silver. Execution Date: {args.date}")

    spark = None
    try:
        spark = create_spark_session()
        process_products(spark, args.date)
        process_comments(spark, args.date)

        logger.info("🎉 Job hoàn thành xuất sắc!")

    except Exception as e:
        logger.error(f"❌ Job thất bại với lỗi: {e}")
        raise e

    finally:
        if spark:
            spark.stop()
            logger.info("Đã đóng Spark Session.")
