import logging
import os

from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


# ==========================================
# 2. SPARK SESSION INITIALIZATION
# (Including MinIO & Iceberg Configuration)
# ==========================================
def create_spark_session():
    logger.info("Khởi tạo Spark Session với cấu hình MinIO/S3 và Iceberg...")

    # Local MinIO: S3_ENDPOINT_URL=http://minio:9000, AWS_ACCESS_KEY_ID=admin, ...
    # AWS S3:      S3_ENDPOINT_URL="" (empty/unset), credentials via IAM role → no key needed
    s3_endpoint = os.getenv("S3_ENDPOINT_URL", "http://minio:9000")
    s3_access = os.getenv("AWS_ACCESS_KEY_ID", "admin")
    s3_secret = os.getenv("AWS_SECRET_ACCESS_KEY", "password")
    pg_host = os.getenv("POSTGRES_HOST", "postgres-catalog")
    pg_port = os.getenv("POSTGRES_PORT", "5432")
    pg_db = os.getenv("POSTGRES_ICEBERG_DB", "iceberg_metadata")
    pg_user = os.getenv("POSTGRES_ICEBERG_USER", "admin")
    pg_password = os.getenv("POSTGRES_ICEBERG_PASSWORD", "password")

    # AWS S3: path-style access and custom endpoint not needed; IAM role provides creds
    use_minio = bool(s3_endpoint)

    builder = (
        SparkSession.builder.appName("Bronze_to_Silver_Clean_Regex")
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        )
        .config("spark.sql.catalog.lakehouse", "org.apache.iceberg.spark.SparkCatalog")
        .config(
            "spark.sql.catalog.lakehouse.io-impl", "org.apache.iceberg.aws.s3.S3FileIO"
        )
        .config(
            "spark.sql.catalog.lakehouse.uri",
            f"jdbc:postgresql://{pg_host}:{pg_port}/{pg_db}",
        )
        .config("spark.sql.catalog.lakehouse.jdbc.user", pg_user)
        .config("spark.sql.catalog.lakehouse.jdbc.password", pg_password)
        .config(
            "spark.sql.catalog.lakehouse.warehouse",
            os.getenv("CATALOG_WAREHOUSE", "s3a://warehouse/"),
        )
        .config("spark.sql.defaultCatalog", "lakehouse")
        .config("spark.sql.legacy.parquet.nanosAsLong", "true")
        .config(
            "spark.hadoop.fs.s3a.endpoint.region", os.getenv("AWS_REGION", "us-east-1")
        )
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    )

    if use_minio:
        # Local MinIO: explicit endpoint + static credentials + path-style
        builder = (
            builder.config("spark.sql.catalog.lakehouse.s3.endpoint", s3_endpoint)
            .config("spark.sql.catalog.lakehouse.s3.path-style-access", "true")
            .config("spark.hadoop.fs.s3a.endpoint", s3_endpoint)
            .config("spark.hadoop.fs.s3a.access.key", s3_access)
            .config("spark.hadoop.fs.s3a.secret.key", s3_secret)
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
            .config(
                "spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
            )
        )
    else:
        aws_region = os.getenv("AWS_REGION", "us-east-1")
        # Hadoop S3A (đọc bronze parquet) cần bare hostname; Iceberg S3FileIO (ghi
        # Iceberg table, dùng AWS SDK v2) lại cần URI đầy đủ có scheme — thiếu
        # "https://" gây "NullPointerException: The URI scheme of endpointOverride
        # must not be null" (bug thật đã gặp, chỉ xảy ra ở bước ghi, không phải đọc).
        s3_regional_endpoint = f"s3.{aws_region}.amazonaws.com"
        builder = (
            builder.config("spark.hadoop.fs.s3a.endpoint", s3_regional_endpoint)
            .config(
                "spark.sql.catalog.lakehouse.s3.endpoint",
                f"https://{s3_regional_endpoint}",
            )
            .config(
                "spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.InstanceProfileCredentialsProvider",
            )
        )

    spark = (
        builder
        # Performance tuning — tunable via env vars, no code change needed
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config(
            "spark.sql.shuffle.partitions", os.getenv("SPARK_SHUFFLE_PARTITIONS", "8")
        )
        .config(
            "spark.sql.autoBroadcastJoinThreshold",
            os.getenv("SPARK_BROADCAST_THRESHOLD", "52428800"),
        )
        .config("spark.sql.files.maxPartitionBytes", "134217728")
        .config("spark.executor.memory", os.getenv("SPARK_EXECUTOR_MEMORY", "2g"))
        .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "1g"))
        .getOrCreate()
    )
    spark.sql("CREATE NAMESPACE IF NOT EXISTS lakehouse.silver")
    spark.sql("CREATE NAMESPACE IF NOT EXISTS lakehouse.default")
    return spark


# .config("spark.sql.catalog.lakehouse.catalog-impl", "org.apache.iceberg.jdbc.JdbcCatalog") \
