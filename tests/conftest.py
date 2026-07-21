import os
import shutil
import tempfile

import pytest
from pyspark.sql import SparkSession

# Iceberg runtime jar (28MB) — không có thì test Iceberg tự skip.
# Tải: repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-3.4_2.12/1.4.3/
ICEBERG_JAR = os.getenv("ICEBERG_JAR", "")


@pytest.fixture(scope="session")
def spark():
    """
    MỘT SparkSession duy nhất cho toàn bộ test session.

    Spark chỉ cho phép 1 SparkContext mỗi JVM. Nếu file test khác tạo session
    không-Iceberg TRƯỚC, thì getOrCreate() ở test Iceberg sẽ trả về đúng session
    cũ đó và config Iceberg bị bỏ qua TRONG IM LẶNG => catalog không tồn tại, test
    fail. Chạy riêng từng file thì pass, chạy cả suite thì đỏ — loại bug rất dễ lọt.

    Nên cấu hình Iceberg ngay từ đầu vào session dùng chung, thay vì để mỗi file
    tự dựng session riêng.
    """
    builder = (
        SparkSession.builder.appName("Test_Spark_Jobs")
        .master("local[2]")
        .config("spark.ui.enabled", "false")
    )

    warehouse = None
    if ICEBERG_JAR and os.path.exists(ICEBERG_JAR):
        warehouse = tempfile.mkdtemp(prefix="iceberg_test_")
        builder = (
            builder.config("spark.jars", ICEBERG_JAR)
            .config(
                "spark.sql.extensions",
                "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
            )
            .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog")
            .config("spark.sql.catalog.local.type", "hadoop")
            .config("spark.sql.catalog.local.warehouse", warehouse)
        )

    session = builder.getOrCreate()
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
    if warehouse:
        shutil.rmtree(warehouse, ignore_errors=True)


@pytest.fixture
def iceberg_spark(spark):
    """Chính session ở trên, nhưng skip nếu không có Iceberg jar."""
    if not ICEBERG_JAR or not os.path.exists(ICEBERG_JAR):
        pytest.skip(
            "Thiếu ICEBERG_JAR — set biến môi trường trỏ tới "
            "iceberg-spark-runtime-3.4_2.12-1.4.3.jar để chạy test Iceberg thật"
        )
    return spark
