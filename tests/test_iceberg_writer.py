"""
Iceberg writer — idempotency.

Bản cũ luôn dùng `.append()`: chạy lại job cùng một ngày là dữ liệu NHÂN ĐÔI.
Đo thật bằng Iceberg thật (không mock):

    lần 1: 2 dòng | lần 2 (rerun): 4 dòng | lần 3 (rerun): 6 dòng

DAG đặt `retries: 3` nên task fail SAU KHI đã ghi xong một phần rồi Airflow retry
sẽ append lần nữa — một task hỏng có thể nhân dữ liệu lên 4 lần. Sai lệch lan
xuống tận cùng: silver trùng -> dbt mart sai -> model train trên dữ liệu nhân đôi.

Test cần Iceberg runtime jar (28MB). Không có thì tự skip — KHÔNG fail giả, nhưng
cũng ghi rõ để không ai tưởng là đã được gác.

Chạy đầy đủ:
    ICEBERG_JAR=/đường/dẫn/iceberg-spark-runtime-3.4_2.12-1.4.3.jar pytest tests/test_iceberg_writer.py
"""

from src.data_pipeline.writers.iceberg_writer import write_iceberg_table

# fixture `iceberg_spark` nằm ở tests/conftest.py — dùng CHUNG SparkSession với
# các test Spark khác (Spark chỉ cho 1 SparkContext mỗi JVM, mỗi file tự dựng
# session riêng là config bị nuốt trong im lặng).


def test_rerun_does_not_duplicate_rows(iceberg_spark):
    """Đây chính là bug: Airflow retry job => .append() cộng dồn thêm 1 bản đầy đủ."""
    spark = iceberg_spark
    table = "local.db.rerun_test"
    df = spark.createDataFrame(
        [("p1", "2026-07-13"), ("p2", "2026-07-13")], ["product_id", "crawl_date"]
    )

    write_iceberg_table(spark, df, table, partition_col="crawl_date")
    assert spark.table(table).count() == 2

    # Mô phỏng Airflow retry: chạy lại y hệt
    write_iceberg_table(spark, df, table, partition_col="crawl_date")
    assert spark.table(table).count() == 2, "rerun đã nhân đôi dữ liệu"

    write_iceberg_table(spark, df, table, partition_col="crawl_date")
    assert spark.table(table).count() == 2, "rerun lần 2 đã nhân đôi dữ liệu"


def test_overwrite_only_touches_partitions_present_in_dataframe(iceberg_spark):
    """
    overwritePartitions() phải THAY THẾ đúng partition có trong DataFrame, không
    được xoá partition của ngày khác — nếu không thì mỗi lần chạy job sẽ xoá sạch
    lịch sử, còn tệ hơn cả bug nhân đôi.
    """
    spark = iceberg_spark
    table = "local.db.partition_test"

    day1 = spark.createDataFrame([("p1", "2026-07-13")], ["product_id", "crawl_date"])
    day2 = spark.createDataFrame([("p9", "2026-07-14")], ["product_id", "crawl_date"])

    write_iceberg_table(spark, day1, table, partition_col="crawl_date")
    write_iceberg_table(spark, day2, table, partition_col="crawl_date")

    rows = {r["product_id"] for r in spark.table(table).collect()}
    assert rows == {"p1", "p9"}, "ghi ngày mới đã xoá mất dữ liệu ngày cũ"


def test_compaction_merges_small_files(iceberg_spark):
    """
    Small-file explosion: mỗi lần ghi đẻ ra vài file nhỏ. Không compaction thì
    query phải mở hàng nghìn file, chậm dần theo thời gian và không bao giờ tự
    khá lên. Đo thật: 10 file nhỏ -> 1 file.
    """
    from src.data_pipeline.jobs.iceberg_maintenance import compact_table

    spark = iceberg_spark
    table = "local.db.small_files"
    spark.sql(
        f"CREATE TABLE {table} (id STRING, crawl_date STRING) "
        "USING iceberg PARTITIONED BY (crawl_date)"
    )
    for i in range(10):
        spark.createDataFrame([(f"id{i}", "2026-07-13")], ["id", "crawl_date"]).writeTo(
            table
        ).append()

    files_before = spark.sql(f"SELECT count(*) c FROM {table}.files").collect()[0]["c"]
    rows_before = spark.table(table).count()
    assert files_before == 10

    compact_table(spark, table)

    files_after = spark.sql(f"SELECT count(*) c FROM {table}.files").collect()[0]["c"]
    assert files_after < files_before, "compaction không gộp được file nào"
    assert spark.table(table).count() == rows_before, "compaction làm mất dữ liệu"


def test_maintenance_never_loses_data(iceberg_spark):
    """
    Bảo trì là việc DỌN DẸP — tuyệt đối không được đụng tới dữ liệu. Đặc biệt
    remove_orphan_files: nếu quên older_than, nó sẽ xoá file mà job khác vừa ghi
    xong nhưng chưa commit snapshot => hỏng dữ liệu.
    """
    from src.data_pipeline.jobs.iceberg_maintenance import maintain

    spark = iceberg_spark
    table = "local.db.maintain_safe"
    df = spark.createDataFrame(
        [("a", "2026-07-13"), ("b", "2026-07-14")], ["product_id", "crawl_date"]
    )
    write_iceberg_table(spark, df, table, partition_col="crawl_date")

    maintain(spark, table)

    rows = {r["product_id"] for r in spark.table(table).collect()}
    assert rows == {"a", "b"}, f"bảo trì làm mất dữ liệu: {rows}"


def test_rewriting_a_day_replaces_that_day_only(iceberg_spark):
    """Nạp lại 1 ngày với dữ liệu ĐÃ SỬA => ngày đó được thay mới, ngày khác nguyên vẹn."""
    spark = iceberg_spark
    table = "local.db.replace_test"

    write_iceberg_table(
        spark,
        spark.createDataFrame(
            [("cũ", "2026-07-13"), ("cũ2", "2026-07-13"), ("giữ_nguyên", "2026-07-14")],
            ["product_id", "crawl_date"],
        ),
        table,
        partition_col="crawl_date",
    )

    # Nạp lại ngày 13 với dữ liệu đã sửa (chỉ còn 1 dòng)
    write_iceberg_table(
        spark,
        spark.createDataFrame([("mới", "2026-07-13")], ["product_id", "crawl_date"]),
        table,
        partition_col="crawl_date",
    )

    rows = {r["product_id"] for r in spark.table(table).collect()}
    assert rows == {"mới", "giữ_nguyên"}, f"kết quả sai: {rows}"
