"""
Bảo trì bảng Iceberg — compaction, dọn snapshot, dọn file mồ côi.

Trước đây KHÔNG có bước nào trong repo (grep rewrite_data_files/expire_snapshots/
remove_orphan_files toàn repo => chỉ được nhắc trong README). Không dọn thì mỗi
lần job ghi, Iceberg lại đẻ thêm file nhỏ + snapshot mới, dẫn tới:

  1. Small-file explosion — mỗi lần ghi tạo vài file nhỏ (job repartition(4) =>
     4 file/partition/lần chạy). Query phải mở hàng nghìn file thay vì vài chục,
     chậm dần theo thời gian và không bao giờ tự khá lên.

  2. Metadata phình vô hạn — mỗi lần ghi = 1 snapshot mới, giữ nguyên tham chiếu
     tới file cũ. Bảng không bao giờ nhỏ lại, S3 tính tiền cho cả những file đã
     bị thay thế từ lâu.

  3. File mồ côi — job fail giữa chừng để lại file data không snapshot nào trỏ
     tới. Không ai đọc, nhưng vẫn tốn tiền lưu trữ mãi mãi.

Ba thao tác này là thủ tục vận hành CHUẨN của lakehouse, không phải tối ưu hoá
màu mè: bỏ qua thì bảng cứ chậm và đắt dần cho tới lúc phải làm lại từ đầu.

Chạy hàng ngày SAU khi bronze_to_silver xong (xem airflow/dags/ecommerce_pipeline.py).
"""

import logging

logger = logging.getLogger(__name__)

# Gộp file nhỏ thành file ~128MB. Dưới ngưỡng này thì mới đáng gộp; file đã to rồi
# mà vẫn rewrite là đốt CPU vô ích.
_TARGET_FILE_SIZE_BYTES = 128 * 1024 * 1024
_MIN_INPUT_FILES = 5  # ít hơn từng này file nhỏ thì chưa bõ công compaction

# Giữ snapshot 7 ngày — đủ để time-travel/rollback khi phát hiện sự cố dữ liệu,
# nhưng không giữ mãi làm phình metadata.
_SNAPSHOT_RETENTION_DAYS = 7
_MIN_SNAPSHOTS_TO_KEEP = 3

# File mồ côi: chỉ xoá file cũ hơn 3 ngày. KHÔNG được để mặc định (0h) — xoá file
# vừa được job khác ghi xong nhưng chưa commit snapshot sẽ làm HỎNG DỮ LIỆU.
_ORPHAN_FILE_MIN_AGE_HOURS = 72


def compact_table(spark, table_name: str) -> None:
    """Gộp file nhỏ (rewrite_data_files)."""
    logger.info("Compacting %s ...", table_name)
    result = spark.sql(
        f"""
        CALL {_catalog_of(table_name)}.system.rewrite_data_files(
            table => '{table_name}',
            options => map(
                'target-file-size-bytes', '{_TARGET_FILE_SIZE_BYTES}',
                'min-input-files', '{_MIN_INPUT_FILES}'
            )
        )
        """
    ).collect()
    if result:
        row = result[0]
        logger.info(
            "Compacted %s: %s file gộp lại thành %s file",
            table_name,
            row["rewritten_data_files_count"],
            row["added_data_files_count"],
        )


def expire_old_snapshots(spark, table_name: str) -> None:
    """Xoá snapshot cũ (expire_snapshots) — giải phóng file không còn ai tham chiếu."""
    logger.info("Expiring snapshots của %s ...", table_name)
    spark.sql(
        f"""
        CALL {_catalog_of(table_name)}.system.expire_snapshots(
            table => '{table_name}',
            older_than => TIMESTAMPADD(DAY, -{_SNAPSHOT_RETENTION_DAYS}, current_timestamp()),
            retain_last => {_MIN_SNAPSHOTS_TO_KEEP}
        )
        """
    ).collect()


def remove_orphan_files(spark, table_name: str) -> None:
    """
    Xoá file không snapshot nào trỏ tới (rác từ job fail giữa chừng).

    LUÔN đặt older_than: xoá file mới ghi mà chưa kịp commit snapshot sẽ làm hỏng
    dữ liệu của job đang chạy song song. Đây là cái bẫy chính của thủ tục này.
    """
    logger.info("Removing orphan files của %s ...", table_name)
    spark.sql(
        f"""
        CALL {_catalog_of(table_name)}.system.remove_orphan_files(
            table => '{table_name}',
            older_than => TIMESTAMPADD(HOUR, -{_ORPHAN_FILE_MIN_AGE_HOURS}, current_timestamp())
        )
        """
    ).collect()


def _catalog_of(table_name: str) -> str:
    """`lakehouse.silver.cleaned_product` -> `lakehouse` (procedure phải gọi đúng catalog)."""
    return table_name.split(".")[0]


def maintain(spark, table_name: str) -> None:
    """
    Chạy đủ 3 bước, ĐÚNG THỨ TỰ:
      1. compact      — gộp file nhỏ (tạo snapshot mới, file cũ thành rác)
      2. expire       — bỏ snapshot cũ (biến file cũ thành không-ai-tham-chiếu)
      3. remove orphan— xoá hẳn file không ai tham chiếu

    Đảo thứ tự là vô nghĩa: xoá orphan trước khi expire thì file cũ vẫn còn snapshot
    trỏ tới, chẳng xoá được gì.

    Lỗi ở một bảng KHÔNG được làm sập cả task — bảo trì là việc dọn dẹp, hỏng thì
    lần sau dọn tiếp, không đáng để chặn cả pipeline.
    """
    for step, fn in (
        ("compact", compact_table),
        ("expire_snapshots", expire_old_snapshots),
        ("remove_orphan_files", remove_orphan_files),
    ):
        try:
            fn(spark, table_name)
        except Exception as exc:
            logger.error("Bảo trì %s bước %s thất bại: %s", table_name, step, exc)


# Đúng các bảng silver mà bronze_to_silver.py ghi ra.
_TABLES = (
    "lakehouse.silver.cleaned_product",
    "lakehouse.silver.cleaned_comment",
)


if __name__ == "__main__":
    import argparse

    from src.data_pipeline.spark.session import create_spark_session

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Spark Job: Iceberg table maintenance")
    parser.add_argument("--date", required=False, help="Execution date từ Airflow")
    args = parser.parse_args()

    spark = None
    try:
        spark = create_spark_session()
        for table in _TABLES:
            maintain(spark, table)
        logger.info("Bảo trì Iceberg xong.")
    finally:
        if spark:
            spark.stop()
