import logging

from pyspark.sql.functions import col

from src.data_pipeline.utils.table_utils import table_exists

logger = logging.getLogger(__name__)


def write_iceberg_table(spark, df, table_name, partition_col=None):
    df = df.repartition(4)

    if not table_exists(spark, table_name):
        logger.info("Creating Iceberg table: %s", table_name)

        writer = df.writeTo(table_name).tableProperty("format-version", "2")

        if partition_col:
            writer = writer.partitionedBy(col(partition_col))

        writer.create()
        return

    if not partition_col:
        logger.warning(
            "Bảng %s không có partition_col — buộc phải append, KHÔNG idempotent. "
            "Chạy lại job sẽ nhân đôi dữ liệu.",
            table_name,
        )
        df.writeTo(table_name).option("fanout-enabled", "true").append()
        return

    logger.info(
        "Overwriting partitions of %s (idempotent — chạy lại không nhân đôi)",
        table_name,
    )
    df.writeTo(table_name).option("fanout-enabled", "true").overwritePartitions()
