import pyspark.sql.functions as F
from pyspark.sql.window import Window


def deduplicate_latest(df, id_column, order_column):

    window_spec = Window.partitionBy(id_column).orderBy(F.col(order_column).desc())

    return (
        df.withColumn("rn", F.row_number().over(window_spec))
        .filter(F.col("rn") == 1)
        .drop("rn")
    )
