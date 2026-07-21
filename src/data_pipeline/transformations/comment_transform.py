import pyspark.sql.functions as F


def transform_comments(df, execution_date):

    return (
        df.withColumn(
            "clean_comment", F.regexp_replace(F.col("comment"), "<[^>]*>", " ")
        )
        .withColumn(
            "clean_comment", F.regexp_replace(F.col("clean_comment"), "\n|\t|\r", " ")
        )
        .withColumn(
            "clean_comment", F.trim(F.regexp_replace(F.col("clean_comment"), " +", " "))
        )
        .withColumn("crawl_date", F.lit(execution_date))
        .withColumn("processed_at", F.current_timestamp())
    )
