import pyspark.sql.functions as F
from pyspark.sql.types import LongType, StringType, StructField, StructType


# ==========================================
# 2. TEST DATA CLEANING LOGIC (REGEX TRANSFORMATIONS)
# ==========================================
def test_clean_comments_logic(spark):
    # Scenario 1: DataFrame with dirty comments
    data = [
        (1, "<b>Sản phẩm tốt</b>\n\nQuá ngon!"),  # Contains HTML tags and newlines
        (2, "   Rất   ok   "),  # Contains extra whitespace
        (3, "Ok"),  # Comment too short (<= 5 characters)
    ]
    schema = StructType(
        [
            StructField("id", LongType(), True),
            StructField("comment", StringType(), True),
        ]
    )
    df_raw = spark.createDataFrame(data, schema)

    # Tái hiện lại logic làm sạch của bạn
    df_clean = (
        df_raw.withColumn(
            "clean_comment", F.regexp_replace(F.col("comment"), "<[^>]*>", " ")
        )
        .withColumn(
            "clean_comment", F.regexp_replace(F.col("clean_comment"), "\n|\t|\r", " ")
        )
        .withColumn(
            "clean_comment", F.trim(F.regexp_replace(F.col("clean_comment"), " +", " "))
        )
        .withColumn(
            "clean_comment",
            F.when(
                F.length(F.col("clean_comment")) > 5, F.col("clean_comment")
            ).otherwise(F.lit(None)),
        )
    )

    results = df_clean.collect()

    # Row 1: HTML tags and newline should be removed
    assert results[0]["clean_comment"] == "Sản phẩm tốt Quá ngon!"

    # Row 2: Extra whitespace should be removed and trimmed
    assert results[1]["clean_comment"] == "Rất ok"

    # Row 3: Comment too short should be set to Null (None in Python)
    assert results[2]["clean_comment"] is None
