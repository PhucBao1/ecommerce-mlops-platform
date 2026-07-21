import ast

import pyspark.sql.functions as F
from pyspark.sql.types import MapType, StringType

# ==========================================
# 4. HÀM XỬ LÝ (TRANSFORM) VÀ GHI ICEBERG
# ==========================================
TARGET_SPECS = {
    "battery_capacity",
    "ram",
    "rom",
    "chip_set",
    "screen_size",
    "display_type",
}


def parse_specs(spec_string):
    if not spec_string or spec_string == "[]":
        return {}

    try:
        specs_list = ast.literal_eval(spec_string)
        parsed = {
            attr.get("code"): attr.get("value")
            for group in specs_list
            if "attributes" in group
            for attr in group["attributes"]
            if attr.get("code") in TARGET_SPECS and attr.get("value")
        }
        return parsed

    except:
        return {}


def transform_products(df, execution_date):

    spark_udf = F.udf(parse_specs, MapType(StringType(), StringType()))

    df = df.withColumn("parsed_specs", spark_udf(F.col("all_specs")))

    for spec in TARGET_SPECS:
        df = df.withColumn(spec, F.col("parsed_specs").getItem(spec))

    return df.drop("all_specs", "parsed_specs")


""".withColumn(
    "crawl_date",
    F.lit(execution_date)
)"""
