import logging

import great_expectations as gx

logger = logging.getLogger(__name__)


# ==========================================
# 3. DATA QUALITY VALIDATION USING GREAT EXPECTATIONS
# ==========================================
def validate_data_quality(df, table_name):
    logger.info(f"Running GX validation for {table_name}")

    # 1. Create an ephemeral (in-memory) GX context
    context = gx.get_context(mode="ephemeral")

    # 2. Register Spark DataFrame into GX context
    datasource = context.data_sources.add_spark(name="my_spark_datasource")
    data_asset = datasource.add_dataframe_asset(name=table_name)
    batch_definition = data_asset.add_batch_definition_whole_dataframe("my_batch_def")

    # 3. Define expectations (data quality rules)
    suite = gx.ExpectationSuite(name=f"{table_name}_suite")

    if table_name == "raw_products":
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column="product_id")
        )
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeUnique(column="product_id")
        )
        # price must exist and be positive — zero/negative price breaks reranker scoring
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column="price")
        )
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="price", min_value=0.01
            )
        )
        # category_id required for Two-Tower item tower and reranker diversity
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column="category_id")
        )
    elif table_name == "raw_comments":
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column="product_id")
        )
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column="customer_id")
        )
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column="review_id")
        )
        # Tiki rating scale is 1–5, not 0–5 — rating=0 indicates missing/error data
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="rating", min_value=0.0, max_value=5.0
            )
        )

    context.suites.add(suite)

    # 4. Run validation
    validation_definition = gx.ValidationDefinition(
        data=batch_definition, suite=suite, name=f"{table_name}_validation"
    )

    # Execute validation by passing Spark DataFrame at runtime
    validation_results = validation_definition.run(batch_parameters={"dataframe": df})

    if not validation_results.success:
        logger.error(f"❌ Dữ liệu bảng {table_name} KHÔNG đạt chuẩn DQ!")
        # Log detailed failing expectations
        for res in validation_results.results:
            if not res.success:
                # ExpectationConfiguration
                ec = res.expectation_config

                expectation_type = getattr(
                    ec, "expectation_type", "unknown_expectation"
                )

                column = getattr(ec, "kwargs", {}).get("column", "N/A")

                logger.error(
                    f"❌ Expectation fail: {expectation_type} trên cột {column}"
                )
                logger.error(
                    f"    Số bản ghi fail: {res.result.get('unexpected_count')}"
                )
                logger.error(f"    Giá trị fail: {res.result.get('unexpected_list')}")
    else:
        logger.info(f"✅ Dữ liệu bảng {table_name} vượt qua bài test DQ!")
