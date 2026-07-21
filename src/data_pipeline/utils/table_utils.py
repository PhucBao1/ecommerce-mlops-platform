def table_exists(spark, table_name):
    try:
        spark.table(table_name)
        return True
    except Exception:
        return False
