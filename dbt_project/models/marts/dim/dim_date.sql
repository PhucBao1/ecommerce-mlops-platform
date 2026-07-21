{{ config(materialized='table', tags=['mart_stg']) }}

WITH dates AS (
    SELECT CAST(crawl_date AS DATE) AS raw_date FROM {{ ref('stg_products') }}
    UNION
    SELECT DATE(FROM_UNIXTIME(purchased_at)) FROM {{ ref('stg_comments') }}
)

SELECT DISTINCT
    raw_date,
    DATE_FORMAT(raw_date, 'yyyyMMdd') AS date_id,
    EXTRACT(DAY FROM raw_date) AS day,
    EXTRACT(MONTH FROM raw_date) AS month,
    EXTRACT(YEAR FROM raw_date) AS year,
    EXTRACT(QUARTER FROM raw_date) AS quarter,
    DATE_FORMAT(raw_date, 'EEEE') AS day_of_week
FROM dates
