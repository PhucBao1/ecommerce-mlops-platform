{{ config(
    materialized='incremental',
    incremental_strategy='merge',
    file_format='iceberg',
    unique_key='review_id',
    tags=['mart_stg']
) }}

WITH review_sentiment AS (
    SELECT *
    FROM (
        SELECT
            *,
            ROW_NUMBER()
                OVER (PARTITION BY review_id ORDER BY inference_time DESC)
            AS rn
        FROM lakehouse.gold.review_sentiments
    ) AS t
    WHERE rn = 1
),

comments_deduped AS (
    SELECT *
    FROM (
        SELECT
            *,
            ROW_NUMBER()
                OVER (PARTITION BY review_id ORDER BY purchased_at DESC)
            AS rn
        FROM {{ ref('stg_comments') }}
    ) AS t
    WHERE rn = 1
)

SELECT
    c.review_id,
    c.product_id,
    c.purchased_at,
    c.customer_id,
    c.customer_name,
    c.is_buyer,
    c.rating,
    c.comment,
    s.sentiment AS sentiment_label,
    s.model_version,
    s.inference_time,
    DATE_FORMAT(FROM_UNIXTIME(c.purchased_at), 'yyyyMMdd') AS date_id
FROM comments_deduped AS c
LEFT JOIN review_sentiment AS s
    ON c.review_id = s.review_id

{% if is_incremental() %}
    WHERE c.purchased_at >= (
        SELECT COALESCE(MAX(purchased_at), 0) FROM {{ this }}
    )
{% endif %}
