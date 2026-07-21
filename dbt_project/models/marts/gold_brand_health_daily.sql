{{ config(
    materialized='table',
    properties={
        "format": "iceberg"
    }
) }}

WITH reviews AS (
    SELECT * FROM {{ ref('fact_review') }}
),

products AS (
    SELECT * FROM {{ ref('dim_product') }}
)

SELECT
    -- Chuyển từ BIGINT -> TIMESTAMP -> DATE
    r.date_id,
    p.product_id,
    r.customer_name AS product_name,
    COUNT(r.comment) AS total_reviews,
    SUM(
        CASE WHEN r.sentiment_label = 'POS' THEN 1 ELSE 0 END
    ) AS positive_reviews,
    SUM(
        CASE WHEN r.sentiment_label = 'NEG' THEN 1 ELSE 0 END
    ) AS negative_reviews,
    SUM(
        CASE WHEN r.sentiment_label = 'NEU' THEN 1 ELSE 0 END
    ) AS neutral_reviews,
    ROUND(AVG(r.rating), 2) AS average_rating
FROM reviews AS r
INNER JOIN products AS p ON r.product_id = p.product_id
WHERE status LIKE 'Active'
GROUP BY 1, 2, 3
