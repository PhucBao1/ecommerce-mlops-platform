{{ config(
    materialized='view',
    tags=['staging']
) }}

WITH source AS (
    SELECT
        product_id,
        review_id,
        customer_id,
        customer_name,
        rating,
        clean_comment,
        is_buyer,
        purchased_at
    FROM {{ source('lakehouse_silver', 'cleaned_comment') }}
)

SELECT
    customer_name,
    clean_comment AS comment,
    purchased_at,
    CAST(product_id AS STRING) AS product_id,
    CAST(review_id AS STRING) AS review_id,
    CAST(customer_id AS STRING) AS customer_id,
    -- Verified Buyer (Khách đã mua hàng thật chưa)
    CAST(rating AS FLOAT) AS rating,

    -- Customer Since (Ngày mua hàng)
    CAST(is_buyer AS BOOLEAN) AS is_buyer
FROM source
