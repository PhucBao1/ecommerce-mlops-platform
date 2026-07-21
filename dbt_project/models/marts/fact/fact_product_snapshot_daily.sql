{{ config(
    materialized='incremental',
    incremental_strategy='merge',
    file_format='iceberg',
    unique_key='snapshot_id',
    tags=['mart_stg']
) }}

WITH deduped_products AS (
    SELECT *
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY product_id, crawl_date ORDER BY crawl_date DESC) AS rn
        FROM {{ ref('stg_products') }}
    ) t
    WHERE rn = 1
)

SELECT
    {{ dbt_utils.generate_surrogate_key(['product_id', 'crawl_date']) }} AS snapshot_id,
    product_id,
    brand_id,
    category_id,
    seller_id,
    price,
    list_price,
    discount_rate,
    stock_qty,
    quantity_sold,
    review_count,
    rating,
    inventory_status,
    crawl_date
FROM deduped_products

{% if is_incremental() %}
WHERE crawl_date >= (SELECT MAX(crawl_date) FROM {{ this }})
{% endif %}
