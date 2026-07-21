{{ config(materialized='table', tags=['mart_snp']) }}

SELECT
    product_id,
    sku,
    product_name,
    short_description,
    brand_id,
    category_id,
    seller_id,
    price,
    quantity_sold,
    stock_qty,
    url,
    thumbnail_url,
    CASE
        WHEN dbt_valid_to IS NULL THEN 'Active'
        ELSE 'Inactive'
    END AS status
FROM {{ ref('snp_product') }}
