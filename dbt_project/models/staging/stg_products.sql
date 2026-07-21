{{ config(
    materialized='view',
    tags=['staging']
) }}
WITH source AS (
    SELECT
        product_id,
        sku,
        product_name,
        short_description,
        price,
        list_price,
        discount_rate,
        rating,
        review_count,
        inventory_status,
        stock_qty,
        quantity_sold,
        brand_id,
        brand_name,
        category_id,
        category_name,
        seller_id,
        seller_name,
        seller_logo,
        seller_link,
        url,
        thumbnail_url,
        crawl_date
    FROM {{ source('lakehouse_silver', 'cleaned_product') }}
)

SELECT
    sku,
    product_name,
    short_description,
    inventory_status,
    brand_name,
    category_name,
    seller_name,
    seller_logo,
    seller_link,
    url,
    thumbnail_url,
    crawl_date,
    CAST(product_id AS STRING) AS product_id,
    CAST(price AS FLOAT) AS price,
    CAST(list_price AS FLOAT) AS list_price,
    CAST(discount_rate AS FLOAT) AS discount_rate,
    CAST(rating AS FLOAT) AS rating,
    CAST(review_count AS INT) AS review_count,
    CAST(stock_qty AS INT) AS stock_qty,
    CAST(quantity_sold AS INT) AS quantity_sold,
    CAST(brand_id AS STRING) AS brand_id,
    CAST(category_id AS STRING) AS category_id,
    CAST(seller_id AS STRING) AS seller_id
FROM source
