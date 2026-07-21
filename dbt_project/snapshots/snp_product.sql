{% snapshot snp_product %}

{{ config(
    target_schema='snapshots',
    unique_key='product_id',
    strategy='check',
    check_cols='all',
    invalidate_hard_deletes=True
) }}

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
    thumbnail_url
FROM {{ ref('stg_products') }}

{% endsnapshot %}
