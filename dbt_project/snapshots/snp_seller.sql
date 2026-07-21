{% snapshot snp_seller %}

{{ config(
    target_schema='snapshots',
    unique_key='seller_id',
    strategy='check',
    check_cols='all',
    invalidate_hard_deletes=True
) }}

SELECT DISTINCT
    seller_id,
    seller_name,
    seller_logo,
    seller_link
FROM {{ ref('stg_products') }}
WHERE seller_id IS NOT NULL

{% endsnapshot %}
