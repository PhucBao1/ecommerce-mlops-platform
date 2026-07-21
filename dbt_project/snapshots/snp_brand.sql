{% snapshot snp_brand %}

{{ config(
    target_schema='snapshots',
    unique_key='brand_id',
    strategy='check',
    check_cols='all',
    invalidate_hard_deletes=True
) }}

SELECT DISTINCT
    brand_id,
    brand_name
FROM {{ ref('stg_products') }}
WHERE brand_id IS NOT NULL

{% endsnapshot %}
