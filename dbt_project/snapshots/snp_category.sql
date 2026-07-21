{% snapshot snp_category %}

{{ config(
    target_schema='snapshots',
    unique_key='category_id',
    strategy='check',
    check_cols='all',
    invalidate_hard_deletes=True
) }}

SELECT DISTINCT
    category_id,
    category_name
FROM {{ ref('stg_products') }}
WHERE category_id IS NOT NULL

{% endsnapshot %}
