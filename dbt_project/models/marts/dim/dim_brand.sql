{{ config(materialized='table', tags=['mart_snp']) }}

SELECT
    brand_id,
    brand_name
FROM {{ ref('snp_brand') }}
WHERE dbt_valid_to IS NULL
