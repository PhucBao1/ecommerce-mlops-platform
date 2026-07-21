{{ config(materialized='table', tags=['mart_snp']) }}

SELECT
    category_id,
    category_name
FROM {{ ref('snp_category') }}
WHERE dbt_valid_to IS NULL
