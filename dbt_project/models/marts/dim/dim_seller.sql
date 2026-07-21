{{ config(materialized='table', tags=['mart_snp']) }}

SELECT
    seller_id,
    seller_name,
    seller_logo,
    seller_link
FROM {{ ref('snp_seller') }}
WHERE dbt_valid_to IS NULL
