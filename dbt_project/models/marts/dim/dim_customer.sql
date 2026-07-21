{{ config(materialized='table', tags=['mart_stg']) }}

WITH ranked AS (
    SELECT
        customer_id,
        customer_name,
        ROW_NUMBER() OVER (
            PARTITION BY customer_id
            ORDER BY customer_name
        ) AS rn
    FROM {{ ref('stg_comments') }}
)

SELECT
    customer_id,
    customer_name
FROM ranked
WHERE rn = 1
