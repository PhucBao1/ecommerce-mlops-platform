{{
    config(
        materialized='table',
        tags=['mart_stg']
    )
}}

-- Aggregates A/B experiment vs control from prediction events logged to DW.
-- Used to decide whether to promote a new model version to production.
select
    experiment_group,
    date_trunc('day', event_time) as date,
    count(*) as total_recommendations,
    count(case when action = 'click' then 1 end) as clicks,
    count(case when action = 'purchase' then 1 end) as purchases,
    count(case when action = 'click' then 1 end)
    / nullif(count(*), 0) as ctr,
    count(case when action = 'purchase' then 1 end)
    / nullif(count(*), 0) as purchase_rate
from {{ source('lakehouse_silver', 'prediction_events') }}
group by 1, 2
order by 2 desc, 1
