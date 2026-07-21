{{
    config(
        materialized='table',
        tags=['mart_stg']
    )
}}

with rfm as (
    select
        customer_id,
        datediff(current_date, to_date(from_unixtime(max(purchased_at) / 1000000000))) as recency_days,
        count(distinct review_id) as frequency,
        sum(rating) as monetary_proxy
    from {{ source('lakehouse_silver', 'cleaned_comment') }}
    group by 1
),

scored as (
    select
        *,
        case
            when recency_days <= 7 then 3
            when recency_days <= 30 then 2
            else 1
        end as r_score,
        case
            when frequency >= 10 then 3
            when frequency >= 3 then 2
            else 1
        end as f_score,
        case
            when monetary_proxy >= 20 then 3
            when monetary_proxy >= 8 then 2
            else 1
        end as m_score
    from rfm
)

select
    customer_id,
    recency_days,
    frequency,
    monetary_proxy,
    r_score,
    f_score,
    m_score,
    r_score + f_score + m_score as rfm_total,
    case
        when r_score + f_score + m_score >= 8 then 'Champion'
        when r_score + f_score + m_score >= 6 then 'Loyal'
        when r_score = 3 then 'New Customer'
        else 'At Risk'
    end as segment
from scored
