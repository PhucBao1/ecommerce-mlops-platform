import json
import os

import redis

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True,
)

CACHE_TTL = 300


def get_cached_recommendations(customer_id):

    key = f"recs:{customer_id}"

    data = redis_client.get(key)

    if data:

        return json.loads(data)

    return None


def cache_recommendations(customer_id, recommendations):

    key = f"recs:{customer_id}"

    redis_client.setex(key, CACHE_TTL, json.dumps(recommendations))


def invalidate_recommendation_cache(customer_id):

    key = f"recs:{customer_id}"

    redis_client.delete(key)
