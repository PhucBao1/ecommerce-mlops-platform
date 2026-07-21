import json
import os

import redis

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True,
)

TTL_SECONDS = 60 * 10


# =====================================================
# GET
# =====================================================


def get_cached_candidates(customer_id):

    key = f"candidate:{customer_id}"

    data = redis_client.get(key)

    if data is None:
        return None

    return json.loads(data)


# =====================================================
# SET
# =====================================================


def cache_candidates(customer_id, candidates):

    key = f"candidate:{customer_id}"

    redis_client.setex(key, TTL_SECONDS, json.dumps(candidates))


# =====================================================
# INVALIDATE
# =====================================================


def invalidate_candidate_cache(customer_id):

    key = f"candidate:{customer_id}"

    redis_client.delete(key)
