import json
import logging
import os

import redis.asyncio as redis

logger = logging.getLogger(__name__)

# db=0 — dành riêng cho recommendation cache.
# redis.asyncio (không phải redis thường) — trước đây redis_client là client
# SYNC gọi trực tiếp trong route `async def` (main.py), chặn nguyên event loop
# mỗi lần get/setex vì Python không tự chuyển sang coroutine khác giữa chừng
# 1 lời gọi socket blocking. Dưới tải đồng thời cao, việc này khiến toàn bộ
# request khác phải xếp hàng dù bản thân mỗi lệnh Redis chỉ mất <1ms — đã đo
# trực tiếp: latency thật (P50) giảm từ giây xuống ms sau khi đổi sang async
# (benchmark GPU vs CPU 16/7/2026, cùng lúc phát hiện thêm 2 bug hạ tầng khác
# — OTLP/Kafka trỏ hostname không resolve được — cũng góp phần vào latency cũ).
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD") or None,
    db=0,
    decode_responses=True,
    socket_connect_timeout=1,
    socket_timeout=1,
)

CACHE_TTL = 300


async def get_cached_recommendations(customer_id):
    key = f"recs:{customer_id}"
    try:
        data = await redis_client.get(key)
        if data:
            return json.loads(data)
    except Exception:
        pass
    return None


async def cache_recommendations(customer_id, recommendations):
    key = f"recs:{customer_id}"
    try:
        await redis_client.setex(key, CACHE_TTL, json.dumps(recommendations))
    except Exception:
        pass


async def invalidate_recommendation_cache(customer_id):
    key = f"recs:{customer_id}"
    try:
        await redis_client.delete(key)
    except Exception:
        pass
