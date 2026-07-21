import json
import logging
import os

import redis

logger = logging.getLogger(__name__)

_MAX_HISTORY = 20
_HISTORY_TTL = 86_400  # 24h
_PREF_TTL = 604_800  # 7 days


def _get_redis() -> redis.Redis:
    # db=2 — tách riêng khỏi recsys cache (db=0) và Feast online store (db=1)
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "redis"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        password=os.getenv("REDIS_PASSWORD", "123"),
        db=2,
        decode_responses=True,
    )


class MemoryStore:
    def __init__(self):
        self._r = _get_redis()

    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------

    def get_history(self, customer_id: str) -> list[dict]:
        try:
            raw = self._r.lrange(f"conv:{customer_id}", 0, _MAX_HISTORY - 1)
        except redis.RedisError as e:
            logger.warning(
                "memory_get_history_failed customer_id=%s error=%s", customer_id, e
            )
            return []
        result = []
        for item in raw:
            try:
                result.append(json.loads(item))
            except json.JSONDecodeError:
                pass
        return result

    def append_turn(self, customer_id: str, role: str, content: str) -> None:
        key = f"conv:{customer_id}"
        try:
            self._r.lpush(key, json.dumps({"role": role, "content": content}))
            self._r.ltrim(key, 0, _MAX_HISTORY - 1)
            self._r.expire(key, _HISTORY_TTL)
        except redis.RedisError as e:
            logger.warning(
                "memory_append_turn_failed customer_id=%s error=%s", customer_id, e
            )

    def clear_history(self, customer_id: str) -> None:
        try:
            self._r.delete(f"conv:{customer_id}")
        except redis.RedisError as e:
            logger.warning(
                "memory_clear_history_failed customer_id=%s error=%s", customer_id, e
            )

    # ------------------------------------------------------------------
    # User preferences (inferred from tool calls)
    # ------------------------------------------------------------------

    def get_preferences(self, customer_id: str) -> dict:
        try:
            raw = self._r.hgetall(f"pref:{customer_id}")
        except redis.RedisError as e:
            logger.warning(
                "memory_get_preferences_failed customer_id=%s error=%s", customer_id, e
            )
            return {}
        prefs: dict = {}
        for k, v in raw.items():
            try:
                prefs[k] = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                prefs[k] = v
        return prefs

    def update_preferences(
        self, customer_id: str, tool_name: str, tool_input: dict
    ) -> None:
        key = f"pref:{customer_id}"
        updates: dict = {}

        if tool_name == "search_products":
            query = tool_input.get("query", "")
            if tool_input.get("max_price"):
                updates["price_max"] = tool_input["max_price"]
            if tool_input.get("min_price"):
                updates["price_min"] = tool_input["min_price"]
            if query:
                updates["last_query"] = query

        elif tool_name == "filter_by_price":
            if tool_input.get("max_price"):
                updates["price_max"] = tool_input["max_price"]
            if tool_input.get("min_price"):
                updates["price_min"] = tool_input["min_price"]

        elif tool_name == "get_recommendations":
            pass  # no preference signal from this call

        if updates:
            try:
                pipe = self._r.pipeline()
                for k, v in updates.items():
                    pipe.hset(key, k, json.dumps(v))
                pipe.expire(key, _PREF_TTL)
                pipe.execute()
            except redis.RedisError as e:
                logger.warning(
                    "memory_update_preferences_failed customer_id=%s error=%s",
                    customer_id,
                    e,
                )

    def build_preference_context(self, customer_id: str) -> str:
        prefs = self.get_preferences(customer_id)
        if not prefs:
            return ""
        parts = []
        if "price_max" in prefs:
            parts.append(f"budget tối đa {int(prefs['price_max']):,} VND")
        if "price_min" in prefs:
            parts.append(f"budget tối thiểu {int(prefs['price_min']):,} VND")
        if "last_query" in prefs:
            parts.append(f"tìm kiếm gần đây: '{prefs['last_query']}'")
        return "Thông tin về khách hàng: " + ", ".join(parts) + "." if parts else ""
