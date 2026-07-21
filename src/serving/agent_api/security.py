"""
Auth + rate limit cho agent_api.
"""

import hmac
import logging
import os
import time

import redis
from fastapi import Header, HTTPException, Request, status

logger = logging.getLogger(__name__)

_ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")

# Rate limit theo IP cho endpoint tốn tiền (mỗi request /chat/stream = token LLM)
_IP_RATE_LIMIT = int(os.getenv("IP_RATE_LIMIT_PER_MIN", "60"))
_IP_RATE_WINDOW = 60


def _get_redis() -> redis.Redis:
    # db=3 — dùng chung với rate-limit sẵn có ở guardrails.py
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "redis"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        password=os.getenv("REDIS_PASSWORD", "123"),
        db=3,
        decode_responses=True,
    )


_redis_client: redis.Redis | None = None


def _redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = _get_redis()
    return _redis_client


async def require_admin_api_key(
    x_api_key: str = Header(default=""),
    authorization: str = Header(default=""),
) -> None:
    """
    FastAPI dependency — chặn mọi endpoint /admin/*.
    """
    if not _ADMIN_API_KEY:
        logger.error("admin_auth_misconfigured: ADMIN_API_KEY chưa được set")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin API chưa được cấu hình.",
        )

    presented = x_api_key or authorization.removeprefix("Bearer ").strip()

    # compare_digest — tránh timing attack khi so sánh secret
    if not presented or not hmac.compare_digest(presented, _ADMIN_API_KEY):
        logger.warning("admin_auth_failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key không hợp lệ.",
        )


def _client_ip(request: Request) -> str:
    # X-Forwarded-For do proxy/nginx set (Phase 2 có nginx LB trước agent-api).
    # Lấy IP đầu tiên = client gốc.
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_ip_rate_limit(request: Request) -> bool:
    """
    True = cho qua, False = vượt quota.
    """
    ip = _client_ip(request)
    key = f"iprate:{ip}"
    try:
        r = _redis()
        count = r.incr(key)
        if count == 1:
            r.expire(key, _IP_RATE_WINDOW)
        if count > _IP_RATE_LIMIT:
            logger.warning("ip_rate_limit_exceeded", extra={"ip": ip, "count": count})
            return False
    except Exception as exc:
        logger.warning("ip_rate_limit_redis_error: %s", exc)
    return True
