"""
Cost governance cho agent_api.

Rate limit (security.py) chặn được số REQUEST, nhưng không chặn được CHI PHÍ:
1 request có thể kéo context dài + nhiều vòng tool-call → tốn gấp hàng chục lần
1 request bình thường. Rate limit 60 req/phút vẫn có thể đốt sạch budget.

Hai lớp, tách bạch vì mục đích khác nhau:

  • Budget theo user (ngày)  — chống 1 user lạm dụng; vượt thì chỉ user đó bị
    chặn, người khác vẫn dùng bình thường.
  • Circuit breaker toàn cục — chống sự cố hệ thống (bug retry loop, bị scrape,
    prompt injection kéo model sinh vô hạn). Vượt trần ngày của CẢ hệ thống thì
    ngắt sạch, vì lúc này vấn đề không nằm ở một user cụ thể.

Chi phí lấy từ chính litellm.cost_per_token() mà graph.py đã dùng để đẩy metric
llm_cost_usd_total — dùng chung một nguồn số, không tự tính lại cho lệch.

Redis chết → CHO QUA (fail-open): chặn hết request khi Redis lỗi là tự gây sự cố
lớn hơn thứ đang muốn phòng. Đánh đổi này là cố ý, khác với admin auth
(security.py) vốn fail-CLOSED vì đó là bảo mật thật.
"""

import datetime as dt
import logging
import os

import redis

logger = logging.getLogger(__name__)

# Trần chi phí — đơn vị USD
_USER_DAILY_BUDGET = float(os.getenv("USER_DAILY_BUDGET_USD", "0.50"))
_GLOBAL_DAILY_BUDGET = float(os.getenv("GLOBAL_DAILY_BUDGET_USD", "5.00"))

_KEY_TTL = 60 * 60 * 36  # 36h — đủ qua ngày, tự dọn, không cần cron


def _get_redis() -> redis.Redis:
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "redis"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        password=os.getenv("REDIS_PASSWORD", "123"),
        db=3,  # dùng chung db với rate-limit (cùng concern: chống lạm dụng)
        decode_responses=True,
    )


_client: redis.Redis | None = None


def _r() -> redis.Redis:
    global _client
    if _client is None:
        _client = _get_redis()
    return _client


def _today() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")


def _user_key(customer_id: str) -> str:
    return f"cost:user:{_today()}:{customer_id}"


def _global_key() -> str:
    return f"cost:global:{_today()}"


class BudgetStatus:
    """Kết quả kiểm tra budget — dùng thay tuple để chỗ gọi đọc dễ hiểu."""

    def __init__(self, allowed: bool, reason: str = "", spent: float = 0.0):
        self.allowed = allowed
        self.reason = reason
        self.spent = spent


def check_budget(customer_id: str) -> BudgetStatus:
    """
    Gọi TRƯỚC khi bắt đầu 1 lượt chat. Chặn dựa trên chi phí ĐÃ tiêu hôm nay
    (không dự đoán chi phí request sắp tới — không thể biết trước độ dài output).
    Nghĩa là user có thể vượt trần một chút ở request cuối; chấp nhận được, vì
    mục tiêu là chặn lạm dụng kéo dài chứ không phải kế toán chính xác từng cent.
    """
    try:
        r = _r()
        global_spent = float(r.get(_global_key()) or 0.0)
        if global_spent >= _GLOBAL_DAILY_BUDGET:
            logger.error(
                "global_budget_exceeded",
                extra={"spent": global_spent, "limit": _GLOBAL_DAILY_BUDGET},
            )
            return BudgetStatus(False, "global_budget_exceeded", global_spent)

        user_spent = float(r.get(_user_key(customer_id)) or 0.0)
        if user_spent >= _USER_DAILY_BUDGET:
            logger.warning(
                "user_budget_exceeded",
                extra={"customer_id": customer_id, "spent": user_spent},
            )
            return BudgetStatus(False, "user_budget_exceeded", user_spent)

        return BudgetStatus(True, spent=user_spent)
    except Exception as exc:
        logger.warning("cost_guard_redis_error: %s", exc)
        return BudgetStatus(True)  # fail-open, xem docstring đầu file


def record_cost(customer_id: str, cost_usd: float) -> None:
    """Cộng dồn chi phí thật sau khi LLM trả lời xong."""
    if cost_usd <= 0:
        return  # Ollama local = free, không cần ghi
    try:
        r = _r()
        pipe = r.pipeline()
        pipe.incrbyfloat(_user_key(customer_id), cost_usd)
        pipe.expire(_user_key(customer_id), _KEY_TTL)
        pipe.incrbyfloat(_global_key(), cost_usd)
        pipe.expire(_global_key(), _KEY_TTL)
        pipe.execute()
    except Exception as exc:
        logger.warning("cost_guard_record_failed: %s", exc)
