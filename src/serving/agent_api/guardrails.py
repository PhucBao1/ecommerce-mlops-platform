import logging
import os
import re
import time
from dataclasses import dataclass

import redis

logger = logging.getLogger(__name__)

_INJECTION_PATTERNS = [
    r"ignore\s+(previous|prior|above|all)\s+instructions",
    r"forget\s+(everything|all|your|the)",
    r"you\s+are\s+now\s+",
    r"new\s+instructions?\s*:",
    r"system\s+prompt",
    r"reveal\s+(your|the)\s+(prompt|instructions?|system)",
    r"act\s+as\s+(if\s+you\s+are|a\s+different|an?\s+)",
    r"disregard\s+(your|all|previous)",
    r"bỏ\s+qua\s+(hướng\s+dẫn|lệnh|quy\s+tắc)",
    r"giả\s+vờ\s+.{0,20}là",
    r"đóng\s+vai",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]
_CJK_RE = re.compile(r"[一-鿿]")

_MAX_LEN = 500
_RATE_LIMIT = int(os.getenv("GUARDRAIL_RATE_LIMIT", "20"))  # requests per minute
_RATE_WINDOW = 60

# PII — redact TRƯỚC KHI gửi lên LLM provider (Claude/OpenAI/vLLM bên ngoài).
# check_output() cũ chỉ redact PII ở câu TRẢ LỜI, nghĩa là số điện thoại/email/
# thẻ của khách trong câu HỎI vẫn bay thẳng sang provider — vấn đề compliance.
_PII_PATTERNS = [
    (re.compile(r"\b(?:\+?84|0)(?:\d[ .-]?){8,10}\d\b"), "[SĐT_ĐÃ_ẨN]"),
    (
        re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
        "[EMAIL_ĐÃ_ẨN]",
    ),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[SỐ_THẺ_ĐÃ_ẨN]"),
]

_RETRIEVED_CONTENT_MAX_LEN = 4000


def _get_redis() -> redis.Redis:
    # db=3 — dành cho rate-limit trên toàn platform (cùng concern với
    # recsys_api/nlp_api rate-limit), tách khỏi cache (db=0),
    # Feast (db=1), agent memory (db=2) theo P2-7.
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "redis"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        password=os.getenv("REDIS_PASSWORD", "123"),
        db=3,
        decode_responses=True,
    )


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str = ""


class Guardrails:
    def __init__(self):
        self._r = _get_redis()

    def check_input(self, message: str, customer_id: str) -> GuardrailResult:
        if len(message) > _MAX_LEN:
            logger.warning(
                "guardrail_blocked",
                extra={"reason": "too_long", "customer_id": customer_id},
            )
            return GuardrailResult(False, "too_long")

        for pattern in _COMPILED:
            if pattern.search(message):
                logger.warning(
                    "guardrail_blocked",
                    extra={
                        "reason": "injection",
                        "customer_id": customer_id,
                        "pattern": pattern.pattern[:40],
                    },
                )
                return GuardrailResult(False, "injection_detected")

        key = f"rate:{customer_id}"
        try:
            count = self._r.incr(key)
            if count == 1:
                self._r.expire(key, _RATE_WINDOW)
            if count > _RATE_LIMIT:
                logger.warning(
                    "guardrail_blocked",
                    extra={"reason": "rate_limit", "customer_id": customer_id},
                )
                return GuardrailResult(False, "rate_limit")
        except Exception:
            pass  # if Redis down, let through

        return GuardrailResult(True)

    def check_output(self, response: str) -> str:
        leaks = [
            r"system\s+prompt",
            r"you\s+are\s+a\s+helpful",
            r"<\|system\|>",
            r"\[INST\]",
        ]
        for pattern in leaks:
            response = re.sub(pattern, "[...]", response, flags=re.IGNORECASE)

        # Bug #14 (BENCHMARK_RESULTS.md, 17/7/2026): Qwen2.5-7B-AWQ đôi khi
        # degenerate giữa chừng, lẫn hẳn 1 đoạn tiếng Trung vào response tiếng
        # Việt (100% tái hiện với 1 câu hỏi search_kb cụ thể qua 3 lần test, dù
        # đã thử giảm context/frequency_penalty, không dứt điểm được ở tầng
        # generation). Cắt response tại điểm bắt đầu xuất hiện CJK — phần trước
        # đó luôn là tiếng Việt hợp lệ, không cần bỏ cả câu trả lời.
        cjk_match = _CJK_RE.search(response)
        if cjk_match:
            response = response[: cjk_match.start()].rstrip()
            logger.warning(
                "guardrail_cjk_leak_truncated", extra={"cut_at": cjk_match.start()}
            )

        return response.strip()


def redact_pii(text: str) -> str:
    """Ẩn SĐT/email/số thẻ trước khi text rời hệ thống đi sang LLM provider."""
    for pattern, placeholder in _PII_PATTERNS:
        text = pattern.sub(placeholder, text)
    return text


def sanitize_retrieved_content(text: str, source: str = "") -> tuple[str, bool]:
    """
    Làm sạch nội dung retrieve từ KB trước khi đưa vào context LLM.

    Trả về (text_đã_làm_sạch, có_phát_hiện_injection).

    KHÔNG chặn cả câu trả lời khi phát hiện injection — vì doc có thể vẫn chứa
    thông tin chính sách hợp lệ bên cạnh payload độc; chặn hết là tự DoS chính
    mình. Thay vào đó vô hiệu hoá câu lệnh (đánh dấu rõ ràng) và log lại để
    truy vết file nào bị nhiễm.
    """
    flagged = False
    for pattern in _COMPILED:
        if pattern.search(text):
            flagged = True
            text = pattern.sub("[NỘI_DUNG_BỊ_CHẶN]", text)

    if flagged:
        logger.warning(
            "indirect_injection_detected_in_kb",
            extra={"source": source or "unknown"},
        )

    if len(text) > _RETRIEVED_CONTENT_MAX_LEN:
        text = text[:_RETRIEVED_CONTENT_MAX_LEN]

    return text, flagged
