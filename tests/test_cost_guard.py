"""
Cost governance — budget theo user + circuit breaker toàn cục.

Chạy với Redis thật (db=3, key có prefix cost: + ngày nên không đụng dữ liệu
khác). Nếu không có Redis, test tự skip thay vì fail giả.
"""

import importlib
import os

import pytest


@pytest.fixture
def cg(monkeypatch):
    monkeypatch.setenv("REDIS_HOST", os.getenv("REDIS_HOST", "localhost"))
    monkeypatch.setenv("REDIS_PASSWORD", os.getenv("REDIS_PASSWORD", "123"))
    monkeypatch.setenv("USER_DAILY_BUDGET_USD", "0.10")
    monkeypatch.setenv("GLOBAL_DAILY_BUDGET_USD", "1.00")

    import src.serving.agent_api.cost_guard as mod

    mod = importlib.reload(mod)

    try:
        mod._r().ping()
    except Exception:
        pytest.skip("Redis không chạy — bỏ qua test cost guard")

    # dọn key của chính ngày hôm nay để test chạy lại được nhiều lần
    r = mod._r()
    for key in r.scan_iter("cost:*"):
        r.delete(key)
    yield mod
    for key in r.scan_iter("cost:*"):
        r.delete(key)


def test_new_user_is_allowed(cg):
    assert cg.check_budget("user_moi").allowed is True


def test_user_blocked_after_exceeding_own_budget(cg):
    cg.record_cost("user_ton_tien", 0.12)  # trần user = 0.10
    status = cg.check_budget("user_ton_tien")
    assert status.allowed is False
    assert status.reason == "user_budget_exceeded"


def test_one_user_hitting_limit_does_not_block_others(cg):
    """Budget theo user phải cô lập — không được để 1 người làm sập cả hệ thống."""
    cg.record_cost("user_xau", 0.12)
    assert cg.check_budget("user_xau").allowed is False
    assert cg.check_budget("user_binh_thuong").allowed is True


def test_global_circuit_breaker_blocks_everyone(cg):
    """Vượt trần TOÀN CỤC thì chặn cả user chưa tiêu gì — đây là sự cố hệ thống."""
    for i in range(12):
        cg.record_cost(f"user_{i}", 0.09)  # tổng ~1.08 > trần global 1.00

    status = cg.check_budget("user_hoan_toan_moi")
    assert status.allowed is False
    assert status.reason == "global_budget_exceeded"


def test_zero_cost_is_not_recorded(cg):
    """Ollama local = free; ghi 0 vào Redis chỉ tổ rác."""
    cg.record_cost("user_ollama", 0.0)
    assert cg.check_budget("user_ollama").spent == 0.0


def test_fails_open_when_redis_down(monkeypatch):
    """Redis chết KHÔNG được làm sập API — chặn hết còn tệ hơn thứ đang phòng."""
    monkeypatch.setenv("REDIS_HOST", "khong-ton-tai.invalid")
    import src.serving.agent_api.cost_guard as mod

    mod = importlib.reload(mod)
    assert mod.check_budget("bat_ky_ai").allowed is True
