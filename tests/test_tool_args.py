"""
Structured output — validate tool args do LLM sinh ra.

Đầu ra của LLM là dữ liệu KHÔNG tin cậy: model nhỏ (qwen2.5:3b) thường xuyên
bịa tham số hoặc vượt ràng buộc. TOOL_ARG_MODELS đã định nghĩa sẵn ràng buộc
(top_k <= 20...) nhưng trước đây KHÔNG chỗ nào gọi tới — các hàm tool đọc thẳng
args.get("top_k", 5), nên top_k=9999 đi thẳng vào FAISS/recsys-api.
"""

import pytest

from src.serving.agent_api.tools import _validate_args, execute_tool


def test_valid_args_pass_through():
    out = _validate_args("search_products", {"query": "laptop", "top_k": 5})
    assert out["query"] == "laptop"
    assert out["top_k"] == 5


def test_defaults_applied_when_llm_omits_optional_args():
    out = _validate_args("search_products", {"query": "tai nghe"})
    assert out["top_k"] == 5  # default từ Pydantic, không phải .get() rải rác


def test_top_k_above_limit_is_rejected():
    """LLM bịa top_k=9999 phải bị chặn, không được đi tới FAISS."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _validate_args("search_products", {"query": "laptop", "top_k": 9999})


def test_missing_required_arg_is_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _validate_args("search_products", {"top_k": 5})  # thiếu `query`


def test_execute_tool_returns_actionable_error_instead_of_crashing():
    """
    Args sai schema => trả message lỗi CÓ CẤU TRÚC để LLM tự sửa ở vòng sau,
    không để tool nổ giữa chừng và cũng không im lặng cho qua.
    """
    result = execute_tool(
        "search_products", {"top_k": 9999}
    )  # thiếu query + quá giới hạn
    assert "không hợp lệ" in result
    assert "gọi lại tool" in result.lower()


def test_unknown_tool_is_reported():
    result = execute_tool("tool_khong_ton_tai", {})
    assert "không được hỗ trợ" in result
