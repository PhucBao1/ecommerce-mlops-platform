"""
LLM resilience — timeout, retry, degradation khi backend chết.

Trước đây lời gọi LLM trong graph.py trần trụi: không timeout (model treo =
request treo vô hạn), không try/except (Ollama/vLLM chết = exception bắn thẳng
ra giữa SSE stream, user nhận stream vỡ). Bản cũ agent.py TỪNG có fallback
"llm_unavailable" (chạy RAG search trực tiếp) nhưng agent.py là file chết —
main.py không import — nên khi viết lại sang graph.py thì mất luôn.

Đặc biệt quan trọng cho Phase 2: GPU Spot bị AWS thu hồi bất cứ lúc nào, agent
phải sống sót chứ không chết theo replica.

Test bằng mock, KHÔNG gọi LLM thật.
"""

import os
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Timeout: model treo không được làm request treo vô hạn
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "backend, timeout_field",
    [
        ("vllm", "request_timeout"),
        ("claude", "default_request_timeout"),
    ],
)
def test_backend_has_timeout_configured(monkeypatch, backend, timeout_field):
    """Mỗi backend gọi tên tham số timeout một kiểu — dễ set nhầm chỗ mà không ai biết."""
    monkeypatch.setenv("AGENT_LLM_BACKEND", backend)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")  # ChatAnthropic đòi key

    from src.serving.agent_api.graph import _get_chat_model

    model, got_backend, _ = _get_chat_model()
    assert got_backend == backend
    assert getattr(model, timeout_field) is not None, f"{backend} thiếu timeout"


def test_ollama_timeout_goes_through_client_kwargs(monkeypatch):
    """ChatOllama KHÔNG có field timeout — phải nhét qua client_kwargs."""
    monkeypatch.setenv("AGENT_LLM_BACKEND", "ollama")

    from src.serving.agent_api.graph import _get_chat_model

    model, _, _ = _get_chat_model()
    assert model.client_kwargs.get("timeout") is not None


def test_retries_configured_for_remote_backends(monkeypatch):
    """
    vLLM chạy sau nginx trên GPU Spot: AWS thu hồi replica giữa chừng thì retry
    cho phép rơi sang replica còn sống mà user không thấy gì.
    """
    monkeypatch.setenv("AGENT_LLM_BACKEND", "vllm")

    from src.serving.agent_api.graph import _get_chat_model

    model, _, _ = _get_chat_model()
    assert model.max_retries >= 1


# ---------------------------------------------------------------------------
# Degradation: LLM chết thì vẫn phải trả được thứ hữu ích
# ---------------------------------------------------------------------------


def test_fallback_returns_search_results_when_llm_dies():
    """
    Retrieval KHÔNG cần LLM — nên LLM chết vẫn phải chạy RAG search và đưa
    danh sách sản phẩm, thay vì chỉ xin lỗi suông.
    """
    from src.serving.agent_api.graph import build_llm_fallback_message

    rag = MagicMock()
    rag.search.return_value = [
        {"product_name": "Laptop Dell XPS", "price": 25_000_000},
        {"product_name": "Laptop HP Envy", "price": 18_000_000},
    ]

    msg = build_llm_fallback_message("tìm laptop", rag)

    assert "không khả dụng" in msg
    assert "Laptop Dell XPS" in msg
    assert "25,000,000 VND" in msg
    rag.search.assert_called_once()


def test_fallback_apologizes_when_retrieval_also_fails():
    """Cả RAG cũng hỏng => xin lỗi, KHÔNG được ném exception ra ngoài."""
    from src.serving.agent_api.graph import build_llm_fallback_message

    rag = MagicMock()
    rag.search.side_effect = RuntimeError("Qdrant down")

    msg = build_llm_fallback_message("tìm laptop", rag)
    assert "Xin lỗi" in msg


def test_fallback_without_rag_pipeline_does_not_crash():
    from src.serving.agent_api.graph import build_llm_fallback_message

    msg = build_llm_fallback_message("tìm laptop", None)
    assert "Xin lỗi" in msg


def test_fallback_increments_metric():
    """
    agent_fallback_total trước đây CHỈ được tăng trong agent.py (file chết) —
    tức là Grafana không bao giờ thấy fallback nào dù có xảy ra thật.
    """
    from src.serving.agent_api.graph import build_llm_fallback_message
    from src.serving.agent_api.tracing import agent_fallback_total

    before = agent_fallback_total.labels(tier="llm_unavailable")._value.get()
    build_llm_fallback_message("test", None)
    after = agent_fallback_total.labels(tier="llm_unavailable")._value.get()

    assert after == before + 1
