"""
LangGraph checkpointer (thread persistence) + memory summarization.

Trước đây run_graph_stream cắt cứng `history[-6:]`: quá 6 lượt là MẤT HẲN —
khách nói ngân sách ở lượt 1 thì tới lượt 8 agent quên sạch. Redis giữ 20 lượt
nhưng 14 lượt kia không ai đọc.

Giờ dùng checkpointer native của LangGraph (thread_id = customer_id) + node
`summarize` nén các lượt cũ thành `summary` và xoá chúng khỏi state bằng
RemoveMessage, giữ nguyên văn N lượt gần nhất.

Test dùng Redis thật (cần RediSearch — redis-stack). Không có thì tự skip.
"""

import os

import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage, HumanMessage

REDIS_URL = os.getenv(
    "TEST_REDIS_URL",
    f"redis://:{os.getenv('REDIS_PASSWORD', '123')}@"
    f"{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}/0",
)


@pytest.fixture
def saver():
    from langgraph.checkpoint.redis import RedisSaver

    try:
        cm = RedisSaver.from_conn_string(REDIS_URL)
        s = cm.__enter__()
        s.setup()
    except Exception as exc:
        pytest.skip(f"Redis không có RediSearch (cần redis-stack): {exc}")
    yield s
    try:
        cm.__exit__(None, None, None)
    except Exception:
        pass


@pytest_asyncio.fixture
async def async_saver():
    from langgraph.checkpoint.redis import AsyncRedisSaver

    try:
        cm = AsyncRedisSaver.from_conn_string(REDIS_URL)
        s = await cm.__aenter__()
        await s.asetup()
    except Exception as exc:
        pytest.skip(f"Redis không có RediSearch (cần redis-stack): {exc}")
    yield s
    try:
        await cm.__aexit__(None, None, None)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Điều kiện kích hoạt summarization
# ---------------------------------------------------------------------------


def test_short_thread_goes_straight_to_router():
    """Hội thoại ngắn thì KHÔNG nén — nén sớm = tốn thêm 1 lời gọi LLM vô ích."""
    from src.serving.agent_api.graph import _needs_summary

    state = {"messages": [HumanMessage(content="xin chào")]}
    assert _needs_summary(state) == "router"


def test_long_thread_triggers_summarize():
    from src.serving.agent_api.graph import _SUMMARIZE_AFTER_MESSAGES, _needs_summary

    msgs = [
        HumanMessage(content=f"lượt {i}") for i in range(_SUMMARIZE_AFTER_MESSAGES + 1)
    ]
    assert _needs_summary({"messages": msgs}) == "summarize"


# ---------------------------------------------------------------------------
# Checkpointer: state có thật sự sống qua các request rời nhau không
# ---------------------------------------------------------------------------


def _echo_graph(checkpointer):
    """Graph tối giản — cô lập hành vi checkpointer, không phụ thuộc LLM."""
    from langchain_core.messages import BaseMessage
    from langgraph.graph import END, StateGraph
    from langgraph.graph.message import add_messages
    from typing_extensions import Annotated, TypedDict

    class S(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    g = StateGraph(S)
    g.add_node(
        "reply",
        lambda s: {
            "messages": [AIMessage(content=f"đã nhận: {s['messages'][-1].content}")]
        },
    )
    g.set_entry_point("reply")
    g.add_edge("reply", END)
    return g.compile(checkpointer=checkpointer)


def test_thread_state_survives_across_separate_invocations(saver):
    """Đây chính là thứ history[-6:] không làm được: lượt sau thấy được lượt trước."""
    app = _echo_graph(saver)
    cfg = {"configurable": {"thread_id": "khach_persist_test"}}

    app.invoke({"messages": [HumanMessage(content="ngân sách 5 triệu")]}, cfg)
    app.invoke({"messages": [HumanMessage(content="gợi ý laptop")]}, cfg)

    contents = [m.content for m in app.get_state(cfg).values["messages"]]
    assert "ngân sách 5 triệu" in contents, "lượt đầu phải còn trong thread state"
    assert "gợi ý laptop" in contents


@pytest.mark.asyncio
async def test_astream_events_works_with_async_checkpointer(async_saver):
    """
    Bug thật bắt được lúc verify deploy: main.py dùng RedisSaver (sync) trong khi
    graph chạy qua astream_events (async) — checkpointer sync KHÔNG implement
    aget_tuple(), astream_events() ném NotImplementedError giữa chừng request
    thật. Phải dùng AsyncRedisSaver cho đường async.
    """
    app = _echo_graph(async_saver)
    cfg = {"configurable": {"thread_id": "khach_async_test"}}

    events = [
        e
        async for e in app.astream_events(
            {"messages": [HumanMessage(content="tìm laptop")]}, cfg, version="v2"
        )
    ]
    assert len(events) > 0

    contents = [m.content for m in (await app.aget_state(cfg)).values["messages"]]
    assert "tìm laptop" in contents


def test_threads_are_isolated_per_customer(saver):
    """thread_id = customer_id — không được rò hội thoại của khách này sang khách khác."""
    app = _echo_graph(saver)
    cfg_a = {"configurable": {"thread_id": "khach_A"}}
    cfg_b = {"configurable": {"thread_id": "khach_B"}}

    app.invoke({"messages": [HumanMessage(content="bí mật của A")]}, cfg_a)
    app.invoke({"messages": [HumanMessage(content="câu hỏi của B")]}, cfg_b)

    b_contents = [m.content for m in app.get_state(cfg_b).values["messages"]]
    assert "bí mật của A" not in b_contents


# ---------------------------------------------------------------------------
# Summarization: nén xong phải THỰC SỰ xoá message cũ, không chỉ thêm summary
# ---------------------------------------------------------------------------


def test_summarize_removes_old_messages_and_keeps_recent(saver):
    """
    Chỉ thêm summary mà không xoá message cũ thì state vẫn phình vô hạn và vẫn
    tràn context — đúng thứ đang muốn tránh. RemoveMessage phải cắt thật.
    """
    from langchain_core.messages import BaseMessage, RemoveMessage
    from langgraph.graph import END, StateGraph
    from langgraph.graph.message import add_messages
    from typing_extensions import Annotated, TypedDict

    from src.serving.agent_api.graph import _KEEP_RECENT_MESSAGES

    class S(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
        summary: str

    def fake_summarize(state):
        old = state["messages"][:-_KEEP_RECENT_MESSAGES]
        return {
            "summary": "Khách có ngân sách 5 triệu, quan tâm laptop.",
            "messages": [RemoveMessage(id=m.id) for m in old if m.id],
        }

    g = StateGraph(S)
    g.add_node("summarize", fake_summarize)
    g.set_entry_point("summarize")
    g.add_edge("summarize", END)
    app = g.compile(checkpointer=saver)

    cfg = {"configurable": {"thread_id": "khach_summary_test"}}
    seed = [HumanMessage(content=f"lượt {i}") for i in range(10)]
    app.invoke({"messages": seed, "summary": ""}, cfg)

    final = app.get_state(cfg).values
    assert (
        len(final["messages"]) == _KEEP_RECENT_MESSAGES
    ), "message cũ phải bị xoá thật"
    assert final["summary"], "phải sinh ra bản tóm tắt"
    # Lượt gần nhất giữ nguyên văn, lượt cũ đã bị nén
    assert final["messages"][-1].content == "lượt 9"
    assert all(m.content != "lượt 0" for m in final["messages"])
