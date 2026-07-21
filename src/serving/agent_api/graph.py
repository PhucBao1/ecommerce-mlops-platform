"""
LangGraph agent — StateGraph tự xây, Router node quyết định luồng tường minh
(kb_search vs product tools) qua add_conditional_edges, thay vì để LLM tự do
quyết định gọi tool nào (ReAct thuần qua create_react_agent).

Một endpoint duy nhất /chat/stream. Streaming qua astream_events(version="v2").
Tất cả tool (bao gồm search_kb) chạy chung qua 1 ToolNode để tool_calls chỉ có
1 nguồn duy nhất (event on_tool_start/on_tool_end), không cần track state riêng.
"""

import json
import logging
import os
import re
import time
import uuid
from typing import AsyncIterator

import litellm
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition
from typing_extensions import Annotated, TypedDict

from src.serving.agent_api.cost_guard import record_cost
from src.serving.agent_api.prompt_registry import PromptRegistry
from src.serving.agent_api.tools import execute_tool
from src.serving.agent_api.tracing import (
    LLMSpan,
    agent_fallback_total,
    conversation_summarized_total,
    llm_cost_usd_total,
    tool_call_total,
)

logger = logging.getLogger(__name__)

_RAW_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# Bug #14 (BENCHMARK_RESULTS.md, 17/7/2026): Qwen2.5-7B-AWQ đôi khi degenerate
# giữa chừng, lẫn tiếng Trung vào response tiếng Việt. guardrails.check_output
# chỉ dọn full_response SAU KHI stream xong — không đủ, vì token đã stream
# thẳng ra client qua SSE trước đó rồi. Phải chặn ngay trong vòng lặp stream.
_CJK_RE = re.compile(r"[一-鿿]")
_FALLBACK_MESSAGE = (
    "Xin lỗi, tôi chưa tìm được thông tin phù hợp. Bạn có thể hỏi cụ thể hơn không?"
)

_SYSTEM_PROMPT_FALLBACK = """Bạn là trợ lý mua sắm của Tiki. Trả lời bằng tiếng Việt, ngắn gọn.

Dùng tool để tìm sản phẩm hoặc gợi ý cá nhân hóa.
TUYỆT ĐỐI KHÔNG bịa URL, số điện thoại, email — nếu không tìm thấy thông tin thì nói thẳng."""

_PROMPT_CACHE_TTL_SECONDS = float(os.getenv("AGENT_PROMPT_CACHE_TTL", "30"))
_prompt_cache: dict = {"text": None, "version": None, "fetched_at": 0.0}
_registry: PromptRegistry | None = None


def _get_registry() -> PromptRegistry:
    global _registry
    if _registry is None:
        _registry = PromptRegistry()
    return _registry


def get_active_prompt() -> tuple[str, str]:
    """
    Trả về (prompt_text, version) đang active.

    Thứ tự fallback: Redis registry → cache cũ (last-known-good) → hằng số
    fallback. Redis chết KHÔNG được làm sập agent — chỉ mất khả năng hot-swap.
    """
    now = time.monotonic()
    cached_text = _prompt_cache["text"]
    if cached_text and (now - _prompt_cache["fetched_at"]) < _PROMPT_CACHE_TTL_SECONDS:
        return cached_text, _prompt_cache["version"]

    try:
        registry = _get_registry()
        version = registry.get_active()
        text = registry.get_prompt_text(version)
        if text:
            _prompt_cache.update(text=text, version=version, fetched_at=now)
            return text, version
        logger.warning("prompt_registry: version %s rỗng/không đọc được", version)
    except Exception as exc:
        logger.warning("prompt_registry không truy cập được: %s", exc)

    if cached_text:
        return cached_text, _prompt_cache["version"]
    return _SYSTEM_PROMPT_FALLBACK, "fallback"


# Keywords báo hiệu câu hỏi chính sách/FAQ — Router dùng để quyết định đi kb_search
_KB_KEYWORDS = (
    "hoàn trả",
    "đổi trả",
    "hoàn tiền",
    "bảo hành",
    "vận chuyển",
    "giao hàng",
    "thanh toán",
    "chính sách",
    "điều khoản",
    "khiếu nại",
    "hỗ trợ",
    "liên hệ",
    "đổi hàng",
    "trả hàng",
    "refund",
    "return",
    "warranty",
    "shipping",
    "phí ship",
    "thời gian giao",
    "làm sao",
    "như thế nào",
    "quy trình",
)


class AgentState(TypedDict):
    """Toàn bộ state của graph — thay cho việc build message list rời rạc trong run_graph_stream cũ."""

    messages: Annotated[list[BaseMessage], add_messages]
    customer_id: str
    pref_context: str
    route: str
    summary: str


_SUMMARIZE_AFTER_MESSAGES = int(os.getenv("AGENT_SUMMARIZE_AFTER", "12"))
_KEEP_RECENT_MESSAGES = int(os.getenv("AGENT_KEEP_RECENT", "6"))


def _is_kb_query(message: str) -> bool:
    msg = message.lower()
    return any(kw in msg for kw in _KB_KEYWORDS)


# Bug #10 (BENCHMARK_RESULTS.md mục 8): Qwen2.5-7B-AWQ đôi khi bỏ qua
# search_products dù system prompt yêu cầu "dùng tool trước khi trả lời",
# tự bịa tên+giá sản phẩm từ parametric knowledge. System prompt chỉ là soft
# instruction, model 7B không đủ tin cậy để tự giác 100%. Ép tường minh qua
# router (cùng cơ chế đã dùng cho kb_search) cho câu có tín hiệu TÌM KIẾM sản
# phẩm rõ ràng — không gồm "gợi ý" một mình vì dễ đụng get_recommendations
# (gợi ý cá nhân hóa dựa lịch sử mua hàng, vẫn để LLM tự quyết định qua ReAct).
_PRODUCT_SEARCH_KEYWORDS = (
    "tìm ",
    "tìm kiếm",
    "tìm giúp",
    "kiếm giúp",
    "cần mua",
    "muốn mua",
    "có bán",
    "mẫu nào",
    "loại nào",
)
_PRICE_CONSTRAINT_KEYWORDS = (
    "giá",
    "đồng",
    "vnd",
    "dưới",
    "trên",
    "khoảng",
    "tầm",
    "ngân sách",
)


def _is_product_search_query(message: str) -> bool:
    msg = message.lower()
    return any(kw in msg for kw in _PRODUCT_SEARCH_KEYWORDS) or any(
        kw in msg for kw in _PRICE_CONSTRAINT_KEYWORDS
    )


# Cùng lỗi như bug #10 nhưng cho get_recommendations: khách hỏi "gợi ý dựa
# trên lịch sử mua hàng" — LLM hỏi ngược lại khách đã mua gì thay vì tự gọi
# tool (customer_id đã có sẵn qua InjectedState, không cần hỏi). Phát hiện
# lúc test bug #10/#11 (17/7/2026).
_RECOMMEND_KEYWORDS = (
    "lịch sử mua hàng",
    "lịch sử mua",
    "dựa trên lịch sử",
    "cá nhân hóa",
    "phù hợp với tôi",
    "phù hợp cho tôi",
    "gợi ý cho tôi",
    "gợi ý giúp tôi",
)


def _is_recommend_query(message: str) -> bool:
    msg = message.lower()
    return any(kw in msg for kw in _RECOMMEND_KEYWORDS)


# Không có timeout thì model treo = request treo VÔ HẠN, giữ connection cho tới
# khi client bỏ cuộc. Mỗi backend gọi tên tham số một kiểu (đã kiểm bằng
# model_fields).
_LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
_LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))


def _get_chat_model():
    """Trả về (model, backend, model_id) — backend/model_id để tag metric
    llm_request_total/llm_latency_seconds trong LLMSpan (xem agent_node)."""
    backend = os.getenv("AGENT_LLM_BACKEND", "ollama")
    if backend == "claude":
        model_id = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
        return (
            ChatAnthropic(
                model=model_id,
                temperature=0.3,
                max_tokens=512,
                default_request_timeout=_LLM_TIMEOUT_SECONDS,
                max_retries=_LLM_MAX_RETRIES,
            ),
            backend,
            model_id,
        )
    if backend == "vllm":
        # vLLM expose API tương thích OpenAI — dùng ChatOpenAI trỏ base_url
        # sang server vLLM (nginx LB trước nhiều replica), api_key giả vì
        # vLLM mặc định không yêu cầu auth (self-hosted, không phải OpenAI
        # thật, chỉ cần field này không rỗng để client không lỗi).
        model_id = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
        return (
            ChatOpenAI(
                model=model_id,
                base_url=os.getenv("VLLM_URL", "http://localhost:8080/v1"),
                api_key=os.getenv("VLLM_API_KEY", "not-needed"),
                # Bug thật gặp phải (16/7/2026): temperature 0.3 khiến Qwen2.5-7B-AWQ
                # thỉnh thoảng sinh tool-call argument JSON có "extra data" thừa sau
                # JSON hợp lệ — vLLM's hermes tool-call parser strict, reject thẳng
                # (HTTP 400 "Extra data: line 1 column N") ngay khi parse lại lịch sử
                # hội thoại có tool_call đó. Hạ xuống 0.1 (giống mức Ollama đã ổn
                # định) để giảm tần suất model lệch format — KHÔNG loại bỏ hoàn toàn
                # (fix triệt để cần guided_json ép JSON hợp schema tận gốc, xem
                # NEXT_STEPS.md mục AI Eng 2.5, chưa làm — đây chỉ giảm nhẹ triệu chứng).
                temperature=0.1,
                # Bug #14 (BENCHMARK_RESULTS.md, 17/7/2026): search_kb context dài
                # (top_k=3 chunk ~1500 ký tự/chunk) đôi khi khiến model lặp lại
                # nguyên đoạn đầu hoặc lẫn token tiếng Trung giữa chừng. Thử
                # frequency_penalty=0.3 trước — KHÔNG cải thiện (1 lần test còn tệ
                # hơn: lặp cả đoạn bằng tiếng Trung) và có rủi ro ảnh hưởng ngược
                # tới độ ổn định JSON tool-call (lý do temperature đang giữ 0.1,
                # xem bug #8) — đã revert. Fix thật nằm ở giảm top_k trong
                # _kb_search_node (giảm tải context) thay vì chỉnh decoding param.
                #
                # Tối ưu throughput (17/7/2026, BENCHMARK_RESULTS.md mục 9): GPU đo
                # được 100% util liên tục dưới tải — compute-bound thật, không phải
                # nghẽn config lãng phí. `vllm:request_generation_tokens_bucket` cho
                # thấy 50% request chỉ cần ≤100 token, 85% ≤200 token — max_tokens=512
                # chỉ có tác dụng ở đúng phần đuôi 10-15% request dài, chiếm GPU lâu
                # hơn cần thiết. Hạ xuống 300 (đủ dư cho câu trả lời chính sách/sản
                # phẩm ngắn gọn đúng yêu cầu system prompt) để cắt đuôi lãng phí,
                # không ảnh hưởng phần lớn request vốn đã dừng sớm qua EOS token.
                max_tokens=300,
                request_timeout=_LLM_TIMEOUT_SECONDS,
                max_retries=_LLM_MAX_RETRIES,
            ),
            backend,
            model_id,
        )
    model_id = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    return (
        ChatOllama(
            model=model_id,
            base_url=os.getenv("OLLAMA_URL", "http://ollama:11434"),
            # Thấp hơn các backend khác (0.3) — Qwen2.5:3b cần bám sát format
            # tool-call gốc để Ollama parse đúng, temperature cao làm tăng
            # xác suất lệch format (xem _RAW_TOOL_CALL_RE ở trên).
            temperature=0.1,
            num_predict=512,
            client_kwargs={"timeout": _LLM_TIMEOUT_SECONDS},
        ),
        backend,
        model_id,
    )


def _make_tools(rag_pipeline=None):
    """4 tool sản phẩm bind cho LLM (search_kb tách riêng — Router ép gọi tường minh,
    không để LLM tự quyết định vì Qwen2.5:3b không đủ tin cậy)."""

    @tool
    def search_products(query: str, top_k: int = 5) -> str:
        """Tìm kiếm sản phẩm trong catalog theo từ khóa tiếng Việt."""
        return execute_tool(
            "search_products",
            {"query": query, "top_k": top_k},
            rag_pipeline=rag_pipeline,
        )

    @tool
    def get_recommendations(
        top_k: int = 5, state: Annotated[AgentState, InjectedState] = None
    ) -> str:
        """Lấy gợi ý sản phẩm cá nhân hóa cho khách hàng dựa trên lịch sử mua hàng."""
        # customer_id lấy từ graph state (InjectedState) — không để LLM tự đoán/transcribe ID,
        # vì Qwen2.5:3b không đủ tin cậy để dùng đúng ID nhìn thấy trong context.
        customer_id = state["customer_id"]
        return execute_tool(
            "get_recommendations", {"customer_id": customer_id, "top_k": top_k}
        )

    @tool
    def filter_by_price(query: str, max_price: float, min_price: float = 0.0) -> str:
        """Lọc sản phẩm theo khoảng giá (VND)."""
        return execute_tool(
            "filter_by_price",
            {"query": query, "max_price": max_price, "min_price": min_price},
            rag_pipeline=rag_pipeline,
        )

    @tool
    def get_product_detail(product_id: str) -> str:
        """Lấy thông tin chi tiết của một sản phẩm theo product_id."""
        return execute_tool(
            "get_product_detail", {"product_id": product_id}, rag_pipeline=rag_pipeline
        )

    return [search_products, get_recommendations, filter_by_price, get_product_detail]


def _make_kb_tool():
    """search_kb — không bind cho LLM, chỉ chạy qua kb_search node (xem _kb_search_node)."""

    @tool
    def search_kb(query: str, top_k: int = 3) -> str:
        """Tìm kiếm chính sách Tiki: đổi trả, bảo hành, vận chuyển, thanh toán, FAQ."""
        return execute_tool("search_kb", {"query": query, "top_k": top_k})

    return search_kb


def _router_node(state: AgentState) -> dict:
    """Router node thật — tự tính route và ghi vào state, thay vì passthrough giả."""
    last_user_msg = state["messages"][-1].content
    if _is_kb_query(last_user_msg):
        route = "kb_search"
    elif _is_product_search_query(last_user_msg):
        route = "product_search"
    elif _is_recommend_query(last_user_msg):
        route = "recommend"
    else:
        route = "agent"
    return {"route": route}


def _read_route(state: AgentState) -> str:
    """add_conditional_edges đọc route đã tính sẵn từ _router_node, không tính lại."""
    return state["route"]


def build_llm_fallback_message(message: str, rag_pipeline) -> str:
    """
    LLM chết => VẪN trả được thứ hữu ích: chạy thẳng RAG search (retrieval không
    cần LLM) và đưa danh sách sản phẩm. Chỉ khi cả retrieval cũng hỏng mới xin lỗi.
    """
    agent_fallback_total.labels(tier="llm_unavailable").inc()

    if rag_pipeline is not None:
        try:
            results = rag_pipeline.search(message, top_k=5)
            if results:
                lines = [
                    f"- {r['product_name']} ({int(r['price']):,} VND)" for r in results
                ]
                return (
                    "Trợ lý AI tạm thời không khả dụng. Đây là kết quả tìm kiếm "
                    "trực tiếp:\n" + "\n".join(lines)
                )
        except Exception as exc:
            logger.warning("fallback_rag_search_failed: %s", exc)

    return "Xin lỗi, hệ thống đang bận. Vui lòng thử lại sau ít phút."


def _needs_summary(state: AgentState) -> str:
    """Entry point: thread đã dài thì nén trước, chưa thì vào thẳng router."""
    if len(state["messages"]) > _SUMMARIZE_AFTER_MESSAGES:
        return "summarize"
    return "router"


def _kb_search_node(state: AgentState) -> dict:
    """Synthesize 1 tool_call cho search_kb rồi để ToolNode chung thực thi —
    tool_calls nhờ vậy chỉ có 1 nguồn (on_tool_start/on_tool_end), không track state riêng.

    top_k=2 (giảm từ 3, bug #14): mỗi chunk ~1500 ký tự — 3 chunk cộng system
    prompt/summary/lịch sử đẩy context đủ dài để Qwen2.5-7B-AWQ đôi khi lặp
    câu/lẫn ngôn ngữ (degenerate generation). 2 chunk vẫn đủ trả lời chính sách
    ngắn gọn, giảm tải context.
    """
    query = state["messages"][-1].content
    synthetic_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_kb",
                "args": {"query": query, "top_k": 2},
                "id": str(uuid.uuid4()),
            }
        ],
    )
    return {"messages": [synthetic_call]}


def _product_search_node(state: AgentState) -> dict:
    """Synthesize 1 tool_call cho search_products — cùng cơ chế _kb_search_node,
    ép gọi tool tường minh thay vì để model 7B tự giác qua system prompt (bug #10).
    """
    query = state["messages"][-1].content
    synthetic_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_products",
                "args": {"query": query, "top_k": 5},
                "id": str(uuid.uuid4()),
            }
        ],
    )
    return {"messages": [synthetic_call]}


def _recommend_node(state: AgentState) -> dict:
    """Synthesize 1 tool_call cho get_recommendations — cùng cơ chế
    _product_search_node/_kb_search_node (bug #10-style fix cho get_recommendations).
    customer_id do ToolNode tự inject qua InjectedState lúc thực thi, không cần
    đưa vào args ở đây.
    """
    synthetic_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_recommendations",
                "args": {"top_k": 5},
                "id": str(uuid.uuid4()),
            }
        ],
    )
    return {"messages": [synthetic_call]}


def build_graph(rag_pipeline=None, checkpointer=None):
    """Build compiled StateGraph — Router node + add_conditional_edges, không dùng create_react_agent.

    `checkpointer` (LangGraph RedisSaver, tạo ở main.py lúc startup): persistence
    NATIVE của LangGraph thay cho việc tự đọc Redis rồi replay history[-6:] vào
    message list mỗi request như trước. Truyền None => graph vẫn chạy (stateless,
    tiện cho test/eval), chỉ mất khả năng nhớ giữa các lượt.
    """
    model, backend, model_id = _get_chat_model()
    tools = _make_tools(rag_pipeline)
    kb_tool = _make_kb_tool()
    model_with_tools = model.bind_tools(
        tools
    )  # LLM chỉ thấy 4 tool sản phẩm, không thấy search_kb

    async def agent_node(state: AgentState, config: RunnableConfig) -> dict:
        # Phải nhận và truyền `config` xuống model.astream() — nếu không, callback/tracing
        # context (bao gồm astream_events() của graph cha) không propagate xuống lời gọi
        # model bên trong, nên on_chat_model_stream không bao giờ được emit ra ngoài.
        system_prompt, prompt_version = get_active_prompt()
        messages = [SystemMessage(content=system_prompt)]
        if state.get("pref_context"):
            messages.append(
                SystemMessage(content=f"Sở thích khách hàng:\n{state['pref_context']}")
            )
        # Không có dòng này thì summarize_node nén xong cũng vô nghĩa: nội dung
        # các lượt cũ đã bị xoá khỏi messages, mà bản tóm tắt lại không được đưa
        # vào prompt => agent vẫn quên y như cũ.
        if state.get("summary"):
            messages.append(
                SystemMessage(
                    content=(
                        "Bối cảnh nền (đã kết thúc, KHÔNG phải câu hỏi hiện tại — "
                        "chỉ dùng để tham khảo sở thích/ngân sách nếu khách nhắc lại, "
                        "TUYỆT ĐỐI không tự lặp lại hay tiếp tục nội dung này trong "
                        f"câu trả lời trừ khi câu hỏi mới liên quan trực tiếp):\n{state['summary']}"
                    )
                )
            )
        messages.extend(state["messages"])
        # CÓ bind tools (dù chỉ để
        # *có thể* gọi tool tiếp, không bắt buộc) → model bịa "chưa tìm
        # thấy". Vì "tools"→"agent" luôn quay lại agent_node dùng
        # model_with_tools bất kể đang ở bước quyết định gọi tool hay bước
        # tổng hợp câu trả lời cuối từ ToolMessage đã có — khi message cuối
        # ĐÃ là ToolMessage (vừa nhận kết quả), đây chắc chắn là bước tổng
        # hợp, không cần khả năng gọi thêm tool → bỏ bind_tools cho đúng bước
        # này, giảm nhiễu cho model nhỏ.
        last_msg = state["messages"][-1] if state["messages"] else None
        active_model = model if isinstance(last_msg, ToolMessage) else model_with_tools

        full_msg = None
        llm_span = LLMSpan(
            model=model_id, backend=backend, prompt_version=prompt_version
        )
        try:
            with llm_span:
                async for chunk in active_model.astream(messages, config=config):
                    full_msg = chunk if full_msg is None else full_msg + chunk
        except Exception as exc:
            logger.error(
                "llm_call_failed",
                extra={
                    "backend": backend,
                    "model": model_id,
                    "customer_id": state.get("customer_id", ""),
                    "error": str(exc),
                },
            )
            last_user_msg = next(
                (
                    m.content
                    for m in reversed(state["messages"])
                    if isinstance(m, HumanMessage)
                ),
                "",
            )
            return {
                "messages": [
                    AIMessage(
                        content=build_llm_fallback_message(last_user_msg, rag_pipeline)
                    )
                ]
            }

        if full_msg is None:
            logger.error("llm_returned_nothing", extra={"backend": backend})
            last_user_msg = next(
                (
                    m.content
                    for m in reversed(state["messages"])
                    if isinstance(m, HumanMessage)
                ),
                "",
            )
            return {
                "messages": [
                    AIMessage(
                        content=build_llm_fallback_message(last_user_msg, rag_pipeline)
                    )
                ]
            }

        try:
            usage = getattr(full_msg, "usage_metadata", None) or {}
            prompt_tokens = usage.get("input_tokens", 0)
            completion_tokens = usage.get("output_tokens", 0)
            if prompt_tokens or completion_tokens:
                llm_span.record_tokens(prompt_tokens, completion_tokens)
                try:
                    litellm_model_id = (
                        model_id if "/" in model_id else f"{backend}/{model_id}"
                    )
                    prompt_cost, completion_cost = litellm.cost_per_token(
                        model=litellm_model_id,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                    total_cost = prompt_cost + completion_cost
                    llm_cost_usd_total.labels(model=model_id, backend=backend).inc(
                        total_cost
                    )
                    # Cộng dồn vào budget của user + toàn hệ thống. Dùng chung
                    # đúng con số vừa tính ở trên, không tự tính lại cho lệch.
                    record_cost(state.get("customer_id", "unknown"), total_cost)
                except Exception as cost_exc:
                    logger.debug(
                        "agent_node: litellm.cost_per_token unavailable for model=%s: %s",
                        model_id,
                        cost_exc,
                    )
        except Exception as exc:
            logger.debug("agent_node: could not record token usage: %s", exc)

        # Native tool-calling thất bại (tool_calls rỗng) nhưng model rõ ràng có ý
        # định gọi tool — tự parse lại thay vì để lộ text thô hoặc coi như câu
        # trả lời thật.
        if not full_msg.tool_calls and isinstance(full_msg.content, str):
            match = _RAW_TOOL_CALL_RE.search(full_msg.content)
            if match:
                try:
                    parsed = json.loads(match.group(1))
                    full_msg = AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": parsed["name"],
                                "args": parsed.get("arguments", {}),
                                "id": str(uuid.uuid4()),
                            }
                        ],
                    )
                    logger.info(
                        "agent_node: recovered malformed tool_call for '%s'",
                        parsed.get("name"),
                    )
                except (json.JSONDecodeError, KeyError, TypeError):
                    logger.warning(
                        "agent_node: malformed tool_call text unparsable, using fallback message"
                    )
                    full_msg = AIMessage(content=_FALLBACK_MESSAGE)

        return {"messages": [full_msg]}

    async def summarize_node(state: AgentState, config: RunnableConfig) -> dict:
        """
        Nén các lượt cũ thành 1 đoạn tóm tắt, giữ nguyên văn _KEEP_RECENT_MESSAGES
        lượt gần nhất. Chạy TRƯỚC router, chỉ khi thread đã dài (xem _needs_summary).

        Dùng RemoveMessage để thực sự XOÁ message cũ khỏi thread state — nếu chỉ
        thêm summary mà không xoá thì state vẫn phình vô hạn và vẫn tràn context.
        """
        messages = state["messages"]
        old_messages = messages[:-_KEEP_RECENT_MESSAGES]
        if not old_messages:
            return {}

        prev_summary = state.get("summary", "")
        transcript = "\n".join(
            f"{'Khách' if isinstance(m, HumanMessage) else 'Trợ lý'}: {m.content}"
            for m in old_messages
            if isinstance(m, (HumanMessage, AIMessage)) and m.content
        )

        instruction = (
            "Tóm tắt hội thoại mua sắm dưới đây trong 3-4 câu tiếng Việt. "
            "BẮT BUỘC giữ lại: ngân sách khách nêu, sản phẩm/danh mục khách quan tâm, "
            "ràng buộc khách đưa ra (thương hiệu, tính năng), và sản phẩm đã được gợi ý.\n\n"
        )
        if prev_summary:
            instruction += f"Tóm tắt trước đó:\n{prev_summary}\n\n"
        instruction += f"Hội thoại mới:\n{transcript}"

        try:
            # model gốc (chưa bind_tools) — summarize không cần gọi tool, bind vào
            # chỉ tổ khiến model nhỏ tưởng phải gọi tool thay vì tóm tắt.
            result = await model.ainvoke(
                [SystemMessage(content=instruction)], config=config
            )
            new_summary = (result.content or "").strip()
        except Exception as exc:
            logger.warning("summarize_node thất bại, giữ nguyên state: %s", exc)
            return {}

        if not new_summary:
            return {}

        conversation_summarized_total.inc()
        logger.info(
            "conversation_summarized",
            extra={
                "customer_id": state.get("customer_id", ""),
                "messages_compacted": len(old_messages),
            },
        )
        return {
            "summary": new_summary,
            "messages": [RemoveMessage(id=m.id) for m in old_messages if m.id],
        }

    workflow = StateGraph(AgentState)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("router", _router_node)
    workflow.add_node("kb_search", _kb_search_node)
    workflow.add_node("product_search", _product_search_node)
    workflow.add_node("recommend", _recommend_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node(
        "tools", ToolNode(tools + [kb_tool])
    )  # 1 ToolNode chung cho cả 5 tool

    workflow.set_conditional_entry_point(
        _needs_summary, {"summarize": "summarize", "router": "router"}
    )
    workflow.add_edge("summarize", "router")
    workflow.add_conditional_edges(
        "router",
        _read_route,
        {
            "kb_search": "kb_search",
            "product_search": "product_search",
            "recommend": "recommend",
            "agent": "agent",
        },
    )
    workflow.add_edge(
        "kb_search", "tools"
    )  # kb_search chỉ tạo tool_call, ToolNode mới thực thi
    workflow.add_edge(
        "product_search", "tools"
    )  # product_search chỉ tạo tool_call, ToolNode mới thực thi
    workflow.add_edge(
        "recommend", "tools"
    )  # recommend chỉ tạo tool_call, ToolNode mới thực thi
    workflow.add_conditional_edges(
        "agent", tools_condition, {"tools": "tools", END: END}
    )
    workflow.add_edge(
        "tools", "agent"
    )  # cả search_kb lẫn tool sản phẩm đều quay lại agent

    return workflow.compile(checkpointer=checkpointer)


def _strip_cjk(text: str) -> tuple[str, bool]:
    """Cắt text tại điểm bắt đầu xuất hiện CJK — trả về (phần_sạch, có_cắt).

    Dùng trong vòng lặp stream (bug #14) để chặn tiếng Trung leak ra SSE
    real-time, không chỉ dọn sau khi full_response đã hoàn chỉnh.
    """
    m = _CJK_RE.search(text)
    if not m:
        return text, False
    return text[: m.start()], True


async def run_graph_stream(
    customer_id: str,
    message: str,
    pref_context: str,
    rag_pipeline=None,
    graph=None,
    history: list[dict] | None = None,
) -> AsyncIterator[dict]:
    """Stream agent response token-by-token via LangGraph astream_events.

    Lịch sử hội thoại KHÔNG còn được replay tay vào message list nữa. Với
    checkpointer, LangGraph tự nạp lại state của thread (thread_id = customer_id)
    nên chỉ cần đẩy vào đúng message MỚI; các lượt cũ + bản tóm tắt đã nằm sẵn
    trong checkpoint. Bản cũ cắt cứng history[-6:] nên quá 6 lượt là quên hẳn.

    `history` chỉ còn dùng làm fallback khi graph chạy KHÔNG có checkpointer
    (test/eval): lúc đó không có state nào để nạp, phải tự dựng lại message list.
    """
    if graph is None:
        graph = build_graph(rag_pipeline)

    has_checkpointer = getattr(graph, "checkpointer", None) is not None

    lc_messages: list = []
    if not has_checkpointer and history:
        for turn in history[-6:]:
            role = turn.get("role")
            content = turn.get("content", "")
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
    lc_messages.append(HumanMessage(content=message))

    initial_state: dict = {
        "messages": lc_messages,
        "customer_id": customer_id,
        "pref_context": pref_context,
        "route": "",
    }
    if not has_checkpointer:
        initial_state["summary"] = ""  # stateless: không có gì để giữ

    # thread_id = customer_id => mỗi khách 1 thread hội thoại riêng, persist
    # trong Redis qua checkpointer.
    config: RunnableConfig = {"configurable": {"thread_id": customer_id}}

    full_response = ""
    tool_calls: list[dict] = []

    _LOOKAHEAD_LEN = 15
    stream_state = "detecting"
    lookahead_buffer = ""

    async for event in graph.astream_events(initial_state, config, version="v2"):
        kind = event["event"]

        if kind == "on_chat_model_start":
            stream_state = "detecting"
            lookahead_buffer = ""

        elif (
            kind == "on_chat_model_end"
            and stream_state == "detecting"
            and lookahead_buffer
        ):
            clean, hit_cjk = _strip_cjk(lookahead_buffer)
            if clean:
                full_response += clean
                yield {"type": "token", "content": clean}
            if hit_cjk:
                logger.warning(
                    "stream_cjk_leak_truncated", extra={"customer_id": customer_id}
                )
            lookahead_buffer = ""

        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            # content is str (Ollama) or list of blocks (Claude)
            if isinstance(content, str) and content:
                if stream_state == "suppressed":
                    continue
                if stream_state == "detecting":
                    lookahead_buffer += content
                    if "<tool_call>" in lookahead_buffer:
                        stream_state = "suppressed"
                        lookahead_buffer = ""
                        continue
                    if len(lookahead_buffer) < _LOOKAHEAD_LEN:
                        continue
                    stream_state = "streaming"
                    clean, hit_cjk = _strip_cjk(lookahead_buffer)
                    if clean:
                        full_response += clean
                        yield {"type": "token", "content": clean}
                    lookahead_buffer = ""
                    if hit_cjk:
                        stream_state = "suppressed"
                        logger.warning(
                            "stream_cjk_leak_truncated",
                            extra={"customer_id": customer_id},
                        )
                else:
                    clean, hit_cjk = _strip_cjk(content)
                    if clean:
                        full_response += clean
                        yield {"type": "token", "content": clean}
                    if hit_cjk:
                        stream_state = "suppressed"
                        logger.warning(
                            "stream_cjk_leak_truncated",
                            extra={"customer_id": customer_id},
                        )
            elif isinstance(content, list):
                if stream_state == "suppressed":
                    continue
                for block in content:
                    if stream_state == "suppressed":
                        break
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            clean, hit_cjk = _strip_cjk(text)
                            if clean:
                                full_response += clean
                                yield {"type": "token", "content": clean}
                            if hit_cjk:
                                stream_state = "suppressed"
                                logger.warning(
                                    "stream_cjk_leak_truncated",
                                    extra={"customer_id": customer_id},
                                )

        elif kind == "on_tool_start":
            tool_name = event.get("name", "")
            tool_input = event["data"].get("input", {})
            tool_calls.append({"tool": tool_name, "input": tool_input, "output": ""})
            tool_call_total.labels(tool_name=tool_name).inc()
            yield {"type": "tool_start", "tools": [tool_name]}

        elif kind == "on_tool_end":
            tool_name = event.get("name", "")
            output = str(event["data"].get("output", ""))
            for tc in reversed(tool_calls):
                if tc["tool"] == tool_name and not tc["output"]:
                    tc["output"] = output[:500]
                    break

    yield {
        "type": "done",
        "full_response": full_response,
        "tool_calls": tool_calls,
    }
