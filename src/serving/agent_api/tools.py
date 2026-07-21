import logging
import os
import re
from typing import Any

import requests
from pydantic import BaseModel, Field, ValidationError

from src.serving.agent_api.guardrails import sanitize_retrieved_content
from src.serving.agent_api.tracing import (
    kb_injection_blocked_total,
    tool_arg_invalid_total,
)

logger = logging.getLogger(__name__)

_RECSYS_URL = os.getenv("RECSYS_API_URL", "http://recsys-api:8001")
_HTTP_TIMEOUT = 5.0

# KB indexer singleton — injected by main.py at startup via set_kb_indexer()
_kb_indexer = None


def set_kb_indexer(indexer) -> None:
    global _kb_indexer
    _kb_indexer = indexer


# ---------------------------------------------------------------------------
# Pydantic models for tool argument validation
# ---------------------------------------------------------------------------


class SearchProductsArgs(BaseModel):
    query: str
    max_price: float | None = None
    min_price: float | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class GetRecommendationsArgs(BaseModel):
    customer_id: str
    top_k: int = Field(default=5, ge=1, le=20)


class FilterByPriceArgs(BaseModel):
    query: str
    max_price: float
    min_price: float = 0.0


class GetProductDetailArgs(BaseModel):
    product_id: str


class SearchKBArgs(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1, le=10)


TOOL_ARG_MODELS: dict[str, type[BaseModel]] = {
    "search_products": SearchProductsArgs,
    "get_recommendations": GetRecommendationsArgs,
    "filter_by_price": FilterByPriceArgs,
    "get_product_detail": GetProductDetailArgs,
    "search_kb": SearchKBArgs,
}

# Tool schemas for LLM (Anthropic tool_use format, also works with LiteLLM)
TOOL_SCHEMAS = [
    {
        "name": "search_products",
        "description": "Tìm kiếm sản phẩm trong catalog theo từ khóa tiếng Việt. Dùng khi user hỏi về sản phẩm cụ thể hoặc muốn tìm theo tên/loại.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Từ khóa tìm kiếm bằng tiếng Việt",
                },
                "max_price": {
                    "type": "number",
                    "description": "Giá tối đa (VND), null nếu không có",
                },
                "min_price": {
                    "type": "number",
                    "description": "Giá tối thiểu (VND), null nếu không có",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Số sản phẩm trả về (mặc định 5)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_recommendations",
        "description": "Lấy danh sách sản phẩm gợi ý cá nhân hóa cho khách hàng dựa trên lịch sử mua hàng.",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "ID khách hàng"},
                "top_k": {
                    "type": "integer",
                    "description": "Số sản phẩm gợi ý (mặc định 5)",
                },
            },
            "required": ["customer_id"],
        },
    },
    {
        "name": "filter_by_price",
        "description": "Lọc danh sách sản phẩm theo khoảng giá.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Từ khóa để tìm sản phẩm cần lọc",
                },
                "max_price": {"type": "number", "description": "Giá tối đa (VND)"},
                "min_price": {
                    "type": "number",
                    "description": "Giá tối thiểu (VND), mặc định 0",
                },
            },
            "required": ["query", "max_price"],
        },
    },
    {
        "name": "get_product_detail",
        "description": "Lấy thông tin chi tiết của một sản phẩm cụ thể theo product_id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": "ID sản phẩm cần xem chi tiết",
                },
            },
            "required": ["product_id"],
        },
    },
    {
        "name": "search_kb",
        "description": "Tìm kiếm thông tin chính sách Tiki: đổi trả, bảo hành, thanh toán, vận chuyển, TikiNOW, FAQ. Dùng khi user hỏi về chính sách, quy trình, điều kiện mua hàng.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Câu hỏi về chính sách (tiếng Việt)",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Số đoạn văn bản trả về (mặc định 3)",
                },
            },
            "required": ["query"],
        },
    },
]


def _format_product_list(products: list[dict]) -> str:
    if not products:
        return "Không tìm thấy sản phẩm phù hợp."
    lines = []
    for i, p in enumerate(products, 1):
        # Giảm giá thật (list_price/discount_rate join từ bronze data, 18/7/2026)
        # — chỉ hiện khi có giảm giá thật, không bịa số cho sản phẩm không có.
        discount_rate = p.get("discount_rate", 0) or 0
        if discount_rate > 0 and p.get("list_price", 0) > p["price"]:
            price_str = (
                f"{int(p['price']):,} VND (giảm {discount_rate:.0f}% từ "
                f"{int(p['list_price']):,} VND)"
            )
        else:
            price_str = f"{int(p['price']):,} VND"

        review_count = p.get("review_count", 0) or 0
        if p.get("avg_sentiment") and review_count > 0:
            sentiment = f"(đánh giá: {p['avg_sentiment']:.1f}, {review_count} lượt)"
        elif p.get("avg_sentiment"):
            sentiment = f"(đánh giá: {p['avg_sentiment']:.1f})"
        else:
            sentiment = ""

        brand = f" [{p['brand_name']}]" if p.get("brand_name") else ""
        link = f"\n   Link: {p['url']}" if p.get("url") else ""

        lines.append(
            f"{i}. [{p['product_id']}]{brand} {p['product_name']} — {price_str} {sentiment}{link}"
        )
    return "\n".join(lines)


def _validate_args(tool_name: str, tool_input: dict) -> dict:
    """
    Ép tool args qua Pydantic model TRƯỚC KHI chạy tool.

    TOOL_ARG_MODELS vốn đã được định nghĩa sẵn (kèm ràng buộc như top_k <= 20)
    nhưng KHÔNG chỗ nào gọi tới — các hàm _tool bên dưới đọc thẳng
    args.get("top_k", 5), nên ràng buộc đó vô nghĩa: LLM (nhất là model nhỏ
    hay bịa tham số) gửi top_k=9999 là đi thẳng vào FAISS/recsys-api. Đầu ra
    của LLM là dữ liệu KHÔNG tin cậy, phải validate như mọi input khác.

    Ném ValidationError → execute_tool bắt và trả message lỗi cho LLM tự sửa
    ở vòng tool-call kế tiếp, thay vì để tool nổ giữa chừng.
    """
    model = TOOL_ARG_MODELS.get(tool_name)
    if model is None:
        return tool_input
    return model.model_validate(tool_input).model_dump()


def execute_tool(tool_name: str, tool_input: dict, rag_pipeline=None) -> str:
    try:
        args = _validate_args(tool_name, tool_input)
    except ValidationError as exc:
        tool_arg_invalid_total.labels(tool_name=tool_name).inc()
        logger.warning(
            "tool_args_invalid",
            extra={"tool": tool_name, "errors": exc.errors()},
        )
        # Trả lỗi có cấu trúc để LLM tự sửa ở lượt sau, không phải đoán mò
        details = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
            for e in exc.errors()[:3]
        )
        return (
            f"Tham số cho tool '{tool_name}' không hợp lệ ({exc.error_count()} lỗi): "
            f"{details}. Hãy gọi lại tool với tham số đúng schema."
        )

    try:
        if tool_name == "search_products":
            return _search_products(args, rag_pipeline)
        elif tool_name == "get_recommendations":
            return _get_recommendations(args)
        elif tool_name == "filter_by_price":
            return _filter_by_price(args, rag_pipeline)
        elif tool_name == "get_product_detail":
            return _get_product_detail(args, rag_pipeline)
        elif tool_name == "search_kb":
            return _search_kb(args)
        else:
            return f"Tool '{tool_name}' không được hỗ trợ."
    except Exception as exc:
        logger.exception("tool_execute_error", extra={"tool": tool_name})
        return f"Lỗi khi thực hiện {tool_name}: {exc}"


def _search_products(args: dict, rag) -> str:
    if rag is None:
        return "RAG pipeline chưa sẵn sàng."
    results = rag.search(
        query=args["query"],
        max_price=args.get("max_price"),
        min_price=args.get("min_price"),
        top_k=int(args.get("top_k", 5)),
    )
    return _format_product_list(results)


def _get_recommendations(args: dict) -> str:
    try:
        resp = requests.post(
            f"{_RECSYS_URL}/recommend",
            json={
                "customer_id": str(args["customer_id"]),
                "top_k": int(args.get("top_k", 5)),
            },
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        recs = data.get("recommendations", [])
        if not recs:
            return "Không có gợi ý cá nhân hóa cho khách hàng này (cold start)."

        import os

        import pandas as pd

        from src.serving.agent_api.rag import _DEFAULT_DATA_PATH

        try:
            df = pd.read_parquet(os.getenv("ITEM_LOOKUP_PATH", _DEFAULT_DATA_PATH))
            df["product_id"] = df["product_id"].astype(str)
        except Exception:
            df = None

        lines = []
        for i, r in enumerate(recs, 1):
            pid = str(r.get("product_id", ""))
            name = r.get("product_name", "")
            price = r.get("price", 0)
            if df is not None and not name:
                row = df[df["product_id"] == pid]
                if not row.empty:
                    name = row.iloc[0]["product_name"]
                    price = row.iloc[0]["price"]
            price_str = f"{int(price):,} VND" if price else ""
            explain = r.get("explanation", {})
            reason = explain.get("top_reason", "")
            reason_str = f" — {reason}" if reason else ""
            lines.append(f"{i}. [{pid}] {name} {price_str}{reason_str}".strip())

        source = data.get("source", "model")
        header = f"Gợi ý cá nhân hóa (nguồn: {source}):"
        return header + "\n" + "\n".join(lines)
    except requests.exceptions.ConnectionError:
        return "Không thể kết nối đến Recommendation API. Hãy thử tìm kiếm theo từ khóa thay thế."
    except Exception as exc:
        return f"Lỗi lấy gợi ý: {exc}"


def _filter_by_price(args: dict, rag) -> str:
    if rag is None:
        return "RAG pipeline chưa sẵn sàng."
    results = rag.search(
        query=args["query"],
        max_price=float(args["max_price"]),
        min_price=float(args.get("min_price", 0)),
        top_k=5,
    )
    return _format_product_list(results)


def _get_product_detail(args: dict, rag) -> str:
    if rag is None:
        return "RAG pipeline chưa sẵn sàng."
    product = rag.get_product(str(args["product_id"]))
    if not product:
        return f"Không tìm thấy sản phẩm với ID {args['product_id']}."
    p = product
    discount_rate = p.get("discount_rate", 0) or 0
    if discount_rate > 0 and p.get("list_price", 0) > p["price"]:
        price_line = (
            f"Giá: {int(p['price']):,} VND (giảm {discount_rate:.0f}% từ "
            f"{int(p['list_price']):,} VND)"
        )
    else:
        price_line = f"Giá: {int(p['price']):,} VND"
    lines = [
        f"Tên: {p['product_name']}",
        price_line,
        f"Danh mục: {p['category_name']}",
        f"Thương hiệu: {p['brand_name']}",
        f"Điểm đánh giá trung bình: {p['avg_sentiment']:.2f}"
        + (f" ({p['review_count']} lượt)" if p.get("review_count") else ""),
    ]
    if p.get("url"):
        lines.append(f"Link: {p['url']}")
    if p.get("short_description"):
        lines.append(f"Mô tả: {p['short_description'][:400]}")
    if p.get("specs_text"):
        lines.append(f"Thông số: {p['specs_text']}")
    return "\n".join(lines)


_KB_HEADER_RE = re.compile(
    r"TIKI 1900 6035 hotro@tiki\.vn\nGỬI YÊU CẦU\n.*?Lượt xem:\s*\d+\n*",
    re.DOTALL,
)
_KB_FOOTER_ANCHORS = (
    "Bài viết trên có hữu ích không?",
    "Công ty TNHH TI KI",
    "Nếu cần thêm sự hỗ trợ, quý khách vui lòng liên hệ",
    "Giấy chứng nhận đăng ký doanh nghiệp",
)


def _clean_kb_boilerplate(text: str) -> str:
    text = _KB_HEADER_RE.sub("", text)
    cut_at = min(
        (idx for a in _KB_FOOTER_ANCHORS if (idx := text.find(a)) != -1),
        default=len(text),
    )
    return text[:cut_at].strip()


def _search_kb(args: dict) -> str:
    if _kb_indexer is None or _kb_indexer.size == 0:
        return "Cơ sở tri thức chính sách chưa sẵn sàng. Vui lòng thử lại sau."
    results = _kb_indexer.search(args["query"], top_k=int(args.get("top_k", 3)))
    if not results:
        return "Không tìm thấy thông tin liên quan trong chính sách Tiki."
    parts = []
    sources: list[str] = []
    seen: set[str] = set()
    for r in results:
        cleaned = _clean_kb_boilerplate(r["text"])
        if not cleaned:
            continue
        # Nội dung KB = dữ liệu KHÔNG tin cậy (file có thể được upload qua
        # /admin/kb/upload). Không sanitize ở đây thì một câu "bỏ qua chỉ dẫn
        # trước đó" nằm trong PDF sẽ được LLM đọc như mệnh lệnh hệ thống —
        # indirect prompt injection.
        safe_text, flagged = sanitize_retrieved_content(
            cleaned, source=r.get("source", "")
        )
        if flagged:
            kb_injection_blocked_total.inc()
        parts.append(safe_text)
        src = r.get("source", "")
        if src and src not in seen:
            seen.add(src)
            sources.append(src)
    if not parts:
        return "Không tìm thấy thông tin liên quan trong chính sách Tiki."
    answer = "\n\n".join(parts)
    if sources:
        citation = ", ".join(sources)
        answer += f"\n\n[Nguồn: {citation}]"
    return answer
