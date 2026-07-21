import logging
import os
import time

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Prometheus metrics
# ------------------------------------------------------------------

llm_request_total = Counter(
    "llm_request_total",
    "Total LLM requests",
    ["model", "backend", "status"],
)

llm_latency_seconds = Histogram(
    "llm_latency_seconds",
    "LLM call latency",
    ["model", "backend"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0, 90.0, 120.0],
)

llm_tokens_total = Counter(
    "llm_tokens_total",
    "Total tokens used",
    ["model", "token_type"],
)

tool_call_total = Counter(
    "tool_call_total",
    "Total tool calls by agent",
    ["tool_name"],
)

guardrail_block_total = Counter(
    "guardrail_block_total",
    "Inputs blocked by guardrails",
    ["reason"],
)

agent_fallback_total = Counter(
    "agent_fallback_total",
    "Fallback activations by tier",
    ["tier"],
)

# Indirect prompt injection phát hiện trong nội dung retrieve từ KB (Qdrant) —
# tách riêng khỏi guardrail_block_total (vốn đếm input độc từ user), vì đây là
# tín hiệu KB đã bị nhiễm, cần đi điều tra file nguồn chứ không chỉ chặn request.
kb_injection_blocked_total = Counter(
    "kb_injection_blocked_total",
    "Injection payloads neutralized in retrieved KB content (indirect injection)",
)

# LLM gọi tool với tham số sai schema (model nhỏ hay bịa/vượt ràng buộc).
# Theo dõi được tỉ lệ này = đo được độ tin cậy tool-calling của từng model/backend
# — chính là con số để chứng minh guided decoding (vLLM) tốt hơn prompt thuần.
tool_arg_invalid_total = Counter(
    "tool_arg_invalid_total",
    "Tool calls rejected because args failed Pydantic schema validation",
    ["tool_name"],
)

# Số lần hội thoại bị nén (summarization). Theo dõi để biết ngưỡng
# AGENT_SUMMARIZE_AFTER có hợp lý không: nén quá thường xuyên = tốn thêm 1 lời
# gọi LLM mỗi vài lượt; không bao giờ nén = ngưỡng đặt quá cao, thread vẫn phình.
conversation_summarized_total = Counter(
    "conversation_summarized_total",
    "Conversations compacted into a running summary",
)

llm_cost_usd_total = Counter(
    "llm_cost_usd_total",
    "Estimated LLM cost in USD",
    ["model", "backend"],
)

search_cache_hit_total = Counter(
    "search_cache_hit_total",
    "Semantic cache hits for /search",
    ["type"],  # semantic | miss
)

chat_cache_hit_total = Counter(
    "chat_cache_hit_total",
    "Semantic cache hits for /chat/stream (chỉ áp dụng câu hỏi policy/KB — xem BENCHMARK_RESULTS.md mục 21)",
    ["type"],  # hit | miss | not_cacheable
)

# ------------------------------------------------------------------
# OTEL tracer setup
# ------------------------------------------------------------------


def setup_tracing() -> trace.Tracer:
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4317")
    resource = Resource.create({"service.name": "agent-api"})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    logger.info("otel_tracing_configured", extra={"endpoint": endpoint})
    return trace.get_tracer("agent-api")


_tracer: trace.Tracer | None = None


def get_tracer() -> trace.Tracer:
    global _tracer
    if _tracer is None:
        _tracer = setup_tracing()
    return _tracer


# ------------------------------------------------------------------
# Context manager for LLM call instrumentation
# ------------------------------------------------------------------


class LLMSpan:
    def __init__(self, model: str, backend: str, prompt_version: str = "unknown"):
        self.model = model
        self.backend = backend
        self.prompt_version = prompt_version
        self._t0 = 0.0
        self._span = None

    def __enter__(self):
        self._t0 = time.time()
        tracer = get_tracer()
        self._span = tracer.start_span("llm.call")
        self._span.set_attribute("llm.model", self.model)
        self._span.set_attribute("llm.backend", self.backend)
        self._span.set_attribute("llm.prompt_version", self.prompt_version)
        return self

    def record_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        llm_tokens_total.labels(model=self.model, token_type="prompt").inc(
            prompt_tokens
        )
        llm_tokens_total.labels(model=self.model, token_type="completion").inc(
            completion_tokens
        )
        if self._span:
            self._span.set_attribute("llm.prompt_tokens", prompt_tokens)
            self._span.set_attribute("llm.completion_tokens", completion_tokens)

    def record_tool_calls(self, count: int) -> None:
        if self._span:
            self._span.set_attribute("llm.tool_calls", count)

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency = time.time() - self._t0
        status = "error" if exc_type else "ok"
        llm_request_total.labels(
            model=self.model, backend=self.backend, status=status
        ).inc()
        llm_latency_seconds.labels(model=self.model, backend=self.backend).observe(
            latency
        )
        if self._span:
            if exc_type:
                self._span.record_exception(exc_val)
            self._span.end()
