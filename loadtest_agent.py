"""
Load test for the Shopping Agent API (src/serving/agent_api), port 8003.

Run: locust -f loadtest_agent.py --host http://localhost:8003

Notes:
- /chat/stream is Server-Sent Events. We consume the full stream and time it
  manually via events.request.fire() so Locust reports true end-to-end
  latency (prefill + decode + tool calls), not just time-to-headers.
- Keep concurrency modest for this one — Ollama here is CPU-only (no GPU
  batching), so it saturates fast compared to /recommend on recsys_api.
"""

import json
import random
import time
import uuid

from locust import HttpUser, between, events, task

# Bug thật phát hiện 17/7/2026 (BENCHMARK_RESULTS.md mục 9): dùng cố định
# 50 customer_id qua NHIỀU lần chạy test trong cùng 1 phiên benchmark khiến
# thread hội thoại (checkpointer Redis, thread_id=customer_id, không tự hết
# hạn) tích lũy dài dần, liên tục trigger summarize_node — mỗi lần summarize
# là 1 lệnh gọi LLM ĐẦY ĐỦ THÊM, cạnh tranh trực tiếp với GPU capacity vốn đã
# giới hạn (kv_cache_max_concurrency ~24.5/replica), làm phồng P95 giả tạo
# không phản ánh đúng hiệu năng baseline. Sinh prefix ngẫu nhiên mỗi lần chạy
# module để đảm bảo mỗi lần loadtest luôn là customer_id sạch, không dây thread cũ.
_RUN_PREFIX = str(random.randint(100000, 999999))
KNOWN_CUSTOMER_IDS = [f"{_RUN_PREFIX}{i}" for i in range(50)]

# Routes to the KB search node (policy questions — see graph.py Router).
POLICY_MESSAGES = [
    "Chính sách đổi trả hàng của shop như thế nào?",
    "Tui mún hoàn trả thì làm sao?",
    "Đơn hàng giao trễ thì xử lý ra sao?",
    "Có được đổi size sau khi mua không?",
    "Thanh toán bằng những hình thức nào?",
]

# Routes to the agent/tool node (search_products tool).
PRODUCT_MESSAGES = [
    "Có áo thun nam nào giá dưới 300k không?",
    "Tìm giúp tui giày sneaker nữ màu trắng",
    "Balo laptop 15 inch giá rẻ có không?",
    "Sản phẩm nào đang bán chạy nhất tuần này?",
]

# Routes to get_recommendations tool (InjectedState auto-fills customer_id).
RECOMMEND_MESSAGES = [
    "Gợi ý cho tui vài sản phẩm phù hợp với tui đi",
    "Dựa vào lịch sử mua hàng, tui nên mua gì tiếp theo?",
]

SEARCH_QUERIES = ["áo thun", "giày sneaker", "balo laptop", "tai nghe bluetooth"]
FEEDBACK_ACTIONS = ["click", "purchase", "ignore"]


class AgentUser(HttpUser):
    wait_time = between(
        1, 3
    )  # LLM turns are slow — don't hammer faster than a real user would

    def on_start(self):
        # Bug thật 17/7/2026 (BENCHMARK_RESULTS.md mục 9, bug #16): mọi request
        # Locust tới từ CÙNG 1 IP máy chạy test → check_ip_rate_limit
        # (security.py, 60 req/60s/IP) chặn phần lớn traffic ở tải cao, trả
        # HTTP 200 kèm body {"type":"blocked"} — không phải lỗi/status khác,
        # nên dễ bị đếm nhầm là "thành công". Spoof X-Forwarded-For RIÊNG mỗi
        # simulated user để mỗi user có rate-limit bucket độc lập, mô phỏng
        # đúng nhiều client thật (agent-api không có cờ TRUST_PROXY_HEADERS
        # như recsys — _client_ip() đọc header này trực tiếp, luôn tin).
        self._fake_ip = f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"

    def _customer_id(self) -> str:
        return random.choice(KNOWN_CUSTOMER_IDS)

    def _chat(self, message: str, request_name: str):
        """POST /chat/stream and consume the whole SSE body, timed manually."""
        customer_id = self._customer_id()
        start = time.perf_counter()
        exception = None
        response_length = 0
        full_body = ""
        try:
            with self.client.post(
                "/chat/stream",
                json={"customer_id": customer_id, "message": message},
                headers={"X-Forwarded-For": self._fake_ip},
                stream=True,
                catch_response=True,
                name=request_name,
            ) as resp:
                if resp.status_code != 200:
                    resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    exception = Exception(f"HTTP {resp.status_code}")
                else:
                    for chunk in resp.iter_lines():
                        if chunk:
                            response_length += len(chunk)
                            full_body += chunk.decode("utf-8", errors="ignore")
                    # "blocked" (ip/budget/policy rate-limit) trả HTTP 200 thật
                    # — phải check nội dung body, không chỉ status_code, nếu
                    # không sẽ đếm nhầm request bị chặn thành thành công.
                    if '"type": "blocked"' in full_body:
                        resp.failure("blocked_by_guardrail")
                        exception = Exception("blocked_by_guardrail")
                    else:
                        resp.success()
        except Exception as exc:
            exception = exc
        total_time = (time.perf_counter() - start) * 1000
        events.request.fire(
            request_type="SSE",
            name=request_name,
            response_time=total_time,
            response_length=response_length,
            exception=exception,
        )

    @task(3)
    def chat_policy_question(self):
        """KB/RAG path — router sends this straight to search_kb, no tool ambiguity."""
        self._chat(random.choice(POLICY_MESSAGES), "/chat/stream [policy]")

    @task(4)
    def chat_product_search(self):
        """Agent path — LLM decides to call search_products."""
        self._chat(random.choice(PRODUCT_MESSAGES), "/chat/stream [product]")

    @task(2)
    def chat_recommendation(self):
        """Agent path — get_recommendations with InjectedState customer_id."""
        self._chat(random.choice(RECOMMEND_MESSAGES), "/chat/stream [recommend]")

    @task(3)
    def search_direct(self):
        """Direct RAG search, no LLM — exercises FAISS + reranker + semantic cache only."""
        self.client.get(
            "/search",
            params={"q": random.choice(SEARCH_QUERIES), "top_k": 10},
            name="/search",
        )

    @task(2)
    def feedback(self):
        self.client.post(
            "/feedback",
            json={
                "customer_id": self._customer_id(),
                "product_id": str(random.randint(1, 500)),
                "action": random.choice(FEEDBACK_ACTIONS),
                "session_id": str(uuid.uuid4()),
                "source": "loadtest",
            },
            name="/feedback",
        )

    @task(1)
    def health(self):
        self.client.get("/health")
