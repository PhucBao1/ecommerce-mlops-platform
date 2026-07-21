import random

from locust import HttpUser, between, task

# ID thật lấy từ USER_HISTORY_DICT (không phải 1000-1049 giả định trước đó —
# range đó không tồn tại trong lịch sử thật, luôn rơi vào nhánh trending/cold
# start, không bao giờ chạm code model.user_tower()/feature-prep thật, khiến
# "cache_hit" task trước đây không đo đúng cái cần đo).
KNOWN_CUSTOMER_IDS = [
    "145739",
    "10799928",
    "17407674",
    "6856857",
    "7685964",
    "17046932",
    "8122065",
    "26714414",
    "8057326",
    "5709555",
    "8082632",
    "5585877",
    "16860472",
    "1001307",
    "8616808",
    "13348864",
    "21157818",
    "636595",
    "19401982",
    "5527680",
    "5406207",
    "16994191",
    "9699172",
    "17347005",
    "29829959",
    "6432831",
    "23795210",
    "19386351",
    "1310091",
    "854224",
    "14444196",
    "8464392",
    "10086133",
    "5426941",
    "28403303",
    "11557420",
    "5912899",
    "10453604",
    "1134152",
    "21713429",
    "12669993",
    "16644559",
    "18435452",
    "10314310",
    "7915401",
    "19235395",
    "1279361",
    "845016",
    "7669033",
    "450444",
]

# Sample item IDs for session-based (SASRec) requests — cold-start friendly,
# doesn't need a customer_id at all.
SAMPLE_ITEM_IDS = [str(i) for i in range(1, 500)]


class RecSysUser(HttpUser):
    """
    Load test for the Recommendation API (src/serving/recsys_api), port 8001.

    Run: locust -f loadtest_recsys.py --host http://localhost:8001

    Mỗi Locust user giả lập 1 client thật riêng biệt bằng cách gửi kèm
    X-Forwarded-For với IP giả khác nhau — khớp với cách production thật
    nhận diện client sau reverse proxy/load balancer (xem main.py
    _client_identity, cần bật TRUST_PROXY_HEADERS=true để server tin header
    này). Nếu không làm vậy, toàn bộ user giả lập trong 1 lần chạy Locust sẽ
    dùng chung 1 IP thật (máy chạy loadtest) → bị rate-limit tính chung 1
    bucket, không phản ánh đúng traffic nhiều user thật.
    """

    wait_time = between(2, 6)

    def on_start(self):
        fake_ip = f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
        self.client.headers.update({"X-Forwarded-For": fake_ip})

    @task(4)
    def recommend_known_user(self):
        """Cache hit path — repeat known IDs to warm Redis cache."""
        self.client.post(
            "/recommend",
            json={"customer_id": random.choice(KNOWN_CUSTOMER_IDS), "top_k": 10},
            name="/recommend [cache_hit]",
        )

    @task(2)
    def recommend_cold_start(self):
        """Cache miss + full FAISS inference path (Feast defaults, no history)."""
        self.client.post(
            "/recommend",
            json={"customer_id": f"cold_{random.randint(100000, 999999)}", "top_k": 10},
            name="/recommend [cold_start]",
        )

    @task(2)
    def recommend_session(self):
        """SASRec session-based recommendation — no customer_id needed."""
        session_len = random.randint(2, 10)
        session_items = random.sample(SAMPLE_ITEM_IDS, k=session_len)
        self.client.post(
            "/recommend/session",
            json={"session_items": session_items, "top_k": 10},
            name="/recommend/session",
        )

    @task(1)
    def health(self):
        self.client.get("/health")
