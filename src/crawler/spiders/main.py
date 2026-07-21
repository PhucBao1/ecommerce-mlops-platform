import asyncio
import json
import logging
import os
import queue
import random
import threading
import time
import uuid
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import pandas as pd

# ============================================================================
# LOGGING
# ============================================================================

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"tiki_crawler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ============================================================================
# HELPER CLASSES
# ============================================================================


class CheckpointManager:
    """📌 Manage crawler state for resume capability"""

    def __init__(self, checkpoint_file="crawler_checkpoint.json"):
        self.checkpoint_file = checkpoint_file
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"products": {}, "comments": {}, "failed_ids": []}

    def save_batch(self, batch_type, category, batch_index, status="completed"):
        self.data.setdefault(batch_type, {}).setdefault(category, {})[
            str(batch_index)
        ] = status
        self._persist()

    def get_checkpoint(self, batch_type, category) -> List[str]:
        return list(self.data.get(batch_type, {}).get(category, {}).keys())

    def add_failed_id(self, item_id):
        self.data.setdefault("failed_ids", [])
        if item_id not in self.data["failed_ids"]:
            self.data["failed_ids"].append(item_id)
        self._persist()

    def _persist(self):
        with open(self.checkpoint_file, "w") as f:
            json.dump(self.data, f, indent=2)


class CircuitBreaker:
    """🔌 Circuit breaker for API resilience"""

    def __init__(self, failure_threshold=4, recovery_timeout=450):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"⚠️ Circuit breaker OPEN (failures: {self.failures})")

    def record_success(self):
        self.failures = max(0, self.failures - 1)
        if self.failures == 0:
            self.state = "closed"

    def call_allowed(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open":
            if (
                self.last_failure_time
                and time.time() - self.last_failure_time > self.recovery_timeout
            ):
                self.state = "half-open"
                logger.info("🔌 Circuit breaker HALF-OPEN (testing recovery)")
                return True
            return False
        return True  # half-open


class DeadLetterQueue:
    """💀 Store failed items for later processing"""

    def __init__(self, dlq_file="dead_letter_queue.jsonl"):
        self.dlq_file = dlq_file

    def add_item(self, item_id, error_msg, batch_type, retry_count=0):
        record = {
            "timestamp": datetime.now().isoformat(),
            "item_id": item_id,
            "error": error_msg,
            "type": batch_type,
            "retry_count": retry_count,
        }
        with open(self.dlq_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def get_items_to_retry(self, max_retries=3) -> List[dict]:
        if not os.path.exists(self.dlq_file):
            return []
        items = []
        with open(self.dlq_file, "r") as f:
            for line in f:
                item = json.loads(line)
                if item.get("retry_count", 0) < max_retries:
                    items.append(item)
        return items


class MetricsCollector:
    """📊 Collect performance metrics"""

    def __init__(self):
        self.metrics: Dict[str, Any] = {
            "api_calls": 0,
            "api_failures": 0,
            "bytes_written": 0,
            "processing_time": {},
        }

    def record_api_call(self, success=True):
        self.metrics["api_calls"] += 1
        if not success:
            self.metrics["api_failures"] += 1

    def record_bytes_written(self, size_bytes: int):
        self.metrics["bytes_written"] += size_bytes

    def record_timing(self, operation: str, duration_ms: float):
        self.metrics["processing_time"].setdefault(operation, []).append(duration_ms)

    def get_summary(self) -> dict:
        summary = {
            "total_api_calls": self.metrics["api_calls"],
            "api_failure_rate": f"{100 * self.metrics['api_failures'] / max(1, self.metrics['api_calls']):.2f}%",
            "mb_written": f"{self.metrics['bytes_written'] / (1024 * 1024):.2f}",
            "avg_timing": {},
        }
        for op, times in self.metrics["processing_time"].items():
            if times:
                summary["avg_timing"][op] = f"{sum(times) / len(times):.2f}ms"
        return summary


# ============================================================================
# ASYNC TIKI CRAWLER
# ============================================================================


class TikiCrawlerAsync:
    # === CONFIG CONSTANTS ===
    SAVE_THRESHOLD_MB = 128  # flush buffer khi đạt 128 MB (uncompressed in-memory)
    PRODUCT_CONCURRENCY = (
        8  # số request đồng thời khi crawl product detail (Tiki rate limit thấp)
    )
    COMMENT_CONCURRENCY = 4  # số request đồng thời khi crawl comments
    PRODUCT_BATCH_SIZE = 200
    COMMENT_BATCH_SIZE = 12
    MAX_CONCURRENT_CATEGORIES = 3

    MINIO_CONFIG = {
        "client_kwargs": {
            "endpoint_url": "http://localhost:9000",
            "aws_access_key_id": "admin",
            "aws_secret_access_key": "password",
            "region_name": "us-east-1",
        }
    }
    PROGRESS_FILE = "crawler_progress.json"

    # Pool header đa dạng — mỗi request random 1 bộ để tránh bị fingerprint
    HEADER_POOL = [
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://tiki.vn/",
            "sec-ch-ua": '"Chromium";v="124", "Google Chrome";v="124"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
        },
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "vi;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://tiki.vn/dien-thoai-may-tinh-bang/c1789",
            "sec-ch-ua": '"Google Chrome";v="123", "Not:A-Brand";v="8"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
        },
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://tiki.vn/",
            "sec-ch-ua": '"Chromium";v="122", "Google Chrome";v="122"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
        },
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "vi-VN,vi;q=0.8,en-US;q=0.5,en;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://tiki.vn/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
        },
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "vi-VN,vi;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://tiki.vn/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
        },
    ]

    @classmethod
    def _random_headers(cls) -> dict:
        """Chọn ngẫu nhiên 1 bộ header từ pool"""
        return random.choice(cls.HEADER_POOL).copy()

    def __init__(self):
        self.checkpoint_mgr = CheckpointManager()
        self.circuit_breaker = CircuitBreaker(failure_threshold=4, recovery_timeout=300)
        self.dlq = DeadLetterQueue()
        self.metrics = MetricsCollector()

        self.progress_data = {
            "start_time": datetime.now().isoformat(),
            "products_crawled": 0,
            "comments_crawled": 0,
            "files_saved": 0,
            "errors": 0,
            "current_batch": None,
            "last_update": datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------

    @staticmethod
    def safe_get_nested(dic, keys, default=""):
        for key in keys:
            if isinstance(dic, dict):
                dic = dic.get(key, default)
            else:
                return default
        return dic

    @staticmethod
    def sanitize_category_name(cat_name: str) -> str:
        return cat_name.replace(" ", "_").replace("&", "and").lower()

    @staticmethod
    def _today_partition() -> str:
        """Return date string: YYYY-MM-DD"""
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _minio_path(self, layer: str, table_name: str, table_type: str) -> str:
        """
        Build MinIO S3 path với date partition và subfolder riêng cho từng loại.

        Result: s3://warehouse/<layer>/<table_type>/YYYY-MM-DD/<table_name>_<uuid>.parquet
        """
        date_partition = self._today_partition()
        file_id = uuid.uuid4().hex
        return f"s3://warehouse/{layer}/{table_type}/{date_partition}/{table_name}_{file_id}.parquet"

    def _save_progress(self):
        self.progress_data["last_update"] = datetime.now().isoformat()
        try:
            with open(self.PROGRESS_FILE, "w") as f:
                json.dump(self.progress_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Không thể lưu progress file: {e}")

    def _print_progress(self):
        p = self.progress_data
        logger.info(
            f"\n{'='*70}\n"
            f"📊 PROGRESS UPDATE:\n"
            f"   Products Crawled : {p['products_crawled']:,}\n"
            f"   Comments Crawled : {p['comments_crawled']:,}\n"
            f"   Files Saved      : {p['files_saved']}\n"
            f"   Errors           : {p['errors']}\n"
            f"   Last Update      : {p['last_update']}\n"
            f"{'='*70}\n"
        )

    # ------------------------------------------------------------------
    # DATA NORMALIZATION
    # ------------------------------------------------------------------

    def normalize_id_columns(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        return df

    def normalize_int_columns(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        for col in cols:
            if col in df.columns:
                numeric = pd.to_numeric(df[col], errors="coerce")
                max_val = numeric.max()
                df[col] = (
                    numeric.astype("Int32")
                    if (pd.notna(max_val) and max_val < 2**31)
                    else numeric.astype("Int64")
                )
        return df

    def normalize_float_columns(
        self, df: pd.DataFrame, cols: List[str]
    ) -> pd.DataFrame:
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        return df

    # ------------------------------------------------------------------
    # VALIDATION & MINIO
    # ------------------------------------------------------------------

    def validate_dataframe(self, df: pd.DataFrame, df_type="product") -> bool:
        if df.empty:
            logger.warning(f"⚠️ Empty DataFrame for {df_type}")
            return False

        # FIX #5: validate df_type để tránh schema sai khi typo
        valid_types = {"product", "comment"}
        if df_type not in valid_types:
            logger.error(
                f"❌ df_type không hợp lệ: '{df_type}'. Phải là một trong {valid_types}"
            )
            return False

        required = (
            {"product_id", "crawl_time"}
            if df_type == "product"
            else {"product_id", "rating", "crawl_time"}
        )
        missing = required - set(df.columns)
        if missing:
            logger.error(f"❌ Missing columns: {missing}")
            return False
        return True

    def save_to_minio(
        self, df: pd.DataFrame, table_name: str, layer="bronze", df_type="product"
    ) -> bool:
        """
        💾 Save DataFrame to MinIO với date partition + subfolder riêng.
        """
        if not self.validate_dataframe(df, df_type=df_type):
            return False

        table_type = f"{df_type}s"  # "product" → "products", "comment" → "comments"
        s3_path = self._minio_path(layer, table_name, table_type)
        date_partition = self._today_partition()
        size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

        try:
            start = time.time()
            logger.info(f"📀 Writing {len(df):,} rows ({size_mb:.1f} MB) → {s3_path}")
            df.to_parquet(
                s3_path,
                storage_options=self.MINIO_CONFIG,
                engine="pyarrow",
                index=False,
                compression="snappy",
            )
            elapsed = time.time() - start
            self.metrics.record_timing("save_to_minio", elapsed * 1000)
            self.metrics.record_bytes_written(int(size_mb * 1024 * 1024))
            logger.info(
                f"✅ Saved [{date_partition}/{table_type}] {table_name} | {size_mb:.1f} MB | {elapsed:.2f}s"
            )
            return True
        except Exception as e:
            logger.error(f"❌ MinIO error: {e}")
            self.dlq.add_item(table_name, str(e), "minio_write")
            return False

    # ------------------------------------------------------------------
    # ASYNC API CALLS
    # ------------------------------------------------------------------

    async def _async_get(
        self,
        session: aiohttp.ClientSession,
        url: str,
        params: dict,
        retries: int = 3,
        base_delay: float = 2.0,
    ) -> Optional[dict]:
        """Single async GET with exponential backoff + circuit breaker"""
        if not self.circuit_breaker.call_allowed():
            wait = self.circuit_breaker.recovery_timeout
            logger.warning(f"⏸️ Circuit breaker OPEN — chờ {wait}s trước khi thử lại...")
            await asyncio.sleep(wait)
            if not self.circuit_breaker.call_allowed():
                logger.error("❌ Circuit breaker vẫn OPEN sau khi chờ, bỏ qua request")
                return None

        for attempt in range(retries):
            try:
                async with session.get(
                    url,
                    params=params,
                    headers=self._random_headers(),
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status == 200:
                        self.metrics.record_api_call(success=True)
                        self.circuit_breaker.record_success()
                        return await resp.json(content_type=None)

                    elif resp.status == 429:
                        retry_after = int(
                            resp.headers.get("Retry-After", base_delay * (2**attempt))
                        )
                        retry_after = max(retry_after, base_delay * (2**attempt))
                        logger.warning(
                            f"🚦 429 Rate limit | chờ {retry_after:.1f}s | attempt {attempt+1} | {url}"
                        )
                        await asyncio.sleep(retry_after + random.uniform(0.5, 2.0))

                    else:
                        logger.warning(
                            f"HTTP {resp.status} | {url} | attempt {attempt+1}"
                        )
                        self.metrics.record_api_call(success=False)
                        self.circuit_breaker.record_failure()

            except Exception as e:
                self.metrics.record_api_call(success=False)
                self.circuit_breaker.record_failure()
                if attempt < retries - 1:
                    delay = base_delay * (2**attempt) + random.uniform(0, 1.0)
                    logger.warning(
                        f"⚡ Retry {attempt+1}/{retries} in {delay:.2f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"❌ Failed after {retries} retries: {e}")

        self.metrics.record_api_call(success=False)
        return None

    # ------------------------------------------------------------------
    # CRAWL PRODUCT IDs  (async)
    # ------------------------------------------------------------------

    async def _crawl_product_ids_async(
        self,
        session: aiohttp.ClientSession,
        category_id: int,
        url_key: str,
        max_pages: Optional[int] = None,
    ) -> List[int]:
        # FIX #4: dùng set để tự dedup IDs giữa các trang
        seen_ids: set = set()
        page = 1
        url = "https://tiki.vn/api/v2/products"

        while True:
            if not self.circuit_breaker.call_allowed():
                logger.error("⚠️ Circuit breaker OPEN - stopping ID crawl")
                break
            if max_pages and page > max_pages:
                break

            params = {
                "limit": 40,
                "category": category_id,
                "page": page,
                "urlKey": url_key,
            }
            data_json = await self._async_get(session, url, params)

            if data_json is None:
                break
            data = data_json.get("data", [])
            if not data:
                logger.info(f"✅ Hết dữ liệu ở trang {page} (category {category_id})")
                break

            before = len(seen_ids)
            for item in data:
                if item.get("id"):
                    seen_ids.add(item["id"])
            new_count = len(seen_ids) - before
            logger.info(f"✓ Page {page}: +{new_count} IDs (category {category_id})")
            page += 1

            await asyncio.sleep(random.uniform(1.5, 3.0))

        return list(seen_ids)

    # ------------------------------------------------------------------
    # CRAWL PRODUCT DETAILS  (async, semaphore-controlled)
    # ------------------------------------------------------------------

    async def _crawl_single_product(
        self,
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
        pid: int,
    ) -> Optional[dict]:
        async with sem:
            url = f"https://tiki.vn/api/v2/products/{pid}"
            params = {"platform": "web"}
            await asyncio.sleep(random.uniform(0.5, 1.5))
            data = await self._async_get(session, url, params)
            if data is None:
                return None

            specs = data.get("specifications", [])
            return {
                "product_id": data.get("id"),
                "sku": data.get("sku"),
                "product_name": data.get("name"),
                "short_description": data.get("short_description"),
                "price": data.get("price"),
                "list_price": data.get("list_price"),
                "discount_rate": data.get("discount_rate"),
                "rating": data.get("rating_average"),
                "review_count": data.get("review_count"),
                "inventory_status": data.get("inventory_status"),
                "stock_qty": self.safe_get_nested(data, ["stock_item", "qty"]),
                "quantity_sold": int(
                    self.safe_get_nested(data, ["quantity_sold", "value"], 0) or 0
                ),
                "brand_id": self.safe_get_nested(data, ["brand", "id"]),
                "brand_name": self.safe_get_nested(data, ["brand", "name"]),
                "category_id": self.safe_get_nested(data, ["categories", "id"]),
                "category_name": self.safe_get_nested(data, ["categories", "name"]),
                "seller_id": self.safe_get_nested(data, ["current_seller", "id"]),
                "seller_name": self.safe_get_nested(data, ["current_seller", "name"]),
                "seller_logo": self.safe_get_nested(data, ["current_seller", "logo"]),
                "seller_link": self.safe_get_nested(data, ["current_seller", "link"]),
                "url": data.get("short_url"),
                "thumbnail_url": data.get("thumbnail_url"),
                "all_specs": str(specs),
            }

    async def _crawl_product_details_async(
        self,
        session: aiohttp.ClientSession,
        product_ids: List[int],
        concurrency: int,
    ) -> pd.DataFrame:
        sem = asyncio.Semaphore(concurrency)
        tasks = [self._crawl_single_product(session, sem, pid) for pid in product_ids]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        rows = [r for r in results if r is not None]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # CRAWL COMMENTS  (async, semaphore-controlled)
    # ------------------------------------------------------------------

    async def _crawl_comments_for_product(
        self,
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
        pid: int,
        max_pages: int = 50,
    ) -> List[dict]:
        # FIX #2: semaphore chỉ wrap từng request, không wrap toàn bộ vòng lặp
        # → các coroutine khác không bị block suốt 50 trang × delay
        comments: List[dict] = []
        url = "https://tiki.vn/api/v2/reviews"

        for page in range(1, max_pages + 1):
            async with sem:
                params = {"product_id": pid, "page": page, "limit": 20}
                data_json = await self._async_get(session, url, params)

            if data_json is None:
                break
            data = data_json.get("data", [])
            if not data:
                break

            for c in data:
                comments.append(
                    {
                        "product_id": pid,
                        "review_id": c.get("id"),
                        "comment": c.get("content"),
                        "rating": c.get("rating"),
                        "customer_id": self.safe_get_nested(c, ["created_by", "id"]),
                        "customer_name": self.safe_get_nested(
                            c, ["created_by", "name"]
                        ),
                        "is_buyer": self.safe_get_nested(
                            c, ["created_by", "purchased"]
                        ),
                        "purchased_at": self.safe_get_nested(
                            c, ["created_by", "purchased_at"]
                        ),
                    }
                )

            await asyncio.sleep(random.uniform(2.0, 3.0))

        logger.info(f"✅ Product {pid}: Thu thập {len(comments)} comments")
        return comments

    async def _crawl_comments_async(
        self,
        session: aiohttp.ClientSession,
        product_ids: List[int],
        concurrency: int,
        max_pages: int = 50,
    ) -> pd.DataFrame:
        sem = asyncio.Semaphore(concurrency)
        tasks = [
            self._crawl_comments_for_product(session, sem, pid, max_pages)
            for pid in product_ids
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        rows = []
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Comment task error: {r}")
            elif r:
                rows.extend(r)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # NORMALIZE BATCH
    # ------------------------------------------------------------------

    def _normalize_product_df(
        self, df: pd.DataFrame, category_name: str
    ) -> pd.DataFrame:
        # FIX #7: pd.Timestamp.utcnow() deprecated → dùng pd.Timestamp.now("UTC")
        df["crawl_time"] = pd.Timestamp.now("UTC")
        df["batch_id"] = uuid.uuid4().hex
        df["general_category"] = category_name
        df = self.normalize_id_columns(
            df, ["product_id", "brand_id", "category_id", "seller_id", "sku"]
        )
        df = self.normalize_int_columns(
            df, ["price", "list_price", "review_count", "stock_qty", "quantity_sold"]
        )
        df = self.normalize_float_columns(df, ["discount_rate", "rating"])
        return df

    def _normalize_comment_df(
        self, df: pd.DataFrame, category_name: str
    ) -> pd.DataFrame:
        # FIX #7: pd.Timestamp.utcnow() deprecated → dùng pd.Timestamp.now("UTC")
        df["crawl_time"] = pd.Timestamp.now("UTC")
        df["batch_id"] = uuid.uuid4().hex
        df["general_category"] = category_name

        for col in ["customer_id", "review_id"]:
            if col in df.columns:
                df[col] = (
                    df[col].fillna(0).astype(str).str.replace(r"\.0$", "", regex=True)
                )

        if "product_id" in df.columns:
            df["product_id"] = (
                pd.to_numeric(df["product_id"], errors="coerce")
                .fillna(0)
                .astype("int64")
            )
        if "rating" in df.columns:
            df["rating"] = (
                pd.to_numeric(df["rating"], errors="coerce").fillna(0).astype("float32")
            )
        if "purchased_at" in df.columns:
            numeric_ts = pd.to_numeric(df["purchased_at"], errors="coerce")
            valid_mask = numeric_ts.between(946_684_800, 4_102_444_800)
            numeric_ts = numeric_ts.where(valid_mask, other=pd.NA)
            df["purchased_at"] = pd.to_datetime(numeric_ts, unit="s", errors="coerce")
        if "is_buyer" in df.columns:
            df["is_buyer"] = df["is_buyer"].replace(["", "nan", "None", "NULL"], pd.NA)
            val_map = {
                True: True,
                False: False,
                1: True,
                0: False,
                "1": True,
                "0": False,
                "true": True,
                "false": False,
                "True": True,
                "False": False,
            }
            df["is_buyer"] = df["is_buyer"].map(val_map)
        return df

    # ------------------------------------------------------------------
    # ORCHESTRATORS
    # ------------------------------------------------------------------

    async def _crawl_category_products(
        self,
        session: aiohttp.ClientSession,
        cat_name: str,
        ids: List[int],
        concurrency: int,
        batch_size: int,
    ) -> int:
        """Crawl products cho 1 category — chạy song song với các category khác"""

        safe_cat = self.sanitize_category_name(cat_name)
        buffer: List[pd.DataFrame] = []
        total_rows = 0

        def _buffer_mb() -> float:
            return (
                sum(df.memory_usage(deep=True).sum() for df in buffer) / (1024 * 1024)
                if buffer
                else 0.0
            )

        logger.info(f"🚀 Products - category: {cat_name} | {len(ids)} IDs")

        for i in range(0, len(ids), batch_size):
            batch = ids[i : i + batch_size]
            logger.info(
                f"  ⏳ [{cat_name}] Batch {i // batch_size + 1} | {len(batch)} products"
            )

            df = await self._crawl_product_details_async(session, batch, concurrency)
            if df.empty:
                continue

            df = self._normalize_product_df(df, cat_name)
            buffer.append(df)
            total_rows += len(df)

            current_mb = _buffer_mb()
            logger.info(
                f"  📦 Buffer [{cat_name}]: {current_mb:.1f} MB / {self.SAVE_THRESHOLD_MB} MB"
            )

            if current_mb >= self.SAVE_THRESHOLD_MB:
                big_df = pd.concat(buffer, ignore_index=True)
                ts = int(time.time())
                self.save_to_minio(
                    big_df, f"products_{safe_cat}_{ts}", df_type="product"
                )
                self.progress_data["files_saved"] += 1
                buffer.clear()

            self.progress_data["products_crawled"] += len(df)
            self._save_progress()

        if buffer:
            big_df = pd.concat(buffer, ignore_index=True)
            remaining_mb = big_df.memory_usage(deep=True).sum() / (1024 * 1024)
            ts = int(time.time())
            logger.info(f"🧹 Flush cuối [{cat_name}]: {remaining_mb:.1f} MB")
            self.save_to_minio(
                big_df, f"products_{safe_cat}_{ts}_final", df_type="product"
            )
            self.progress_data["files_saved"] += 1

        logger.info(f"✅ Xong [{cat_name}]: {total_rows:,} rows")
        return total_rows

    async def _run_products_async(
        self,
        all_product_ids: Dict[str, List[int]],
        concurrency: int,
        batch_size: int,
    ):
        """Async orchestrator — tất cả categories crawl SONG SONG"""
        connector = aiohttp.TCPConnector(
            limit=concurrency * len(all_product_ids) + 10, ttl_dns_cache=300
        )
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                self._crawl_category_products(
                    session, cat_name, ids, concurrency, batch_size
                )
                for cat_name, ids in all_product_ids.items()
            ]
            # FIX #6: log các exception thay vì nuốt im lặng
            results = await asyncio.gather(*tasks, return_exceptions=True)

        total_rows = 0
        for cat_name, result in zip(all_product_ids.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"❌ Category '{cat_name}' failed: {result}")
                self.progress_data["errors"] += 1
            else:
                total_rows += result

        self._print_progress()
        self._save_progress()
        logger.info(f"✨ HOÀN THÀNH PRODUCTS: {total_rows:,} rows")

    async def _crawl_category_comments(
        self,
        session: aiohttp.ClientSession,
        cat_name: str,
        ids: List[int],
        concurrency: int,
        batch_size: int,
        max_pages: int,
    ) -> int:
        """Crawl comments cho 1 category — chạy song song với các category khác"""
        safe_cat = self.sanitize_category_name(cat_name)
        buffer: List[pd.DataFrame] = []
        total_rows = 0

        def _buffer_mb() -> float:
            return (
                sum(df.memory_usage(deep=True).sum() for df in buffer) / (1024 * 1024)
                if buffer
                else 0.0
            )

        logger.info(f"🚀 Comments - category: {cat_name} | {len(ids)} IDs")

        for i in range(0, len(ids), batch_size):
            batch = ids[i : i + batch_size]
            logger.info(
                f"  ⏳ [{cat_name}] Batch {i // batch_size + 1} | {len(batch)} products"
            )

            df = await self._crawl_comments_async(
                session, batch, concurrency, max_pages
            )
            if df.empty:
                continue

            df = self._normalize_comment_df(df, cat_name)
            buffer.append(df)
            total_rows += len(df)

            current_mb = _buffer_mb()
            logger.info(
                f"  📦 Buffer [{cat_name}]: {current_mb:.1f} MB / {self.SAVE_THRESHOLD_MB} MB"
            )

            if current_mb >= self.SAVE_THRESHOLD_MB:
                big_df = pd.concat(buffer, ignore_index=True)
                ts = int(time.time())
                self.save_to_minio(
                    big_df, f"comments_{safe_cat}_{ts}", df_type="comment"
                )
                self.progress_data["files_saved"] += 1
                buffer.clear()

            self.progress_data["comments_crawled"] += len(df)
            self._save_progress()

        if buffer:
            big_df = pd.concat(buffer, ignore_index=True)
            remaining_mb = big_df.memory_usage(deep=True).sum() / (1024 * 1024)
            ts = int(time.time())
            logger.info(f"🧹 Flush cuối [{cat_name}]: {remaining_mb:.1f} MB")
            self.save_to_minio(
                big_df, f"comments_{safe_cat}_{ts}_final", df_type="comment"
            )
            self.progress_data["files_saved"] += 1

        logger.info(f"✅ Xong [{cat_name}]: {total_rows:,} rows")
        return total_rows

    async def _run_comments_async(
        self,
        all_product_ids: Dict[str, List[int]],
        concurrency: int,
        batch_size: int,
        max_pages: int = 50,
    ):
        """Async orchestrator — tất cả categories crawl SONG SONG"""
        n_cats = max(len(all_product_ids), 1)
        connector = aiohttp.TCPConnector(
            limit=concurrency * self.MAX_CONCURRENT_CATEGORIES, ttl_dns_cache=300
        )
        async with aiohttp.ClientSession(connector=connector) as session:
            # Giới hạn số category chạy song song
            semaphore_cat = asyncio.Semaphore(self.MAX_CONCURRENT_CATEGORIES)

            async def wrapped_category(cat_name, ids):
                async with semaphore_cat:
                    logger.info(
                        f"🚀 BẮT ĐẦU COMMENT - {cat_name} ({len(ids)} products)"
                    )
                    return await self._crawl_category_comments(
                        session, cat_name, ids, concurrency, batch_size, max_pages
                    )

            tasks = [
                wrapped_category(cat_name, ids)
                for cat_name, ids in all_product_ids.items()
            ]

            # FIX #6: log các exception thay vì nuốt im lặng
            results = await asyncio.gather(*tasks, return_exceptions=True)

        total_rows = 0
        for cat_name, result in zip(all_product_ids.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"❌ Category '{cat_name}' failed: {result}")
                self.progress_data["errors"] += 1
            else:
                total_rows += result

        self._print_progress()
        self._save_progress()
        logger.info(f"✨ HOÀN THÀNH COMMENTS: {total_rows:,} rows")

    async def _crawl_all_ids_async(
        self,
        categories: List[dict],
        max_pages: Optional[int] = 20,
    ) -> Dict[str, List[int]]:
        """Crawl product IDs cho tất cả categories song song"""
        connector = aiohttp.TCPConnector(limit=len(categories) + 5, ttl_dns_cache=300)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                self._crawl_product_ids_async(
                    session,
                    cat["category_id"],
                    cat["url_key"],
                    max_pages=max_pages,
                )
                for cat in categories
            ]
            results = await asyncio.gather(*tasks)

        all_product_ids = {}
        for cat, ids in zip(categories, results):
            logger.info(f"✅ {cat['name']}: {len(ids)} IDs")
            all_product_ids[cat["name"]] = ids
        return all_product_ids

    # ------------------------------------------------------------------
    # PUBLIC API
    # FIX #1: sync wrappers dùng asyncio.run() — KHÔNG gọi từ trong event loop.
    # Nếu cần gọi từ async context (Airflow async, Jupyter), dùng trực tiếp
    # các method _run_products_async / _run_comments_async bằng await.
    # ------------------------------------------------------------------

    def run_all_products(
        self,
        all_product_ids: Dict[str, List[int]],
        concurrency: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Entry point đồng bộ — dùng cho Airflow PythonOperator hoặc __main__.
        ⚠️ KHÔNG gọi method này từ bên trong async function/event loop đang chạy.
           Nếu đang ở async context, dùng: await crawler._run_products_async(...)
        """
        asyncio.run(
            self._run_products_async(
                all_product_ids,
                concurrency=concurrency or self.PRODUCT_CONCURRENCY,
                batch_size=batch_size or self.PRODUCT_BATCH_SIZE,
            )
        )

    def run_all_comments(
        self,
        all_product_ids: Dict[str, List[int]],
        concurrency: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_pages: int = 50,
    ):
        """
        Entry point đồng bộ — dùng cho Airflow PythonOperator hoặc __main__.
        ⚠️ KHÔNG gọi method này từ bên trong async function/event loop đang chạy.
           Nếu đang ở async context, dùng: await crawler._run_comments_async(...)
        """
        asyncio.run(
            self._run_comments_async(
                all_product_ids,
                concurrency=concurrency or self.COMMENT_CONCURRENCY,
                batch_size=batch_size or self.COMMENT_BATCH_SIZE,
                max_pages=max_pages,
            )
        )

    def _export_metrics_summary(self):
        summary = self.metrics.get_summary()
        logger.info("\n" + "=" * 70)
        logger.info("📈 PERFORMANCE METRICS:")
        for k, v in summary.items():
            logger.info(f"   {k}: {v}")
        logger.info("=" * 70)

        with open("metrics_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # FIX #8: thêm trailing newline để Prometheus scraper parse đúng
        prom_lines = [
            "# HELP api_calls Total API calls made",
            f"api_calls {summary['total_api_calls']}",
            "# HELP mb_written Total MB written to MinIO",
            f"mb_written {summary['mb_written']}",
            "",  # trailing newline
        ]
        with open("metrics.prom", "w") as f:
            f.write("\n".join(prom_lines))

        dlq_items = self.dlq.get_items_to_retry()
        if dlq_items:
            logger.warning(f"⚠️ Dead Letter Queue: {len(dlq_items)} items chờ retry")

    @staticmethod
    def check_progress():
        if os.path.exists("crawler_progress.json"):
            with open("crawler_progress.json") as f:
                return json.load(f)
        return None


# ============================================================================
# MAIN
# ============================================================================


async def main():
    crawler = TikiCrawlerAsync()

    logger.info("=" * 70)
    logger.info("🚀 ASYNC TIKI CRAWLER — ENTERPRISE EDITION")
    logger.info("   ✅ Async I/O với aiohttp (không block thread)")
    logger.info("   ✅ Semaphore-controlled concurrency")
    logger.info("   ✅ Date-partitioned MinIO output (YYYY/MM/DD)")
    logger.info("   ✅ Circuit Breaker + Dead Letter Queue")
    logger.info("   ✅ Checkpoint / Resume")
    logger.info("   ✅ Metrics Tracking")
    logger.info("=" * 70)

    categories = [
        {
            "name": "Điện Thoại & Máy Tính Bảng",
            "category_id": 1789,
            "url_key": "dien-thoai-may-tinh-bang",
        },
        {
            "name": "Laptop, Máy Tính & Linh Kiện",
            "category_id": 1846,
            "url_key": "laptop-may-vi-tinh-linh-kien",
        },
        {
            "name": "Thiết Bị Số - Phụ Kiện Số",
            "category_id": 1815,
            "url_key": "thiet-bi-kts-phu-kien-so",
        },
        {
            "name": "Nhà Sách Online Tiki",
            "category_id": 8322,
            "url_key": "nha-sach-online",
        },
        {
            "name": "Nhà Cửa - Đời Sống",
            "category_id": 1883,
            "url_key": "nha-cua-doi-song",
        },
        {
            "name": "Đồ Chơi An Toàn Cho Bé",
            "category_id": 2549,
            "url_key": "do-choi-me-be",
        },
        {
            "name": "Thiết Bị Điện Gia Dụng",
            "category_id": 1882,
            "url_key": "dien-gia-dung",
        },
        {
            "name": "Chăm Sóc Sắc Đẹp và Sức Khỏe",
            "category_id": 1520,
            "url_key": "lam-dep-suc-khoe",
        },
        {
            "name": "Ô Tô - Xe Máy - Xe Đạp",
            "category_id": 8594,
            "url_key": "o-to-xe-may-xe-dap",
        },
        {
            "name": "Thời Trang Nữ Thiết Kế Cao Cấp",
            "category_id": 931,
            "url_key": "thoi-trang-nu",
        },
        {
            "name": "Bách Hóa Online - Đi Chợ Tại Nhà",
            "category_id": 4384,
            "url_key": "bach-hoa-online",
        },
        {
            "name": "Thể Thao - Dã Ngoại",
            "category_id": 1975,
            "url_key": "the-thao-da-ngoai",
        },
        {
            "name": "Thời Trang Nam Cao Cấp",
            "category_id": 915,
            "url_key": "thoi-trang-nam",
        },
        {
            "name": "Giày Dép Nam Thời Trang",
            "category_id": 1686,
            "url_key": "giay-dep-nam",
        },
        {
            "name": "Điện Tử - Điện Lạnh",
            "category_id": 4221,
            "url_key": "dien-tu-dien-lanh",
        },
        {"name": "Giày Dép Nữ Xinh Xắn", "category_id": 1703, "url_key": "giay-dep-nu"},
        {"name": "Máy Ảnh - Máy Quay Phim", "category_id": 1801, "url_key": "may-anh"},
        {
            "name": "Phụ Kiện Thời Trang Nam Nữ",
            "category_id": 27498,
            "url_key": "phu-kien-thoi-trang",
        },
        {
            "name": "Đồng Hồ & Trang Sức Cao Cấp",
            "category_id": 8371,
            "url_key": "dong-ho-va-trang-suc",
        },
        {
            "name": "Balo, Vali, Túi Kéo Du Lịch",
            "category_id": 6000,
            "url_key": "balo-va-vali",
        },
        {
            "name": "Túi Xách & Ví Nữ Thời Trang",
            "category_id": 976,
            "url_key": "tui-vi-nu",
        },
        {
            "name": "Túi & Ví Nam Cao Cấp",
            "category_id": 27616,
            "url_key": "tui-thoi-trang-nam",
        },
        {
            "name": "Chăm sóc nhà cửa",
            "category_id": 15078,
            "url_key": "cham-soc-nha-cua",
        },
    ]

    # 1. Crawl tất cả product IDs song song
    all_product_ids = await crawler._crawl_all_ids_async(categories, max_pages=50)

    total_ids = sum(len(v) for v in all_product_ids.values())
    logger.info(f"📦 Tổng IDs: {total_ids:,}")

    # 2. Crawl product details (async) — dùng await trực tiếp, không dùng sync wrapper
    """await crawler._run_products_async(
        all_product_ids,
        concurrency=crawler.PRODUCT_CONCURRENCY,
        batch_size=crawler.PRODUCT_BATCH_SIZE,
    )"""

    # 3. Crawl comments (async) — dùng await trực tiếp, không dùng sync wrapper
    await crawler._run_comments_async(
        all_product_ids,
        concurrency=crawler.COMMENT_CONCURRENCY,
        batch_size=crawler.COMMENT_BATCH_SIZE,
        max_pages=50,
    )

    crawler._export_metrics_summary()


if __name__ == "__main__":
    asyncio.run(main())
