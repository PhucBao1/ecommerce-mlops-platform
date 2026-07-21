import json
import logging
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime

import pandas as pd
from confluent_kafka import Consumer, KafkaError, Producer, TopicPartition
from dotenv import load_dotenv
from feast import FeatureStore
from prometheus_client import start_http_server

load_dotenv()
from src.data_pipeline.streaming.consumer.cache import invalidate_recommendation_cache
from src.data_pipeline.streaming.consumer.metrics import *
from src.data_pipeline.streaming.consumer.sentiment_model import (
    load_model,
    predict_sentiment,
)


# -----------------------------------------------------------------
# CẤU HÌNH JSON LOGGING (Chuẩn hóa để phân tích log tự động)
# -----------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logger = logging.getLogger("recsys_stream")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
# Tắt log mặc định để tránh trùng lặp
logger.propagate = False

start_http_server(8010)


class StreamProcessor:

    def __init__(self, consumer, dlq_producer, dlq_topic, fs):
        self.consumer = consumer
        self.dlq_producer = dlq_producer
        self.dlq_topic = dlq_topic
        self.fs = fs

        self.buffer_size = int(os.getenv("BUFFER_SIZE", 100))
        self.flush_interval = int(os.getenv("FLUSH_INTERVAL_SEC", 30))
        self.max_io_workers = int(os.getenv("MAX_IO_WORKERS", 10))
        self.lag_check_interval = int(os.getenv("LAG_CHECK_INTERVAL_SEC", 10))

        self.buffer = []
        self.last_flush = time.time()
        self.last_lag_check = 0.0
        self.stop_flag = False

        self.executor = ThreadPoolExecutor(max_workers=self.max_io_workers)

        self.lock = threading.Lock()

    def send_to_dlq(self, event, error, stage):
        """Đẩy thẳng dữ liệu lỗi vào một Kafka DLQ Topic riêng biệt"""
        dlq_record = {
            "event": event,
            "error": str(error),
            "stage": stage,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        try:
            # default=str — event có thể chứa pandas Timestamp (event_timestamp)
            # không tự serialize được, làm crash cả đường DLQ (bug thật đã gặp)
            self.dlq_producer.produce(
                topic=self.dlq_topic,
                value=json.dumps(dlq_record, default=str).encode("utf-8"),
            )
            # Gọi poll(0) để giải phóng hàng đợi nội bộ của Producer
            self.dlq_producer.poll(0)
            logger.warning(
                f"Event sent to DLQ topic '{self.dlq_topic}' at stage '{stage}'"
            )
        except Exception as e:
            logger.critical(
                f"CRITICAL: Failed to send event to Kafka DLQ! Error: {e} | Data: {dlq_record}"
            )

        STREAM_ERRORS_TOTAL.inc()

    def handle_event(self, event_data):
        try:
            customer_id = str(event_data["customer_id"])
            comment_text = str(event_data["comment"])
            product_id = str(event_data["product_id"])

            start = time.time()
            sentiment_score = predict_sentiment(comment_text)
            logger.info(f"SENTIMENT DEBUG: {comment_text} -> {sentiment_score}")
            INFERENCE_LATENCY.observe(time.time() - start)

            row = {
                "customer_id": customer_id,
                "recent_sentiment_score": sentiment_score,
                "last_commented_product_id": product_id,
                "event_timestamp": datetime.now(),
            }
            STREAM_EVENTS_TOTAL.inc()
            return row

        except Exception as e:
            self.send_to_dlq(event_data, e, "handle_event")
            return None

    def flush(self):
        with self.lock:
            if not self.buffer:
                return
            batch_buffer = self.buffer
            self.buffer = []

        self.last_flush = time.time()
        BUFFER_SIZE_GAUGE.set(0)

        if not batch_buffer:
            return
        rows, msgs = zip(*batch_buffer)
        pdf = pd.DataFrame(rows)

        # STAGE 1: Push sang Feast
        start = time.time()
        try:
            self.fs.push(push_source_name="recent_sentiment_push_source", df=pdf)
            FEAST_PUSH_LATENCY.observe(time.time() - start)
        except Exception as e:
            logger.error(f"Feast push failed. Routing batch to DLQ.")
            for row in pdf.to_dict("records"):
                self.send_to_dlq(row, e, "feast_push")
            return

        # STAGE 2: Invalidate Cache song song kèm theo BOUNDED TIMEOUT nhằm khống chế Latency Spike
        affected_users = pdf["customer_id"].astype(str).unique()

        def _safe_invalidate_with_retry(uid):
            # Cài đặt Retry cơ bản ngay tại tầng tác vụ I/O bound để tăng tính kiên cố (Idempotency Safe)
            for attempt in range(3):
                try:
                    invalidate_recommendation_cache(uid)
                    CACHE_INVALIDATIONS_TOTAL.inc()
                    return True
                except Exception as e:
                    if attempt == 2:
                        logger.error(
                            f"Failed to invalidate cache for user {uid} after 3 attempts: {e}"
                        )
                        STREAM_ERRORS_TOTAL.inc()
                    time.sleep(0.1 * (attempt + 1))  # Bounded backoff
            return False

        # Đẩy các tác vụ vào pool
        futures = [
            self.executor.submit(_safe_invalidate_with_retry, uid)
            for uid in affected_users
        ]

        # Bắt buộc phải ĐỢI (để giữ tính At-least-once) nhưng giới hạn thời gian tối đa là 5 giây tránh nghẽn chết luồng chính
        done, not_done = wait(futures, timeout=5.0)

        if not_done:
            logger.warning(
                f"Cache invalidation timed out for {len(not_done)} users. Moving forward to prevent pipeline stall."
            )

        # STAGE 3: SỬA BUG MẤT DỮ LIỆU - COMMIT CHÍNH XÁC THEO TỪNG BATCH BOUNDARY PER PARTITION
        latest_offsets = {}
        for msg in msgs:
            tp = (msg.topic(), msg.partition())
            # Vị trí commit kế tiếp phải là offset hiện tại + 1
            latest_offsets[tp] = max(latest_offsets.get(tp, -1), msg.offset() + 1)

        # Chuyển đổi thành cấu trúc TopicPartition chuẩn của confluent_kafka
        offsets_to_commit = [
            TopicPartition(topic, partition, offset)
            for (topic, partition), offset in latest_offsets.items()
        ]

        try:
            # Commit chính xác danh sách offset vừa xử lý xong của đúng batch này
            self.consumer.commit(offsets=offsets_to_commit, asynchronous=False)
            logger.info(
                f"Successfully flushed {len(pdf)} events and committed precise offsets."
            )
        except Exception as e:
            logger.error(f"Kafka precise commit error: {e}")

        now = time.time()
        for msg in msgs:
            _, ts_ms = msg.timestamp()
            if ts_ms > 0:
                E2E_FRESHNESS.observe(now - ts_ms / 1000.0)

    def _update_lag(self):
        """
        lag = high_watermark (offset mới nhất broker có) - committed_position
        (offset consumer đã xác nhận xử lý). Hỏi broker qua network nên chỉ gọi
        định kỳ (lag_check_interval), KHÔNG gọi mỗi vòng poll() 0.5s.
        """
        try:
            partitions = self.consumer.assignment()
            if not partitions:
                return
            positions = self.consumer.position(partitions)
            for tp in positions:
                _, high = self.consumer.get_watermark_offsets(tp, cached=False)
                # position=-1001 nghĩa là chưa fetch message nào ở partition này
                committed = tp.offset if tp.offset >= 0 else 0
                lag = max(0, high - committed)
                CONSUMER_LAG_GAUGE.labels(
                    topic=tp.topic, partition=str(tp.partition)
                ).set(lag)
        except Exception as e:
            logger.warning(f"Không đo được consumer lag: {e}")

    def shutdown(self, sig, frame):
        logger.info("Shutdown signal received. Exiting loop...")
        self.stop_flag = True

    def run(self):
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

        logger.info("Stream Processor started running.")

        while not self.stop_flag:
            msg = self.consumer.poll(0.5)

            # Kiểm tra xem khoảng lặng thời gian đã đến hạn flush chưa
            if time.time() - self.last_flush >= self.flush_interval and self.buffer:
                logger.info("Flush interval reached, triggering flush.")
                self.flush()

            if time.time() - self.last_lag_check >= self.lag_check_interval:
                self.last_lag_check = time.time()
                self._update_lag()

            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error(f"Kafka Error: {msg.error()}")
                STREAM_ERRORS_TOTAL.inc()
                continue

            try:
                event = json.loads(msg.value().decode("utf-8"))
                row = self.handle_event(event)

                if row is None:
                    continue
                with self.lock:
                    self.buffer.append((row, msg))
                    BUFFER_SIZE_GAUGE.set(len(self.buffer))

                    if len(self.buffer) >= self.buffer_size:
                        self.flush()

            except Exception as e:
                STREAM_ERRORS_TOTAL.inc()
                logger.error(f"Error decoding/processing message: {e}")

        # ---- KHU VỰC DỌN DẸP KHI SHUTDOWN GỒM ĐỦ 4 BƯỚC ----
        logger.info("Executing graceful shutdown pipelines...")
        self.flush()
        self.executor.shutdown(wait=True)

        # Đảm bảo toàn bộ message trong DLQ gửi đi hết trước khi tắt app
        logger.info("Flushing DLQ Kafka Producer...")
        self.dlq_producer.flush(timeout=5)

        try:
            self.consumer.close()
        except Exception as e:
            logger.error(f"Kafka consumer close error: {e}")

        logger.info("Processor shutdown cleanly.")


if __name__ == "__main__":
    logger.info("Warming sentiment model...")
    load_model()

    kafka_brokers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "broker:29092")

    # 1. Khởi tạo Kafka Consumer
    consumer_conf = {
        "bootstrap.servers": kafka_brokers,
        "group.id": "recsys_streaming_group",
        "auto.offset.reset": "latest",
        "enable.auto.commit": False,  # Ép buộc tắt auto commit
    }
    consumer = Consumer(consumer_conf)
    consumer.subscribe(["new_reviews"])

    # 2. Khởi tạo Kafka DLQ Producer
    dlq_producer_conf = {
        "bootstrap.servers": kafka_brokers,
        "acks": "all",  # Đảm bảo tin nhắn DLQ đã ghi xuống đĩa của Broker mới trả về thành công
    }
    dlq_producer = Producer(dlq_producer_conf)
    dlq_topic = "new_reviews_dlq"

    # 3. Khởi tạo Feature Store
    fs = FeatureStore(repo_path="src/feature_store/feature_repo")

    # Run
    processor = StreamProcessor(consumer, dlq_producer, dlq_topic, fs)
    processor.run()
