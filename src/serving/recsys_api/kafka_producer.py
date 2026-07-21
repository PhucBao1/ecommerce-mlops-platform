import atexit
import json
import logging
import os

from confluent_kafka import Producer

logger = logging.getLogger(__name__)

# Local: "broker:29092" | EC2 cluster: "kafka-1:9092,kafka-2:9092,kafka-3:9092"
conf = {"bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "broker:29092")}

producer = Producer(conf)


def send_prediction_event(event: dict):
    producer.produce(
        topic="recsys_predictions", value=json.dumps(event).encode("utf-8")
    )
    # poll(0) chỉ phục vụ hàng đợi callback nội bộ (không chặn) — KHÔNG dùng
    # flush() ở đây vì flush() chờ broker xác nhận xong mới trả về, biến
    # mỗi prediction event thành 1 round-trip đồng bộ, phá mất batching.
    producer.poll(0)


def _flush_on_exit() -> None:
    """Chỉ chạy 1 lần lúc process tắt — đảm bảo message còn trong hàng đợi
    được gửi hết trước khi container dừng hẳn."""
    remaining = producer.flush(timeout=10)
    if remaining > 0:
        logger.warning("kafka_producer_flush_incomplete remaining=%d", remaining)


atexit.register(_flush_on_exit)
