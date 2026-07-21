import json
import logging
import os
from datetime import datetime

import pandas as pd
from confluent_kafka import Consumer, KafkaError
from feast import FeatureStore

logger = logging.getLogger(__name__)

# click=0.7, purchase=1.0, ignore=0.3 — implicit feedback weights
FEEDBACK_WEIGHTS: dict[str, float] = {
    "purchase": 1.0,
    "click": 0.7,
    "ignore": 0.3,
}

conf = {
    "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "broker:29092"),
    "group.id": "feedback_processor_group",
    "auto.offset.reset": "latest",
}

consumer = Consumer(conf)
consumer.subscribe(["user_actions"])

fs = FeatureStore(repo_path="src/feature_store/feature_repo")

BUFFER_SIZE = 50
buffer: list = []


def process_feedback(event: dict) -> dict:
    action = event.get("action", "click")
    return {
        "customer_id": str(event["customer_id"]),
        "product_id": str(event["product_id"]),
        "implicit_score": FEEDBACK_WEIGHTS.get(action, 0.5),
        "event_timestamp": datetime.now(),
        "created_at": datetime.now(),
    }


def flush_buffer() -> None:
    global buffer
    if not buffer:
        return

    pdf = pd.DataFrame(buffer)
    fs.push(push_source_name="user_feedback_push_source", df=pdf)
    logger.info("Flushed %d feedback events to Feast", len(buffer))
    buffer = []


logger.info("Listening for user actions from Kafka topic 'user_actions'...")

try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            logger.error("Kafka error: %s", msg.error())
            continue
        try:
            event_data = json.loads(msg.value().decode("utf-8"))
            row = process_feedback(event_data)
            buffer.append(row)
            logger.info(
                "customer=%s product=%s action=%s score=%.1f buffer=%d/%d",
                row["customer_id"],
                row["product_id"],
                event_data.get("action"),
                row["implicit_score"],
                len(buffer),
                BUFFER_SIZE,
            )
            if len(buffer) >= BUFFER_SIZE:
                flush_buffer()
        except Exception as e:
            logger.error("Failed to process feedback event: %s", e)

except KeyboardInterrupt:
    logger.info("Shutting down feedback processor...")
    flush_buffer()
finally:
    consumer.close()
