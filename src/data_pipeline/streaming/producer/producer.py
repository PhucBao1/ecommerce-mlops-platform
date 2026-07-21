import json
import logging

from confluent_kafka import Producer

from src.data_pipeline.streaming.producer.config import settings
from src.data_pipeline.streaming.producer.metrics import (
    DLQ_ERROR_COUNTER,
    KAFKA_STATUS_GAUGE,
)

logger = logging.getLogger(__name__)


class KafkaProducerManager:
    def __init__(self):
        self.producer = Producer(
            {
                "bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS,
                "acks": "all",
                "enable.idempotence": True,
                "compression.type": "snappy",
                "linger.ms": 10,
                "batch.num.messages": 1000,
                "queue.buffering.max.messages": 100000,
            }
        )

    def _delivery_report(self, err, msg):
        if err:
            logger.error("Kafka delivery failed: %s", err)
            DLQ_ERROR_COUNTER.inc()
            KAFKA_STATUS_GAUGE.set(0)
        else:
            KAFKA_STATUS_GAUGE.set(1)
            logger.debug(
                "Message delivered topic=%s partition=%s offset=%s",
                msg.topic(),
                msg.partition(),
                msg.offset(),
            )

    def send(self, topic: str, key: str, value: dict):
        try:
            self.producer.produce(
                topic=topic,
                key=key,
                value=json.dumps(value, ensure_ascii=False),
                callback=self._delivery_report,
            )

            self.producer.poll(0)
        except Exception as e:
            logger.exception("Kafka produce error: %s", e)
            DLQ_ERROR_COUNTER.inc()
            raise

    def flush(self, timeout=10):
        remaining = self.producer.flush(timeout)

        if remaining > 0:
            logger.warning("%s messages were not delivered before shutdown", remaining)


kafka_producer = KafkaProducerManager()
