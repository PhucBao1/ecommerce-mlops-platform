"""
Avro producer with Schema Registry integration.

Wraps confluent_kafka SerializingProducer to auto-register Avro schemas
and serialize messages with schema ID prefix (Confluent wire format).
Falls back to JSON if Schema Registry is unavailable at startup.

Usage:
    producer = AvroProducerManager()
    producer.send("new_reviews", key="product_123", value={...})

Schema files: src/data_pipeline/streaming/schemas/*.avsc
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA_REGISTRY_URL = os.getenv("SCHEMA_REGISTRY_URL", "http://schema-registry:8081")
_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "broker:29092")
_SCHEMAS_DIR = Path(__file__).parents[1] / "schemas"

_TOPIC_SCHEMA_MAP: dict[str, str] = {
    "new_reviews": "new_reviews",
    "recsys_predictions": "recsys_predictions",
    "agent_feedback": "agent_feedback",
}

# Try to import Avro dependencies at module level
try:
    from confluent_kafka import SerializingProducer
    from confluent_kafka.schema_registry import Schema, SchemaRegistryClient
    from confluent_kafka.schema_registry.avro import AvroSerializer
    from confluent_kafka.serialization import MessageField, SerializationContext

    _AVRO_AVAILABLE = True
except ImportError:
    _AVRO_AVAILABLE = False
    logger.warning(
        "avro_producer: confluent_kafka schema_registry not installed — JSON fallback only"
    )


def _load_schema_str(schema_name: str) -> str:
    return (_SCHEMAS_DIR / f"{schema_name}.avsc").read_text(encoding="utf-8")


def _delivery_callback(err, msg) -> None:
    if err:
        logger.error("avro_producer delivery failed: %s", err)
    else:
        logger.debug(
            "avro_producer delivered topic=%s partition=%d",
            msg.topic(),
            msg.partition(),
        )


class AvroProducerManager:
    """
    Kafka producer that serializes messages with Avro + Schema Registry.
    Falls back to JSON if Schema Registry is unreachable.
    """

    def __init__(self):
        self._serializers: dict[str, AvroSerializer] = {}
        self._avro_producer: SerializingProducer | None = None
        self._json_producer = None
        self._use_avro = False

        if not _AVRO_AVAILABLE:
            self._init_json_fallback()
            return

        try:
            sr_client = SchemaRegistryClient({"url": _SCHEMA_REGISTRY_URL})
            sr_client.get_subjects()  # connectivity check
            self._sr_client = sr_client
            self._use_avro = True
            logger.info(
                "avro_producer: Schema Registry connected at %s", _SCHEMA_REGISTRY_URL
            )
        except Exception as e:
            logger.warning(
                "avro_producer: Schema Registry unavailable (%s) — JSON fallback", e
            )
            self._init_json_fallback()

    def _init_json_fallback(self) -> None:
        from confluent_kafka import Producer

        self._json_producer = Producer({"bootstrap.servers": _BOOTSTRAP})

    def _get_serializer(self, topic: str) -> AvroSerializer:
        if topic not in self._serializers:
            schema_name = _TOPIC_SCHEMA_MAP.get(topic)
            if not schema_name:
                raise ValueError(f"No Avro schema mapped for topic '{topic}'")
            self._serializers[topic] = AvroSerializer(
                self._sr_client,
                Schema(_load_schema_str(schema_name), "AVRO"),
                to_dict=lambda obj, ctx: obj,
            )
        return self._serializers[topic]

    def _get_avro_producer(self) -> SerializingProducer:
        if self._avro_producer is None:
            self._avro_producer = SerializingProducer(
                {
                    "bootstrap.servers": _BOOTSTRAP,
                    "acks": "all",
                    "enable.idempotence": True,
                    "compression.type": "snappy",
                    "linger.ms": 10,
                }
            )
        return self._avro_producer

    def send(self, topic: str, key: str, value: dict) -> None:
        if not self._use_avro:
            self._json_producer.produce(
                topic=topic,
                key=key.encode(),
                value=json.dumps(value, ensure_ascii=False).encode(),
                callback=_delivery_callback,
            )
            self._json_producer.poll(0)
            return

        serializer = self._get_serializer(topic)
        ctx = SerializationContext(topic, MessageField.VALUE)
        self._get_avro_producer().produce(
            topic=topic,
            key=key,
            value=serializer(value, ctx),
            on_delivery=_delivery_callback,
        )
        self._get_avro_producer().poll(0)

    def flush(self, timeout: int = 10) -> None:
        if self._use_avro and self._avro_producer:
            self._avro_producer.flush(timeout)
        elif self._json_producer:
            self._json_producer.flush(timeout)
