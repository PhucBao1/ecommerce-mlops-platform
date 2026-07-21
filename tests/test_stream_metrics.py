"""
Consumer lag + E2E freshness — 2 metric quan trọng nhất của streaming mà
trước đây KHÔNG được đo. Không có lag thì không trả lời được câu hỏi cơ bản
nhất: "consumer có theo kịp producer không?"

Test bằng mock Kafka consumer, KHÔNG cần broker thật.
"""

import time
from unittest.mock import MagicMock

from confluent_kafka import TopicPartition

from src.data_pipeline.streaming.consumer.metrics import (
    CONSUMER_LAG_GAUGE,
    E2E_FRESHNESS,
)
from src.data_pipeline.streaming.consumer.stream import StreamProcessor


def _make_processor():
    return StreamProcessor(
        consumer=MagicMock(),
        dlq_producer=MagicMock(),
        dlq_topic="new_reviews_dlq",
        fs=MagicMock(),
    )


# ---------------------------------------------------------------------------
# Consumer lag
# ---------------------------------------------------------------------------


def test_lag_is_high_watermark_minus_committed_position():
    proc = _make_processor()
    tp = TopicPartition("new_reviews", 0, 42)  # committed position = 42
    proc.consumer.assignment.return_value = [tp]
    proc.consumer.position.return_value = [tp]
    proc.consumer.get_watermark_offsets.return_value = (0, 100)  # high watermark

    proc._update_lag()

    value = CONSUMER_LAG_GAUGE.labels(topic="new_reviews", partition="0")._value.get()
    assert value == 100 - 42


def test_lag_never_negative_when_no_position_yet():
    """position=-1001 (OFFSET_INVALID) khi consumer chưa fetch message nào."""
    proc = _make_processor()
    tp = TopicPartition("new_reviews", 0, -1001)
    proc.consumer.assignment.return_value = [tp]
    proc.consumer.position.return_value = [tp]
    proc.consumer.get_watermark_offsets.return_value = (0, 50)

    proc._update_lag()

    value = CONSUMER_LAG_GAUGE.labels(topic="new_reviews", partition="0")._value.get()
    assert value >= 0


def test_lag_check_does_not_crash_when_broker_unreachable():
    """Lỗi hỏi watermark KHÔNG được làm sập vòng lặp chính của consumer."""
    proc = _make_processor()
    proc.consumer.assignment.side_effect = Exception("broker down")

    proc._update_lag()  # không raise


def test_lag_check_skips_when_no_partitions_assigned():
    proc = _make_processor()
    proc.consumer.assignment.return_value = []

    proc._update_lag()

    proc.consumer.position.assert_not_called()


# ---------------------------------------------------------------------------
# E2E freshness — dùng timestamp Kafka tự gắn, KHÔNG dùng purchased_at
# ---------------------------------------------------------------------------


def test_freshness_uses_kafka_produce_timestamp_not_business_time():
    """
    purchased_at (payload) là thời điểm khách MUA HÀNG — có thể lệch nhiều ngày
    so với lúc event vào pipeline. Freshness phải dùng msg.timestamp() (Kafka
    tự gắn lúc produce), không phải field nào trong payload.
    """
    proc = _make_processor()
    proc.fs.push = MagicMock()
    proc.consumer.commit = MagicMock()

    produced_5s_ago_ms = int((time.time() - 5) * 1000)
    msg = MagicMock()
    msg.timestamp.return_value = (1, produced_5s_ago_ms)  # (CreateTime, ms)
    msg.topic.return_value = "new_reviews"
    msg.partition.return_value = 0
    msg.offset.return_value = 10

    row = {
        "customer_id": "c1",
        "recent_sentiment_score": 0.8,
        "last_commented_product_id": "p1",
        "event_timestamp": None,
    }
    proc.buffer = [(row, msg)]

    before = E2E_FRESHNESS._sum.get()
    proc.flush()
    after = E2E_FRESHNESS._sum.get()

    observed = after - before
    assert 4.5 <= observed <= 6.0, f"freshness phải ~5s, đo được {observed}"


def test_freshness_skipped_when_kafka_timestamp_unavailable():
    """ts_ms <= 0 nghĩa là broker không gắn timestamp (topic cấu hình cũ) —
    bỏ qua thay vì ghi số âm/rác vào histogram."""
    proc = _make_processor()
    proc.fs.push = MagicMock()
    proc.consumer.commit = MagicMock()

    msg = MagicMock()
    msg.timestamp.return_value = (0, -1)  # không có timestamp
    msg.topic.return_value = "new_reviews"
    msg.partition.return_value = 0
    msg.offset.return_value = 10

    row = {
        "customer_id": "c1",
        "recent_sentiment_score": 0.8,
        "last_commented_product_id": "p1",
        "event_timestamp": None,
    }
    proc.buffer = [(row, msg)]

    sum_before = E2E_FRESHNESS._sum.get()
    proc.flush()
    sum_after = E2E_FRESHNESS._sum.get()

    assert sum_after == sum_before, "không được observe khi thiếu timestamp"
