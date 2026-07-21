from prometheus_client import Counter, Gauge, Histogram

# ==========================================
# EVENT COUNTERS
# ==========================================

STREAM_EVENTS_TOTAL = Counter("stream_events_total", "Total streaming events consumed")

STREAM_ERRORS_TOTAL = Counter("stream_errors_total", "Total streaming errors")

# ==========================================
# LATENCY
# ==========================================

INFERENCE_LATENCY = Histogram("inference_latency_seconds", "PhoBERT inference latency")

FEAST_PUSH_LATENCY = Histogram("feast_push_latency_seconds", "Feast push latency")

# ==========================================
# BUFFER
# ==========================================

BUFFER_SIZE_GAUGE = Gauge("stream_buffer_size", "Current streaming buffer size")

# ==========================================
# CACHE INVALIDATION
# ==========================================

CACHE_INVALIDATIONS_TOTAL = Counter(
    "cache_invalidations_total", "Redis cache invalidations"
)

# ==========================================
# CONSUMER LAG + FRESHNESS
# ==========================================

CONSUMER_LAG_GAUGE = Gauge(
    "kafka_consumer_lag_messages",
    "high_watermark - committed_position, theo từng partition",
    ["topic", "partition"],
)

E2E_FRESHNESS = Histogram(
    "e2e_event_freshness_seconds",
    "Thời gian từ lúc Kafka nhận message tới lúc xử lý xong (Feast push thành công)",
    buckets=(1, 5, 10, 30, 60, 120, 300, 600),
)
