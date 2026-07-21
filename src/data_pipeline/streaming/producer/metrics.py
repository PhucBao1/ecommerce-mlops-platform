from prometheus_client import Counter, Gauge

REQUEST_COUNTER = Counter(
    "api_producer_requests_total", "Total producer number of ingestion API requests"
)


DLQ_ERROR_COUNTER = Counter(
    "kafka__producer_dlq_errors_total", "Total producer number of DLQ fallback events"
)


KAFKA_STATUS_GAUGE = Gauge(
    "kafka_connection_status", "Total producer number of Kafka connection status"
)
