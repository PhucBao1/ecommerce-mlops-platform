from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, PushSource
from feast.types import Float32, String

# 1. Định nghĩa Entity (Thực thể chính là Khách hàng)
customer = Entity(name="customer", join_keys=["customer_id"])

# 2. Định nghĩa Nguồn dữ liệu offline (Dùng làm dummy backup cho PushSource)
dummy_offline_source = FileSource(
    path="data/dummy_sentiment.parquet",
    timestamp_field="event_timestamp",
)

# 3. Định nghĩa Push Source (Cổng nhận dữ liệu Real-time)
sentiment_push_source = PushSource(
    name="recent_sentiment_push_source",
    batch_source=dummy_offline_source,
)

# 4. Định nghĩa Feature View kết nối với Push Source
customer_recent_sentiment_view = FeatureView(
    name="customer_recent_sentiment",
    entities=[customer],
    ttl=timedelta(days=7),  # Feature này có giá trị trong 7 ngày
    schema=[
        Field(name="recent_sentiment_score", dtype=Float32),
        Field(name="last_commented_product_id", dtype=String),
    ],
    online=True,
    source=sentiment_push_source,
)
