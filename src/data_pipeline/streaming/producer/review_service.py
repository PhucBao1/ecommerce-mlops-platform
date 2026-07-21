from src.data_pipeline.streaming.producer.CommentRequest import CustomerReviewEvent
from src.data_pipeline.streaming.producer.config import settings
from src.data_pipeline.streaming.producer.producer import kafka_producer


class ReviewService:

    @staticmethod
    def publish_review(review: CustomerReviewEvent):

        kafka_producer.send(
            topic=settings.KAFKA_TOPIC_REVIEWS,
            key=review.customer_id,
            value=review.model_dump(mode="json"),
        )
