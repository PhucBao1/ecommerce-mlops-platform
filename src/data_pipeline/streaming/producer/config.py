from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Local dev: "broker:29092" | EC2 cluster: "kafka-1:9092,kafka-2:9092,kafka-3:9092"
    KAFKA_BOOTSTRAP_SERVERS: str = "broker:29092"
    KAFKA_TOPIC_REVIEWS: str = "new_reviews"

    class Config:
        env_file = ".env"


settings = Settings()
