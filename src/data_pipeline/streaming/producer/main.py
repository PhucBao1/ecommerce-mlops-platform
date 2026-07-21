import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.data_pipeline.streaming.producer.metrics import KAFKA_STATUS_GAUGE
from src.data_pipeline.streaming.producer.producer import kafka_producer
from src.data_pipeline.streaming.producer.reviews import router as review_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting application")

    KAFKA_STATUS_GAUGE.set(1)

    yield

    logger.info("Flushing kafka producer...")
    kafka_producer.flush(10)
    logger.info("Application stopped")


app = FastAPI(title="Review Event API", version="1.0.0", lifespan=lifespan)

app.include_router(review_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
