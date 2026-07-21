from fastapi import APIRouter, BackgroundTasks, status
from fastapi.responses import JSONResponse

from src.data_pipeline.streaming.producer.CommentRequest import CustomerReviewEvent
from src.data_pipeline.streaming.producer.review_service import ReviewService

router = APIRouter(prefix="/api/v1/reviews", tags=["reviews"])


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def create_review(review: CustomerReviewEvent, background_tasks: BackgroundTasks):

    ReviewService.publish_review(review)

    return JSONResponse(
        status_code=202,
        content={"status": "accepted", "customer_id": review.customer_id},
    )
