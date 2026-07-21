from datetime import datetime

from pydantic import BaseModel, Field


class CustomerReviewEvent(BaseModel):
    customer_id: str = Field(..., example="CUST_1001")
    product_id: str = Field(..., example="PROD_25")
    comment: str = Field(..., example="Sản phẩm rất tốt")
    rating: int = Field(..., ge=1, le=5)
    purchased_at: datetime = Field(..., example="2024-06-01T12:34:56Z")
