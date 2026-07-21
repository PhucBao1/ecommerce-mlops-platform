from typing import Optional

from pydantic import BaseModel, Field, field_validator


class TikiProduct(BaseModel):
    product_id: str
    name: str
    price: float = Field(gt=0)
    category_id: str
    seller_id: Optional[str] = None
    rating_average: Optional[float] = Field(default=None, ge=0, le=5)
    review_count: Optional[int] = Field(default=None, ge=0)

    @field_validator("product_id", "category_id")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("field must not be empty")
        return v


class TikiReview(BaseModel):
    review_id: str
    product_id: str
    customer_id: str
    rating: int = Field(ge=1, le=5)
    comment: Optional[str] = None
    created_at: str

    @field_validator("comment")
    @classmethod
    def comment_max_length(cls, v: Optional[str]) -> Optional[str]:
        if v and len(v) > 5000:
            return v[:5000]
        return v
