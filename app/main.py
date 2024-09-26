from fastapi import FastAPI
from pydantic import BaseModel

from .model import relevance_model

app = FastAPI()


@app.get("/")
async def root():
    """Validation endpoint"""
    return {"message": "Hello World"}


class ReviewRequest(BaseModel):
    """Represents the reviews request body"""

    review: str


@app.post("/review")
async def check_review_relevance(request: ReviewRequest):
    """Handles the evaulation of reviews to check for relevancy"""
    relevance_score = relevance_model.predict_relevance(request.review)
    return {"relevance_score": relevance_score}
