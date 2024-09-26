from fastapi import FastAPI
from pydantic import BaseModel

from .model import relevance_model

app = FastAPI()


class ReviewRequest(BaseModel):
    review: str


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/review")
async def check_review_relevance(request: ReviewRequest):
    relevance_score = relevance_model.predict_relevance(request.review)
    return {"relevance_score": relevance_score}
