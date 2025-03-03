from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import wandb
from typing import List, Optional
import uvicorn
from model import ReviewClassifier

# Initialize the model
model = ReviewClassifier()

# Start Weights & Biases run
wandb.init(project="customer-review-classifier")

# FastAPI app
app = FastAPI(title="Amazon Review Classifier API")

# Pydantic stuff
# request body for single classification.
class ReviewRequest(BaseModel):
    text: str

# request body for batch classification.
class BatchReviewRequest(BaseModel):
    reviews: List[str]
    log_results: Optional[bool] = False

# api response format 
class ClassificationResponse(BaseModel):
    category: str
    confidence: float
    all_scores: dict



@app.post("/classify", response_model=ClassificationResponse)
async def classify_review(review: ReviewRequest):
    try:
        result = model.classify_review(review.text)
        
        # Log to tracking platforms
        model.log_to_wandb(review.text, result)
        model.log_to_mlflow(review.text, result)
        
        return ClassificationResponse(
            category=result["top_category"],
            confidence=result["confidence"],
            all_scores=result["all_scores"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-classify")
async def batch_classify(request: BatchReviewRequest):
    try:
        results = model.batch_classify(request.reviews)
        
        # Log to tracking platforms if requested
        if request.log_results:
            for review, result in zip(request.reviews, results):
                model.log_to_wandb(review, result)
                model.log_to_mlflow(review, result)
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)