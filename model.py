import torch
from transformers import pipeline
import wandb
import mlflow
import pandas


class ReviewClassifier:
    def __init__(self, model_name="facebook/bart-large-mnli"):
        self.model_name = model_name
        self.categories = [
            "Product Quality", 
            "Delivery Issues", 
            "Customer Service", 
            "Price Concerns", 
            "Packaging Issues", 
            "Product Expectations", 
            "Technical Issues", 
            "Others"
        ]

        self.classifier = pipeline(task="zero-shot-classification", model=model_name)
    
    def classify_review(self, review_text: str):
        """ Classify a single review into one of the predefined categories """
        result = self.classifier(
            sequences = review_text,
            candidate_labels = self.categories,
            hypothesis_template = "This review is about {}"
        )

        # Returns the most likely category and its scores
        top_category = result["labels"][0]
        top_score = result["scores"][0]

        # Full result for logging
        category_scores = {
            category: score
            for category, score in zip(result["labels"], result["scores"])
        }

        return {
            "top_category": top_category,
            "top_score": top_score,
            "all_scores": category_scores
        }

    def batch_classify(self, reviews: list):
        """ Classify a batch of reviews"""
        results = []

        for review in reviews:
            results.append(self.classify_review(review))
        return results

    def log_to_wandb(self, review, classification_result):
        """ Log classification results in weights and bias """
        wandb.log({
            "review": review,
            "predicted_category": classification_result["top_category"],
            "confidence": classification_result["confidence"],
            "category_scores": classification_result["all_scores"]
        })

    def log_to_mlflow(self, review, classification_result):
    
        """Log classification results to MLflow"""
    
        with mlflow.start_run():
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("review", review)
            mlflow.log_metric("confidence", classification_result["confidence"])
            
            # Log the category as a tag
            mlflow.set_tag("predicted_category", classification_result["top_category"])
            
            # Log all scores
            for category, score in classification_result["all_scores"].items():
                mlflow.log_metric(f"score_{category.replace(' ', '_')}", score)