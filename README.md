# Customer Feedback Categorization System

A semi full-stack ML platform that automatically classifies negative Amazon reviews into specific issue categories using zero-shot learning. This project demonstrates the integration of modern ML tools and technologies.

## Technologies Used
- **Hugging Face Transformers**: For the pre-trained transformer models.
- **FastAPI**: For serving the model via a REST API.
- **Gradio**: For building an interactive UI.
- **MLflow**: For model versioning and tracking.
- **Weights & Biases (W&B)**: For experiment tracking.

## Features
- Zero-shot classification (no labeled data needed)
- Issue categorization into 8 predefined categories
- Single review and batch processing capabilities
- Visualizations of classification results
- Experiment tracking and logging
- REST API for integration with other systems

## Steps to Run the Project

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
