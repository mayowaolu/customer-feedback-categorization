import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import io
from model import ReviewClassifier
from PIL import Image

# Initialize the model
model = ReviewClassifier()


def classify_and_visualize(review_text):
    """
    Classifies a review and returns visualization of category scores
    """
    # Classify the review
    result = model.classify_review(review_text)
    
    # Create a DataFrame for visualization
    df = pd.DataFrame({
        'Category': list(result["all_scores"].keys()),
        'Score': list(result["all_scores"].values())
    })
    df = df.sort_values('Score', ascending=False)
    
    # Create the bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.barh(df['Category'], df['Score'])
    plt.xlabel('Confidence Score')
    plt.title('Customer Review Classification')
    plt.xlim(0, 1)
    
    # Highlight the top category
    top_index = df[df['Category'] == result["top_category"]].index[0]
    bars[top_index].set_color('red')
    
    # Add annotations
    for i, (score, category) in enumerate(zip(df['Score'], df['Category'])):
        plt.text(score + 0.01, i, f"{score:.2f}", va='center')
    
    # Save plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()

    image = Image.open(buf)
    
    # Return the results
    return [
        result["top_category"],
        f"{result['top_score']:.2f}",
        image
    ]

def batch_process(file):
    """
    Process a CSV file of reviews and return classified results
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file.name)
        
        # Check if there's a column named 'review' or 'text'
        if 'review' in df.columns:
            review_column = 'review'
        elif 'text' in df.columns:
            review_column = 'text'
        else:
            return "Error: CSV file must have a column named 'review' or 'text'"
        
        # Classify each review
        results = []
        for review in df[review_column]:
            result = model.classify_review(review)
            results.append({
                'review': review[:100] + "..." if len(review) > 100 else review,
                'category': result["top_category"],
                'confidence': result["top_score"]
            })
        
        # Create a DataFrame with the results
        results_df = pd.DataFrame(results)
        
        # Count categories
        category_counts = results_df['category'].value_counts()
        
        # Plot category distribution
        plt.figure(figsize=(10, 6))
        category_counts.plot(kind='bar')
        plt.title('Distribution of Review Categories')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        
        # Save plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return {
            "Results Table": results_df,
            "Category Distribution": buf
        }
    except Exception as e:
        return f"Error processing file: {str(e)}"

# Create the Gradio interface
with gr.Blocks(title="Customer Review Classifier") as demo:
    gr.Markdown("# Customer Negative Review Classifier")
    gr.Markdown("This app classifies negative Customer reviews into different issue categories.")
    gr.Markdown(f"Model {str(model.model_name)}")
    
    with gr.Tab("Single Review"):
        with gr.Row():
            with gr.Column():
                review_input = gr.Textbox(
                    label="Enter a negative Customer review", 
                    placeholder="Type or paste a negative Customer review here...",
                    lines=5
                )
                classify_btn = gr.Button("Classify")
            
            with gr.Column():
                top_category = gr.Textbox(label="Top Category")
                confidence = gr.Textbox(label="Confidence")
                viz_output = gr.Image(label="Category Scores")
    
    with gr.Tab("Batch Processing"):
        gr.Markdown("CSV file should have a column with ""review"" or ""text"" as the header/label")
        with gr.Row():
            with gr.Column():
                file_input = gr.File(label="Upload CSV file with reviews")
                process_btn = gr.Button("Process Batch")
            
            with gr.Column():
                results_table = gr.DataFrame(label="Classification Results")
                category_dist = gr.Image(label="Category Distribution")
    
    # Connect the components
    classify_btn.click(
        classify_and_visualize, 
        inputs=review_input, 
        outputs=[top_category, confidence, viz_output]
    )
    
    process_btn.click(
        batch_process, 
        inputs=file_input, 
        outputs=[results_table, category_dist]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)