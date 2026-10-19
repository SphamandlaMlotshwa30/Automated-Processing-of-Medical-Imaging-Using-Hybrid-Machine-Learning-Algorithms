import re

import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from data_preprocessing import IMAGE_SIZE, CLASS_NAMES

METRICS_FILE = "Metrics.txt"


def load_model_metrics():
    """Read model evaluation metrics from the project metrics file if it exists."""
    metrics = {
        "Training Accuracy": "N/A",
        "Validation Accuracy": "N/A",
        "Precision": "N/A",
        "Recall": "N/A",
        "F1-Score": "N/A",
        "AUC-ROC": "N/A"
    }

    if not os.path.exists(METRICS_FILE):
        return metrics

    try:
        with open(METRICS_FILE, "r", encoding="utf-8") as file:
            content = file.read()

        metric_map = {
            "Training Accuracy": r"Training Accuracy\s*:\s*([0-9.]+)",
            "Validation Accuracy": r"Validation Accuracy\s*:\s*([0-9.]+)",
            "Precision": r"Precision\s*:\s*([0-9.]+)",
            "Recall": r"Recall\s*:\s*([0-9.]+)",
            "F1-Score": r"F1-Score\s*:\s*([0-9.]+)",
            "AUC-ROC": r"AUC-ROC\s*:\s*([0-9.]+)"
        }

        for name, pattern in metric_map.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                metrics[name] = f"{value * 100:.2f}%" if value <= 1.0 else f"{value:.2f}%"
    except Exception:
        return metrics

    return metrics

# Load the trained model
results_dir = "training_results"
model_files = []
if os.path.isdir(results_dir):
    model_files = [
        filename
        for filename in os.listdir(results_dir)
        if filename.startswith("multi_cancer_cnn_model_epoch_") and filename.endswith(".keras")
    ]
model_files.sort(
    key=lambda filename: int(filename.rsplit("_", 1)[-1].replace(".keras", ""))
)
model_path = os.path.join(results_dir, model_files[-1]) if model_files else None

if model_path and os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from: {model_path}")
else:
    print("No trained model found. Please run train_evaluate.py first.")
    model = None

class_names = CLASS_NAMES

def predict_cancer_type(image):
    """
    Predict the cancer type from an image and return confidence scores.
    """
    if model is None:
        return "Error: No trained model available. Please train the model first.", {}
    
    try:
        # Preprocess the image
        if image is None:
            return "Error: No image provided.", {}
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Resize image to match model input
        image = image.resize(IMAGE_SIZE)
        
        image = image.convert("L")
        
        # Convert to numpy array and normalize
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Add batch dimension
        image_batch = np.expand_dims(image_array, axis=(0, -1))
        
        # Make prediction
        predictions = model.predict(image_batch, verbose=0)
        
        # Get confidence scores
        confidence_scores = tf.nn.softmax(predictions[0]).numpy()
        
        # Create confidence dictionary
        confidence_dict = {}
        for i, class_name in enumerate(class_names):
            confidence_dict[class_name] = float(confidence_scores[i])
        
        # Get predicted class
        predicted_class_idx = np.argmax(confidence_scores)
        predicted_class = class_names[predicted_class_idx]
        confidence = confidence_scores[predicted_class_idx]
        
        # Create result message
        result_message = f"Prediction: {predicted_class.upper()}\nConfidence: {confidence:.2%}"
        
        return result_message, confidence_dict
        
    except Exception as e:
        return f"Error during prediction: {str(e)}", {}

def create_interface():
    """
    Create and configure the Gradio interface.
    """
    # Custom CSS for styling
    css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
        max-width: 900px;
        margin: 0 auto;
    }
    .header {
        text-align: center;
        margin-bottom: 30px;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
    }
    .prediction-output {
        font-size: 18px;
        font-weight: bold;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    """
    
    # Create the interface
    with gr.Blocks(css=css, title="Multi-Cancer Image Classifier") as interface:
        gr.HTML("""
        <div class="header">
            <h1>Multi-Cancer Image Classifier</h1>
            <p>AI-powered image classification with a convolutional neural network</p>
            <p><em>Upload an image to classify it into one of the supported cancer types</em></p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### Upload Image")
                image_input = gr.Image(
                    type="pil",
                    label="Image",
                    height=400
                )
                
                predict_button = gr.Button(
                    "Classify Image",
                    variant="primary",
                    size="lg"
                )
                
            with gr.Column(scale=1):
                # Output section
                gr.Markdown("### Classification Results")
                
                prediction_output = gr.Textbox(
                    label="Prediction Result",
                    lines=3,
                    max_lines=5
                )
                
                confidence_plot = gr.Label(
                    label="Confidence Scores",
                    num_top_classes=len(class_names)
                )
        
        # Model information section
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Model Performance")
                metrics_output = gr.JSON(label="Evaluation Metrics")

            with gr.Column(scale=1):
                gr.Markdown("""
                ### ℹ️ Model Information
                - **Architecture**: Residual U-Net CNN
                - **Classes**: 8 cancer types
                - **Input Size**: 128x128 pixels
                
                ### ⚠️ Important Disclaimer
                This tool is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical advice.
                """)

        # Set up the prediction function
        predict_button.click(
            fn=predict_cancer_type,
            inputs=[image_input],
            outputs=[prediction_output, confidence_plot]
        )

        interface.load(
            fn=load_model_metrics,
            outputs=[metrics_output]
        )

    return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()
    
    print("Starting Multi-Cancer Image Classifier...")
    print("Model status:", "Loaded" if model is not None else "Not available")
    
    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
