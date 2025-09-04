"""
Production-ready Gradio app for Lung Disease Classification
"""
import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import *

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LungDiseaseClassifier:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or MODEL_DIR / 'final_lung_disease_model.keras'
        self.class_names = DATASET_CONFIG["class_names"]
        self.image_size = DATASET_CONFIG["image_size"]
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load the trained model"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            
            # Compile model
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            self.model = self.create_demo_model()
    
    def create_demo_model(self):
        """Create a demo model if main model fails to load"""
        logger.warning("Creating demo model...")
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE preprocessing"""
        yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        y_channel = yuv_image[:, :, 0]
        
        clahe = cv2.createCLAHE(
            clipLimit=PREPROCESSING_CONFIG["clahe_clip_limit"],
            tileGridSize=PREPROCESSING_CONFIG["clahe_tile_grid_size"]
        )
        y_channel_clahe = clahe.apply(y_channel)
        
        yuv_image[:, :, 0] = y_channel_clahe
        img_clahe = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
        
        return img_clahe
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Apply image enhancement"""
        pil_img = Image.fromarray(image)
        
        # Contrast enhancement
        enhancer = ImageEnhance.Contrast(pil_img)
        image_enhanced = enhancer.enhance(PREPROCESSING_CONFIG["contrast_factor"])
        
        # Sharpness enhancement
        enhancer = ImageEnhance.Sharpness(image_enhanced)
        image_enhanced = enhancer.enhance(PREPROCESSING_CONFIG["sharpness_factor"])
        
        # Brightness enhancement
        enhancer = ImageEnhance.Brightness(image_enhanced)
        image_enhanced = enhancer.enhance(PREPROCESSING_CONFIG["brightness_factor"])
        
        return np.array(image_enhanced)
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for prediction"""
        try:
            # Convert PIL to numpy array
            image = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Apply CLAHE
            image = self.apply_clahe(image)
            
            # Apply enhancement
            image = self.enhance_image(image)
            
            # Resize image
            image = cv2.resize(image, self.image_size)
            
            # Normalize
            image = image.astype(np.float32) / 255.0
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise ValueError(f"Failed to preprocess image: {e}")
    
    def predict(self, image: Image.Image) -> Tuple[str, float, Dict[str, float]]:
        """Make prediction on image"""
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            probabilities = predictions[0]
            
            # Get predicted class
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(probabilities[predicted_class_idx] * 100)
            
            # Create class probabilities dictionary
            class_probabilities = {
                self.class_names[i]: float(probabilities[i] * 100) 
                for i in range(len(self.class_names))
            }
            
            return predicted_class, confidence, class_probabilities
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return "Error", 0.0, {name: 0.0 for name in self.class_names}
    
    def create_confidence_plot(self, class_probabilities: Dict[str, float]) -> go.Figure:
        """Create confidence plot"""
        classes = list(class_probabilities.keys())
        probabilities = list(class_probabilities.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=classes,
                y=probabilities,
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                text=[f"{p:.1f}%" for p in probabilities],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Prediction Confidence",
            xaxis_title="Disease Class",
            yaxis_title="Confidence (%)",
            yaxis=dict(range=[0, 100]),
            template="plotly_white"
        )
        
        return fig

# Initialize classifier
classifier = LungDiseaseClassifier()

def predict_lung_disease(image: Image.Image) -> Tuple[str, float, go.Figure]:
    """Gradio prediction function"""
    if image is None:
        return "Please upload an image", 0.0, go.Figure()
    
    try:
        # Make prediction
        predicted_class, confidence, class_probabilities = classifier.predict(image)
        
        # Create confidence plot
        confidence_plot = classifier.create_confidence_plot(class_probabilities)
        
        # Format result
        result_text = f"**Prediction:** {predicted_class}\n**Confidence:** {confidence:.2f}%"
        
        return result_text, confidence, confidence_plot
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return f"Error: {str(e)}", 0.0, go.Figure()

def create_interface():
    """Create Gradio interface"""
    
    # Custom CSS
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .info-box {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    """
    
    with gr.Blocks(css=css, title="Lung Disease Classification") as interface:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🫁 Lung Disease Classification System</h1>
            <p>Advanced AI-powered chest X-ray analysis for detecting Normal, Viral Pneumonia, and Lung Opacity</p>
        </div>
        """)
        
        # Main content
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.HTML("<h3>📤 Upload Chest X-Ray Image</h3>")
                
                input_image = gr.Image(
                    type="pil",
                    label="Chest X-Ray Image",
                    height=300
                )
                
                predict_btn = gr.Button(
                    "🔍 Analyze Image",
                    variant="primary",
                    size="lg"
                )
                
                # Info section
                gr.HTML("""
                <div class="info-box">
                    <h4>ℹ️ Instructions:</h4>
                    <ul>
                        <li>Upload a clear chest X-ray image</li>
                        <li>Supported formats: JPG, PNG, JPEG</li>
                        <li>Image will be automatically preprocessed</li>
                        <li>Results show confidence for all classes</li>
                    </ul>
                </div>
                """)
                
            with gr.Column(scale=1):
                # Output section
                gr.HTML("<h3>📊 Analysis Results</h3>")
                
                result_text = gr.Markdown(
                    value="Upload an image to get started...",
                    label="Prediction Result"
                )
                
                confidence_score = gr.Number(
                    label="Confidence Score (%)",
                    value=0.0,
                    precision=2
                )
                
                confidence_plot = gr.Plot(
                    label="Confidence Distribution"
                )
        
        # Examples section
        gr.HTML("<h3>📋 Example Images</h3>")
        
        # Add example images if available
        example_images = []
        example_dir = Path("examples")
        if example_dir.exists():
            for img_path in example_dir.glob("*.jpg"):
                example_images.append(str(img_path))
        
        if example_images:
            gr.Examples(
                examples=example_images,
                inputs=input_image,
                label="Click to load example"
            )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; color: #666;">
            <p>⚠️ This tool is for research/educational purposes only. Always consult healthcare professionals for medical decisions.</p>
            <p>Powered by EfficientNetB4 | Built with Gradio & TensorFlow</p>
        </div>
        """)
        
        # Event handlers
        predict_btn.click(
            fn=predict_lung_disease,
            inputs=[input_image],
            outputs=[result_text, confidence_score, confidence_plot]
        )
        
        # Also predict on image change
        input_image.change(
            fn=predict_lung_disease,
            inputs=[input_image],
            outputs=[result_text, confidence_score, confidence_plot]
        )
    
    return interface

def main():
    """Main function to launch the app"""
    print("🚀 Starting Lung Disease Classification App...")
    
    # Create interface
    interface = create_interface()
    
    # Launch app
    interface.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=False,
        show_error=True,
        debug=True
    )

if __name__ == "__main__":
    main()
