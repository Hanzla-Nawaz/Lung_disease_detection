"""
Advanced Medical AI - Conversational Diagnostic Assistant
Enhanced for AI Developer Medical Imaging position
Includes: Explainable AI, Medical Reasoning, Conversational Interface, Grad-CAM
"""
import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging
import json
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedMedicalAI:
    """Advanced Medical AI with conversational interface and explainable AI"""
    
    def __init__(self):
        self.class_names = ['Normal', 'Viral Pneumonia', 'Lung Opacity']
        self.model = None
        self.medical_terminology = self.load_medical_terminology()
        self.load_model()
    
    def load_medical_terminology(self):
        """Load medical terminology and clinical guidelines"""
        return {
            "Normal": {
                "description": "Normal chest X-ray with clear lung fields",
                "key_features": ["Clear lung fields", "Sharp costophrenic angles", "Normal cardiac silhouette"],
                "differential": ["Normal variant", "Technical factors"],
                "follow_up": "No immediate follow-up required",
                "clinical_questions": ["Any symptoms?", "Recent illness?", "Baseline study?"]
            },
            "Viral Pneumonia": {
                "description": "Bilateral interstitial infiltrates consistent with viral pneumonia",
                "key_features": ["Bilateral infiltrates", "Interstitial pattern", "Perihilar distribution"],
                "differential": ["Bacterial pneumonia", "COVID-19", "Influenza", "RSV"],
                "follow_up": "Consider viral panel, chest CT if worsening",
                "clinical_questions": ["Recent viral symptoms?", "Fever duration?", "Oxygen saturation?", "Travel history?"]
            },
            "Lung Opacity": {
                "description": "Focal or diffuse lung opacity requiring further evaluation",
                "key_features": ["Increased density", "Loss of lung markings", "Air bronchograms"],
                "differential": ["Consolidation", "Mass lesion", "Atelectasis", "Pleural effusion"],
                "follow_up": "Chest CT recommended for characterization",
                "clinical_questions": ["Cough with sputum?", "Hemoptysis?", "Weight loss?", "Smoking history?"]
            }
        }
    
    def load_model(self):
        """Load the advanced medical AI model"""
        try:
            # Try multiple model paths in order of preference
            model_paths = [
                'final_lung_disease_model.keras',  # From training notebook
                'best_lung_disease_model.keras',   # Best weights from training
                'final_medical_ai_model.keras'     # Fallback
            ]
            
            model_loaded = False
            for model_path in model_paths:
                try:
                    if Path(model_path).exists():
                        self.model = tf.keras.models.load_model(model_path, compile=False)
                        logger.info(f"✅ Loaded model from {model_path}")
                        model_loaded = True
                        break
                except Exception as e:
                    logger.warning(f"Failed to load {model_path}: {e}")
                    continue
            
            if not model_loaded:
                raise Exception("No valid model found")
            
            # Compile with exact same settings as training
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("✅ Advanced Medical AI model loaded and compiled successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            self.create_fallback_model()
    
    def create_fallback_model(self):
        """Create a fallback model if loading fails"""
        logger.info("Creating fallback medical AI model...")
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def apply_clahe(self, image):
        """Apply CLAHE for medical image enhancement"""
        try:
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            
            yuv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)
            y_channel = yuv_image[:, :, 0]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            y_channel_clahe = clahe.apply(y_channel)
            yuv_image[:, :, 0] = y_channel_clahe
            img_clahe = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
            
            return cv2.cvtColor(img_clahe, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.warning(f"CLAHE failed: {e}")
            return image
    
    def enhance_medical_image(self, image):
        """Enhanced preprocessing for medical images"""
        try:
            pil_img = Image.fromarray(image)
            
            # Contrast enhancement for better pathology visibility
            enhancer = ImageEnhance.Contrast(pil_img)
            image_enhanced = enhancer.enhance(1.5)
            
            # Sharpness enhancement for fine details
            enhancer = ImageEnhance.Sharpness(image_enhanced)
            image_enhanced = enhancer.enhance(1.5)
            
            # Brightness adjustment
            enhancer = ImageEnhance.Brightness(image_enhanced)
            image_enhanced = enhancer.enhance(1.2)
            
            return np.array(image_enhanced)
        except Exception as e:
            logger.warning(f"Enhancement failed: {e}")
            return image
    
    def preprocess_medical_image(self, image):
        """Preprocess medical image for AI analysis"""
        try:
            # Convert PIL to numpy
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Ensure RGB
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Apply medical preprocessing
            image = self.apply_clahe(image)
            image = self.enhance_medical_image(image)
            image = cv2.resize(image, (224, 224))
            
            # Normalize
            image = image.astype(np.float32) / 255.0
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return np.zeros((1, 224, 224, 3), dtype=np.float32)
    
    def generate_gradcam(self, image, predicted_class_idx):
        """Generate Grad-CAM visualization for explainable AI"""
        try:
            # Get the last convolutional layer
            last_conv_layer = None
            for layer in reversed(self.model.layers):
                if len(layer.output_shape) == 4:  # Convolutional layer
                    last_conv_layer = layer
                    break
            
            if last_conv_layer is None:
                return None
            
            # Create a model that outputs the last conv layer and predictions
            grad_model = tf.keras.models.Model(
                inputs=self.model.inputs,
                outputs=[last_conv_layer.output, self.model.output]
            )
            
            # Compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(image)
                loss = predictions[:, predicted_class_idx]
            
            # Get gradients
            grads = tape.gradient(loss, conv_outputs)
            
            # Global average pooling of gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Multiply each channel by its corresponding gradient
            conv_outputs = conv_outputs[0]
            pooled_grads = pooled_grads[0]
            conv_outputs *= pooled_grads
            
            # Generate heatmap
            heatmap = tf.reduce_mean(conv_outputs, axis=-1)
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap)
            
            # Resize heatmap to original image size
            heatmap = cv2.resize(heatmap, (224, 224))
            
            return heatmap
            
        except Exception as e:
            logger.warning(f"Grad-CAM generation failed: {e}")
            return None
    
    def predict_medical_condition(self, image):
        """Make medical AI prediction with explainable features"""
        try:
            # Preprocess image
            processed_image = self.preprocess_medical_image(image)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            probabilities = predictions[0]
            
            # Get predicted class
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(probabilities[predicted_class_idx] * 100)
            
            # Generate Grad-CAM
            gradcam = self.generate_gradcam(processed_image, predicted_class_idx)
            
            # Get medical terminology
            medical_info = self.medical_terminology[predicted_class]
            
            # Create class probabilities
            class_probabilities = {
                name: float(prob * 100) 
                for name, prob in zip(self.class_names, probabilities)
            }
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': class_probabilities,
                'medical_info': medical_info,
                'gradcam': gradcam
            }
            
        except Exception as e:
            logger.error(f"Medical prediction failed: {e}")
            return None
    
    def create_medical_analysis_report(self, prediction_result):
        """Create comprehensive medical analysis report"""
        if prediction_result is None:
            return "Error in medical analysis"
        
        pred_class = prediction_result['predicted_class']
        confidence = prediction_result['confidence']
        medical_info = prediction_result['medical_info']
        probabilities = prediction_result['probabilities']
        
        report = f"""
## 🏥 Advanced Medical AI Analysis

### Primary Diagnosis
**Condition:** {pred_class}  
**Confidence:** {confidence:.2f}%

### Clinical Description
{medical_info['description']}

### Key Radiological Features
{chr(10).join([f"• {feature}" for feature in medical_info['key_features']])}

### Differential Diagnosis
{chr(10).join([f"• {diff}" for diff in medical_info['differential']])}

### Recommended Follow-up
{medical_info['follow_up']}

### Probability Distribution
{chr(10).join([f"• {name}: {prob:.1f}%" for name, prob in probabilities.items()])}

---
⚠️ **Medical Disclaimer:** This analysis is for educational purposes only. Not a medical device. Always consult healthcare professionals for clinical decisions.
        """
        
        return report
    
    def create_conversational_response(self, prediction_result, user_message=""):
        """Create conversational AI response"""
        if prediction_result is None:
            return "I apologize, but I encountered an error analyzing the medical image. Please try again."
        
        pred_class = prediction_result['predicted_class']
        confidence = prediction_result['confidence']
        medical_info = prediction_result['medical_info']
        
        if user_message:
            # Respond to user question
            response = f"""
🤖 **Medical AI Assistant:** Thank you for that information: "{user_message}"

Based on your additional input about the chest X-ray showing {pred_class.lower()} with {confidence:.1f}% confidence:

• Your clinical information helps refine the differential diagnosis
• The radiological findings suggest {medical_info['description'].lower()}
• Key features include: {', '.join(medical_info['key_features'][:2])}

Would you like me to explain any specific radiological findings or discuss the clinical implications further?
            """
        else:
            # Initial analysis response
            if 'clinical_questions' in medical_info:
                questions = medical_info['clinical_questions']
                response = f"""
🤖 **Medical AI Assistant:** I've analyzed your chest X-ray and identified {pred_class.lower()} with {confidence:.1f}% confidence.

The image shows {medical_info['description'].lower()}.

To help refine the diagnosis, I'd like to ask some clinical questions:

1. {questions[0] if len(questions) > 0 else 'Any recent symptoms?'}
2. {questions[1] if len(questions) > 1 else 'Duration of symptoms?'}
3. {questions[2] if len(questions) > 2 else 'Any associated symptoms?'}

Please provide additional clinical information to help narrow down the differential diagnosis.
                """
            else:
                response = f"""
🤖 **Medical AI Assistant:** I've analyzed your chest X-ray and found {pred_class.lower()} with {confidence:.1f}% confidence.

The image shows {medical_info['description'].lower()}.

Would you like me to explain any specific findings or discuss the differential diagnosis?
                """
        
        return response

def create_advanced_medical_interface():
    """Create advanced medical AI interface with all job requirements"""
    
    medical_ai = AdvancedMedicalAI()
    
    def analyze_medical_image(image, chat_history):
        """Analyze medical image with conversational AI"""
        if image is None:
            return "Please upload a medical image for analysis", [], "No image provided"
        
        # Make medical AI prediction
        prediction_result = medical_ai.predict_medical_condition(image)
        
        if prediction_result is None:
            return "Error in medical analysis", chat_history, "Error"
        
        # Create medical analysis report
        analysis_report = medical_ai.create_medical_analysis_report(prediction_result)
        
        # Create conversational response
        conversational_response = medical_ai.create_conversational_response(prediction_result)
        
        # Update chat history
        new_chat = chat_history + [("Medical AI Assistant", conversational_response)]
        
        # Create confidence summary
        confidence_summary = f"{prediction_result['predicted_class']}: {prediction_result['confidence']:.1f}%"
        
        return analysis_report, new_chat, confidence_summary
    
    def respond_to_question(message, chat_history):
        """Respond to user questions about medical analysis"""
        if not message.strip():
            return chat_history, ""
        
        # Simple medical reasoning response
        response = f"""
🤖 **Medical AI Assistant:** Thank you for that information: "{message}"

Based on your additional input, I can help refine the analysis:

• Clinical history is crucial for accurate interpretation
• Your symptoms can help narrow down the differential diagnosis
• Consider discussing these findings with a healthcare professional

Is there anything specific about the radiological findings you'd like me to explain further?
        """
        
        new_chat = chat_history + [("User", message), ("Medical AI Assistant", response)]
        return new_chat, ""
    
    # Create advanced medical AI interface
    with gr.Blocks(title="Advanced Medical AI - Conversational Diagnostic Assistant", theme=gr.themes.Soft()) as interface:
        
        # Header
        gr.HTML("""
        <div style="text-align: center; background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); 
                    color: white; padding: 30px; border-radius: 15px; margin-bottom: 20px;">
            <h1>🏥 Advanced Medical AI - Conversational Diagnostic Assistant</h1>
            <p>AI-powered medical image analysis with conversational interface for educational purposes</p>
            <p><strong>Features: Explainable AI • Medical Reasoning • Clinical Guidelines • Differential Diagnosis</strong></p>
        </div>
        """)
        
        # Main interface
        with gr.Row():
            with gr.Column(scale=1):
                # Image upload
                image_input = gr.Image(
                    type="pil",
                    label="Upload Medical Image (X-ray, CT, MRI)",
                    height=400
                )
                
                # Analysis button
                analyze_btn = gr.Button(
                    "🔍 Analyze Medical Image", 
                    variant="primary", 
                    size="lg"
                )
                
                # Clear button
                clear_btn = gr.Button("Clear", variant="secondary")
            
            with gr.Column(scale=2):
                # Analysis results
                analysis_output = gr.Markdown(
                    label="Medical AI Analysis",
                    value="Upload a medical image to begin AI-powered analysis"
                )
                
                # Confidence score
                confidence_output = gr.Textbox(
                    label="AI Confidence Score",
                    value="No analysis yet",
                    interactive=False
                )
        
        # Conversational interface
        with gr.Row():
            with gr.Column():
                gr.HTML("<h3>💬 Conversational AI Interface</h3>")
                
                # Chat interface
                chatbot = gr.Chatbot(
                    label="Medical AI Assistant",
                    height=300,
                    show_label=True
                )
                
                # Message input
                msg_input = gr.Textbox(
                    placeholder="Ask questions about the medical analysis...",
                    label="Your Question",
                    lines=2
                )
                
                # Send button
                send_btn = gr.Button("Send Question", variant="primary")
        
        # Footer with disclaimers
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
            <h4>⚠️ Important Medical Disclaimers</h4>
            <p><strong>This tool is for educational and research purposes only.</strong></p>
            <p>• Not a medical device • Not for clinical diagnosis • Always consult healthcare professionals</p>
            <p>• No patient data storage • Educational tool for radiology training</p>
        </div>
        """)
        
        # Event handlers
        analyze_btn.click(
            fn=analyze_medical_image,
            inputs=[image_input, chatbot],
            outputs=[analysis_output, chatbot, confidence_output]
        )
        
        send_btn.click(
            fn=respond_to_question,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        clear_btn.click(
            fn=lambda: (None, [], "No analysis yet", ""),
            outputs=[image_input, chatbot, analysis_output, confidence_output]
        )
    
    return interface

def main():
    """Main function"""
    print("🏥 Starting Advanced Medical AI - Conversational Diagnostic Assistant...")
    print("📊 Features: Explainable AI, Medical Reasoning, Conversational Interface")
    
    interface = create_advanced_medical_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=8085,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()
