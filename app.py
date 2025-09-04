import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance
import cv2

def load_model():
    try:
        print("Loading model...")
        
        # Try different model files in order of preference
        model_files = [
            'quick_trained_model.keras',
            'pretrained_medical_model.keras',
            'input_adapter_model.keras',
            'compatible_efficientnet.keras',
            'simple_effective_model.keras', 
            'working_model.keras',
            'pretrained_model.keras',
            'best_model.keras'
        ]
        
        for model_file in model_files:
            try:
                print(f"Trying to load {model_file}...")
                model = tf.keras.models.load_model(model_file, compile=False)
                print(f"✅ {model_file} loaded successfully!")
                return model
            except Exception as e:
                print(f"❌ {model_file} failed: {e}")
                continue
        
        # If all fail, create demo model
        print("All model files failed. Creating demo model...")
        return create_demo_model()
        
    except Exception as e:
        print(f"Error in load_model: {e}")
        return create_demo_model()

def create_demo_model():
    """Create a demo model for testing"""
    print("Creating a demo model for demonstration...")
    
    # Create a more sophisticated demo model that mimics your architecture
    input_layer = tf.keras.layers.Input(shape=(224, 224, 3))
    
    # Mimic EfficientNetB4-like architecture
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(3, activation='softmax')(x)
    
    demo_model = tf.keras.Model(inputs=input_layer, outputs=output)
    demo_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("Demo model created successfully!")
    print("⚠️  WARNING: This is a demo model with random weights. For production, fix the original model loading.")
    return demo_model

model = load_model()

def apply_clahe(image):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y_channel = yuv_image[:, :, 0]
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_channel_clahe = clahe.apply(y_channel)
    
    yuv_image[:, :, 0] = y_channel_clahe
    img_clahe = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
    
    return img_clahe

def enhance_image(image):
    """Apply image enhancement (contrast, sharpness, brightness)"""
    pil_img = Image.fromarray(image)  
    enhancer = ImageEnhance.Contrast(pil_img)
    image_enhanced = enhancer.enhance(1.5)
    
    enhancer = ImageEnhance.Sharpness(image_enhanced)
    image_enhanced = enhancer.enhance(1.5)
    
    enhancer = ImageEnhance.Brightness(image_enhanced)
    image_enhanced = enhancer.enhance(1.2)
    
    return np.array(image_enhanced)

def preprocess_image(image):
    """Preprocess image to match training pipeline exactly"""
    if image is None:
        return None
    
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert RGB to BGR for OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image = Image.fromarray(image)
            image = np.array(image.convert('RGB'))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        # PIL Image
        image = np.array(image.convert('RGB'))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Apply CLAHE
    image = apply_clahe(image)
    
    # Apply enhancement
    image = enhance_image(image)
    
    # Resize to match training size (224x224)
    image = cv2.resize(image, (224, 224))
    
    # Convert BGR back to RGB for model input
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0,1] range
    image = image.astype(np.float32) / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

def predict_lung_disease(image):
    if model is None:
        return "Error: Model not loaded"
    
    if image is None:
        return "Please upload an image"
    
    try:
        processed_image = preprocess_image(image)
        if processed_image is None:
            return "Error processing image"
        
        prediction = model.predict(processed_image)
        
        # Match your training classes exactly
        classes = ['Normal', 'Viral Pneumonia', 'Lung Opacity']
        
        confidence = float(np.max(prediction))
        predicted_class = classes[np.argmax(prediction)]
        
        # Create detailed result with all class probabilities
        class_probs = prediction[0]
        result = f"Prediction: {predicted_class}\nConfidence: {confidence:.2%}\n\nClass Probabilities:\n"
        for i, (class_name, prob) in enumerate(zip(classes, class_probs)):
            result += f"{class_name}: {prob:.2%}\n"
        
        return result, confidence, predicted_class
        
    except Exception as e:
        return f"Error during prediction: {str(e)}", 0.0, "Error"

def create_interface():
    with gr.Blocks(title="Lung Disease Detection", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🫁 Lung Disease Detection")
        gr.Markdown("Upload a chest X-ray image to detect lung diseases using AI. This model can classify between Normal, Viral Pneumonia, and Lung Opacity.")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    label="Upload Chest X-ray Image",
                    type="pil",
                    height=300
                )
                predict_btn = gr.Button("🔍 Analyze Image", variant="primary")
                
            with gr.Column():
                output_text = gr.Textbox(
                    label="Analysis Result",
                    lines=3,
                    interactive=False
                )
                confidence_gauge = gr.Number(
                    label="Confidence Level",
                    value=0,
                    interactive=False
                )
                predicted_class = gr.Textbox(
                    label="Predicted Class",
                    interactive=False
                )
        
        gr.Markdown("""
        ### Instructions:
        1. Upload a chest X-ray image (JPG, PNG, or other common formats)
        2. Click 'Analyze Image' to get the AI prediction
        3. The model will classify the image as Normal, Viral Pneumonia, or Lung Opacity
        4. Confidence level shows how certain the model is about its prediction
        5. All class probabilities are displayed for transparency
        """)
        
        predict_btn.click(
            fn=predict_lung_disease,
            inputs=[image_input],
            outputs=[output_text, confidence_gauge, predicted_class]
        )
        
        gr.Examples(
            examples=[],
            inputs=image_input,
            label="Example Images (Upload your own)"
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=False
    )
