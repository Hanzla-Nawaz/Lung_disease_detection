"""
Advanced model builder with latest best practices
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB4, EfficientNetB5
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, 
    LearningRateScheduler, TensorBoard
)
import numpy as np
from config import *

class AdvancedModelBuilder:
    def __init__(self, input_shape=MODEL_CONFIG["input_shape"], num_classes=DATASET_CONFIG["num_classes"]):
        self.input_shape = input_shape
        self.num_classes = num_classes
        
    def create_efficientnet_model(self, base_model_name="EfficientNetB4", dropout_rate=0.3):
        """Create EfficientNet model with advanced architecture"""
        print(f"Creating {base_model_name} model...")
        
        # Choose base model
        if base_model_name == "EfficientNetB4":
            base_model = EfficientNetB4(
                include_top=False,
                weights='imagenet',
                input_shape=self.input_shape
            )
        elif base_model_name == "EfficientNetB5":
            base_model = EfficientNetB5(
                include_top=False,
                weights='imagenet',
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unsupported base model: {base_model_name}")
        
        # Freeze initial layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False
            
        # Unfreeze last 20 layers for fine-tuning
        for layer in base_model.layers[-20:]:
            layer.trainable = True
        
        # Build model
        inputs = layers.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dropout for regularization
        x = layers.Dropout(dropout_rate)(x)
        
        # Dense layers with batch normalization
        x = layers.Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        
        return model
    
    def create_ensemble_model(self):
        """Create ensemble model with multiple EfficientNet variants"""
        print("Creating ensemble model...")
        
        # Create multiple base models
        models = []
        for base_name in ["EfficientNetB4", "EfficientNetB5"]:
            model = self.create_efficientnet_model(base_name)
            models.append(model)
        
        # Create ensemble
        inputs = layers.Input(shape=self.input_shape)
        
        # Get predictions from each model
        predictions = []
        for i, model in enumerate(models):
            # Rename layers to avoid conflicts
            for layer in model.layers[1:]:  # Skip input layer
                layer._name = f"{layer.name}_{i}"
            
            pred = model(inputs)
            predictions.append(pred)
        
        # Average predictions
        ensemble_output = layers.Average()(predictions)
        
        ensemble_model = Model(inputs, ensemble_output)
        
        return ensemble_model
    
    def compile_model(self, model, learning_rate=MODEL_CONFIG["learning_rate"]):
        """Compile model with advanced optimizer and metrics"""
        print("Compiling model...")
        
        # Use AdamW optimizer for better generalization
        optimizer = AdamW(
            learning_rate=learning_rate,
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        return model
    
    def create_callbacks(self, model_name="lung_disease_model"):
        """Create advanced callbacks for training"""
        callbacks = []
        
        # Model checkpoint
        checkpoint = ModelCheckpoint(
            filepath=MODEL_DIR / f"{model_name}_best.keras",
            monitor=MODEL_CONFIG["monitor"],
            mode=MODEL_CONFIG["mode"],
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=MODEL_CONFIG["factor"],
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=MODEL_CONFIG["patience"],
            min_delta=MODEL_CONFIG["min_delta"],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate scheduler
        def lr_schedule(epoch):
            """Learning rate schedule"""
            if epoch < 10:
                return MODEL_CONFIG["learning_rate"]
            elif epoch < 20:
                return MODEL_CONFIG["learning_rate"] * 0.5
            else:
                return MODEL_CONFIG["learning_rate"] * 0.1
        
        lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
        callbacks.append(lr_scheduler)
        
        # TensorBoard
        tensorboard = TensorBoard(
            log_dir=LOGS_DIR / model_name,
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        callbacks.append(tensorboard)
        
        return callbacks
    
    def build_model(self, model_type="efficientnet", ensemble=False):
        """Build the complete model"""
        if ensemble:
            model = self.create_ensemble_model()
        elif model_type == "efficientnet":
            model = self.create_efficientnet_model()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Compile model
        model = self.compile_model(model)
        
        # Print model summary
        model.summary()
        
        return model

def main():
    """Test model builder"""
    builder = AdvancedModelBuilder()
    
    # Create model
    model = builder.build_model(model_type="efficientnet")
    
    # Create callbacks
    callbacks = builder.create_callbacks()
    
    print("✅ Model builder completed successfully!")
    
    return model, callbacks

if __name__ == "__main__":
    main()
