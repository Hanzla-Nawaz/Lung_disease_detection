"""
MLOps Training Pipeline with experiment tracking and hyperparameter optimization
"""
import os
import json
import mlflow
import mlflow.tensorflow
import wandb
from wandb.integration.keras import WandbMetricsLogger
import optuna
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import *
from model_builder import AdvancedModelBuilder
from data_pipeline import DataPipeline

class MLflowLogger(Callback):
    """Custom callback for MLflow logging"""
    def __init__(self, experiment_name):
        super().__init__()
        self.experiment_name = experiment_name
        
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            with mlflow.start_run(nested=True):
                mlflow.log_metrics({
                    f"epoch_{k}": v for k, v in logs.items()
                }, step=epoch)

class AdvancedTrainer:
    def __init__(self, experiment_name=MLOPS_CONFIG["experiment_name"]):
        self.experiment_name = experiment_name
        self.setup_mlops()
        
    def setup_mlops(self):
        """Setup MLOps tools"""
        print("Setting up MLOps tools...")
        
        # Setup MLflow
        mlflow.set_tracking_uri(MLOPS_CONFIG["mlflow_tracking_uri"])
        mlflow.set_experiment(self.experiment_name)
        
        # Setup Weights & Biases
        wandb.init(
            project=MLOPS_CONFIG["wandb_project"],
            config={
                **MODEL_CONFIG,
                **DATASET_CONFIG,
                **AUGMENTATION_CONFIG
            }
        )
        
        print("✅ MLOps tools initialized")
    
    def load_data(self):
        """Load preprocessed data"""
        print("Loading preprocessed data...")
        
        # Check if data exists
        if not (ARTIFACTS_DIR / 'X_train.npy').exists():
            print("Data not found. Running data pipeline...")
            pipeline = DataPipeline()
            train_dataset, val_dataset, test_dataset, class_weights = pipeline.main()
        else:
            # Load saved data
            X_train = np.load(ARTIFACTS_DIR / 'X_train.npy')
            y_train = np.load(ARTIFACTS_DIR / 'y_train.npy')
            X_val = np.load(ARTIFACTS_DIR / 'X_val.npy')
            y_val = np.load(ARTIFACTS_DIR / 'y_val.npy')
            X_test = np.load(ARTIFACTS_DIR / 'X_test.npy')
            y_test = np.load(ARTIFACTS_DIR / 'y_test.npy')
            
            # Load metadata
            with open(ARTIFACTS_DIR / 'metadata.json', 'r') as f:
                metadata = json.load(f)
            
            class_weights = metadata['class_weights']
            
            # Create datasets
            pipeline = DataPipeline()
            train_dataset, val_dataset, test_dataset = pipeline.create_tf_datasets(
                X_train, y_train, X_val, y_val, X_test, y_test
            )
        
        return train_dataset, val_dataset, test_dataset, class_weights
    
    def train_model(self, model, train_dataset, val_dataset, class_weights, trial=None):
        """Train the model with advanced techniques"""
        print("Starting model training...")
        
        # Create callbacks
        builder = AdvancedModelBuilder()
        callbacks = builder.create_callbacks()
        
        # Add MLOps callbacks
        mlflow_callback = MLflowLogger(self.experiment_name)
        wandb_callback = WandbMetricsLogger()
        callbacks.extend([mlflow_callback, wandb_callback])
        
        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(MODEL_CONFIG)
            mlflow.log_params(DATASET_CONFIG)
            
            # Train model
            history = model.fit(
                train_dataset,
                epochs=MODEL_CONFIG["epochs"],
                validation_data=val_dataset,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1
            )
            
            # Log model
            mlflow.tensorflow.log_model(
                model, 
                "model",
                registered_model_name=f"{self.experiment_name}_model"
            )
            
            return history
    
    def evaluate_model(self, model, test_dataset, class_names):
        """Evaluate model performance"""
        print("Evaluating model...")
        
        # Get predictions
        y_true = []
        y_pred = []
        y_prob = []
        
        for batch in test_dataset:
            images, labels = batch
            predictions = model.predict(images, verbose=0)
            
            y_true.extend(labels.numpy())
            y_pred.extend(np.argmax(predictions, axis=1))
            y_prob.extend(predictions)
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        
        # Calculate metrics
        test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Log metrics to MLflow
        with mlflow.start_run():
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("test_loss", test_loss)
            
            for class_name in class_names:
                if class_name in report:
                    mlflow.log_metric(f"precision_{class_name}", report[class_name]['precision'])
                    mlflow.log_metric(f"recall_{class_name}", report[class_name]['recall'])
                    mlflow.log_metric(f"f1_{class_name}", report[class_name]['f1-score'])
        
        # Log to wandb
        wandb.log({
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=class_names
            )
        })
        
        return {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'classification_report': report,
            'confusion_matrix': cm,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
    
    def plot_training_history(self, history, save_path=ARTIFACTS_DIR):
        """Plot training history"""
        print("Plotting training history...")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Training Loss')
        axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        if 'precision' in history.history:
            axes[1, 0].plot(history.history['precision'], label='Training Precision')
            axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Recall
        if 'recall' in history.history:
            axes[1, 1].plot(history.history['recall'], label='Training Recall')
            axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Log to wandb
        wandb.log({"training_history": wandb.Image(str(save_path / 'training_history.png'))})
    
    def plot_confusion_matrix(self, cm, class_names, save_path=ARTIFACTS_DIR):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Log to wandb
        wandb.log({"confusion_matrix": wandb.Image(str(save_path / 'confusion_matrix.png'))})
    
    def hyperparameter_optimization(self, train_dataset, val_dataset, class_weights, n_trials=20):
        """Perform hyperparameter optimization using Optuna"""
        print("Starting hyperparameter optimization...")
        
        def objective(trial):
            # Suggest hyperparameters
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            
            # Create model with suggested parameters
            builder = AdvancedModelBuilder()
            model = builder.create_efficientnet_model(dropout_rate=dropout_rate)
            model = builder.compile_model(model, learning_rate=learning_rate)
            
            # Train model
            history = model.fit(
                train_dataset,
                epochs=10,  # Reduced for faster optimization
                validation_data=val_dataset,
                class_weight=class_weights,
                verbose=0
            )
            
            # Return validation accuracy
            return max(history.history['val_accuracy'])
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        print(f"Best parameters: {study.best_params}")
        print(f"Best validation accuracy: {study.best_value}")
        
        return study.best_params
    
    def run_training_pipeline(self, use_hyperopt=False):
        """Run the complete training pipeline"""
        print("🚀 Starting MLOps Training Pipeline...")
        
        # Load data
        train_dataset, val_dataset, test_dataset, class_weights = self.load_data()
        
        # Hyperparameter optimization
        if use_hyperopt:
            best_params = self.hyperparameter_optimization(train_dataset, val_dataset, class_weights)
            print(f"Using optimized parameters: {best_params}")
        else:
            best_params = MODEL_CONFIG
        
        # Build model
        builder = AdvancedModelBuilder()
        model = builder.build_model()
        
        # Train model
        history = self.train_model(model, train_dataset, val_dataset, class_weights)
        
        # Evaluate model
        results = self.evaluate_model(model, test_dataset, DATASET_CONFIG["class_names"])
        
        # Plot results
        self.plot_training_history(history)
        self.plot_confusion_matrix(results['confusion_matrix'], DATASET_CONFIG["class_names"])
        
        # Save final model
        model.save(MODEL_DIR / 'final_lung_disease_model.keras')
        print(f"✅ Final model saved to {MODEL_DIR / 'final_lung_disease_model.keras'}")
        
        # Print results
        print(f"\n🎯 Final Results:")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"Test Loss: {results['test_loss']:.4f}")
        
        # Close wandb
        wandb.finish()
        
        return model, results

def main():
    """Main training function"""
    trainer = AdvancedTrainer()
    model, results = trainer.run_training_pipeline(use_hyperopt=False)
    
    return model, results

if __name__ == "__main__":
    main()
