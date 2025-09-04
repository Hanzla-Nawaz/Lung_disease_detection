"""
Configuration file for the Lung Disease Classification project
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "archive" / "Lung X-Ray Image" / "Lung X-Ray Image"
MODEL_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Create directories if they don't exist
for dir_path in [MODEL_DIR, LOGS_DIR, ARTIFACTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Dataset configuration
DATASET_CONFIG = {
    "image_size": (224, 224),
    "batch_size": 32,
    "validation_split": 0.2,
    "test_split": 0.1,
    "random_seed": 42,
    "class_names": ["Normal", "Viral Pneumonia", "Lung Opacity"],
    "num_classes": 3
}

# Model configuration
MODEL_CONFIG = {
    "base_model": "EfficientNetB4",
    "input_shape": (224, 224, 3),
    "learning_rate": 2e-4,
    "epochs": 50,
    "patience": 10,
    "min_delta": 0.001,
    "factor": 0.5,
    "monitor": "val_accuracy",
    "mode": "max"
}

# Data augmentation configuration
AUGMENTATION_CONFIG = {
    "rotation_range": 15,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "horizontal_flip": True,
    "zoom_range": 0.1,
    "brightness_range": [0.8, 1.2],
    "contrast_range": [0.8, 1.2]
}

# MLOps configuration
MLOPS_CONFIG = {
    "experiment_name": "lung_disease_classification",
    "mlflow_tracking_uri": f"file://{ARTIFACTS_DIR}/mlruns",
    "wandb_project": "lung-disease-classification",
    "log_interval": 10
}

# Preprocessing configuration
PREPROCESSING_CONFIG = {
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid_size": (8, 8),
    "contrast_factor": 1.5,
    "sharpness_factor": 1.5,
    "brightness_factor": 1.2,
    "resize_method": "bilinear"
}
