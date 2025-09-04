"""
Advanced data pipeline for lung disease classification
"""
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import albumentations as A
from albumentations.pytorch import ToTensorV2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from config import *

class DataPipeline:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = Path(data_dir)
        self.class_names = DATASET_CONFIG["class_names"]
        self.image_size = DATASET_CONFIG["image_size"]
        self.setup_directories()
        
    def setup_directories(self):
        """Setup data directories"""
        self.normal_dir = self.data_dir / "Normal"
        self.pneumonia_dir = self.data_dir / "Viral Pneumonia"
        self.opacity_dir = self.data_dir / "Lung_Opacity"
        
    def apply_clahe(self, image):
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
    
    def enhance_image(self, image):
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
    
    def preprocess_image(self, image_path, apply_clahe=True, apply_enhancement=True):
        """Preprocess a single image"""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply CLAHE
        if apply_clahe:
            image = self.apply_clahe(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply enhancement
        if apply_enhancement:
            image = self.enhance_image(image)
        
        # Resize image
        image = cv2.resize(image, self.image_size)
        
        return image
    
    def load_dataset(self):
        """Load and preprocess the entire dataset"""
        print("Loading dataset...")
        
        images = []
        labels = []
        file_paths = []
        
        # Load Normal images
        print("Loading Normal images...")
        for img_path in self.normal_dir.glob("*.jpg"):
            try:
                image = self.preprocess_image(img_path)
                images.append(image)
                labels.append(0)  # Normal
                file_paths.append(str(img_path))
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        # Load Viral Pneumonia images
        print("Loading Viral Pneumonia images...")
        for img_path in self.pneumonia_dir.glob("*.jpg"):
            try:
                image = self.preprocess_image(img_path)
                images.append(image)
                labels.append(1)  # Viral Pneumonia
                file_paths.append(str(img_path))
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        # Load Lung Opacity images
        print("Loading Lung Opacity images...")
        for img_path in self.opacity_dir.glob("*.jpg"):
            try:
                image = self.preprocess_image(img_path)
                images.append(image)
                labels.append(2)  # Lung Opacity
                file_paths.append(str(img_path))
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        images = np.array(images)
        labels = np.array(labels)
        
        print(f"Dataset loaded: {len(images)} images, {len(set(labels))} classes")
        print(f"Class distribution: {np.bincount(labels)}")
        
        return images, labels, file_paths
    
    def create_dataframe(self, images, labels, file_paths):
        """Create a pandas DataFrame for analysis"""
        df = pd.DataFrame({
            'file_path': file_paths,
            'label': labels,
            'class_name': [self.class_names[label] for label in labels]
        })
        
        return df
    
    def analyze_dataset(self, df):
        """Analyze dataset distribution and quality"""
        print("\n=== Dataset Analysis ===")
        print(f"Total images: {len(df)}")
        print(f"Classes: {df['class_name'].value_counts().to_dict()}")
        
        # Plot class distribution
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        df['class_name'].value_counts().plot(kind='bar')
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        df['class_name'].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title('Class Distribution (Percentage)')
        
        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR / 'dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return df
    
    def split_dataset(self, images, labels, file_paths):
        """Split dataset into train, validation, and test sets"""
        print("Splitting dataset...")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test, paths_temp, paths_test = train_test_split(
            images, labels, file_paths,
            test_size=DATASET_CONFIG["test_split"],
            random_state=DATASET_CONFIG["random_seed"],
            stratify=labels
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val, paths_train, paths_val = train_test_split(
            X_temp, y_temp, paths_temp,
            test_size=DATASET_CONFIG["validation_split"] / (1 - DATASET_CONFIG["test_split"]),
            random_state=DATASET_CONFIG["random_seed"],
            stratify=y_temp
        )
        
        print(f"Train set: {len(X_train)} images")
        print(f"Validation set: {len(X_val)} images")
        print(f"Test set: {len(X_test)} images")
        
        return (X_train, y_train, paths_train), (X_val, y_val, paths_val), (X_test, y_test, paths_test)
    
    def compute_class_weights(self, y_train):
        """Compute class weights for imbalanced dataset"""
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = dict(enumerate(class_weights))
        print(f"Class weights: {class_weights}")
        return class_weights
    
    def create_augmentation_pipeline(self):
        """Create advanced augmentation pipeline using Albumentations"""
        train_transform = A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            A.Rotate(limit=AUGMENTATION_CONFIG["rotation_range"], p=0.5),
            A.ShiftScaleRotate(
                shift_limit=AUGMENTATION_CONFIG["width_shift_range"],
                scale_limit=AUGMENTATION_CONFIG["zoom_range"],
                rotate_limit=AUGMENTATION_CONFIG["rotation_range"],
                p=0.5
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=AUGMENTATION_CONFIG["brightness_range"],
                contrast_limit=AUGMENTATION_CONFIG["contrast_range"],
                p=0.3
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
                A.Blur(blur_limit=3, p=0.3),
            ], p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        val_transform = A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        return train_transform, val_transform
    
    def create_tf_datasets(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Create TensorFlow datasets with augmentation"""
        print("Creating TensorFlow datasets...")
        
        # Normalize images
        X_train = X_train.astype(np.float32) / 255.0
        X_val = X_val.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0
        
        # Create datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        
        # Apply augmentation to training set
        def augment_image(image, label):
            image = tf.cast(image, tf.uint8)
            # Apply random augmentation
            image = tf.image.random_brightness(image, 0.2)
            image = tf.image.random_contrast(image, 0.8, 1.2)
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_rotation(image, 0.1)
            image = tf.cast(image, tf.float32)
            return image, label
        
        train_dataset = train_dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Batch and prefetch
        train_dataset = train_dataset.batch(DATASET_CONFIG["batch_size"]).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(DATASET_CONFIG["batch_size"]).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(DATASET_CONFIG["batch_size"]).prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset, test_dataset

def main():
    """Main function to run data pipeline"""
    pipeline = DataPipeline()
    
    # Load dataset
    images, labels, file_paths = pipeline.load_dataset()
    
    # Create DataFrame
    df = pipeline.create_dataframe(images, labels, file_paths)
    
    # Analyze dataset
    pipeline.analyze_dataset(df)
    
    # Split dataset
    (X_train, y_train, paths_train), (X_val, y_val, paths_val), (X_test, y_test, paths_test) = pipeline.split_dataset(images, labels, file_paths)
    
    # Compute class weights
    class_weights = pipeline.compute_class_weights(y_train)
    
    # Create TensorFlow datasets
    train_dataset, val_dataset, test_dataset = pipeline.create_tf_datasets(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Save processed data
    np.save(ARTIFACTS_DIR / 'X_train.npy', X_train)
    np.save(ARTIFACTS_DIR / 'y_train.npy', y_train)
    np.save(ARTIFACTS_DIR / 'X_val.npy', X_val)
    np.save(ARTIFACTS_DIR / 'y_val.npy', y_val)
    np.save(ARTIFACTS_DIR / 'X_test.npy', X_test)
    np.save(ARTIFACTS_DIR / 'y_test.npy', y_test)
    
    # Save metadata
    metadata = {
        'class_weights': class_weights,
        'class_names': pipeline.class_names,
        'image_size': pipeline.image_size,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test)
    }
    
    import json
    with open(ARTIFACTS_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Data pipeline completed successfully!")
    print(f"Processed data saved to {ARTIFACTS_DIR}")
    
    return train_dataset, val_dataset, test_dataset, class_weights

if __name__ == "__main__":
    main()