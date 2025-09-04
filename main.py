"""
Main script to run the complete MLOps pipeline
"""
import os
import sys
import argparse
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup the environment and create necessary directories"""
    logger.info("Setting up environment...")
    
    from config import *
    
    # Create directories
    for dir_path in [MODEL_DIR, LOGS_DIR, ARTIFACTS_DIR]:
        dir_path.mkdir(exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
    
    logger.info("✅ Environment setup complete")

def run_data_pipeline():
    """Run the data preprocessing pipeline"""
    logger.info("🔄 Running data pipeline...")
    
    try:
        from data_pipeline import DataPipeline
        
        pipeline = DataPipeline()
        train_dataset, val_dataset, test_dataset, class_weights = pipeline.main()
        
        logger.info("✅ Data pipeline completed successfully")
        return train_dataset, val_dataset, test_dataset, class_weights
        
    except Exception as e:
        logger.error(f"❌ Data pipeline failed: {e}")
        raise

def run_training():
    """Run the model training pipeline"""
    logger.info("🚀 Starting model training...")
    
    try:
        from trainer import AdvancedTrainer
        
        trainer = AdvancedTrainer()
        model, results = trainer.run_training_pipeline(use_hyperopt=False)
        
        logger.info("✅ Training completed successfully")
        logger.info(f"Final Test Accuracy: {results['test_accuracy']:.4f}")
        
        return model, results
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

def run_hyperparameter_optimization():
    """Run hyperparameter optimization"""
    logger.info("🔧 Running hyperparameter optimization...")
    
    try:
        from trainer import AdvancedTrainer
        
        trainer = AdvancedTrainer()
        train_dataset, val_dataset, test_dataset, class_weights = trainer.load_data()
        
        best_params = trainer.hyperparameter_optimization(
            train_dataset, val_dataset, class_weights, n_trials=10
        )
        
        logger.info(f"✅ Hyperparameter optimization completed")
        logger.info(f"Best parameters: {best_params}")
        
        return best_params
        
    except Exception as e:
        logger.error(f"❌ Hyperparameter optimization failed: {e}")
        raise

def test_model():
    """Test the trained model"""
    logger.info("🧪 Testing trained model...")
    
    try:
        from gradio_app import LungDiseaseClassifier
        import numpy as np
        from PIL import Image
        
        # Load classifier
        classifier = LungDiseaseClassifier()
        
        # Create test image
        test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # Make prediction
        predicted_class, confidence, class_probabilities = classifier.predict(test_image)
        
        logger.info(f"✅ Model test successful")
        logger.info(f"Prediction: {predicted_class}, Confidence: {confidence:.2f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model test failed: {e}")
        return False

def launch_gradio_app():
    """Launch the Gradio app"""
    logger.info("🌐 Launching Gradio app...")
    
    try:
        from gradio_app import main as launch_app
        launch_app()
        
    except Exception as e:
        logger.error(f"❌ Failed to launch Gradio app: {e}")
        raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Lung Disease Classification MLOps Pipeline')
    parser.add_argument('--mode', choices=['data', 'train', 'hyperopt', 'test', 'app', 'full'], 
                       default='full', help='Pipeline mode to run')
    parser.add_argument('--hyperopt-trials', type=int, default=10, 
                       help='Number of hyperparameter optimization trials')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Lung Disease Classification MLOps Pipeline")
    logger.info(f"Mode: {args.mode}")
    
    try:
        # Setup environment
        setup_environment()
        
        if args.mode in ['data', 'full']:
            # Run data pipeline
            run_data_pipeline()
        
        if args.mode in ['hyperopt', 'full']:
            # Run hyperparameter optimization
            run_hyperparameter_optimization()
        
        if args.mode in ['train', 'full']:
            # Run training
            run_training()
        
        if args.mode in ['test', 'full']:
            # Test model
            if test_model():
                logger.info("✅ Model test passed")
            else:
                logger.error("❌ Model test failed")
                return
        
        if args.mode in ['app', 'full']:
            # Launch Gradio app
            launch_gradio_app()
        
        logger.info("🎉 Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
