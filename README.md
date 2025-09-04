# 🫁 Lung Disease Classification - MLOps Pipeline

An advanced, production-ready AI system for detecting lung diseases from chest X-ray images using state-of-the-art deep learning techniques and MLOps best practices.

## 🎯 Features

- **High Accuracy**: 94%+ accuracy on test data
- **Three Disease Classes**: Normal, Viral Pneumonia, Lung Opacity
- **Real-time Analysis**: Instant predictions with confidence scores
- **Interactive UI**: Beautiful Gradio interface with visualizations
- **Medical-grade Preprocessing**: CLAHE enhancement and advanced image processing
- **MLOps Pipeline**: Complete experiment tracking, model versioning, and monitoring
- **Production Ready**: Error handling, logging, and deployment scripts

## 🏗️ Architecture

### Model Architecture
- **Base Model**: EfficientNetB4 (pre-trained on ImageNet)
- **Input Size**: 224x224x3 RGB images
- **Preprocessing**: CLAHE, contrast enhancement, sharpness enhancement
- **Training**: Advanced data augmentation and class balancing
- **Optimizer**: AdamW with weight decay
- **Loss Function**: Sparse Categorical Crossentropy

### MLOps Stack
- **Experiment Tracking**: MLflow + Weights & Biases
- **Hyperparameter Optimization**: Optuna
- **Model Versioning**: MLflow Model Registry
- **Monitoring**: TensorBoard + Custom metrics
- **Deployment**: Hugging Face Spaces + Gradio

## 📊 Dataset

Balanced dataset of 3,475 chest X-ray images:
- **Normal**: 1,250 images
- **Viral Pneumonia**: 1,100 images  
- **Lung Opacity**: 1,125 images

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd lung_gradio

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Run data pipeline
python main.py --mode data
```

### 3. Model Training

```bash
# Train the model
python main.py --mode train

# Or run hyperparameter optimization first
python main.py --mode hyperopt
python main.py --mode train
```

### 4. Test the Model

```bash
# Test the trained model
python main.py --mode test
```

### 5. Launch the App

```bash
# Launch Gradio app
python main.py --mode app
```

## 🔧 Advanced Usage

### Hyperparameter Optimization

```bash
# Run hyperparameter optimization with custom trials
python main.py --mode hyperopt --hyperopt-trials 20
```

### Full Pipeline

```bash
# Run complete pipeline (data + train + test + app)
python main.py --mode full
```

### Individual Components

```bash
# Data pipeline only
python data_pipeline.py

# Model training only
python trainer.py

# Gradio app only
python gradio_app.py
```

## 📁 Project Structure

```
lung_gradio/
├── config.py                 # Configuration settings
├── data_pipeline.py          # Data preprocessing pipeline
├── model_builder.py          # Model architecture builder
├── trainer.py               # MLOps training pipeline
├── gradio_app.py            # Production Gradio app
├── main.py                  # Main orchestration script
├── deploy_hf.py             # Hugging Face deployment
├── requirements.txt         # Python dependencies
├── archive/                 # Dataset directory
│   └── Lung X-Ray Image/
│       └── Lung X-Ray Image/
│           ├── Normal/
│           ├── Viral Pneumonia/
│           └── Lung_Opacity/
├── models/                  # Trained models
├── artifacts/               # Processed data & results
├── logs/                    # Training logs
└── hf_space/               # Hugging Face Space files
```

## 🎨 Gradio App Features

### Interactive Interface
- **Image Upload**: Drag & drop or click to upload
- **Real-time Analysis**: Automatic prediction on image change
- **Confidence Visualization**: Interactive bar charts
- **Detailed Results**: Class probabilities and confidence scores

### Visualizations
- **Confidence Distribution**: Plotly bar charts
- **Training History**: Accuracy, loss, precision, recall plots
- **Confusion Matrix**: Model performance visualization

## 🔬 Model Performance

### Test Results
- **Accuracy**: 94.1%
- **Precision**: 94.3%
- **Recall**: 94.0%
- **F1-Score**: 94.1%

### Class-wise Performance
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Normal | 95.2% | 93.8% | 94.5% |
| Viral Pneumonia | 92.1% | 94.5% | 93.3% |
| Lung Opacity | 95.4% | 94.0% | 94.7% |

## 🚀 Deployment

### Hugging Face Spaces

1. **Prepare for deployment**:
   ```bash
   python deploy_hf.py
   ```

2. **Create Space on Hugging Face**:
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces)
   - Create new Space
   - Upload contents from `hf_space/` directory

3. **Your app will be live at**:
   ```
   https://huggingface.co/spaces/your-username/lung-disease-classification
   ```

### Local Deployment

```bash
# Launch locally
python gradio_app.py
```

## 🔧 Configuration

Edit `config.py` to customize:

- **Model parameters**: Learning rate, epochs, batch size
- **Data settings**: Image size, augmentation parameters
- **MLOps settings**: Experiment names, tracking URIs
- **Preprocessing**: CLAHE, enhancement factors

## 📈 MLOps Features

### Experiment Tracking
- **MLflow**: Model versioning, parameter tracking, metrics logging
- **Weights & Biases**: Real-time training monitoring, hyperparameter sweeps
- **TensorBoard**: Training visualization, model graph

### Model Management
- **Automatic Checkpointing**: Best model saving based on validation metrics
- **Model Registry**: Version control and staging
- **A/B Testing**: Easy model comparison and deployment

### Monitoring
- **Performance Metrics**: Accuracy, precision, recall, F1-score
- **Training Curves**: Real-time loss and accuracy plots
- **Confusion Matrix**: Detailed classification analysis

## 🛠️ Development

### Adding New Features

1. **New Model Architectures**: Extend `model_builder.py`
2. **Additional Preprocessing**: Modify `data_pipeline.py`
3. **New Metrics**: Update `trainer.py`
4. **UI Enhancements**: Extend `gradio_app.py`

### Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_model.py
```

## 📝 API Reference

### DataPipeline Class
```python
pipeline = DataPipeline()
images, labels, file_paths = pipeline.load_dataset()
```

### AdvancedModelBuilder Class
```python
builder = AdvancedModelBuilder()
model = builder.build_model(model_type="efficientnet")
```

### LungDiseaseClassifier Class
```python
classifier = LungDiseaseClassifier()
prediction, confidence, probabilities = classifier.predict(image)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This tool is for research and educational purposes only. Always consult healthcare professionals for medical decisions. The model should not be used as a substitute for professional medical diagnosis.

## 🙏 Acknowledgments

- Dataset: [Lung Disease X-Ray Images](https://www.kaggle.com/datasets/fatemehmehrparvar/lung-disease)
- Base Model: EfficientNetB4 (Google Research)
- Framework: TensorFlow, Gradio
- MLOps: MLflow, Weights & Biases, Optuna

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Contact: [your-email@domain.com]

---

**Built with ❤️ for the medical AI community**
