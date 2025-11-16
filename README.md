# 🌱 Crop Disease Detection using Deep Learning

A comprehensive deep learning solution for detecting crop diseases from leaf images, helping farmers identify plant health issues early and providing organic treatment recommendations.

## 📋 Project Overview

This project leverages state-of-the-art deep learning techniques to automatically detect and classify crop diseases from leaf images. Using transfer learning with EfficientNetB0, the system can identify multiple diseases across various crops including tomatoes, apples, corn, potatoes, grapes, and more.

### 🎯 Goals

- **Early Disease Detection**: Enable farmers to identify crop diseases at an early stage through image analysis
- **Accessibility**: Provide an easy-to-use web interface that doesn't require technical expertise
- **Organic Solutions**: Offer sustainable, organic treatment recommendations for each detected disease
- **Scalability**: Support mobile deployment through TensorFlow Lite conversion
- **Accuracy**: Achieve high classification accuracy using transfer learning on EfficientNetB0

### 🌍 Sustainability Impact

1. **Reduced Pesticide Use**: Early detection allows for targeted, minimal intervention, reducing overall pesticide usage
2. **Organic Alternatives**: Promotes organic farming practices through natural remedy suggestions
3. **Crop Yield Protection**: Early disease identification helps prevent crop loss, ensuring food security
4. **Resource Efficiency**: Reduces water and fertilizer waste by addressing issues before they spread
5. **Economic Benefits**: Helps small-scale farmers protect their livelihoods through timely intervention
6. **Environmental Protection**: Minimizes chemical runoff and soil contamination

## 📁 Project Structure

```
CropDiseaseDetection/
│
├── app/                          # Streamlit web application
│   ├── __init__.py
│   ├── main.py                  # Main Streamlit app
│   └── remedies.py              # Organic remedy database
│
├── dataset/                      # Dataset directory
│   └── raw/                     # Raw PlantVillage dataset
│       ├── Apple___Apple_scab/
│       ├── Apple___Black_rot/
│       └── ... (other disease folders)
│
├── model/                        # Saved models
│   ├── crop_disease_model.h5    # Trained model
│   └── crop_disease_model.tflite # TensorFlow Lite model (optional)
│
├── scripts/                      # Utility scripts
│   ├── prepare_dataset.py       # Dataset preparation script
│   └── convert_to_tflite.py     # TensorFlow Lite conversion
│
├── train_model.py               # Main training script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Environment Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for cloning)

### Step 1: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
python -c "import streamlit; print('Streamlit installed successfully')"
```

## 📊 Dataset Preparation

The project uses the PlantVillage dataset, which contains images of healthy and diseased crop leaves.

### Option 1: Using Existing Dataset

If you already have the PlantVillage dataset in `dataset/raw/`, run:

```bash
python scripts/prepare_dataset.py
```

This script will:
- Organize images into train/validation/test splits (70/15/15)
- Resize images to 224x224 pixels
- Apply data augmentation
- Create class mappings

### Option 2: Download PlantVillage Dataset

1. Visit [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
2. Download and extract to `dataset/raw/`
3. Run the preparation script:

```bash
python scripts/prepare_dataset.py
```

## 🧠 Model Architecture

The project uses **EfficientNetB0** as the base model with transfer learning:

- **Base Model**: EfficientNetB0 (pre-trained on ImageNet)
- **Input Size**: 224x224x3 (RGB images)
- **Output**: Multi-class classification (38 disease classes)
- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy, Precision, Recall, F1-Score

### Transfer Learning Strategy

1. Load pre-trained EfficientNetB0 (ImageNet weights)
2. Freeze base layers initially
3. Add custom classification head
4. Fine-tune on crop disease dataset
5. Unfreeze top layers for final training

## 🏋️ Training the Model

### Basic Training

```bash
python train_model.py
```

### Training with Custom Parameters

Edit `train_model.py` to modify:
- Batch size
- Number of epochs
- Learning rate
- Image size
- Data augmentation parameters

### Training Output

The training process will:
- Display training progress and metrics
- Save the best model to `model/crop_disease_model.h5`
- Generate training history plots
- Save class labels mapping

### Expected Training Time

- **CPU**: ~8-12 hours (depending on hardware)
- **GPU**: ~2-4 hours (recommended)
- **TPU**: ~1-2 hours (if available)

## 🌐 Running the Streamlit Web App

### Start the Application

```bash
streamlit run app/main.py
```

The app will open in your default web browser at `http://localhost:8501`

### Features

1. **Image Upload**: Drag and drop or browse to upload leaf images
2. **Real-time Prediction**: Instant disease classification with confidence scores
3. **Visual Results**: Display predicted disease with probability
4. **Organic Remedies**: Get natural treatment recommendations for each disease
5. **History**: View prediction history
6. **Mobile Responsive**: Works on desktop, tablet, and mobile devices

### Usage

1. Upload a leaf image (JPG, PNG, JPEG formats)
2. Click "Predict Disease"
3. View the prediction results
4. Read organic remedy suggestions
5. Download or share results

## 📱 Mobile Deployment (Optional)

### Convert to TensorFlow Lite

```bash
python scripts/convert_to_tflite.py
```

This creates `model/crop_disease_model.tflite` optimized for mobile devices.

### Integration

The `.tflite` model can be integrated into:
- Android apps (using TensorFlow Lite Android API)
- iOS apps (using TensorFlow Lite iOS API)
- Edge devices (Raspberry Pi, etc.)

## 🔧 Configuration

### Model Parameters

Edit `train_model.py` to adjust:
- `IMG_SIZE = 224` - Input image dimensions
- `BATCH_SIZE = 32` - Training batch size
- `EPOCHS = 30` - Number of training epochs
- `LEARNING_RATE = 0.001` - Initial learning rate

### App Configuration

Edit `app/main.py` to customize:
- Theme colors
- Page title and description
- Image upload limits
- Display preferences

## 📈 Model Performance

Expected performance metrics:
- **Accuracy**: 95-98% on test set
- **Precision**: 94-97%
- **Recall**: 93-96%
- **F1-Score**: 94-97%

*Note: Actual performance depends on dataset quality and training configuration*

## 🛠️ Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce batch size in `train_model.py`
   - Use smaller image size
   - Enable mixed precision training

2. **Slow Training**
   - Use GPU acceleration
   - Reduce number of epochs for testing
   - Use smaller dataset subset

3. **Import Errors**
   - Ensure virtual environment is activated
   - Reinstall requirements: `pip install -r requirements.txt --upgrade`

4. **Model Not Loading**
   - Check model file path
   - Verify model file exists in `model/` directory
   - Ensure TensorFlow version compatibility

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional disease classes
- Better data augmentation techniques
- Model optimization
- UI/UX improvements
- Mobile app development

## 📄 License

This project is open source and available for educational and research purposes.

## 🙏 Acknowledgments

- **PlantVillage Dataset**: For providing comprehensive crop disease images
- **TensorFlow Team**: For EfficientNet architecture
- **Streamlit**: For the amazing web framework
- **Open Source Community**: For continuous support and improvements

## 📧 Contact

For questions, issues, or suggestions, please open an issue on the project repository.

---

**Made with ❤️ for sustainable agriculture**

$$