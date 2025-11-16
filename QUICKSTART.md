# 🚀 Quick Start Guide

Get up and running with the Crop Disease Detection project in minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- 8GB+ RAM (16GB recommended)
- GPU (optional but recommended for training)

## Step-by-Step Setup

### 1. Clone or Navigate to Project

```bash
cd CropDiseaseDetection
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare Dataset

If you have the PlantVillage dataset in `dataset/raw/`:

```bash
python scripts/prepare_dataset.py
```

This will:
- Split data into train/val/test (70/15/15)
- Resize images to 224x224
- Create class mappings

**Note:** If you don't have the dataset, download it from [Kaggle PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

### 5. Train the Model

```bash
python train_model.py
```

**Training Time:**
- CPU: ~8-12 hours
- GPU: ~2-4 hours
- TPU: ~1-2 hours

**Tip:** For testing, you can reduce epochs in `train_model.py` (change `EPOCHS = 30` to `EPOCHS = 5`)

### 6. Run the Web App

```bash
streamlit run app/main.py
```

The app will open automatically in your browser at `http://localhost:8501`

### 7. (Optional) Convert to TensorFlow Lite

For mobile deployment:

```bash
python scripts/convert_to_tflite.py
```

## Using Pre-trained Model

If you have a pre-trained model:

1. Place `crop_disease_model.h5` in `model/` directory
2. Place `class_names.json` in `model/` directory
3. Run the Streamlit app:

```bash
streamlit run app/main.py
```

## Troubleshooting

### Issue: "Model not found"
- Ensure you've trained the model or have a pre-trained model in `model/` directory
- Check that `crop_disease_model.h5` exists

### Issue: "Out of memory"
- Reduce `BATCH_SIZE` in `train_model.py` (try 16 or 8)
- Use smaller image size (change `IMG_SIZE = 224` to `IMG_SIZE = 128`)

### Issue: "Dataset not found"
- Run `python scripts/prepare_dataset.py` first
- Ensure PlantVillage dataset is in `dataset/raw/`

### Issue: Import errors
- Activate virtual environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (macOS/Linux)
- Reinstall requirements: `pip install -r requirements.txt --upgrade`

## Next Steps

1. **Test the Model**: Upload various leaf images through the web app
2. **Improve Accuracy**: Train for more epochs or use data augmentation
3. **Deploy**: Convert to TFLite for mobile apps
4. **Extend**: Add more disease classes or crops

## Need Help?

- Check the main [README.md](README.md) for detailed documentation
- Review error messages carefully
- Ensure all dependencies are installed correctly

---

**Happy Farming! 🌱**

