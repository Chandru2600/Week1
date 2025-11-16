"""
Dataset Preparation Script for PlantVillage Dataset
Organizes images into train/validation/test splits and applies preprocessing
"""

import os
import shutil
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
import json
from tqdm import tqdm

# Configuration
RAW_DATA_DIR = Path("dataset/raw")
PROCESSED_DATA_DIR = Path("dataset/processed")
TRAIN_DIR = PROCESSED_DATA_DIR / "train"
VAL_DIR = PROCESSED_DATA_DIR / "val"
TEST_DIR = PROCESSED_DATA_DIR / "test"

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Image settings
IMG_SIZE = (224, 224)
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')


def get_all_images(class_dir):
    """Get all image files from a class directory"""
    images = []
    for ext in SUPPORTED_FORMATS:
        images.extend(list(class_dir.glob(f"*{ext}")))
    return images


def resize_and_save_image(src_path, dst_path, size=IMG_SIZE):
    """Resize image and save to destination"""
    try:
        img = Image.open(src_path)
        # Convert RGBA to RGB if necessary
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img = img.resize(size, Image.Resampling.LANCZOS)
        img.save(dst_path, 'JPEG', quality=95)
        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False


def prepare_dataset():
    """Main function to prepare the dataset"""
    print("=" * 60)
    print("PlantVillage Dataset Preparation")
    print("=" * 60)
    
    # Check if raw data directory exists
    if not RAW_DATA_DIR.exists():
        print(f"Error: Raw data directory not found at {RAW_DATA_DIR}")
        print("Please ensure the PlantVillage dataset is in dataset/raw/")
        return
    
    # Create processed directories
    for dir_path in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in RAW_DATA_DIR.iterdir() if d.is_dir()]
    class_dirs.sort()
    
    if not class_dirs:
        print(f"Error: No class directories found in {RAW_DATA_DIR}")
        return
    
    print(f"\nFound {len(class_dirs)} disease classes")
    
    # Create class mapping
    class_names = [d.name for d in class_dirs]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    
    # Save class mappings
    mappings = {
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'num_classes': len(class_names)
    }
    
    with open(PROCESSED_DATA_DIR / 'class_mappings.json', 'w') as f:
        json.dump(mappings, f, indent=2)
    
    print("\nClass Mappings:")
    for name, idx in list(class_to_idx.items())[:10]:
        print(f"  {idx}: {name}")
    if len(class_names) > 10:
        print(f"  ... and {len(class_names) - 10} more classes")
    
    # Process each class
    total_images = 0
    stats = {'train': 0, 'val': 0, 'test': 0}
    
    print("\n" + "=" * 60)
    print("Processing classes...")
    print("=" * 60)
    
    for class_dir in tqdm(class_dirs, desc="Classes"):
        class_name = class_dir.name
        images = get_all_images(class_dir)
        
        if not images:
            print(f"Warning: No images found in {class_name}")
            continue
        
        # Split images
        train_imgs, temp_imgs = train_test_split(
            images, test_size=(1 - TRAIN_RATIO), random_state=42
        )
        val_imgs, test_imgs = train_test_split(
            temp_imgs, test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)), random_state=42
        )
        
        # Create class directories in splits
        for split_dir, split_imgs in [(TRAIN_DIR, train_imgs), 
                                      (VAL_DIR, val_imgs), 
                                      (TEST_DIR, test_imgs)]:
            class_split_dir = split_dir / class_name
            class_split_dir.mkdir(exist_ok=True)
            
            for img_path in split_imgs:
                dst_path = class_split_dir / f"{img_path.stem}.jpg"
                if resize_and_save_image(img_path, dst_path):
                    stats[split_dir.name] += 1
                    total_images += 1
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Dataset Preparation Complete!")
    print("=" * 60)
    print(f"\nTotal images processed: {total_images}")
    print(f"Train images: {stats['train']} ({stats['train']/total_images*100:.1f}%)")
    print(f"Validation images: {stats['val']} ({stats['val']/total_images*100:.1f}%)")
    print(f"Test images: {stats['test']} ({stats['test']/total_images*100:.1f}%)")
    print(f"\nNumber of classes: {len(class_names)}")
    print(f"\nProcessed dataset saved to: {PROCESSED_DATA_DIR}")
    print(f"Class mappings saved to: {PROCESSED_DATA_DIR / 'class_mappings.json'}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    prepare_dataset()

