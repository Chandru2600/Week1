"""
Create class_names.json from dataset or model
This script generates the class names file needed for the Streamlit app
"""

import json
from pathlib import Path

MODEL_DIR = Path("model")
DATASET_DIR = Path("dataset/processed")
RAW_DATASET_DIR = Path("dataset/raw")
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
MAPPINGS_PATH = DATASET_DIR / "class_mappings.json"


def create_class_names_from_mappings():
    """Create class_names.json from class_mappings.json"""
    if not MAPPINGS_PATH.exists():
        print(f"❌ Class mappings not found at {MAPPINGS_PATH}")
        return False
    
    print(f"📖 Loading class mappings from {MAPPINGS_PATH}...")
    with open(MAPPINGS_PATH, 'r') as f:
        mappings = json.load(f)
    
    num_classes = mappings['num_classes']
    class_names = [mappings['idx_to_class'][str(i)] for i in range(num_classes)]
    
    # Save class names
    MODEL_DIR.mkdir(exist_ok=True)
    with open(CLASS_NAMES_PATH, 'w') as f:
        json.dump(class_names, f, indent=2)
    
    print(f"✅ Created {CLASS_NAMES_PATH} with {num_classes} classes")
    return True


def create_class_names_from_raw_dataset():
    """Create class_names.json from raw dataset directory structure"""
    if not RAW_DATASET_DIR.exists():
        print(f"❌ Raw dataset not found at {RAW_DATASET_DIR}")
        return False
    
    print(f"📂 Scanning raw dataset directory...")
    class_dirs = [d for d in RAW_DATASET_DIR.iterdir() if d.is_dir()]
    class_dirs.sort()
    
    if not class_dirs:
        print(f"❌ No class directories found in {RAW_DATASET_DIR}")
        return False
    
    class_names = [d.name for d in class_dirs]
    
    # Save class names
    MODEL_DIR.mkdir(exist_ok=True)
    with open(CLASS_NAMES_PATH, 'w') as f:
        json.dump(class_names, f, indent=2)
    
    print(f"✅ Created {CLASS_NAMES_PATH} with {len(class_names)} classes")
    print(f"   Classes: {', '.join(class_names[:5])}...")
    return True


def create_class_names_from_model():
    """Try to extract class names from model (if possible)"""
    model_path = MODEL_DIR / "crop_disease_model.h5"
    
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return False
    
    print(f"🤖 Model found, but cannot extract class names from model file.")
    print(f"   Need to use dataset to create class names.")
    return False


def main():
    """Main function to create class_names.json"""
    print("=" * 60)
    print("Creating class_names.json")
    print("=" * 60)
    
    # Check if already exists
    if CLASS_NAMES_PATH.exists():
        print(f"✅ {CLASS_NAMES_PATH} already exists!")
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
        print(f"   Contains {len(class_names)} classes")
        return
    
    # Try different methods
    print("\n🔍 Searching for class information...")
    
    # Method 1: From processed dataset mappings
    if MAPPINGS_PATH.exists():
        if create_class_names_from_mappings():
            return
    
    # Method 2: From raw dataset structure
    if RAW_DATASET_DIR.exists():
        if create_class_names_from_raw_dataset():
            return
    
    # Method 3: From model (not possible, but check anyway)
    if create_class_names_from_model():
        return
    
    # If all methods fail
    print("\n❌ Could not create class_names.json")
    print("\nPlease ensure one of the following exists:")
    print(f"  1. {MAPPINGS_PATH} (from prepare_dataset.py)")
    print(f"  2. {RAW_DATASET_DIR} (raw dataset with class folders)")
    print("\nOr run: python scripts/prepare_dataset.py")


if __name__ == "__main__":
    main()

