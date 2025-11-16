"""
Setup Verification Script
Checks if all required files and directories are in place
"""

from pathlib import Path
import sys

def verify_setup():
    """Verify project setup"""
    print("=" * 60)
    print("Project Setup Verification")
    print("=" * 60)
    
    errors = []
    warnings = []
    
    # Required directories
    required_dirs = [
        "app",
        "scripts",
        "model",
        "dataset"
    ]
    
    print("\n[DIRECTORIES] Checking directories...")
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"  [OK] {dir_name}/")
        else:
            print(f"  [ERROR] {dir_name}/ (missing)")
            errors.append(f"Directory '{dir_name}' not found")
    
    # Required files
    required_files = [
        "README.md",
        "requirements.txt",
        "train_model.py",
        "app/main.py",
        "app/remedies.py",
        "scripts/prepare_dataset.py",
        "scripts/convert_to_tflite.py"
    ]
    
    print("\n[FILES] Checking files...")
    for file_name in required_files:
        file_path = Path(file_name)
        if file_path.exists():
            size = file_path.stat().st_size
            if size > 0:
                print(f"  [OK] {file_name} ({size} bytes)")
            else:
                print(f"  [WARN] {file_name} (empty)")
                warnings.append(f"File '{file_name}' is empty")
        else:
            print(f"  [ERROR] {file_name} (missing)")
            errors.append(f"File '{file_name}' not found")
    
    # Optional but recommended
    print("\n[OPTIONAL] Checking optional files...")
    optional_files = [
        ".gitignore",
        "QUICKSTART.md"
    ]
    
    for file_name in optional_files:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"  [OK] {file_name}")
        else:
            print(f"  [WARN] {file_name} (recommended)")
            warnings.append(f"Optional file '{file_name}' not found")
    
    # Check dataset
    print("\n[DATASET] Checking dataset...")
    dataset_raw = Path("dataset/raw")
    if dataset_raw.exists():
        class_dirs = [d for d in dataset_raw.iterdir() if d.is_dir()]
        if class_dirs:
            print(f"  [OK] Found {len(class_dirs)} disease classes in dataset/raw/")
        else:
            print(f"  [WARN] dataset/raw/ exists but is empty")
            warnings.append("Dataset directory is empty")
    else:
        print(f"  [WARN] dataset/raw/ not found (dataset preparation needed)")
        warnings.append("Dataset not found - run prepare_dataset.py after adding data")
    
    # Check model
    print("\n[MODEL] Checking model...")
    model_file = Path("model/crop_disease_model.h5")
    if model_file.exists():
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"  [OK] Trained model found ({size_mb:.2f} MB)")
    else:
        print(f"  [WARN] No trained model found (run train_model.py)")
        warnings.append("Model not trained yet")
    
    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    if errors:
        print(f"\n[ERRORS] Found {len(errors)} error(s):")
        for error in errors:
            print(f"   - {error}")
        print("\n[WARNING] Please fix these errors before proceeding.")
    
    if warnings:
        print(f"\n[WARNINGS] Found {len(warnings)} warning(s):")
        for warning in warnings:
            print(f"   - {warning}")
        print("\n[INFO] These are recommendations but won't prevent the project from running.")
    
    if not errors and not warnings:
        print("\n[SUCCESS] All checks passed! Project setup is complete.")
        return True
    elif not errors:
        print("\n[SUCCESS] No critical errors. Project should work, but check warnings above.")
        return True
    else:
        print("\n[ERROR] Critical errors found. Please fix them before proceeding.")
        return False


if __name__ == "__main__":
    success = verify_setup()
    sys.exit(0 if success else 1)

