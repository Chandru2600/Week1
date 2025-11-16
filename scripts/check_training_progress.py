"""
Training Progress Checker
Analyzes training log to help decide if training should continue
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

TRAINING_LOG = Path("model/training_log.csv")
MODEL_PATH = Path("model/crop_disease_model.h5")


def check_progress():
    """Check training progress and provide recommendations"""
    print("=" * 60)
    print("Training Progress Analysis")
    print("=" * 60)
    
    # Check if log exists
    if not TRAINING_LOG.exists():
        print("\n❌ Training log not found. Training may not have started yet.")
        return
    
    # Load log
    try:
        log = pd.read_csv(TRAINING_LOG)
        if len(log) == 0:
            print("\n⚠️  Training log is empty. Training may not have started.")
            return
    except Exception as e:
        print(f"\n❌ Error reading training log: {e}")
        return
    
    # Get latest epoch
    latest_epoch = log['epoch'].iloc[-1]
    total_epochs = 30  # From train_model.py
    
    print(f"\n📊 Current Status:")
    print(f"   Epochs completed: {latest_epoch + 1} / {total_epochs}")
    print(f"   Progress: {(latest_epoch + 1) / total_epochs * 100:.1f}%")
    
    # Get latest metrics
    latest = log.iloc[-1]
    print(f"\n📈 Latest Metrics (Epoch {latest_epoch}):")
    print(f"   Training Accuracy:   {latest['accuracy']:.4f} ({latest['accuracy']*100:.2f}%)")
    print(f"   Validation Accuracy: {latest['val_accuracy']:.4f} ({latest['val_accuracy']*100:.2f}%)")
    print(f"   Training Loss:   {latest['loss']:.4f}")
    print(f"   Validation Loss: {latest['val_loss']:.4f}")
    print(f"   Top-3 Accuracy:  {latest['val_top_3_accuracy']:.4f} ({latest['val_top_3_accuracy']*100:.2f}%)")
    
    # Get best metrics
    best_val_acc_epoch = log['val_accuracy'].idxmax()
    best_val_acc = log['val_accuracy'].max()
    
    print(f"\n🏆 Best Performance:")
    print(f"   Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"   Achieved at Epoch: {int(log['epoch'].iloc[best_val_acc_epoch])}")
    
    # Analyze trends
    print(f"\n📉 Trend Analysis:")
    
    if len(log) >= 5:
        # Last 5 epochs trend
        recent = log.tail(5)
        val_acc_trend = recent['val_accuracy'].values
        
        # Check if improving
        if val_acc_trend[-1] > val_acc_trend[0]:
            improvement = val_acc_trend[-1] - val_acc_trend[0]
            print(f"   ✅ Validation accuracy improving: +{improvement:.4f} over last 5 epochs")
            print(f"   💡 Recommendation: Continue training")
        elif val_acc_trend[-1] == val_acc_trend[0]:
            print(f"   ⚠️  Validation accuracy plateauing (no change)")
            print(f"   💡 Recommendation: EarlyStopping will stop soon if no improvement")
        else:
            decline = val_acc_trend[0] - val_acc_trend[-1]
            print(f"   ⚠️  Validation accuracy declining: -{decline:.4f} over last 5 epochs")
            print(f"   💡 Recommendation: EarlyStopping will restore best weights")
    else:
        print(f"   📊 Not enough data yet (need 5+ epochs for trend analysis)")
    
    # Check for overfitting
    if len(log) > 0:
        latest_gap = latest['accuracy'] - latest['val_accuracy']
        if latest_gap > 0.10:  # 10% gap
            print(f"\n⚠️  Overfitting Warning:")
            print(f"   Large gap between training ({latest['accuracy']*100:.2f}%) and validation ({latest['val_accuracy']*100:.2f}%) accuracy")
            print(f"   Gap: {latest_gap*100:.2f}%")
        else:
            print(f"\n✅ No overfitting detected (gap: {latest_gap*100:.2f}%)")
    
    # Phase detection
    print(f"\n🔄 Training Phase:")
    if latest_epoch < 10:
        print(f"   Phase 1: Frozen Base Model Training")
        print(f"   Status: Training new classification layers")
        print(f"   Remaining: {10 - latest_epoch - 1} epochs until fine-tuning")
    elif latest_epoch < 30:
        print(f"   Phase 2: Fine-tuning (Unfrozen Top Layers)")
        print(f"   Status: Fine-tuning EfficientNetB0")
        print(f"   Remaining: {30 - latest_epoch - 1} epochs")
    else:
        print(f"   ✅ Training Complete!")
    
    # Recommendations
    print(f"\n" + "=" * 60)
    print("💡 Recommendations:")
    print("=" * 60)
    
    if best_val_acc >= 0.95:
        print("✅ Excellent! Validation accuracy ≥ 95%")
        print("   Your model is performing exceptionally well.")
        print("   You can stop training if needed, or continue for marginal gains.")
    elif best_val_acc >= 0.90:
        print("✅ Very Good! Validation accuracy ≥ 90%")
        print("   Your model is production-ready.")
        print("   Continuing training may improve to 95%+ accuracy.")
    elif best_val_acc >= 0.85:
        print("✅ Good! Validation accuracy ≥ 85%")
        print("   Model is usable but could be better.")
        print("   Recommend continuing training to reach 90%+.")
    else:
        print("⚠️  Validation accuracy < 85%")
        print("   Model needs more training.")
        print("   Continue training to improve performance.")
    
    # Check if model exists
    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"\n💾 Saved Model:")
        print(f"   Location: {MODEL_PATH}")
        print(f"   Size: {size_mb:.2f} MB")
        print(f"   Status: ✅ Ready to use!")
    else:
        print(f"\n⚠️  Model file not found at {MODEL_PATH}")
    
    # Early stopping status
    if len(log) >= 5:
        recent_val_acc = log['val_accuracy'].tail(5).values
        if len(set(recent_val_acc)) == 1:  # All same values
            print(f"\n⚠️  Early Stopping Alert:")
            print(f"   Validation accuracy unchanged for 5 epochs")
            print(f"   EarlyStopping will trigger soon (patience=5)")
        elif all(recent_val_acc[i] >= recent_val_acc[i+1] for i in range(len(recent_val_acc)-1)):
            print(f"\n⚠️  Early Stopping Alert:")
            print(f"   Validation accuracy declining")
            print(f"   EarlyStopping will restore best weights soon")
    
    print(f"\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("1. Continue training: python train_model.py")
    print("2. Use current model: streamlit run app/main.py")
    print("3. Check this again: python scripts/check_training_progress.py")
    print("=" * 60)


if __name__ == "__main__":
    check_progress()

