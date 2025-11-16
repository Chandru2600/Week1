# 🌐 Network Error Handling Guide

## What Happens During Network Errors

### Scenario 1: Network Error During Weight Download (Startup)

**When it happens:**
- At the very beginning when the script tries to download EfficientNetB0 ImageNet weights
- This is a one-time download (~30-50 MB)

**What happens:**
```
❌ Error downloading weights: <connection error>
Trying to use cached weights or continue without pre-trained weights...
⚠️  Warning: Using random initialization instead of pre-trained weights.
   Training will take longer and may achieve lower accuracy.
```

**Solutions:**

1. **Wait and Retry:**
   ```bash
   # Simply run again - TensorFlow caches downloaded weights
   python train_model.py
   ```

2. **Check Internet Connection:**
   - Ensure you have stable internet
   - Check firewall/proxy settings
   - Try downloading manually from: https://storage.googleapis.com/keras-applications/

3. **Use Cached Weights:**
   - TensorFlow automatically caches weights in `~/.keras/models/`
   - If you've downloaded before, it will use cached version
   - Location: `C:\Users\<username>\.keras\models\` (Windows)

4. **Download Weights Manually:**
   - Download EfficientNetB0 weights manually
   - Place in `.keras/models/` directory
   - Script will use cached version

5. **Continue Without Pre-trained Weights:**
   - The script will automatically fall back to random initialization
   - Training will take longer (2-3x more epochs needed)
   - Final accuracy may be 5-10% lower

### Scenario 2: Network Error During Training

**When it happens:**
- After model is built and training has started
- During an epoch (mid-training)

**What happens:**
```
Epoch 5/10
500/1698 [=========>................] - ETA: 2:30:00
❌ Error during training: <network error>
💾 Best model so far saved to: model/crop_disease_model.h5
You can resume training or use the saved model.
```

**Recovery Options:**

1. **Use Saved Model:**
   - The `ModelCheckpoint` callback saves the best model after each epoch
   - Your best model is already saved in `model/crop_disease_model.h5`
   - You can use it immediately for predictions

2. **Resume Training:**
   - Check `model/training_log.csv` to see how many epochs completed
   - Modify the script to load the saved model and continue from last epoch
   - Or simply restart - the checkpoint saves the best weights

3. **Check Training Progress:**
   ```python
   import pandas as pd
   log = pd.read_csv('model/training_log.csv')
   print(log.tail())  # See last few epochs
   ```

### Scenario 3: Interrupted Training (Ctrl+C)

**What happens:**
```
⚠️  Training interrupted by user.
💾 Best model saved to: model/crop_disease_model.h5
You can resume training later or use the saved model.
```

**This is safe!** The best model is already saved.

## Built-in Protection Mechanisms

### 1. ModelCheckpoint Callback
- **Saves best model** after each epoch based on validation accuracy
- **Location:** `model/crop_disease_model.h5`
- **Frequency:** After every epoch
- **What's saved:** Best weights so far (not just last epoch)

### 2. CSVLogger Callback
- **Saves training history** to `model/training_log.csv`
- **Contains:** Loss, accuracy, validation metrics for each epoch
- **Useful for:** Checking progress, resuming training

### 3. EarlyStopping Callback
- **Stops training** if validation accuracy doesn't improve for 5 epochs
- **Restores best weights** automatically
- **Prevents overfitting** and saves time

## How to Resume Training After Network Error

### Option 1: Use the Saved Model (Recommended)

The saved model is already trained and ready to use:

```bash
# Run the Streamlit app with saved model
streamlit run app/main.py
```

### Option 2: Resume Training from Checkpoint

Create a resume script or modify `train_model.py`:

```python
# Load saved model
model = tf.keras.models.load_model('model/crop_disease_model.h5')

# Continue training from where you left off
# (modify epochs to continue from last completed epoch)
```

### Option 3: Start Fresh

If network is stable now:

```bash
# Simply run again - it will download weights and start fresh
python train_model.py
```

## Prevention Tips

### 1. Download Weights Before Training

```python
# Run this once to download weights
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0

model = EfficientNetB0(weights='imagenet', include_top=False)
print("✅ Weights downloaded and cached!")
```

### 2. Use Stable Internet Connection

- Use wired connection if possible
- Avoid training during peak hours
- Check connection stability before starting

### 3. Monitor Training Progress

```bash
# In another terminal, watch the training log
tail -f model/training_log.csv
```

### 4. Save More Frequently (Optional)

Modify the ModelCheckpoint to save every N epochs:

```python
callbacks.ModelCheckpoint(
    'model/checkpoint_epoch_{epoch:02d}.h5',
    save_freq='epoch',  # Save every epoch
    period=1
)
```

## Common Network Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `URLError: <urlopen error>` | No internet connection | Check connection, retry |
| `ConnectionTimeout` | Slow/unstable connection | Wait and retry, use better connection |
| `SSL Certificate Error` | Certificate issues | Update certificates or use `weights=None` |
| `HTTP 403 Forbidden` | Access denied | Check firewall, use VPN if needed |
| `Connection reset by peer` | Server closed connection | Retry - TensorFlow will resume download |

## Offline Training (No Internet)

If you need to train completely offline:

1. **Download weights first** (when online):
   ```python
   from tensorflow.keras.applications import EfficientNetB0
   EfficientNetB0(weights='imagenet')
   ```

2. **Weights are cached** in `~/.keras/models/`

3. **Train offline** - script will use cached weights

4. **If cache missing**, use `weights=None` (random initialization)

## Summary

✅ **Good News:**
- ModelCheckpoint saves your progress automatically
- Best model is saved after each epoch
- Training can be interrupted safely
- Saved model can be used immediately

⚠️ **If Network Fails:**
- During download: Script tries cached weights or random init
- During training: Best model is already saved, can resume later
- Always check `model/crop_disease_model.h5` exists before restarting

💡 **Best Practice:**
- Download weights once when online
- They'll be cached for future use
- Training can continue offline after initial download

