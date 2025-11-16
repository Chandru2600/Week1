"""
TensorFlow Lite Conversion Script
Converts the trained Keras model to TensorFlow Lite format for mobile deployment
"""

import tensorflow as tf
from pathlib import Path
import json

# Paths
MODEL_DIR = Path("model")
KERAS_MODEL_PATH = MODEL_DIR / "crop_disease_model.h5"
TFLITE_MODEL_PATH = MODEL_DIR / "crop_disease_model.tflite"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

# Conversion settings
OPTIMIZATION = "DEFAULT"  # Options: "DEFAULT", "OPTIMIZE_FOR_SIZE", "OPTIMIZE_FOR_LATENCY"


def convert_to_tflite():
    """Convert Keras model to TensorFlow Lite format"""
    print("=" * 60)
    print("TensorFlow Lite Conversion")
    print("=" * 60)
    
    # Check if Keras model exists
    if not KERAS_MODEL_PATH.exists():
        print(f"\nError: Keras model not found at {KERAS_MODEL_PATH}")
        print("Please train the model first using train_model.py")
        return False
    
    print(f"\nLoading Keras model from: {KERAS_MODEL_PATH}")
    
    try:
        # Load the Keras model
        model = tf.keras.models.load_model(KERAS_MODEL_PATH)
        print("✅ Model loaded successfully")
        
        # Convert to TensorFlow Lite
        print("\nConverting to TensorFlow Lite format...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Apply optimizations
        if OPTIMIZATION == "OPTIMIZE_FOR_SIZE":
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
            print("   Optimization: Size")
        elif OPTIMIZATION == "OPTIMIZE_FOR_LATENCY":
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
            print("   Optimization: Latency")
        else:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            print("   Optimization: Default")
        
        # Convert
        tflite_model = converter.convert()
        
        # Save the TFLite model
        with open(TFLITE_MODEL_PATH, 'wb') as f:
            f.write(tflite_model)
        
        # Get model size
        keras_size = KERAS_MODEL_PATH.stat().st_size / (1024 * 1024)  # MB
        tflite_size = TFLITE_MODEL_PATH.stat().st_size / (1024 * 1024)  # MB
        
        print(f"\n✅ Conversion successful!")
        print(f"\nModel saved to: {TFLITE_MODEL_PATH}")
        print(f"Keras model size: {keras_size:.2f} MB")
        print(f"TFLite model size: {tflite_size:.2f} MB")
        print(f"Size reduction: {(1 - tflite_size/keras_size)*100:.1f}%")
        
        # Test the TFLite model
        print("\n" + "=" * 60)
        print("Testing TFLite Model")
        print("=" * 60)
        test_tflite_model(TFLITE_MODEL_PATH)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tflite_model(tflite_path):
    """Test the converted TFLite model with a dummy input"""
    try:
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"\nInput shape: {input_details[0]['shape']}")
        print(f"Input type: {input_details[0]['dtype']}")
        print(f"Output shape: {output_details[0]['shape']}")
        print(f"Output type: {output_details[0]['dtype']}")
        
        # Create dummy input
        input_shape = input_details[0]['shape']
        dummy_input = tf.random.normal(input_shape, dtype=tf.float32)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"\n✅ TFLite model test successful!")
        print(f"Output shape: {output.shape}")
        print(f"Sample output values: {output[0][:5]}")
        
    except Exception as e:
        print(f"\n⚠️ Warning: Could not test TFLite model: {e}")


def create_mobile_inference_guide():
    """Create a guide for using the TFLite model in mobile apps"""
    guide = """
# TensorFlow Lite Model Usage Guide

## Model Information
- Model file: crop_disease_model.tflite
- Input size: 224x224x3 (RGB image)
- Output: 38 classes (disease categories)
- Input type: float32, normalized [0, 1]

## Android Integration

### 1. Add TensorFlow Lite to build.gradle
```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
}
```

### 2. Load and run inference
```java
// Load model
Interpreter interpreter = new Interpreter(loadModelFile("crop_disease_model.tflite"));

// Preprocess image
TensorImage image = TensorImage.fromBitmap(bitmap);
ImageProcessor processor = new ImageProcessor.Builder()
    .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
    .add(new NormalizeOp(0f, 255f))
    .build();
image = processor.process(image);

// Run inference
float[][] output = new float[1][38];
interpreter.run(image.getBuffer(), output);

// Get prediction
int maxIndex = 0;
for (int i = 1; i < output[0].length; i++) {
    if (output[0][i] > output[0][maxIndex]) {
        maxIndex = i;
    }
}
```

## iOS Integration

### 1. Add TensorFlow Lite to Podfile
```ruby
pod 'TensorFlowLiteSwift'
```

### 2. Load and run inference
```swift
import TensorFlowLite

// Load model
let interpreter = try Interpreter(modelPath: modelPath)

// Preprocess image
let inputData = preprocessImage(image) // Resize to 224x224, normalize

// Allocate tensors
try interpreter.allocateTensors()

// Copy input data
try interpreter.copy(inputData, toInputAt: 0)

// Run inference
try interpreter.invoke()

// Get output
let outputTensor = try interpreter.output(at: 0)
let output = outputTensor.data.toArray(type: Float32.self)
```

## Python Inference (for testing)

```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="crop_disease_model.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocess image
img = Image.open("test_image.jpg").resize((224, 224))
img_array = np.array(img, dtype=np.float32) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Run inference
interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()

# Get prediction
output = interpreter.get_tensor(output_details[0]['index'])
predicted_class = np.argmax(output[0])
```
"""
    
    guide_path = MODEL_DIR / "TFLITE_USAGE_GUIDE.md"
    with open(guide_path, 'w') as f:
        f.write(guide)
    
    print(f"\n📖 Usage guide saved to: {guide_path}")


def main():
    """Main function"""
    success = convert_to_tflite()
    
    if success:
        create_mobile_inference_guide()
        print("\n" + "=" * 60)
        print("Conversion Complete!")
        print("=" * 60)
        print(f"\n✅ TFLite model ready for mobile deployment")
        print(f"📱 See {MODEL_DIR / 'TFLITE_USAGE_GUIDE.md'} for integration instructions")
    else:
        print("\n❌ Conversion failed. Please check the error messages above.")


if __name__ == "__main__":
    main()

