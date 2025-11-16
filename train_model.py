"""
Crop Disease Detection Model Training Script
Uses EfficientNetB0 with transfer learning
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
NUM_CLASSES = 38  # Will be updated from dataset

# Paths
DATASET_DIR = Path("dataset/processed")
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

# Model paths
MODEL_PATH = MODEL_DIR / "crop_disease_model.h5"
MAPPINGS_PATH = DATASET_DIR / "class_mappings.json"


def load_class_mappings():
    """Load class mappings from JSON file"""
    if not MAPPINGS_PATH.exists():
        raise FileNotFoundError(f"Class mappings not found at {MAPPINGS_PATH}. Run prepare_dataset.py first.")
    
    with open(MAPPINGS_PATH, 'r') as f:
        mappings = json.load(f)
    return mappings


def create_data_generators():
    """Create data generators with augmentation"""
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2]
    )
    
    # Only rescaling for validation and test
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator


def build_model(num_classes):
    """Build EfficientNetB0 model with transfer learning"""
    print("\nBuilding EfficientNetB0 model...")
    
    # Load pre-trained EfficientNetB0
    try:
        print("Downloading EfficientNetB0 ImageNet weights (this may take a few minutes)...")
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        print("✅ Weights downloaded successfully!")
    except Exception as e:
        print(f"\n❌ Error downloading weights: {e}")
        print("\nTrying to use cached weights or continue without pre-trained weights...")
        try:
            # Try to load without weights (random initialization)
            base_model = EfficientNetB0(
                weights=None,
                include_top=False,
                input_shape=(IMG_SIZE, IMG_SIZE, 3)
            )
            print("⚠️  Warning: Using random initialization instead of pre-trained weights.")
            print("   Training will take longer and may achieve lower accuracy.")
        except Exception as e2:
            print(f"\n❌ Fatal error: {e2}")
            raise
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build model
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Data preprocessing (EfficientNet expects inputs in range [0, 255])
    x = layers.Rescaling(scale=255.0)(inputs)
    
    # Base model
    x = base_model(x, training=False)
    
    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dropout for regularization
    x = layers.Dropout(0.3)(x)
    
    # Dense layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    return model, base_model


def train_model(model, train_gen, val_gen):
    """Train the model"""
    print("\n" + "=" * 60)
    print("Starting Model Training")
    print("=" * 60)
    
    # Callbacks
    callbacks_list = [
        callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.CSVLogger(MODEL_DIR / 'training_log.csv')
    ]
    
    # Step 1: Train with frozen base model
    print("\nPhase 1: Training with frozen base model...")
    try:
        history1 = model.fit(
            train_gen,
            epochs=10,
            validation_data=val_gen,
            callbacks=callbacks_list,
            verbose=1
        )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user.")
        print(f"💾 Best model saved to: {MODEL_PATH}")
        print("You can resume training later or use the saved model.")
        raise
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print(f"💾 Best model so far saved to: {MODEL_PATH}")
        print("You can resume training or use the saved model.")
        raise
    
    # Step 2: Unfreeze top layers and fine-tune
    print("\nPhase 2: Fine-tuning with unfrozen top layers...")
    base_model = model.layers[2]  # EfficientNetB0 is the 3rd layer
    base_model.trainable = True
    
    # Freeze bottom layers, unfreeze top layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    # Continue training
    try:
        history2 = model.fit(
            train_gen,
            epochs=EPOCHS - 10,
            validation_data=val_gen,
            callbacks=callbacks_list,
            verbose=1,
            initial_epoch=10
        )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user.")
        print(f"💾 Best model saved to: {MODEL_PATH}")
        print("You can resume training later or use the saved model.")
        raise
    except Exception as e:
        print(f"\n❌ Error during fine-tuning: {e}")
        print(f"💾 Best model so far saved to: {MODEL_PATH}")
        print("You can resume training or use the saved model.")
        raise
    
    # Combine histories
    history = {
        'loss': history1.history['loss'] + history2.history['loss'],
        'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
        'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
        'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy']
    }
    
    return history


def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['loss'], label='Training Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history['accuracy'], label='Training Accuracy')
    axes[1].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'training_history.png', dpi=300, bbox_inches='tight')
    print(f"\nTraining history plot saved to {MODEL_DIR / 'training_history.png'}")


def evaluate_model(model, test_gen, class_names):
    """Evaluate model on test set"""
    print("\n" + "=" * 60)
    print("Evaluating Model on Test Set")
    print("=" * 60)
    
    # Evaluate
    test_results = model.evaluate(test_gen, verbose=1)
    print(f"\nTest Loss: {test_results[0]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f}")
    if len(test_results) > 2:
        print(f"Test Top-3 Accuracy: {test_results[2]:.4f}")
    
    # Predictions
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_gen.classes
    
    # Classification report
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names,
        digits=4
    ))
    
    # Confusion matrix (sample for visualization)
    if len(class_names) <= 20:
        cm = confusion_matrix(true_classes, predicted_classes)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(MODEL_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to {MODEL_DIR / 'confusion_matrix.png'}")
    
    return test_results


def main():
    """Main training function"""
    print("=" * 60)
    print("Crop Disease Detection - Model Training")
    print("=" * 60)
    
    # Check if dataset is prepared
    if not DATASET_DIR.exists():
        print(f"\nError: Processed dataset not found at {DATASET_DIR}")
        print("Please run: python scripts/prepare_dataset.py")
        return
    
    # Load class mappings
    mappings = load_class_mappings()
    num_classes = mappings['num_classes']
    class_names = [mappings['idx_to_class'][str(i)] for i in range(num_classes)]
    
    print(f"\nNumber of classes: {num_classes}")
    print(f"Classes: {', '.join(class_names[:10])}...")
    
    # Create data generators
    print("\nCreating data generators...")
    train_gen, val_gen, test_gen = create_data_generators()
    
    # Verify class mapping matches
    if train_gen.num_classes != num_classes:
        print(f"Warning: Generator found {train_gen.num_classes} classes, but mappings have {num_classes}")
        num_classes = train_gen.num_classes
    
    # Build model
    model, base_model = build_model(num_classes)
    model.summary()
    
    # Train model
    history = train_model(model, train_gen, val_gen)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    evaluate_model(model, test_gen, class_names)
    
    # Save class names for inference
    with open(MODEL_DIR / 'class_names.json', 'w') as f:
        json.dump(class_names, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Class names saved to: {MODEL_DIR / 'class_names.json'}")
    print(f"Training log saved to: {MODEL_DIR / 'training_log.csv'}")


if __name__ == "__main__":
    # Set memory growth for GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    main()

