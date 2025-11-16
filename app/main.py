"""
Crop Disease Detection - Streamlit Web Application
Main application file for image upload, prediction, and remedy suggestions
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
from pathlib import Path
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.remedies import get_remedy

# Page configuration
st.set_page_config(
    page_title="🌱 Crop Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #E8F5E9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E7D32;
        margin: 1rem 0;
    }
    .remedy-box {
        background-color: #FFF3E0;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #FF9800;
        margin: 1rem 0;
    }
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# Paths
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "crop_disease_model.h5"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

# Image settings
IMG_SIZE = 224


@st.cache_resource
def load_model():
    """Load the trained model"""
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}. Please train the model first.")
        st.stop()
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


@st.cache_data
def load_class_names():
    """Load class names from JSON file"""
    if not CLASS_NAMES_PATH.exists():
        # Try to create it from dataset mappings
        st.warning(f"Class names file not found. Attempting to create it...")
        mappings_path = Path("dataset/processed/class_mappings.json")
        
        if mappings_path.exists():
            try:
                with open(mappings_path, 'r') as f:
                    mappings = json.load(f)
                num_classes = mappings['num_classes']
                class_names = [mappings['idx_to_class'][str(i)] for i in range(num_classes)]
                
                # Save for future use
                MODEL_DIR.mkdir(exist_ok=True)
                with open(CLASS_NAMES_PATH, 'w') as f:
                    json.dump(class_names, f, indent=2)
                
                st.success(f"✅ Created class_names.json with {num_classes} classes!")
                return class_names
            except Exception as e:
                st.error(f"Error creating class names: {e}")
                st.error("Please run: python scripts/create_class_names.py")
                st.stop()
        else:
            st.error(f"Class names file not found at {CLASS_NAMES_PATH}")
            st.error("Please run: python scripts/create_class_names.py")
            st.stop()
    
    try:
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
        return class_names
    except Exception as e:
        st.error(f"Error loading class names: {e}")
        st.stop()


def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Resize image
    image = image.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert to array and normalize
    img_array = np.array(image)
    
    # Convert RGBA to RGB if necessary
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    # Normalize to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def predict_disease(model, image, class_names):
    """Predict disease from image"""
    # Preprocess image
    processed_image = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(processed_image, verbose=0)
    
    # Get top 3 predictions
    top_3_indices = np.argsort(predictions[0])[-3:][::-1]
    top_3_probs = predictions[0][top_3_indices]
    
    results = []
    for idx, prob in zip(top_3_indices, top_3_probs):
        results.append({
            'class': class_names[idx],
            'confidence': float(prob)
        })
    
    return results


def format_class_name(class_name):
    """Format class name for display"""
    # Replace underscores with spaces
    formatted = class_name.replace("_", " ")
    # Capitalize words
    formatted = " ".join(word.capitalize() for word in formatted.split())
    return formatted


def main():
    """Main application"""
    # Header
    st.markdown('<h1 class="main-header">🌱 Crop Disease Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a leaf image to detect diseases and get organic treatment recommendations</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.markdown("""
        This application uses deep learning (EfficientNetB0) to detect crop diseases from leaf images.
        
        **Features:**
        - 🖼️ Image upload and analysis
        - 🔍 Disease classification
        - 💊 Organic remedy suggestions
        - 📊 Confidence scores
        
        **Supported Crops:**
        - Tomatoes, Apples, Corn, Potatoes
        - Grapes, Peaches, Peppers, and more
        """)
        
        st.header("📝 Instructions")
        st.markdown("""
        1. Upload a clear image of a crop leaf
        2. Click 'Predict Disease'
        3. View results and remedies
        4. Follow organic treatment recommendations
        """)
        
        st.header("⚠️ Note")
        st.info("For best results, use clear, well-lit images of individual leaves.")
    
    # Load model and class names
    with st.spinner("Loading model..."):
        model = load_model()
        class_names = load_class_names()
    
    st.success("✅ Model loaded successfully!")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a leaf image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a JPG, JPEG, or PNG image of a crop leaf"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Prediction button
            if st.button("🔍 Predict Disease", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    # Make prediction
                    results = predict_disease(model, image, class_names)
                    
                    # Store results in session state
                    st.session_state['prediction_results'] = results
                    st.session_state['uploaded_image'] = image
    
    with col2:
        st.header("📊 Prediction Results")
        
        if 'prediction_results' in st.session_state:
            results = st.session_state['prediction_results']
            top_result = results[0]
            
            # Display top prediction
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown(f"### 🎯 Predicted Disease")
            st.markdown(f"**{format_class_name(top_result['class'])}**")
            
            # Confidence bar
            confidence = top_result['confidence'] * 100
            st.progress(confidence / 100)
            st.markdown(f"**Confidence: {confidence:.2f}%**")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show top 3 predictions
            with st.expander("View Top 3 Predictions"):
                for i, result in enumerate(results, 1):
                    st.markdown(f"**{i}. {format_class_name(result['class'])}** - {result['confidence']*100:.2f}%")
            
            # Get remedy information
            remedy_info = get_remedy(top_result['class'])
            
            # Display remedies
            st.markdown('<div class="remedy-box">', unsafe_allow_html=True)
            st.markdown(f"### 💊 Organic Treatment Recommendations")
            st.markdown(f"**For: {remedy_info['name']}**")
            
            st.markdown("#### 🌿 Treatment Remedies:")
            for i, remedy in enumerate(remedy_info['remedies'], 1):
                st.markdown(f"{i}. {remedy}")
            
            st.markdown("#### 🛡️ Prevention Tips:")
            for i, tip in enumerate(remedy_info['prevention'], 1):
                st.markdown(f"{i}. {tip}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Download results button
            st.download_button(
                label="📥 Download Results",
                data=f"""
Predicted Disease: {format_class_name(top_result['class'])}
Confidence: {confidence:.2f}%

Treatment Recommendations:
{chr(10).join([f"{i+1}. {r}" for i, r in enumerate(remedy_info['remedies'])])}

Prevention Tips:
{chr(10).join([f"{i+1}. {p}" for i, p in enumerate(remedy_info['prevention'])])}
                """,
                file_name="disease_prediction_results.txt",
                mime="text/plain"
            )
        else:
            st.info("👆 Upload an image and click 'Predict Disease' to see results here.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🌱 Made for sustainable agriculture | Powered by Deep Learning</p>
        <p>For best results, ensure images are clear and well-lit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

