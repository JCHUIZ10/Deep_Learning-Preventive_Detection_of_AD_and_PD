import streamlit as st
from PIL import Image
import numpy as np
import gdown
import os
from tensorflow.keras.models import load_model

@st.cache_resource
def load_model_func():
    model_path = "02_modelo.keras"
    drive_id = "1G3_ysyKP4uokQSnoIbcACWliRruUDSUx"
    message = st.empty()

    if not os.path.exists(model_path):
        message.info("Downloading model...")
        url = f"https://drive.google.com/uc?id={drive_id}"
        gdown.download(url, model_path, quiet=False)
        
    try:
        model = load_model(model_path)
        message.empty()  # Clear download message
        return model
    except Exception as e:
        message.error(f"Error loading model: {e}")
        return None

# Function to preprocess the image
def preprocess_image(image):
    """Preprocesses the image to be compatible with the model"""
    try:
        # Open image with PIL (compatible with Streamlit and TensorFlow)
        img = Image.open(image)
        
        # Convert to RGB if necessary (in case the image has alpha channel)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to 150x150 (same as in training)
        img_resized = img.resize((150, 150))
        
        # Convert to numpy array
        img_array = np.array(img_resized)
        
        # Normalize (divide by 255.0 as in training)
        img_array = img_array / 255.0
        
        # Add batch dimension (model expects a batch of images)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Function to make prediction
def predict(model, processed_image):
    """Makes prediction using the loaded model"""
    try:
        # Make the prediction
        prediction = model.predict(processed_image)
        
        # Classes are in the same order as in training
        labels = ['AD','CONTROL', 'PD'] # Alzheimer, Control, Parkinson
        
        # Get the class with highest probability
        predicted_class_idx = np.argmax(prediction[0])
        predicted_class = labels[predicted_class_idx]
        
        # Get confidence (maximum probability)
        confidence = np.max(prediction[0])
        
        # Get all probabilities to display
        probabilities = {labels[i]: prediction[0][i] for i in range(len(labels))}
        
        return predicted_class, confidence, probabilities
    
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None

# Main interface
st.markdown("<h1 style='text-align: center;'>Neuroimaging Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Automatic Classification: Alzheimer (AD) | Control | Parkinson (PD)</p>", unsafe_allow_html=True)
st.markdown("---")

# Load model at startup
model = load_model_func()

if model is None:
    st.stop()  # Stop execution if model cannot be loaded

# Image upload
image = st.file_uploader(
    "Upload Neuroimaging",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
    help="Supported formats: JPG, JPEG, PNG"
)

if image:
    # Button to process
    if st.button("🔍 Analyze Image", type="primary"):
        with st.spinner("Analyzing neuroimaging..."):
            # Preprocess the image
            processed_image = preprocess_image(image)
            
            if processed_image is not None:
                # Make the prediction
                predicted_class, confidence, probabilities = predict(model, processed_image)
                
                if predicted_class is not None:
                    # Show results
                    st.markdown("---")
                    st.markdown("## **-> Analysis Results**")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.image(image, caption="Analyzed Neuroimaging", use_container_width=True)
                    
                    with col2:
                        # Determine color according to result
                        if predicted_class == "CONTROL":
                            color = "#4CAF50"  # Green for healthy control
                        elif predicted_class == "AD":
                            color = "#FF5722"  # Red for Alzheimer
                        else:  # PD
                            color = "#FF9800"  # Orange for Parkinson
                        
                        # Main result
                        st.markdown(f"""
                            <div style='
                                border: 2px solid {color};
                                border-radius: 15px;
                                padding: 20px;
                                background-color: #f9f9f9;
                                box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.1);
                                margin-bottom: 20px;
                            '>
                               <h3 style='color: #333; margin-bottom: 15px;'>Diagnosis</h3>
                                <p style='font-size: 24px; margin: 10px 0;'><strong>Class:</strong> <span style='color: {color}; font-weight: bold;'>{predicted_class}</span></p>
                                <p style='font-size: 20px; margin: 10px 0;'><strong>Confidence:</strong> <span style='color: #2196F3; font-weight: bold;'>{confidence:.2%}</span></p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Show all probabilities
                        st.markdown("### 📈 Probabilities by class:")
                        for class_name, prob in probabilities.items():
                            st.progress(float(prob), text=f"{class_name}: {prob:.2%}")
                    
                    # Results interpretation
                    st.markdown("---")
                    st.markdown("### Interpretation:")
                    
                    if predicted_class == "CONTROL":
                        st.success("**Healthy Control**: The image suggests normal neurological patterns.")
                    elif predicted_class == "AD":
                        st.error("**Alzheimer's**: The image shows patterns compatible with Alzheimer's disease.")
                    else:  # PD
                        st.warning("**Parkinson's**: The image presents characteristics associated with Parkinson's disease.")
                    
                    st.info("**Important note**: This is an automated analysis for informational purposes. Always consult with a medical professional for a definitive diagnosis.")

# Additional information in sidebar
with st.sidebar:
    st.markdown("## Model Information")
    st.markdown("""
    **Features:**
    - **Architecture:** EfficientNetB0 Architecture with Sigmoid
    - **Resolution:** 150x150 pixels
    - **Type:** Neuroimaging classification
    
    **Diagnostic Classes:**
    """)
    
    # Classes section 
    st.markdown("""
    - **AD (Alzheimer's Disease)**
    - **Control (Healthy)**
    - **PD (Parkinson's Disease)**
    """)
    
    st.markdown("---")
    
    st.markdown("""
    **Usage Instructions:**
    1. Upload a neuroimaging
    2. Click "Analyze Image"
    3. Review the diagnosis results
    
    **Supported Formats:**
    - JPG, JPEG, PNG
    """)
    
    if model:
        st.success("Model loaded successfully")
    else:
        st.error("Error loading model")