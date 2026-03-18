# Import required libraries
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import tensorflow as tf
from keras.src.legacy.saving import legacy_h5_format
from keras.models import load_model

# Set page configuration
st.set_page_config(
    page_title="Age & Gender Detector",
    page_icon="👤",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .result-text {
        font-size: 1.5rem;
        font-weight: 500;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .image-container {
        margin-bottom: 2rem;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: rgba(237, 242, 247, 0.5);
    }
    
    .app-footer {
        text-align: center;
        margin-top: 2rem;
        opacity: 0.7;
    }
    
    .stButton>button {
        background-color: #2563EB;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #1E40AF;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Load Age & Gender Model
@st.cache_resource
def load_age_gender_model():
    try:

        # Current directory (Task 1)
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

        # Go to project root
        PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

        # Correct model path
        model_path = os.path.join(PROJECT_ROOT, "models", "Age_Sex_Detection.h5")

        model = legacy_h5_format.load_model_from_hdf5(
            model_path, custom_objects={"mae": "mae"}
        )

        return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Image Preprocessing
def preprocess_image(uploaded_image):

    # Convert to grayscale
    image = uploaded_image.convert("L")

    # Resize to training size
    image = image.resize((128,128))

    # Convert to numpy
    image_array = np.array(image) / 255.0

    # Reshape to match model input
    image_array = image_array.reshape(1,128,128,1)

    return image_array

# Hair Detection Logic (Task 1)
def detect_hair_type(image):

    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray,(128,128))

    hair_region = gray[0:40,:]

    avg_intensity = np.mean(hair_region)

    if avg_intensity < 120:
        return "Long"
    else:
        return "Short"

# Prediction Function
def predict_age_gender(model, image_array, original_image):

    try:
        predictions = model.predict(image_array)

        predicted_age = int(np.round(predictions[1][0]))

        gender_prob = predictions[0][0]
        predicted_gender = "Female" if gender_prob > 0.5 else "Male"
        gender_confidence = (
            gender_prob if predicted_gender == "Female" else 1 - gender_prob
        )

        # Detect hair type
        hair_type = detect_hair_type(original_image)

        # Task-1 Logic
        if 20 <= predicted_age <= 30:

            if hair_type == "Long":
                predicted_gender = "Female"
            else:
                predicted_gender = "Male"

        return predicted_age, predicted_gender, float(gender_confidence), hair_type

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None, None


# Helper function to convert hex color to RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

# Main App
def main():

    st.markdown(
        '<div class="main-header">Age and Gender Detector</div>', unsafe_allow_html=True
    )

    with st.spinner("Loading model... This may take a moment."):
        model = load_age_gender_model()

    if model is None:
        st.warning("Please make sure the model file exists.")
        return

    st.markdown('<div class="sub-header">Upload Images</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Choose one or more images...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Detect Age & Gender", key="detect_button"):

        with st.spinner("Analyzing images..."):

            for i, uploaded_file in enumerate(uploaded_files):

                with st.container():

                    st.markdown(
                        f'<div class="image-container">', unsafe_allow_html=True
                    )

                    st.markdown(f"<h3>Image {i+1}</h3>", unsafe_allow_html=True)

                    col1, col2 = st.columns([1, 1])

                    image = Image.open(uploaded_file)

                    col1.image(image, caption=f"Image {i+1}: {uploaded_file.name}", width=300)

                    processed_image = preprocess_image(image)

                    age, gender, confidence, hair_type = predict_age_gender(
                        model, processed_image, image
                    )

                    if age is not None:

                        col2.markdown(
                            '<div class="sub-header">Results:</div>',
                            unsafe_allow_html=True,
                        )

                        col2.markdown(
                            f'<div class="result-text" style="background-color: rgba(37, 99, 235, 0.1);">Age: {age}</div>',
                            unsafe_allow_html=True,
                        )

                        gender_color = "#9F7AEA" if gender == "Female" else "#4F46E5"

                        col2.markdown(
                            f'<div class="result-text" style="background-color: rgba({", ".join(map(str, hex_to_rgb(gender_color)))}, 0.1);">'
                            f"Gender: {gender}<br>"
                            f"<small>Confidence: {confidence:.2%}</small>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                        col2.markdown(
                            f'<div class="result-text">Hair Type: {hair_type}</div>',
                            unsafe_allow_html=True,
                        )

                    else:
                        col2.error("Failed to process this image")

                    st.markdown("</div>", unsafe_allow_html=True)

                    if i < len(uploaded_files) - 1:
                        st.markdown("<hr>", unsafe_allow_html=True)

    elif st.button("Detect Age & Gender", key="detect_button_empty"):
        st.info("Please upload one or more images first.")

    st.markdown(
        '<div class="app-footer">Powered by NULLCLASS🧑‍💻</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()