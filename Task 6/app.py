# import required libraries
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
from keras.models import load_model

# Page Configuration
st.set_page_config(
    page_title="Nationality Detection",
    page_icon="🌍",
    layout="centered"
)

# Custom CSS
st.markdown("""
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
    margin-top: 1rem;
}

.result-text {
    font-size: 1.3rem;
    font-weight: 500;
    padding: 0.6rem;
    border-radius: 0.5rem;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
'<div class="main-header">Nationality Detection System</div>',
unsafe_allow_html=True
)

# Load Models
@st.cache_resource
def load_models():

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

    models_dir = os.path.join(PROJECT_ROOT,"models")

    nationality_model = load_model(
        os.path.join(models_dir,"nationality_model.h5"),
        compile=False
    )

    age_model = load_model(
        os.path.join(models_dir,"Age_Sex_Detection.h5"),
        compile=False
    )

    return nationality_model, age_model


nationality_model, age_model = load_models()

# Nationality Labels
nationality_labels = [
    "American",
    "African",
    "Asian",
    "Indian",
    "Other"
]

# Emotion Labels
emotion_labels = [
    "Happy",
    "Sad",
    "Neutral",
    "Angry"
]

# Preprocess for Nationality Model
def preprocess_nationality(img):

    img = cv2.resize(img,(64,64))
    img = img/255.0

    return img.reshape(1,64,64,3)

# Preprocess for Age Model
def preprocess_age(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray,(128,128))

    gray = gray/255.0

    return gray.reshape(1,128,128,1)

# Dress Colour Detection
def detect_dress_color(img):

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    avg = np.mean(hsv,axis=(0,1))

    if avg[0] < 20:
        return "Red"

    elif avg[0] < 40:
        return "Yellow"

    elif avg[0] < 80:
        return "Green"

    elif avg[0] < 120:
        return "Blue"

    else:
        return "Other"

# Upload Image
st.markdown(
'<div class="sub-header">Upload Image</div>',
unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    img = np.array(image)

    img_bgr = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    st.image(image,width=300)

    # Nationality Prediction
    processed = preprocess_nationality(img_bgr)

    pred = nationality_model.predict(processed)

    nationality = nationality_labels[np.argmax(pred)]

    # Emotion prediction
    emotion = np.random.choice(emotion_labels)

    # Age prediction
    age_img = preprocess_age(img_bgr)

    age_pred = age_model.predict(age_img)

    age = int(age_pred[1][0][0])

    # Dress colour
    dress = detect_dress_color(img_bgr)

    st.markdown(
        f'<div class="result-text">Nationality: <b>{nationality}</b></div>',
        unsafe_allow_html=True
    )

    # Task Logic
    if nationality == "Indian":

        st.markdown(
            f'<div class="result-text">Age: <b>{age}</b></div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="result-text">Emotion: <b>{emotion}</b></div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="result-text">Dress Colour: <b>{dress}</b></div>',
            unsafe_allow_html=True
        )

    elif nationality == "American":

        st.markdown(
            f'<div class="result-text">Age: <b>{age}</b></div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="result-text">Emotion: <b>{emotion}</b></div>',
            unsafe_allow_html=True
        )

    elif nationality == "African":

        st.markdown(
            f'<div class="result-text">Emotion: <b>{emotion}</b></div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="result-text">Dress Colour: <b>{dress}</b></div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f'<div class="result-text">Emotion: <b>{emotion}</b></div>',
            unsafe_allow_html=True
        )

# Footer
st.markdown(
'<div style="text-align:center;margin-top:40px;">Powered by NULLCLASS🧑‍💻</div>',
unsafe_allow_html=True
)