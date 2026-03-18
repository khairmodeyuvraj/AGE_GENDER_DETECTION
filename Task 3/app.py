# Import required libraries
import streamlit as st
import librosa
import numpy as np
import os
from keras.models import load_model

# Page Configuration
st.set_page_config(
    page_title="Voice Age & Emotion Detection",
    page_icon="🎤",
    layout="centered"
)

st.title("Age and Emotion Detection through Voice")

# Custom CSS
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
    margin-top: 1rem;
    margin-bottom: 1rem;
}

.result-text {
    font-size: 1.3rem;
    font-weight: 500;
    padding: 0.6rem;
    border-radius: 0.5rem;
    margin-bottom: 0.5rem;
}
</style>
""",
unsafe_allow_html=True
)

# Load the Models
@st.cache_resource
def load_models():

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

    models_dir = os.path.join(PROJECT_ROOT, "models")

    gender_model = load_model(
        os.path.join(models_dir, "voice_gender_model.h5"),
        compile=False
    )

    age_model = load_model(
        os.path.join(models_dir, "voice_age_model.h5"),
        compile=False
    )

    emotion_model = load_model(
        os.path.join(models_dir, "emotion_model.h5"),
        compile=False
    )

    return gender_model, age_model, emotion_model


gender_model, age_model, emotion_model = load_models()


# Feature Extraction (MFCC)
def extract_features(audio_file):

    audio, sr = librosa.load(audio_file, duration=3, offset=0.5)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    mfcc = np.mean(mfcc.T, axis=0)

    return mfcc.reshape(1, -1)

# Upload Voice File
st.subheader("Upload Voice Note")

uploaded_file = st.file_uploader(
    "Upload a .wav voice file",
    type=["wav"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    features = extract_features(uploaded_file)

    # Gender Prediction
    gender_pred = gender_model.predict(features)

    gender = "Male" if gender_pred > 0.5 else "Female"

    if gender == "Female":

        st.error("Upload male voice")

    else:
        # Age Prediction
        age_pred = age_model.predict(features)

        age = int(age_pred[0][0])

        st.success(f"Predicted Age: {age}")

        # Senior Citizen Check
        if age > 60:

            st.warning("Senior Citizen Detected")

            # Emotion Detection
            emotion_pred = emotion_model.predict(features)

            emotions = ["Neutral", "Happy", "Sad", "Angry"]

            emotion = emotions[np.argmax(emotion_pred)]

            st.info(f"Detected Emotion: {emotion}")

        else:

            st.info("Emotion detection only available for senior citizens.")

# Footer
st.markdown(
    "<div style='text-align:center;margin-top:40px;'>Powered by NULLCLASS🧑‍💻</div>",
    unsafe_allow_html=True
)