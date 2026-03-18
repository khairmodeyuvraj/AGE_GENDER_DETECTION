# Import required libraries
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import pandas as pd
from datetime import datetime
from keras.models import load_model

# Page Configuration
st.set_page_config(
    page_title="Senior Citizen Detection",
    page_icon="🧓",
    layout="centered",
)

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
}

.result-text {
    font-size: 1.3rem;
    padding: 0.5rem;
}
</style>
""",
unsafe_allow_html=True,
)

# Load the Model
@st.cache_resource
def load_age_gender_model():

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

    model_path = os.path.join(PROJECT_ROOT, "models", "Age_Sex_Detection.h5")

    model = load_model(model_path, compile=False)

    return model


model = load_age_gender_model()

# CSV Visit Log
TASK2_DIR = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(TASK2_DIR, "visit_log.csv")

if not os.path.exists(log_file):

    df = pd.DataFrame(columns=["Age","Gender","Senior Citizen","Visit Time"])
    df.to_csv(log_file, index=False)

# Face Detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

gender_dict = {0:"Male",1:"Female"}

# Preprocess Image
def preprocess_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face,(128,128))
    face = face / 255.0
    face = face.reshape(1,128,128,1)
    return face


# Header
st.markdown(
'<div class="main-header">Senior Citizen Identification</div>',
unsafe_allow_html=True
)

# Start Camera Button
start = st.button("Start Camera")

if start:

    cap = cv2.VideoCapture(0)

    frame_window = st.image([])

    st.markdown('<div class="sub-header">Live Detection</div>', unsafe_allow_html=True)

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            st.error("Camera not detected")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:

            face = frame[y:y+h, x:x+w]

            processed_face = preprocess_face(face)

            pred = model.predict(processed_face)

            gender = gender_dict[int(round(pred[0][0][0]))]

            age = int(pred[1][0][0])

            senior = "Yes" if age > 60 else "No"

            label = f"{gender}, {age}"

            if senior == "Yes":
                label += " (Senior Citizen)"

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            cv2.putText(
                frame,
                label,
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

            # Save visit log
            visit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            data = pd.DataFrame(
                [[age, gender, senior, visit_time]],
                columns=["Age","Gender","Senior Citizen","Visit Time"]
            )

            data.to_csv(log_file, mode='a', header=False, index=False)

        frame_window.image(frame, channels="BGR")

    cap.release()

# Footer
st.markdown(
'<div style="text-align:center;margin-top:2rem;">Powered by NULLCLASS🧑‍💻</div>',
unsafe_allow_html=True
)