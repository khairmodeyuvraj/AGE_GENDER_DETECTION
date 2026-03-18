#import required libraries
import streamlit as st
import numpy as np
import cv2
import os
from datetime import datetime
from keras.models import load_model
from PIL import Image

# Page Config
st.set_page_config(
    page_title="Sign Language Detection",
    page_icon="✋",
    layout="centered"
)

st.title("Sign Language Detection System")


# Time Restriction (6 PM to 10 PM)
current_hour = datetime.now().hour

if not (18 <= current_hour <= 22):
    st.error("⚠️ This system works only between 6 PM and 10 PM.")
    st.stop()


# Load Model
@st.cache_resource
def load_model_file():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)

    model_path = os.path.join(PROJECT_ROOT, "models", "sign_language_model.h5")

    model = load_model(model_path, compile=False)
    return model

model = load_model_file()

# Classes
classes = ["Hello","Bye","No","Yes","Perfect"]

# Preprocess
def preprocess(img):
    img = cv2.resize(img, (64,64))
    img = img / 255.0
    img = img.reshape(1,64,64,3)
    return img

# Upload Image Section
st.subheader("Upload Image")

uploaded_file = st.file_uploader(
    "Upload a hand gesture image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    img = np.array(image)

    processed = preprocess(img)

    pred = model.predict(processed)
    confidence = np.max(pred)
    label = classes[np.argmax(pred)]

    st.image(image, width=300)

    if confidence > 0.85:
        st.success(f"Detected Gesture: {label} ({confidence:.2f})")
    else:
        st.warning("Low confidence prediction")

# Real-Time Detection
st.subheader("Real-Time Detection")

run = st.checkbox("Start Camera")

frame_window = st.image([])

cap = cv2.VideoCapture(0)

while run:

    ret, frame = cap.read()

    if not ret:
        st.error("Camera not working")
        break

    frame = cv2.flip(frame,1)

    h, w, _ = frame.shape

    # ROI box
    x1, y1 = int(w*0.3), int(h*0.2)
    x2, y2 = int(w*0.7), int(h*0.8)

    roi = frame[y1:y2, x1:x2]

    # Draw box
    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)

    processed = preprocess(roi)

    pred = model.predict(processed)
    confidence = np.max(pred)
    label = classes[np.argmax(pred)]

    if confidence < 0.85:
        label = "No Gesture"

    cv2.putText(
        frame,
        f"{label} ({confidence:.2f})",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    frame_window.image(frame, channels="BGR")

cap.release()

# Footer
st.markdown(
    "<div style='text-align:center;margin-top:40px;'>Powered by NULLCLASS🧑‍💻</div>",
    unsafe_allow_html=True
)