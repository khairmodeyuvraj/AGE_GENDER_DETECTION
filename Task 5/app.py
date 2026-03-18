# Import required libraries
import streamlit as st
import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Car Colour Detection",
    page_icon="🚗",
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

.result-box {
    font-size: 1.3rem;
    padding: 0.6rem;
    border-radius: 0.5rem;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
'<div class="main-header">Car Colour Detection System</div>',
unsafe_allow_html=True
)

# Load YOLO Model
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")
    return model

model = load_model()

# Detect Car Colour
def detect_car_color(car_img):

    hsv = cv2.cvtColor(car_img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100,150,50])
    upper_blue = np.array([140,255,255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    blue_pixels = cv2.countNonZero(mask)

    total_pixels = car_img.shape[0] * car_img.shape[1]

    blue_ratio = blue_pixels / total_pixels

    if blue_ratio > 0.20:
        return "Blue"
    else:
        return "Other"

# Process Frame
def process_frame(frame):

    results = model(frame)

    people_count = 0
    car_count = 0

    for r in results:

        boxes = r.boxes

        for box in boxes:

            cls = int(box.cls)
            label = model.names[cls]

            x1,y1,x2,y2 = map(int,box.xyxy[0])

            # Person detection
            if label == "person":

                people_count += 1

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                cv2.putText(frame,"Person",(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,(0,255,0),2)

            # Car detection
            if label in ["car","truck","bus"]:

                car_count += 1

                car_img = frame[y1:y2,x1:x2]

                if car_img.size == 0:
                    continue

                # crop center of car to avoid background
                h,w,_ = car_img.shape
                car_img = car_img[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]

                color = detect_car_color(car_img)

                # Blue car → Red rectangle
                if color == "Blue":

                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)

                    cv2.putText(frame,"Blue Car",(x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,(0,0,255),2)

                # Other cars → Blue rectangle
                else:

                    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)

                    cv2.putText(frame,"Car",(x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,(255,0,0),2)

    return frame, people_count, car_count

# Upload Image
st.markdown(
'<div class="sub-header">Upload Traffic Image</div>',
unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    frame = np.array(image)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    processed_frame, people_count, car_count = process_frame(frame)

    st.image(cv2.cvtColor(processed_frame,cv2.COLOR_BGR2RGB))

    st.markdown(
        f'<div class="result-box">🚗 Cars Detected: <b>{car_count}</b></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="result-box">🧍 People Detected: <b>{people_count}</b></div>',
        unsafe_allow_html=True
    )

# Footer
st.markdown(
'<div style="text-align:center;margin-top:40px;">Powered by NULLCLASS🧑‍💻</div>',
unsafe_allow_html=True
)