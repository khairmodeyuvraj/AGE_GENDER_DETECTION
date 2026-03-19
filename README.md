# AGE_GENDER_DETECTION

A comprehensive AI-based multi-task detection system developed using Python, OpenCV, TensorFlow/Keras, and Streamlit. This project integrates multiple computer vision and machine learning applications with real-time processing and GUI support.

---

## 🚀 Features

This project consists of multiple AI-based modules:

---

### 🔹 Task 1: Age & Gender Detection with Hair Logic
- Predicts age and gender from image input
- Implements custom logic:
  - Long hair → Female
  - Short hair → Male
- Applies logic only for age group **20–30**
- GUI-based image upload detection

---

### 🔹 Task 2: Senior Citizen Identification
- Real-time webcam detection
- Predicts age and gender
- Marks person as **Senior Citizen (Age > 60)**
- Stores:
  - Age
  - Gender
  - Visit time
- Saves data into **CSV file**

---

### 🔹 Task 3: Voice-Based Age & Emotion Detection
- Accepts voice input (.wav file)
- Detects:
  - Gender (only processes male voice)
  - Age prediction
- If age > 60:
  - Detects emotion
- Rejects female voice input

---

### 🔹 Task 4: Sign Language Detection
- Recognizes hand gestures:
  - Hello, Bye, Yes, No, Perfect
- Supports:
  - Image upload
  - Real-time webcam detection
- Works only between **6 PM – 10 PM**
- Uses CNN-based gesture classification

---

### 🔹 Task 5: Car Color Detection in Traffic
- Detects cars in traffic images
- Identifies car color
- Logic:
  - Blue car → Red bounding box
  - Other cars → Blue bounding box
- Counts number of cars
- Detects number of people in image

---

### 🔹 Task 6: Nationality Detection with Conditional Logic
- Predicts nationality from image
- Detects emotion
- Applies conditional outputs:
  - 🇮🇳 Indian → Age + Emotion + Dress Color
  - 🇺🇸 American → Age + Emotion
  - 🌍 African → Emotion + Dress Color
  - Others → Emotion only

---

## 🧠 Technologies Used

- Python
- OpenCV
- TensorFlow / Keras
- NumPy
- Streamlit (GUI)
- Librosa (for audio processing)

---

## 💻 How to Run the Project

Follow the steps below to run this project on your system:

---

### 🔹 1. Clone the Repository

```bash
git clone https://github.com/khairmodeyuvraj/AGE_GENDER_DETECTION.git
```
### 🔹 2. Navigate to Project Folder
```bash
cd AGE_GENDER_DETECTION
```
### 🔹 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```
### 🔹4. Run Streamlit Applications

### ▶️ Task 1: Age & Gender Detection
```bash
cd Task 1
streamlit run app.py
```
### ▶️ Task 2: Senior Citizen Identification
```bash
cd Task 2
streamlit run app.py
```
### ▶️ Task 3: Voice-Based Detection
```bash
cd Task 3
streamlit run app.py
```
### ▶️ Task 4: Sign Language Detection
```bash
cd Task 4
streamlit run app.py
```
### ▶️ Task 5: Car Colour Detection
```bash
cd Task 5
streamlit run app.py
```
### ▶️ Task 6: Nationality Detection
```bash
cd Task 6
streamlit run app.py
```
### 🔹 5. Open in Browser
After running the command, Streamlit will automatically open in your browser:
```bash
http://localhost:8501
```
