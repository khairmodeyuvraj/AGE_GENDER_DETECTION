Age & Gender Detection with Long Hair Identification
📌 Project Overview

This project is an Age and Gender Detection system built using Deep Learning and Computer Vision. It predicts a person’s age and gender from facial images and provides results through an interactive Streamlit-based graphical user interface (GUI).

During the internship, the project was extended to include Task 1 – Long Hair Identification, implemented as an additional feature on top of the existing training project.

🎯 Internship Task 1 – Long Hair Identification

Task 1 focuses on logic-building and decision-making rather than only model accuracy.

🔹 Task Logic

For individuals aged 20 to 30:

Long hair → Female

Short hair → Male

For individuals below 20 or above 30:

The system retains the original gender prediction from the trained model, regardless of hair length.

This logic is applied after age and gender prediction, ensuring the task works as an extension of the training project.

🧠 Methodology

Used the same dataset and trained model from the training phase

CNN-based model for:

Age prediction

Gender prediction

Added a hair-length detection module (logic-based placeholder)

Implemented rule-based decision logic for Task 1

Integrated all components into a single Streamlit GUI

🖥️ Features

Upload one or multiple face images

Predict:

Age

Gender (with confidence)

Hair length

Final gender after applying Task-1 logic

Clean and user-friendly interface

🛠️ Technologies Used

Python

TensorFlow / Keras

OpenCV

NumPy

Streamlit

Pillow

▶️ How to Run the Project
pip install -r requirements.txt
python -m streamlit run app.py


The application will open in your browser at:

http://localhost:8501

📂 Project Structure
AGE_GENDER_DETECTION/
├── app.py
├── README.md
├── requirements.txt
├── Age_Sex_Detection.h5
├── notebooks/
│   └── training_notebook.ipynb

✅ Internship Compliance

✔ Built on the same training project

✔ No new dataset used

✔ Task implemented as an additional feature

✔ Includes GUI, logic, and documentation

✔ Clean, reproducible, and professional code

📌 Conclusion

This project demonstrates how machine learning models can be combined with rule-based logic to meet specific real-world constraints. The internship extension highlights problem-solving skills, logical reasoning, and practical deployment using a GUI.
