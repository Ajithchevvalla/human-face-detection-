import streamlit as st
import cv2
import numpy as np
from gtts import gTTS
import tempfile
import os
from PIL import Image

st.set_page_config(page_title="Live Face Detection", layout="centered")

st.title("👋 Live Face Detection & Greeting")

# Ask for camera input
img_file = st.camera_input("Take a photo", key="camera1")

if img_file is not None:
    # Convert uploaded image to OpenCV format
    img = Image.open(img_file)
    img_array = np.array(img)
    frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Load OpenCV pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangle around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Detected Faces", use_column_width=True)

    if len(faces) > 0:
        greeting_text = "Hello! I see you!"
        st.success(greeting_text)

        # Convert greeting to speech
        tts = gTTS(greeting_text)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        st.audio(temp_file.name, format="audio/mp3")
        os.unlink(temp_file.name)
    else:
        st.info("No face detected. Please try again.")
