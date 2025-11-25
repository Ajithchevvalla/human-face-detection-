import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64

st.set_page_config(page_title="Live Face Greeting", page_icon="🤖")
st.title("🤖 RoboKalam Live Face Detection & Greeting")

# Ask user to allow camera
snapshot = st.camera_input("Allow camera for face detection", key="camera1")

if snapshot is not None:
    # Convert snapshot to OpenCV image
    img = np.array(Image.open(snapshot))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        st.success("Hello friend! Welcome to RoboKalam 🤖")

        # Play greeting audio
        audio_file = "greeting.mp3"  # Place your greeting audio in the same folder
        audio_bytes = open(audio_file, "rb").read()
        st.audio(audio_bytes, format="audio/mp3")

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        st.info("No face detected. Please stay in front of the camera.")

    # Show image with rectangles
    st.image(img, channels="BGR")
