import streamlit as st
import cv2
from gtts import gTTS
import numpy as np
import threading
import os

st.title("RoboKalam Live Face Detection")

# -----------------------------
# 1️⃣ Prepare greeting
# -----------------------------
greeting_text = "Hello friend! Welcome to RoboKalam."
greeting_file = "greeting.mp3"

# Generate greeting once
if not os.path.exists(greeting_file):
    tts = gTTS(greeting_text)
    tts.save(greeting_file)

# Function to play greeting
def play_greeting():
    threading.Thread(target=lambda: os.system(f"mpg123 {greeting_file}")).start()

# -----------------------------
# 2️⃣ Load face detection cascade
# -----------------------------
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

st.info("Click the camera button below to take a photo. Faces will be detected and greeting played automatically.")

# -----------------------------
# 3️⃣ Camera input loop
# -----------------------------
last_greeting_time = 0
cooldown_seconds = 5  # Wait 5 seconds before next greeting

while True:
    img_file = st.camera_input("Take a photo")
    if img_file is None:
        break  # No image yet

    # Convert image to OpenCV format
    bytes_data = img_file.getvalue()
    nparr = np.frombuffer(bytes_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Play greeting if face detected and cooldown passed
    import time
    current_time = time.time()
    if len(faces) > 0 and (current_time - last_greeting_time) > cooldown_seconds:
        play_greeting()
        last_greeting_time = current_time

    # Display result
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Detected Faces")
