import streamlit as st
import cv2
from gtts import gTTS
from playsound import playsound
import os
import threading
import time

st.title("RoboKalam Face Detection with Voice Greeting")

# -----------------------------
# 1️⃣ Prepare greeting
# -----------------------------
greeting_text = "Hello friend! Welcome to RoboKalam."
greeting_file = "greeting.mp3"

# Generate mp3 if it doesn't exist
if not os.path.exists(greeting_file):
    tts = gTTS(greeting_text, lang="en")
    tts.save(greeting_file)

def play_greeting():
    playsound(greeting_file)

# -----------------------------
# 2️⃣ Initialize face detection
# -----------------------------
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
frame_placeholder = st.empty()

# -----------------------------
# 3️⃣ Start camera
# -----------------------------
cap = cv2.VideoCapture(0)
st.info("Camera is starting...")

# Variables for cooldown
last_greeting_time = 0
cooldown_seconds = 5  # Wait 5 seconds before next greeting

while True:
    ret, frame = cap.read()
    if not ret:
        st.warning("Camera not available.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Play greeting if faces detected AND cooldown has passed
    current_time = time.time()
    if len(faces) > 0 and (current_time - last_greeting_time) > cooldown_seconds:
        threading.Thread(target=play_greeting).start()
        last_greeting_time = current_time  # Reset cooldown timer

    # Display frame in Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB")

    # Stop if Streamlit session is stopped
    if st.session_state.get("stop_camera", False):
        break

cap.release()
