import cv2
import streamlit as st
from gtts import gTTS
import tempfile
import base64
import numpy as np
from PIL import Image

st.title("🤖 RoboKalam Live Face Detection & Greeting")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Generate greeting audio once
greeting_text = "Hello friend! Welcome to RoboKalam"
tts = gTTS(greeting_text)
temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
tts.save(temp_audio.name)

# This flag ensures audio plays only once per session
if "audio_played" not in st.session_state:
    st.session_state.audio_played = False

st.write("Allow camera for face detection")

# Single camera input widget
img_file = st.camera_input("Camera", key="camera1")

if img_file is not None:
    image = Image.open(img_file)
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if len(faces) > 0 and not st.session_state.audio_played:
        st.session_state.audio_played = True
        with open(temp_audio.name, "rb") as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        """
        st.components.v1.html(audio_html, height=0, width=0)
