import cv2
import streamlit as st
from gtts import gTTS
import tempfile
import base64

st.title("🤖 RoboKalam Live Face Detection & Greeting")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Access webcam
video_capture = cv2.VideoCapture(0)

# Generate greeting audio (if not already done)
greeting_text = "Hello friend! Welcome to RoboKalam"
tts = gTTS(greeting_text)
temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
tts.save(temp_audio.name)

# Flag to check if audio already played
audio_played = False

stframe = st.empty()

while True:
    ret, frame = video_capture.read()
    if not ret:
        st.write("Cannot access camera")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangle around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert frame to RGB for Streamlit
    stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    # If face detected and audio not played yet
    if len(faces) > 0 and not audio_played:
        audio_played = True  # prevent replaying continuously

        # Read mp3 and encode to base64
        with open(temp_audio.name, "rb") as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode()

        # HTML audio tag with autoplay
        audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        """
        st.components.v1.html(audio_html, height=0, width=0)
