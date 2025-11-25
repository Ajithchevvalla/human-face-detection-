import streamlit as st
import cv2
import av
import time
import numpy as np
from gtts import gTTS
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.title("RoboKalam Live Face Detection with Auto Greeting")

# -----------------------------
# 1️⃣ Prepare greeting
# -----------------------------
greeting_text = "Hello friend! Welcome to RoboKalam."
greeting_file = "greeting.mp3"

# Generate greeting file once
tts = gTTS(greeting_text, lang="en")
tts.save(greeting_file)

# -----------------------------
# 2️⃣ Face detection model
# -----------------------------
cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# -----------------------------
# 3️⃣ Video Processing Class
# -----------------------------
class FaceDetector(VideoTransformerBase):

    last_greet = 0
    cooldown = 5   # seconds

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.2, 5)

        # Draw rectangles and trigger voice
        if len(faces) > 0:
            now = time.time()

            # Play greeting every cooldown seconds
            if now - FaceDetector.last_greet > FaceDetector.cooldown:
                st.audio("greeting.mp3", autoplay=True)
                FaceDetector.last_greet = now

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

        return img


# -----------------------------
# 4️⃣ Start WebRTC Camera
# -----------------------------
webrtc_streamer(
    key="face-detection",
    video_transformer_factory=FaceDetector,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
)
