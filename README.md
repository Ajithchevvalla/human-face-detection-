# RoboKalam Live Face Detection (WebRTC Version)

This project uses Streamlit + WebRTC to provide:

- Live webcam streaming
- Real-time face detection
- Automatic greeting voice using gTTS
- Works on cloud platforms like Render, Railway, Heroku, etc.

## 🚀 Deployment on Render.com

1. Upload files to GitHub
2. Go to https://render.com
3. Click New → Web Service
4. Choose your GitHub repo
5. Use these commands:

### Build Command
pip install -r requirements.txt

### Start Command
streamlit run app.py --server.port $PORT --server.address 0.0.0.0

6. Deploy and open the link.
