import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
import cv2
import numpy as np
from ultralytics import YOLO
import datetime
import urllib.request
import os

st.set_page_config(page_title="Privacy Exam Monitor", layout="wide")

st.title("🛡️ Privacy-Aware Exam Integrity Monitor")
st.caption("Event-Level Explainability & Local Real-time Redaction")

# Ensure cascade file is guaranteed to exist locally
CASCADE_FILE = "haarcascade_frontalface_default.xml"
if not os.path.exists(CASCADE_FILE):
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, CASCADE_FILE)

@st.cache_resource
def load_models():
    yolo = YOLO("yolov8n.pt")
    face_cascade = cv2.CascadeClassifier(CASCADE_FILE)
    return yolo, face_cascade

yolo_model, face_cascade = load_models()

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class ExamVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.status_text = "Status: Clear"
        self.status_color = (0, 255, 0)
        self.detected_phones = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1
            h, w, _ = img.shape
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")

            # 1. Real-Time Face Anonymization (Runs on every frame)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Downscale for instant cascade detection
            small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
            faces = face_cascade.detectMultiScale(small_gray, scaleFactor=1.2, minNeighbors=3, minSize=(25, 25))

            face_count = len(faces)

            for (fx, fy, fw, fh) in faces:
                # Upscale coordinates back to actual frame size
                x1, y1 = fx * 2, fy * 2
                x2, y2 = (fx + fw) * 2, (fy + fh) * 2

                # Enforce boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Privacy Obfuscation (Gaussian/Box blur)
                face_roi = img[y1:y2, x1:x2]
                if face_roi.size > 0:
                    img[y1:y2, x1:x2] = cv2.blur(face_roi, (41, 41))

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # 2. Object & Absence Detection (Evaluated once every 6 frames)
            if self.frame_count % 6 == 0:
                self.detected_phones = []
                current_status = "Status: Clear"
                current_color = (0, 255, 0)

                if face_count == 0:
                    current_status = "ALERT: Candidate Missing (No Face)"
                    current_color = (0, 0, 255)
                elif face_count > 1:
                    current_status = f"ALERT: Multiple Faces Detected ({face_count})"
                    current_color = (0, 0, 255)

                # Nano YOLO inference on compact frame
                blob = cv2.resize(img, (160, 160))
                preds = yolo_model(blob, verbose=False, conf=0.35)

                for r in preds:
                    for box in r.boxes:
                        if int(box.cls[0].item()) == 67:  # Cell phone
                            bx = box.xyxy[0].cpu().numpy()
                            bx_full = [
                                int(bx[0] * (w / 160.0)),
                                int(bx[1] * (h / 160.0)),
                                int(bx[2] * (w / 160.0)),
                                int(bx[3] * (h / 160.0))
                            ]
                            conf = float(box.conf[0].item())
                            self.detected_phones.append((bx_full, conf))
                            current_status = "ALERT: Prohibited Device (Phone)"
                            current_color = (0, 0, 255)

                self.status_text = current_status
                self.status_color = current_color

            # Draw cached phone bounding boxes
            for (bx, conf) in self.detected_phones:
                cv2.rectangle(img, (bx[0], bx[1]), (bx[2], bx[3]), (0, 0, 255), 3)
                cv2.putText(img, f"Phone: {conf:.2f}", (bx[0], max(30, bx[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Explainability HUD Banner
            cv2.rectangle(img, (0, 0), (w, 45), (30, 30, 30), -1)
            cv2.putText(img, f"[{timestamp}] {self.status_text}", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.status_color, 2)
            cv2.putText(img, "Privacy Mode: Real-time Biometric Masking", (12, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

        except Exception as e:
            # Fallback keeps stream alive even if an individual frame drops
            return frame

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Secure Proctoring Stream")
    webrtc_streamer(
        key="exam-secure-monitor",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=ExamVideoProcessor,
        media_stream_constraints={"video": {"width": 480, "height": 360}, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("Viva Highlights")
    st.markdown("""
    - **Privacy Protection:** Identifiable facial features are blurred in real time locally.
    - **Event-Level Explainability:** Explicit rule flags (`FACE_ABSENT`, `MULTIPLE_FACES`, `PROHIBITED_DEVICE_PHONE`) ensure transparent proctoring.
    """)
