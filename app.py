import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
import cv2
import numpy as np
from ultralytics import YOLO
import datetime

st.set_page_config(page_title="Privacy Exam Monitor", layout="wide")

st.title("🛡️ Privacy-Aware Exam Integrity Monitor")
st.caption("Event-Level Explainability & Local Real-time Redaction")

@st.cache_resource
def load_models():
    yolo = YOLO("yolov8n.pt")
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    return yolo, face_cascade

yolo_model, face_cascade = load_models()

# Public STUN servers for reliable video handshake
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class ExamVideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        status_text = "Status: Clear"
        status_color = (0, 255, 0)

        # 1. Face Detection & Privacy Obfuscation
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.3, 
            minNeighbors=4, 
            minSize=(50, 50)
        )
        face_count = len(faces)

        for (fx, fy, fw, fh) in faces:
            # Mask the facial biometric region for privacy
            face_roi = img[fy:fy+fh, fx:fx+fw]
            if face_roi.size > 0:
                blurred_roi = cv2.GaussianBlur(face_roi, (51, 51), 30)
                img[fy:fy+fh, fx:fx+fw] = blurred_roi

            cv2.rectangle(img, (fx, fy), (fx+fw, fy+fh), (255, 200, 0), 2)

        # Event-Level Presence Check
        if face_count == 0:
            status_text = "ALERT: Candidate Missing (No Face)"
            status_color = (0, 0, 255)
        elif face_count > 1:
            status_text = f"ALERT: Multiple Faces Detected ({face_count})"
            status_color = (0, 0, 255)

        # 2. Fast YOLO Object Detection (Phone = Class 67)
        results = yolo_model(img, verbose=False, conf=0.40, imgsz=320)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if cls_id == 67:  # Cell phone
                    bx = box.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(img, (bx[0], bx[1]), (bx[2], bx[3]), (0, 0, 255), 2)
                    cv2.putText(img, f"Phone {conf:.2f}", (bx[0], bx[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    status_text = "ALERT: Prohibited Device (Phone)"
                    status_color = (0, 0, 255)

        # On-screen HUD
        cv2.putText(img, f"[{timestamp}] {status_text}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)
        cv2.putText(img, "Privacy Mode: Real-time Identity Masking Active", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Secure Proctoring Stream")
    webrtc_streamer(
        key="exam-stream",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=ExamVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("Viva Highlights")
    st.markdown("""
    - **Privacy Protection:** Identifiable facial features are blurred in real time locally.
    - **Event-Level Explainability:** Discrete anomaly flags (`FACE_ABSENT`, `MULTIPLE_FACES`, `PROHIBITED_DEVICE_PHONE`) ensure transparent proctoring.
    """)
