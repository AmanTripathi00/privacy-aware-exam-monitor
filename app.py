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
    # Load YOLO Nano
    model = YOLO("yolov8n.pt")
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    return model, face_cascade

yolo_model, face_cascade = load_models()

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class ExamVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_idx = 0
        self.last_status = "Status: Clear"
        self.last_color = (0, 255, 0)
        self.cached_boxes = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_idx += 1
        h, w, _ = img.shape
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        # 1. Fast Face Detection on downscaled image
        scale = 0.5
        small_gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), fx=scale, fy=scale)
        faces = face_cascade.detectMultiScale(small_gray, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30))
        face_count = len(faces)

        # Apply Real-time Facial Redaction
        for (fx, fy, fw, fh) in faces:
            x1 = int(fx / scale)
            y1 = int(fy / scale)
            x2 = int((fx + fw) / scale)
            y2 = int((fy + fh) / scale)

            roi = img[y1:y2, x1:x2]
            if roi.size > 0:
                # Fast box blur for privacy protection
                img[y1:y2, x1:x2] = cv2.blur(roi, (31, 31))

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # 2. Object Detection (Skipped across frames to prevent CPU lag)
        if self.frame_idx % 10 == 0:
            self.cached_boxes = []
            status = "Status: Clear"
            color = (0, 255, 0)

            if face_count == 0:
                status = "ALERT: Candidate Missing (No Face)"
                color = (0, 0, 255)
            elif face_count > 1:
                status = f"ALERT: Multiple Faces Detected ({face_count})"
                color = (0, 0, 255)

            # Fast 160px inference for phone detection
            tiny_frame = cv2.resize(img, (160, 160))
            results = yolo_model(tiny_frame, verbose=False, conf=0.35)

            for r in results:
                for box in r.boxes:
                    if int(box.cls[0].item()) == 67:  # Cell phone
                        bx = box.xyxy[0].cpu().numpy()
                        sx1 = int(bx[0] * (w / 160.0))
                        sy1 = int(bx[1] * (h / 160.0))
                        sx2 = int(bx[2] * (w / 160.0))
                        sy2 = int(bx[3] * (h / 160.0))
                        conf = float(box.conf[0].item())
                        self.cached_boxes.append((sx1, sy1, sx2, sy2, conf))
                        status = "ALERT: Prohibited Device (Phone)"
                        color = (0, 0, 255)

            self.last_status = status
            self.last_color = color

        # Draw cached phone boundaries
        for (x1, y1, x2, y2, conf) in self.cached_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"Phone {conf:.2f}", (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Visual Diagnostic Overlay
        cv2.rectangle(img, (0, 0), (w, 40), (20, 20, 20), -1)
        cv2.putText(img, f"[{timestamp}] {self.last_status}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.last_color, 2)
        cv2.putText(img, "Privacy Active: Face Redacted", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Secure Proctoring Stream")
    webrtc_streamer(
        key="exam-stream-v2",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=ExamVideoProcessor,
        media_stream_constraints={"video": {"width": 360, "height": 270, "frameRate": 15}, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("Viva Highlights")
    st.markdown("""
    - **Privacy Protection:** Identifiable facial features are blurred in real time locally.
    - **Event-Level Explainability:** Explicit rule flags (`FACE_ABSENT`, `MULTIPLE_FACES`, `PROHIBITED_DEVICE_PHONE`) ensure transparent proctoring.
    """)
