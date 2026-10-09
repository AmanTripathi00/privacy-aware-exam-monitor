import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
import cv2
import numpy as np
from ultralytics import YOLO
import datetime
import urllib.request
import os
import pandas as pd

st.set_page_config(page_title="AI Exam Integrity Copilot (BAI-06)", layout="wide", page_icon="🛡️")

# Custom CSS Styling for Dashboard Elements
st.markdown("""
    <style>
    .metric-card {
        background: #1E222D;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #00D4B2;
        margin-bottom: 10px;
        color: white;
    }
    .status-clear {
        background-color: #0E2F1F;
        border: 1px solid #00FF7F;
        color: #00FF7F;
        padding: 10px;
        border-radius: 8px;
        font-weight: bold;
    }
    .status-alert {
        background-color: #3B1212;
        border: 1px solid #FF4B4B;
        color: #FF4B4B;
        padding: 10px;
        border-radius: 8px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Privacy-Aware Exam Integrity Monitor")
st.caption("BAI-06 | Event-Level Explainability Engine with Active Evidence Capture")

# Absolute path setup for flagged evidence directory
EVIDENCE_DIR = os.path.join(os.getcwd(), "flagged_evidence")
if not os.path.exists(EVIDENCE_DIR):
    os.makedirs(EVIDENCE_DIR)

# Initialize Session State Variables
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

CASCADES = {
    "frontal": "haarcascade_frontalface_default.xml",
    "profile": "haarcascade_profileface.xml"
}

BASE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"
for name, filename in CASCADES.items():
    if not os.path.exists(filename):
        urllib.request.urlretrieve(BASE_URL + filename, filename)

@st.cache_resource
def load_models():
    yolo = YOLO("yolov8n.pt")
    frontal_cascade = cv2.CascadeClassifier(CASCADES["frontal"])
    profile_cascade = cv2.CascadeClassifier(CASCADES["profile"])
    return yolo, frontal_cascade, profile_cascade

yolo_model, frontal_cascade, profile_cascade = load_models()

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class ExamIntegrityProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_idx = 0
        self.status = "Status: Clear"
        self.color = (0, 255, 0)
        self.cached_boxes = []
        self.latest_event = None
        self.last_capture_time = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            self.frame_idx += 1
            h, w, _ = img.shape
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)

            # 1. Frontal & Profile Face Detection via Haar Cascades
            frontal_faces = frontal_cascade.detectMultiScale(
                small_gray, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30)
            )
            profile_faces = profile_cascade.detectMultiScale(
                small_gray, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30)
            )
            flipped_small_gray = cv2.flip(small_gray, 1)
            left_profile_faces = profile_cascade.detectMultiScale(
                flipped_small_gray, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30)
            )

            all_faces = list(frontal_faces) + list(profile_faces)
            for (lx, ly, lw, lh) in left_profile_faces:
                all_faces.append((small_gray.shape[1] - lx - lw, ly, lw, lh))

            face_count = len(frontal_faces) + len(profile_faces) + len(left_profile_faces)

            # 2. Real-Time Facial Redaction (Privacy Protocol)
            for (fx, fy, fw, fh) in all_faces:
                x1, y1 = max(0, fx * 2), max(0, fy * 2)
                x2, y2 = min(w, (fx + fw) * 2), min(h, (fy + fh) * 2)

                roi = img[y1:y2, x1:x2]
                if roi.size > 0:
                    img[y1:y2, x1:x2] = cv2.blur(roi, (45, 45))

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # 3. Rule Reasoning Engine (Evaluated every 6 frames)
            event_type = None
            explanation = "Candidate compliant."

            if self.frame_idx % 6 == 0:
                self.cached_boxes = []
                cur_status = "Status: Clear"
                cur_color = (0, 255, 0)

                if face_count == 0:
                    cur_status = "ALERT: Candidate Missing"
                    cur_color = (0, 0, 255)
                    event_type = "Absence Violation"
                    explanation = "No primary face detected in viewport."
                elif len(frontal_faces) == 0 and (len(profile_faces) > 0 or len(left_profile_faces) > 0):
                    cur_status = "ALERT: Head Turned / Gaze Away"
                    cur_color = (0, 165, 255)
                    event_type = "Pose Violation"
                    explanation = "Side profile detected; gaze vector lost."
                elif face_count > 1:
                    cur_status = f"ALERT: Multiple People ({face_count})"
                    cur_color = (0, 0, 255)
                    event_type = "Collusion Violation"
                    explanation = f"Detected {face_count} distinct face candidates."

                # Prohibited Object Detection (Phone = 67, Book = 73, Laptop = 63)
                blob = cv2.resize(img, (160, 160))
                preds = yolo_model(blob, verbose=False, conf=0.35)

                for r in preds:
                    for box in r.boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        label = None
                        if cls_id == 67:
                            label = f"Phone {conf:.2f}"
                        elif cls_id == 73:
                            label = f"Book {conf:.2f}"
                        elif cls_id == 63:
                            label = f"Laptop {conf:.2f}"

                        if label:
                            bx = box.xyxy[0].cpu().numpy()
                            bx_full = [
                                int(bx[0] * (w / 160.0)),
                                int(bx[1] * (h / 160.0)),
                                int(bx[2] * (w / 160.0)),
                                int(bx[3] * (h / 160.0))
                            ]
                            self.cached_boxes.append((bx_full, label))
                            cur_status = f"ALERT: Prohibited Object ({label.split()[0]})"
                            cur_color = (0, 0, 255)
                            event_type = "Prohibited Object"
                            explanation = f"Unauthorized object detected: {label}."

                self.status = cur_status
                self.color = cur_color

            # 4. Draw Red Bounding Boxes & Text Labels BEFORE Saving Image
            for (bx, label) in self.cached_boxes:
                cv2.rectangle(img, (bx[0], bx[1]), (bx[2], bx[3]), (0, 0, 255), 2)
                cv2.putText(img, label, (bx[0], max(25, bx[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

            # Draw HUD Status Banner
            cv2.rectangle(img, (0, 0), (w, 40), (25, 25, 25), -1)
            cv2.putText(img, f"[{timestamp}] {self.status}", (12, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color, 2)
            cv2.putText(img, "Privacy Layer: Active Redaction", (12, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            # 5. Save Evidence Screenshot (Includes drawn bounding boxes & top HUD)
            now_sec = datetime.datetime.now().timestamp()
            if event_type and (now_sec - self.last_capture_time > 3.0):
                self.last_capture_time = now_sec
                file_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                img_filename = os.path.join(EVIDENCE_DIR, f"{file_stamp}_{event_type.replace(' ', '_')}.jpg")
                cv2.imwrite(img_filename, img)

                self.latest_event = {
                    "Timestamp": timestamp,
                    "Event": event_type,
                    "Status": self.status,
                    "Explanation": explanation,
                    "Image": img_filename
                }

            return av.VideoFrame.from_ndarray(img, format="bgr24")

        except Exception:
            return frame

# Dashboard Layout Setup
stream_col, audit_col = st.columns([1.5, 1])

with stream_col:
    st.subheader("📹 Live Stream & Privacy Filter")
    ctx = webrtc_streamer(
        key="proctoring-copilot",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=ExamIntegrityProcessor,
        media_stream_constraints={"video": {"width": 480, "height": 360}, "audio": False},
        async_processing=True,
    )

with audit_col:
    st.subheader("📋 Proctor Control Panel")
    
    # Live Refresh / Sync Button
    if st.button("🔄 Sync Live Stream Data & Refresh Gallery"):
        st.rerun()

    # Catch live violation alerts
    if ctx.video_processor and ctx.video_processor.latest_event:
        evt = ctx.video_processor.latest_event
        if not st.session_state.audit_log or st.session_state.audit_log[-1]["Timestamp"] != evt["Timestamp"]:
            st.session_state.audit_log.append(evt)

    # Active State Banner Display
    if st.session_state.audit_log:
        last_evt = st.session_state.audit_log[-1]
        st.markdown(f"""
        <div class="status-alert">
            ⚠️ <b>Active State:</b> {last_evt['Event']}<br>
            💬 <b>XAI Reason:</b> {last_evt['Explanation']}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-clear">
            ✅ <b>Active State:</b> Compliant Session
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown("#### ⚖️ Human Override Actions")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("✅ Approve"):
            st.success("Alert dismissed.")
    with col_b:
        if st.button("🚩 Flag Session"):
            st.warning("Session escalated.")
    with col_c:
        if st.button("🗑️ Clear Logs"):
            st.session_state.audit_log = []
            st.rerun()

    st.markdown("#### 📜 Incident Log")
    if st.session_state.audit_log:
        df_log = pd.DataFrame(st.session_state.audit_log)
        st.dataframe(df_log[["Timestamp", "Event", "Explanation"]], height=150, use_container_width=True)
        csv = df_log.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export CSV Log", data=csv, file_name="exam_audit.csv", mime="text/csv")

# Section Divider and Photo Management Gallery
st.markdown("---")
st.subheader("🖼️ Captured Evidence Gallery (Photo Audit Trail)")

photo_files = [f for f in os.listdir(EVIDENCE_DIR) if f.endswith(".jpg")]

if photo_files:
    cols = st.columns(3)
    for index, img_name in enumerate(photo_files):
        img_path = os.path.join(EVIDENCE_DIR, img_name)
        col = cols[index % 3]
        
        with col:
            st.image(img_path, caption=img_name, use_container_width=True)
            
            c1, c2 = st.columns(2)
            with c1:
                with open(img_path, "rb") as file:
                    st.download_button(
                        label="💾 Download",
                        data=file,
                        file_name=img_name,
                        mime="image/jpeg",
                        key=f"dl_{img_name}"
                    )
            with c2:
                if st.button("🗑️ Delete", key=f"del_{img_name}"):
                    os.remove(img_path)
                    st.toast(f"Deleted {img_name}")
                    st.rerun()

    if st.button("🔥 Delete All Evidence Screenshots"):
        for f in photo_files:
            os.remove(os.path.join(EVIDENCE_DIR, f))
        st.success("All evidence photos removed.")
        st.rerun()
else:
    st.info("No prohibited evidence screenshots recorded yet. (Click '🔄 Sync Live Stream Data & Refresh Gallery' above during video streaming to update).")

from streamlit_drawable_canvas import st_canvas

# Whiteboard / Scratchpad Section
st.markdown("---")
st.subheader("📝 Interactive Whiteboard / Proctor Notes Canvas")

col_tool, col_size, col_color = st.columns(3)
with col_tool:
    drawing_mode = st.selectbox(
        "Tool:", ("freedraw", "line", "rect", "circle", "transform")
    )
with col_size:
    stroke_width = st.slider("Brush Size: ", 1, 25, 3)
with col_color:
    stroke_color = st.color_picker("Brush Color: ", "#FF0000")

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color="#FFFFFF",
    height=350,
    width=700,
    drawing_mode=drawing_mode,
    key="proctor_whiteboard",
)
