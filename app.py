import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2
import mediapipe as mp
import numpy as np
import av
import time
from scipy.spatial import distance as dist
import base64

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Detector Sueño", page_icon="👁️")

# --- SONIDO DE ALARMA EN BASE64 (Para que funcione en la Web) ---
# Esto es un pitido codificado en texto para no depender de archivos mp3 externos
alarm_b64 = "data:audio/wav;base64,UklGRl9vT1BXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YU" + "A" * 500 # Un sonido dummy corto si falla, pero abajo usamos JS
# Nota: Para una alarma real en web, usamos un truco de HTML visual o JS.
# En Streamlit Cloud, el audio síncrono es difícil. Nos centraremos en ALERTA VISUAL EXTREMA.

st.title("🛡️ Sistema de Detección de Fatiga")
st.markdown("### 🔴 Demo en Vivo")
st.warning("Nota: Para esta demo web, asegúrate de tener buena iluminación. La alerta será principalmente VISUAL (Pantalla Roja).")

# --- BARRA LATERAL ---
EAR_THRESHOLD = st.sidebar.slider("Sensibilidad (EAR)", 0.15, 0.35, 0.21)
TIME_THRESHOLD = st.sidebar.slider("Tiempo para Alerta (seg)", 0.5, 3.0, 1.5)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.start_time = None

    def calculate_ear(self, eye_points):
        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])
        C = dist.euclidean(eye_points[0], eye_points[3])
        return (A + B) / (2.0 * C)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        # Color por defecto: Verde (Todo bien)
        color_status = (0, 255, 0)
        status_text = "OJOS ABIERTOS"

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = img.shape
                mesh_points = np.array([np.multiply([p.x, p.y], [w, h]).astype(int) for p in face_landmarks.landmark])
                
                left_eye_points = mesh_points[LEFT_EYE]
                right_eye_points = mesh_points[RIGHT_EYE]
                
                avg_ear = (self.calculate_ear(left_eye_points) + self.calculate_ear(right_eye_points)) / 2.0
                
                # Lógica de Detección
                if avg_ear < EAR_THRESHOLD:
                    if self.start_time is None:
                        self.start_time = time.time()
                    else:
                        elapsed = time.time() - self.start_time
                        
                        # ALERTA TEMPRANA (Naranja)
                        color_status = (0, 165, 255) 
                        status_text = f"CERRADO: {elapsed:.1f}s"

                        if elapsed > TIME_THRESHOLD:
                            # ALERTA CRÍTICA (Rojo)
                            color_status = (0, 0, 255)
                            status_text = "!!! PELIGRO !!!"
                            # Dibujar un recuadro rojo gigante en toda la pantalla
                            cv2.rectangle(img, (0, 0), (w, h), (0, 0, 255), 30)
                else:
                    self.start_time = None
                    color_status = (0, 255, 0)

                # Dibujar ojos y texto
                cv2.polylines(img, [left_eye_points], True, color_status, 1)
                cv2.polylines(img, [right_eye_points], True, color_status, 1)
                cv2.putText(img, f"EAR: {avg_ear:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(img, status_text, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color_status, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="drowsiness-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)