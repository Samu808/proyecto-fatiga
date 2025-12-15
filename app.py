import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2
import numpy as np
import av
import time

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Detector Fatiga (Universal)", page_icon="👁️")

st.title("🛡️ Sistema de Detección de Fatiga")
st.warning("Modo Compatibilidad: Usando detección óptica estándar (Haar Cascades).")

# Cargar los clasificadores (vienen incluidos en OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.start_time = None
        # Buffer para suavizar la detección (evitar falsos positivos por parpadeo)
        self.closed_eyes_frames = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        status_text = "BUSCANDO CONDUCTOR..."
        color_status = (200, 200, 200)

        for (x, y, w, h) in faces:
            # Dibujar recuadro de la cara
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Recortar la zona de la cara (ROI) para buscar ojos solo ahí
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            # Detectar ojos dentro de la cara
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
            
            # Lógica Simplificada: Si detecta menos de 2 ojos, asume cerrados/no visibles
            if len(eyes) < 2:
                self.closed_eyes_frames += 1
            else:
                self.closed_eyes_frames = 0 # Resetear si ve ojos
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            # Umbral de tiempo (aprox 30 frames = 1 segundo)
            if self.closed_eyes_frames > 20:
                status_text = "¡ALERTA DE SUEÑO!"
                color_status = (0, 0, 255)
                # Recuadro Rojo Gigante
                cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 30)
            else:
                status_text = "OJOS ABIERTOS"
                color_status = (0, 255, 0)
        
        cv2.putText(img, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_status, 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="universal-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
