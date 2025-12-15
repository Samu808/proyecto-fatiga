import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2
import numpy as np
import av

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Detector Fatiga (Ajustable)", page_icon="👁️")

st.title("🛡️ Sistema de Detección - Modo Calibración")
st.info("ℹ️ Instrucciones: Usa los controles de la izquierda hasta que los recuadros verdes SOLO aparezcan en tus ojos abiertos.")

# --- CONTROLES DE CALIBRACIÓN (SIDEBAR) ---
st.sidebar.header("🔧 Calibración del Modelo")
# Scale Factor: Qué tanto reduce la imagen buscando. 1.1 es estándar.
scale_factor = st.sidebar.slider("Scale Factor (Precisión)", 1.05, 1.30, 1.20, 0.05)
# Min Neighbors: Cuántos rectángulos vecinos necesita para confirmar un ojo.
# MÁS ALTO = Menos falsos positivos (más estricto).
# MÁS BAJO = Detecta más fácil (pero confunde nariz/cejas).
min_neighbors = st.sidebar.slider("Min Neighbors (Estrictez)", 3, 10, 5)

# Umbral de Frames para Alerta
FRAMES_PARA_ALERTA = st.sidebar.slider("Sensibilidad de Sueño (Frames)", 10, 50, 20)

# Cargar clasificadores
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
except Exception as e:
    st.error("Error cargando modelos de OpenCV")

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.closed_eyes_frames = 0
        # Necesitamos leer los valores globales de los sliders
        self.scale_factor = 1.2
        self.min_neighbors = 5

    def update_params(self, scale, neighbors):
        self.scale_factor = scale
        self.min_neighbors = neighbors

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Detectar Rostro
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        status_text = "Esperando rostro..."
        color_status = (200, 200, 200)
        eyes_found = 0

        for (x, y, w, h) in faces:
            # Dibujar cara (Azul)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Recortar zona de la cara (ROI)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            # 2. Detectar Ojos (Usando parámetros dinámicos)
            # Nota: Leemos los valores globales definidos por los sliders
            eyes = eye_cascade.detectMultiScale(
                roi_gray, 
                scaleFactor=scale_factor, 
                minNeighbors=min_neighbors
            )
            
            eyes_found = len(eyes)

            # Dibujar ojos encontrados (Verde)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            # Lógica de Sueño:
            # Si detecta MENOS de 2 ojos, asumimos que están cerrados o parpadeando
            if eyes_found < 2:
                self.closed_eyes_frames += 1
            else:
                self.closed_eyes_frames = 0 

            # Alerta
            if self.closed_eyes_frames > FRAMES_PARA_ALERTA:
                status_text = "¡DORMIDO!"
                color_status = (0, 0, 255)
                cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 30)
            else:
                status_text = "DESPIERTO"
                color_status = (0, 255, 0)
        
        # Información en pantalla para depurar
        cv2.putText(img, f"Estado: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_status, 2)
        cv2.putText(img, f"Ojos detectados: {eyes_found}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Renderizar WebRTC
ctx = webrtc_streamer(
    key="universal-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Truco para pasar los valores de los sliders al procesador de video en tiempo real
if ctx.video_processor:
    ctx.video_processor.update_params(scale_factor, min_neighbors)

