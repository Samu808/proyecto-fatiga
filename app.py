import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2
import numpy as np
import av

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Detector Fatiga", page_icon="🚫")

st.title("🛡️ Demo: Detector de Sueño")
st.markdown("""
**Cómo funciona esta Demo:**
1. El sistema busca tu **ROSTRO** (Cuadro Azul).
2. Dentro del rostro, busca **OJOS ABIERTOS** (Cuadros Verdes).
3. Si el cuadro verde **DESAPARECE**, el sistema asume que cerraste los ojos y la **Barra de Peligro** sube.
""")

# --- CONTROLES ---
st.sidebar.header("Ajustes")
min_neighbors = st.sidebar.slider("Exigencia del Ojo", 2, 10, 5, help="Baja este número si no te detecta los ojos abiertos.")
umbral_sueno = st.sidebar.slider("Velocidad de Alerta", 10, 60, 20, help="Qué tan rápido salta la alarma.")

# Cargar modelos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.drowsy_counter = 0
        self.min_neighbors = 5
        self.umbral = 20

    def update_params(self, neighbors, umbral):
        self.min_neighbors = neighbors
        self.umbral = umbral

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Detectar Rostro
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Variables de estado para este frame
        face_detected = False
        eyes_detected = 0
        status_color = (0, 255, 0)
        status_text = "SISTEMA ACTIVO"

        if len(faces) > 0:
            face_detected = True
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                
                # 2. Detectar Ojos
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, self.min_neighbors)
                eyes_detected = len(eyes)
                
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # --- LÓGICA DE DETECCIÓN POR AUSENCIA ---
        if face_detected:
            # Si veo la cara, PERO veo menos de 2 ojos -> Posible sueño
            if eyes_detected < 2:
                self.drowsy_counter += 1
                status_text = f"¡OJOS PERDIDOS! ({self.drowsy_counter}/{self.umbral})"
                status_color = (0, 165, 255) # Naranja
            else:
                # Si veo ojos, bajo el contador (recuperación)
                if self.drowsy_counter > 0:
                    self.drowsy_counter -= 1
                status_text = "OJOS VISIBLES"
                status_color = (0, 255, 0) # Verde
        else:
            status_text = "BUSCANDO ROSTRO..."
            status_color = (200, 200, 200) # Gris
            # Si no hay cara, mantenemos el contador o lo bajamos lento
            if self.drowsy_counter > 0:
                self.drowsy_counter -= 1

        # --- DISPARO DE ALARMA ---
        # Dibujar barra de progreso del sueño
        bar_width = int((self.drowsy_counter / self.umbral) * 200)
        bar_width = min(bar_width, 200) # Tope visual
        
        # Fondo barra
        cv2.rectangle(img, (20, 100), (220, 130), (50, 50, 50), -1) 
        # Relleno barra (Cambia de color si es peligroso)
        bar_color = (0, 255, 0)
        if self.drowsy_counter > self.umbral / 2: bar_color = (0, 165, 255)
        if self.drowsy_counter >= self.umbral: bar_color = (0, 0, 255)
            
        cv2.rectangle(img, (20, 100), (20 + bar_width, 130), bar_color, -1)
        cv2.putText(img, "NIVEL DE SUENO", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        if self.drowsy_counter >= self.umbral:
            status_text = "!!! ALERTA DE SUEÑO !!!"
            # Pantalla roja intermitente o borde grueso
            cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 40)

        # Textos de debug
        cv2.putText(img, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

ctx = webrtc_streamer(
    key="demo-final",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if ctx.video_processor:
    ctx.video_processor.update_params(min_neighbors, umbral_sueno)

