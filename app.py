import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2
import numpy as np
import av

# --- CONFIGURACIÓN DE PÁGINA ---
# Cambié el ícono y el título para que sepas si se actualizó
st.set_page_config(page_title="Detector PRO v2", page_icon="👀")

st.title("🛡️ Detector de Sueño - Versión Alta Sensibilidad")
st.markdown("""
**Instrucciones de Calibración:**
1. Sube la **"Exigencia del Ojo"** hasta que los cuadros verdes **DESAPAREZCAN** cuando cierras los ojos relajadamente.
2. Si desaparecen incluso con los ojos abiertos, baja un poco el número.
""")

# --- CONTROLES DE CALIBRACIÓN (Rango Ampliado) ---
st.sidebar.header("🔧 Panel de Control")

# RANGO AMPLIADO: De 1 a 30.
# Valor por defecto: 12 (Bastante estricto para empezar)
min_neighbors = st.sidebar.slider("Exigencia del Ojo (Strictness)", 1, 30, 12, 
    help="Sube este número para eliminar falsos positivos. Valor alto = El modelo exige un ojo perfecto.")

umbral_sueno = st.sidebar.slider("Velocidad de Alerta (Frames)", 5, 50, 15, 
    help="Menor número = Alarma más rápida.")

# Cargar modelos de OpenCV (Haar Cascades)
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
except Exception as e:
    st.error(f"Error cargando modelos: {e}")

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.drowsy_counter = 0
        self.min_neighbors = 12 # Valor inicial igual al del slider
        self.umbral = 15

    def update_params(self, neighbors, umbral):
        self.min_neighbors = neighbors
        self.umbral = umbral

    def recv(self, frame):
        # Convertir frame de video a formato NumPy
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Efecto espejo
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Detectar Rostro
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        face_detected = False
        eyes_detected = 0
        status_color = (0, 255, 0)
        status_text = "SISTEMA ACTIVO"

        if len(faces) > 0:
            face_detected = True
            for (x, y, w, h) in faces:
                # Dibujar recuadro de la cara (Azul)
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                
                # 2. Detectar Ojos (Usando el parámetro dinámico min_neighbors)
                eyes = eye_cascade.detectMultiScale(
                    roi_gray, 
                    scaleFactor=1.1, 
                    minNeighbors=self.min_neighbors # <--- AQUÍ ESTÁ LA CLAVE
                )
                eyes_detected = len(eyes)
                
                # Dibujar ojos encontrados (Verde)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # --- LÓGICA DE ALERTA ---
        if face_detected:
            # Si veo la cara pero NO veo al menos 2 ojos...
            if eyes_detected < 2:
                self.drowsy_counter += 1
                status_text = f"¡OJOS NO DETECTADOS! ({self.drowsy_counter})"
                status_color = (0, 165, 255) # Naranja
            else:
                # Si veo ojos, el contador baja (te estás recuperando)
                if self.drowsy_counter > 0:
                    self.drowsy_counter -= 1
                status_text = "OJOS VISIBLES - OK"
                status_color = (0, 255, 0) # Verde
        else:
            status_text = "BUSCANDO ROSTRO..."
            status_color = (200, 200, 200)
            if self.drowsy_counter > 0:
                self.drowsy_counter -= 1

        # --- DIBUJAR BARRA DE PROGRESO DE SUEÑO ---
        # Calcular ancho de la barra
        progreso = min(self.drowsy_counter / self.umbral, 1.0)
        bar_width = int(progreso * 200)
        
        # Color de la barra (Verde -> Naranja -> Rojo)
        bar_color = (0, 255, 0)
        if progreso > 0.5: bar_color = (0, 165, 255)
        if progreso >= 1.0: bar_color = (0, 0, 255)

        # Fondo Gris
        cv2.rectangle(img, (20, 100), (220, 130), (50, 50, 50), -1)
        # Barra de Color
        cv2.rectangle(img, (20, 100), (20 + bar_width, 130), bar_color, -1)
        cv2.putText(img, "NIVEL DE FATIGA", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # --- DISPARO DE ALARMA VISUAL ---
        if self.drowsy_counter >= self.umbral:
            status_text = "!!! ALERTA DE SUEÑO !!!"
            # RECUADRO ROJO GIGANTE EN TODA LA PANTALLA
            h, w, _ = img.shape
            cv2.rectangle(img, (0, 0), (w, h), (0, 0, 255), 50)

        # Poner texto de estado
        cv2.putText(img, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- COMPONENTE WEBRTC ---
ctx = webrtc_streamer(
    key="detector-pro-v2",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# --- ENVIAR PARÁMETROS DE SLIDERS AL PROCESADOR ---
if ctx.video_processor:
    ctx.video_processor.update_params(min_neighbors, umbral_sueno)

if ctx.video_processor:
    ctx.video_processor.update_params(min_neighbors, umbral_sueno)



