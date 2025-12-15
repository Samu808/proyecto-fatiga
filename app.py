import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2
import numpy as np
import av
import queue
import time

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Detector PRO + Sonido", page_icon="🔊")

# --- SONIDO BEEP (Base64) ---
# Esto es un sonido de alarma codificado en texto para no necesitar archivos externos
BEEP_SOUND = """
<audio autoplay>
  <source src="data:audio/wav;base64,UklGRl9vT1BXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YU" type="audio/wav">
</audio>
"""
# Nota: El string de arriba es un placeholder corto. 
# Para un sonido real, inyectaremos un script de JS abajo que es más confiable.

st.title("🔊 Detector de Sueño con Alarma")

st.markdown("""
**Instrucciones para activar audio:**
1. Sube el volumen de tu computador.
2. Cuando le des a **START**, el navegador puede pedirte permiso de sonido o mostrar un icono de "silenciado" en la barra de dirección. ¡Habilítalo!
""")

# --- CONTROLES ---
st.sidebar.header("🔧 Panel de Control")
min_neighbors = st.sidebar.slider("Exigencia del Ojo (Strictness)", 1, 30, 12)
umbral_sueno = st.sidebar.slider("Velocidad de Alerta (Frames)", 5, 50, 15)

# Cargar modelos
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
except Exception as e:
    st.error(f"Error cargando modelos: {e}")

# --- COLA DE COMUNICACIÓN ---
# Aquí el video enviará mensajes a la web
result_queue = queue.Queue()

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.drowsy_counter = 0
        self.min_neighbors = 12
        self.umbral = 15

    def update_params(self, neighbors, umbral):
        self.min_neighbors = neighbors
        self.umbral = umbral

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        face_detected = False
        eyes_detected = 0
        alarm_trigger = False

        if len(faces) > 0:
            face_detected = True
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, self.min_neighbors)
                eyes_detected = len(eyes)
                
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # Lógica de Alarma
        if face_detected:
            if eyes_detected < 2:
                self.drowsy_counter += 1
            else:
                if self.drowsy_counter > 0: self.drowsy_counter -= 1
        else:
            if self.drowsy_counter > 0: self.drowsy_counter -= 1

        # DIBUJAR GUI
        progreso = min(self.drowsy_counter / self.umbral, 1.0)
        bar_width = int(progreso * 200)
        bar_color = (0, 255, 0)
        if progreso >= 1.0: 
            bar_color = (0, 0, 255)
            # --- SEÑAL DE ALARMA ---
            alarm_trigger = True
            
            # Recuadro Rojo
            h, w, _ = img.shape
            cv2.rectangle(img, (0, 0), (w, h), (0, 0, 255), 50)
            cv2.putText(img, "!!! ALERTA !!!", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)

        elif progreso > 0.5: 
            bar_color = (0, 165, 255)

        cv2.rectangle(img, (20, 100), (20 + bar_width, 130), bar_color, -1)
        cv2.putText(img, f"FATIGA: {int(progreso*100)}%", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Enviar señal a la cola (solo si hay alarma)
        if alarm_trigger:
            try:
                result_queue.put_nowait(True)
            except queue.Full:
                pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- COMPONENTE WEBRTC ---
ctx = webrtc_streamer(
    key="detector-con-sonido",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Actualizar parámetros
if ctx.video_processor:
    ctx.video_processor.update_params(min_neighbors, umbral_sueno)

# --- ESCUCHADOR DE AUDIO (El truco sucio) ---
# Esto revisa la cola constantemente para ver si hay que gritar
status_placeholder = st.empty()
audio_placeholder = st.empty()

if ctx.state.playing:
    while True:
        try:
            # Esperar 0.1s por un mensaje del video
            msg = result_queue.get(timeout=0.1)
            if msg:
                status_placeholder.error("🚨 ¡DETECTADO SUEÑO! 🚨")
                # Reproducir sonido usando HTML oculto
                # Usamos un sonido de alarma real online (link corto)
                audio_placeholder.markdown(
                    """
                    <audio autoplay>
                    <source src="https://www.soundjay.com/buttons/beep-01a.mp3" type="audio/mp3">
                    </audio>
                    """,
                    unsafe_allow_html=True
                )
                time.sleep(1) # Esperar a que suene para no spammear
                audio_placeholder.empty() # Limpiar para que pueda sonar de nuevo
        except queue.Empty:
            # Si no hay alarma, limpiamos el status
            status_placeholder.empty()





