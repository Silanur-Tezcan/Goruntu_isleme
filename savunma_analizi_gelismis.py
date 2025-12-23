import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
import os
import pygame

from playsound import playsound
# 1. Ses Sistemini Hazırla
if not pygame.mixer.get_init():
    pygame.mixer.init()

# 2. YOLOv8 Modelini Yükle
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt") 

model = load_yolo_model()

# Hedef Nesne Listesi (Sadece bunlar tespit edildiğinde alarm çalar)
TARGET_OBJECTS = ["airplane", "truck", "bus"] # İsteğe göre artırılabilir



def play_alarm():
    try:
        if os.path.exists("alarm.mp3"):
            playsound("alarm.mp3", block=False) # block=False videonun donmasını engeller
    except Exception as e:
        print(f"Hata: {e}")

# --- ARAYÜZ ---
st.set_page_config(page_title="YZ Hedef Tespit Sistemi", page_icon="🚨", layout="wide")

st.title("🚨 Stratejik Video Analiz ve Alarm Sistemi")
st.markdown("Video akışında kritik bir hedef belirlendiğinde sistem otomatik olarak sesli uyarı verir.")

# Yan Panel Ayarları
st.sidebar.header("Sistem Ayarları")
conf_threshold = st.sidebar.slider("Güven Eşiği (Hassasiyet)", 0.1, 1.0, 0.3)
ses_aktif = st.sidebar.toggle("Sesli Alarmı Etkinleştir", value=True)

# --- VİDEO İŞLEME ALANI ---
uploaded_video = st.file_uploader("Analiz edilecek videoyu yükleyin", type=["mp4", "avi", "mov"])

if uploaded_video:
    # Geçici dosya oluşturma
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    
    vf = cv2.VideoCapture(tfile.name)
    st_video_frame = st.empty() # Video karesi için boş alan
    st_warning_area = st.empty() # Uyarı metni için boş alan

    while vf.isOpened():
        ret, frame = vf.read()
        if not ret:
            break

        # YOLO ile Tespit Yap
        results = model.predict(frame, conf=conf_threshold, verbose=False)
        
        detected_objects = []
        alert_trigger = False

        # Kare içindeki nesneleri kontrol et
        for box in results[0].boxes:
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            detected_objects.append(f"{label} (%{conf*100:.1f})")

            # Eğer kritik bir nesne bulunduysa alarmı tetikle
            if label in TARGET_OBJECTS:
                alert_trigger = True

        # Görseli Hazırla (YOLO çizimleri ile)
        annotated_frame = results[0].plot()
        st_video_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Alarm ve Uyarı Yönetimi
        if alert_trigger:
            st_warning_area.error(f"⚠️ KRİTİK HEDEF TESPİT EDİLDİ: {', '.join(detected_objects)}")
            if ses_aktif:
                play_alarm()
        else:
            st_warning_area.empty()

    vf.release()
    tfile.close()
    os.remove(tfile.name)
    st.success("Analiz tamamlandı.")