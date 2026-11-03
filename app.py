import streamlit as st
import torch
import cv2
import numpy as np
import joblib
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
import os

# Page Config
st.set_page_config(page_title="DeepGuard AI - Deepfake Detector", page_icon="🛡️", layout="centered")

@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224').to(device)
    vit_model.eval()
    
    # Load trained Random Forest model
    clf = joblib.load("deepguard_supervised_rf_model.pkl")
    return processor, vit_model, clf, device

processor, vit_model, clf, device = load_models()

st.title("🛡️ DeepGuard AI: Deepfake Video Detection System")
st.write("Apni video upload karein aur AI-powered model se check karein ki video **Real** hai ya **Fake**!")

uploaded_file = st.file_uploader("Ek .mp4 video upload karein", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())
        
    st.video(temp_video_path)
    
    if st.button("🔍 Video Analyze Karein"):
        with st.spinner("ViT embeddings extract ho rahi hain aur model analyze kar raha hai..."):
            cap = cv2.VideoCapture(temp_video_path)
            embeddings = []
            count = 0
            max_frames = 10
            
            while cap.isOpened() and count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    
                    inputs = processor(images=pil_image, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = vit_model(**inputs)
                    vec = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
                    embeddings.append(vec)
                    count += 1
                except Exception as e:
                    continue
            cap.release()
            
            if len(embeddings) > 0:
                final_feat = np.mean(embeddings, axis=0).reshape(1, -1)
                pred = clf.predict(final_feat)[0]
                proba = clf.predict_proba(final_feat)[0]
                
                confidence = proba[pred] * 100
                
                st.markdown("---")
                st.subheader("📊 Prediction Result:")
                if pred == 1:
                    st.error(f"🚨 Prediction: **FAKE (Manipulated Video)** (Confidence: {confidence:.2f}%)")
                else:
                    st.success(f"✅ Prediction: **REAL (Authentic Video)** (Confidence: {confidence:.2f}%)")
            else:
                st.error("Video se frames extract nahi ho paaye. Kripya doosri video try karein.")
    
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)