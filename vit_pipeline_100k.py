import os
import glob
import cv2
import torch
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# =====================================================================
# 1. GPU CHECK & SETUP (RTX 5060 Enforcement)
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    print("\n⚠️ WARNING: GPU detect nahi hua! Code CPU par chalega.")
else:
    print(f"\n🚀 SUCCESS: Running on NVIDIA GPU -> {torch.cuda.get_device_name(0)}")

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTModel.from_pretrained('google/vit-base-patch16-224').to(device)
model.eval()

def extract_vit_embeddings(video_path, max_frames=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    embeddings = []
    count = 0
    
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            inputs = processor(images=pil_image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            vec = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
            embeddings.append(vec)
            count += 1
        except Exception as e:
            continue
        
    cap.release()
    if len(embeddings) == 0:
        return None
    return np.mean(embeddings, axis=0)

# =====================================================================
# 2. STAGE 2: AUTO-LABELING USING ViT EMBEDDINGS (768-D)
# =====================================================================
def run_stage2_auto_labeling():
    print("\n[STAGE 2] Starting Auto-Labeling on 768-D ViT Embeddings...")

    # Real videos path
    real_dir = "FaceForensics++_C23/original"
    real_vids = glob.glob(os.path.join(real_dir, "**", "*.mp4"), recursive=True)

    # All Fake folders paths combine karna
    base_path = "FaceForensics++_C23"
    fake_subfolders = ["Deepfakes", "Face2Face", "FaceSwap", "FaceShifter", "NeuralTextures", "DeepFakeDetection"]
    
    fake_vids = []
    for sub in fake_subfolders:
        folder_path = os.path.join(base_path, sub)
        found = glob.glob(os.path.join(folder_path, "**", "*.mp4"), recursive=True)
        fake_vids.extend(found)

    all_videos = real_vids + fake_vids
    if len(all_videos) == 0:
        print("[ERROR] Koi bhi video nahi mili! Dataset ka path check karein.")
        return

    print(f"Total Combined Videos for Processing: {len(all_videos)} (Real: {len(real_vids)}, Fake: {len(fake_vids)})")

    features_list, valid_paths = [], []
    for vpath in tqdm(all_videos, desc="Extracting ViT Embeddings on RTX 5060"):
        feat = extract_vit_embeddings(vpath, max_frames=10)
        if feat is not None:
            features_list.append(feat)
            valid_paths.append(vpath)

    if len(features_list) == 0:
        print("[ERROR] Koi bhi valid video features extract nahi ho paaye.")
        return

    X = np.array(features_list)
    print(f"ViT Embeddings Shape: {X.shape}")

    # Scale and PCA for 768-D ViT Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=32, random_state=42) # Reducing dimensions for better clustering
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA Reduced Shape for Clustering: {X_pca.shape}")

    # KMeans Unsupervised Clustering for Auto-Labeling
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    raw_clusters = kmeans.fit_predict(X_pca)

    df_dataset = pd.DataFrame(X)
    df_dataset.insert(0, "Video_Path", valid_paths)
    df_dataset["Cluster_Label"] = raw_clusters  # 0 or 1

    output_csv = "final_kaggle_auto_labeled_dataset.csv"
    df_dataset.to_csv(output_csv, index=False)
    print(f"✅ Success! Auto-labeled dataset saved to '{output_csv}' with {len(df_dataset)} samples.")

# =====================================================================
# 3. STAGE 3: SUPERVISED MODEL TRAINING
# =====================================================================
def run_stage3_supervised_training():
    csv_file = "final_kaggle_auto_labeled_dataset.csv"
    if not os.path.exists(csv_file):
        print(f"[ERROR] '{csv_file}' nahi mila. Pehle Stage 2 run karein.")
        return

    print(f"\n[STAGE 3] Training Supervised Classifier using '{csv_file}'...")
    df = pd.read_csv(csv_file)
    
    X = df.drop(columns=["Video_Path", "Cluster_Label"]).values
    y = df["Cluster_Label"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("\n================ SUPERVISED MODEL RESULTS ================")
    print(classification_report(y_test, preds))
    print(f"🎯 Final Test Accuracy: {acc * 100:.2f}%")

    # Save trained supervised model
    joblib.dump(rf, "deepguard_supervised_rf_model.pkl")
    print("✅ Supervised model saved as 'deepguard_supervised_rf_model.pkl'!")

if __name__ == "__main__":
    # Agar aapne features pehle hi 7,000 videos ke nikal liye hain aur CSV direct train karni hai, 
    # toh run_stage2_auto_labeling() ko comment karke sirf run_stage3_supervised_training() bhi chala sakte hain.
    run_stage2_auto_labeling()
    run_stage3_supervised_training()