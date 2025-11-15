import os
import cv2
import joblib
import numpy as np
from skimage.feature import hog

# ==========================
# Hàm trích xuất đặc trưng HOG
# ==========================
def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (64, 64))

    hog_features = hog(
        gray_resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )
    return hog_features

# ==========================
# Load model + PCA
# ==========================
BASE_DIR = r"C:\Users\ADMIN\Downloads\UKTFace"
MODEL_PATH = os.path.join(BASE_DIR, "models", r"C:\Users\ADMIN\Downloads\UTKFace_split\models\age_group_hog_pca_svm_5class.joblib")
PCA_PATH   = os.path.join(BASE_DIR, "models", r"C:\Users\ADMIN\Downloads\UTKFace_split\models\pca_transformer.joblib")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Không tìm thấy model: {MODEL_PATH}")
if not os.path.exists(PCA_PATH):
    raise FileNotFoundError(f"❌ Không tìm thấy PCA: {PCA_PATH}")

model = joblib.load(MODEL_PATH)
pca = joblib.load(PCA_PATH)

labels = ["Tre em", "Thieu nhi", "Thanh nien", "Trung nien", "Nguoi gia"]

# ==========================
# Mở webcam
# ==========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Không thể mở webcam. Kiểm tra kết nối.")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("✅ Webcam đang chạy. Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Không đọc được khung hình từ webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        features = extract_features(face).reshape(1, -1)
        features_pca = pca.transform(features)

        pred = model.predict(features_pca)[0]
        label = labels[pred]

        # Vẽ khung và nhãn
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Dự đoán nhóm tuổi - Nhấn 'q' để thoát", frame)

    # Nhấn q để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
