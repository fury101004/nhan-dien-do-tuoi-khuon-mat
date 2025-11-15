import os
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from tkinter import Tk, filedialog

# ==========================
# Hàm trích xuất đặc trưng HOG
# ==========================
def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (64, 64))

    # Chỉ lấy HOG features
    hog_features = hog(
        gray_resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )
    return hog_features  # Trả về mảng 1D

# ==========================
# Load model + PCA
# ==========================
BASE_DIR = r"C:\Users\ADMIN\Downloads\UKTFace"
MODEL_PATH = os.path.join(BASE_DIR, "models", r"C:\Users\ADMIN\Downloads\UTKFace_split\models\age_group_hog_pca_svm_5class.joblib")
PCA_PATH   = os.path.join(BASE_DIR, r"C:\Users\ADMIN\Downloads\UTKFace_split\models\pca_transformer.joblib")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Không tìm thấy model: {MODEL_PATH}")
if not os.path.exists(PCA_PATH):
    raise FileNotFoundError(f"❌ Không tìm thấy PCA: {PCA_PATH}")

model = joblib.load(MODEL_PATH)
pca = joblib.load(PCA_PATH)

labels = ["Tre em", "Thieu nhi", "Thanh nien", "Trung nien", "Nguoi gia"]

# ==========================
# Chọn ảnh trực tiếp từ máy
# ==========================
Tk().withdraw()
file_paths = filedialog.askopenfilenames(
    title="Chọn ảnh khuôn mặt để dự đoán",
    filetypes=[("Ảnh", "*.jpg *.jpeg *.png")]
)

if not file_paths:
    print("❌ Bạn chưa chọn ảnh nào.")
else:
    for img_path in file_paths:
        print(f"\n🔍 Đang xử lý ảnh: {os.path.basename(img_path)}")

        # Đọc ảnh an toàn với Unicode
        try:
            img_array = np.fromfile(img_path, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"❌ Lỗi khi đọc ảnh {img_path}: {e}")
            continue

        if img is None:
            print(f"❌ Không thể đọc ảnh: {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Phát hiện khuôn mặt
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            print(f"❌ Không phát hiện khuôn mặt trong ảnh: {img_path}")
        else:
            for (x, y, w, h) in faces:
                face = img[y:y+h, x:x+w]
                features = extract_features(face).reshape(1, -1)
                features_pca = pca.transform(features)

                pred = model.predict(features_pca)[0]
                label = labels[pred]

                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f"Dự đoán: {label}")
            plt.axis("off")
            plt.show()
