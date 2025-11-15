# ==========================================================
# KIỂM THỬ MÔ HÌNH MobileNetV2 — DỰ ĐOÁN NHÓM TUỔI QUA WEBCAM
# ==========================================================
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ========================= 1. Cấu hình =========================
IMG_SIZE = (96, 96)
CLASS_NAMES = ["Tre em", "Thieu nhi", "Thanh nien", "Trung nien", "Nguoi gia"]
MODEL_PATH = r"C:\Users\ADMIN\UTKFace_split\models\mobilenetv2_final_fixed.h5"

# ========================= 2. Tải mô hình =========================
print("🧠 Đang tải mô hình...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Mô hình đã tải thành công!")

# ========================= 3. Mở webcam =========================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, IMG_SIZE)
        face_array = np.expand_dims(face_resized / 255.0, axis=0)

        preds = model.predict(face_array)
        predicted_idx = np.argmax(preds)
        predicted_label = CLASS_NAMES[predicted_idx]
        confidence = preds[0][predicted_idx] * 100

        # Vẽ khung và nhãn
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{predicted_label} ({confidence:.1f}%)", 
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Dự đoán nhóm tuổi - Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
