# ==========================================================
# KIỂM THỬ MÔ HÌNH MobileNetV2 — DỰ ĐOÁN ẢNH TỪ FILE
# ==========================================================
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tkinter import filedialog, Tk
import os

# ========================= 1. Cấu hình =========================
IMG_SIZE = (96, 96)
CLASS_NAMES = ["Tre em", "Thieu nhi", "Thanh nien", "Trung nien", "Nguoi gia"]
MODEL_PATH = r"C:\Users\ADMIN\UTKFace_split\models\mobilenetv2_final.h5"

# ========================= 2. Tải mô hình =========================
print("🧠 Đang tải mô hình...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Mô hình đã tải thành công!")

# ========================= 3. Hàm dự đoán =========================
def predict_image_from_file():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh khuôn mặt cần dự đoán",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
    )

    if not file_path:
        print("❌ Không có ảnh nào được chọn.")
        return

    # Tiền xử lý ảnh
    img = image.load_img(file_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Dự đoán
    preds = model.predict(img_array)
    predicted_idx = np.argmax(preds)
    predicted_label = CLASS_NAMES[predicted_idx]
    confidence = preds[0][predicted_idx] * 100

    # Hiển thị kết quả
    plt.imshow(image.load_img(file_path))
    plt.title(f"🔮 {predicted_label} ({confidence:.2f}%)")
    plt.axis("off")
    plt.show()

    print(f"📸 Ảnh: {os.path.basename(file_path)}")
    print(f"➡️ Dự đoán: {predicted_label}")
    print(f"📈 Độ tin cậy: {confidence:.2f}%")

# ========================= 4. Chạy =========================
if __name__ == "__main__":
    predict_image_from_file()
