# ==========================================================
# 🔍 KIỂM THỬ MÔ HÌNH PHÂN LOẠI NHÓM TUỔI (TEST PHASE)
# ==========================================================
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

# ========================== 1. Cấu hình & đường dẫn ==========================
BASE_DIR = r"C:\Users\ADMIN\UTKFace_split"
TEST_DIR = os.path.join(BASE_DIR, "test")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "mobilenetv2_final_fixed.h5")  # hoặc "mobilenetv2_final.h5"

target_names = ["Tre em", "Thieu nhi", "Thanh nien", "Trung nien", "Nguoi gia"]
num_classes = len(target_names)
IMG_SIZE = (96, 96)
BATCH_SIZE = 16

# ========================== 2. Tải mô hình ==========================
print("🧠 Đang tải mô hình đã huấn luyện...")
model = load_model(MODEL_PATH)
print("✅ Mô hình đã tải thành công!")

# ========================== 3. Chuẩn bị dữ liệu test ==========================
test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ========================== 4. Đánh giá tổng quan ==========================
loss, acc = model.evaluate(test_gen, verbose=1)
print(f"\n🎯 Độ chính xác (Test Accuracy): {acc*100:.2f}%")
print(f"💡 Hàm mất mát (Test Loss): {loss:.4f}")

# ========================== 5. Dự đoán ==========================
y_pred = np.argmax(model.predict(test_gen), axis=1)
y_true = test_gen.classes

# ========================== 6. Báo cáo chi tiết ==========================
print("\n📋 Báo cáo phân loại (Classification Report):")
print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

# ========================== 7. Ma trận nhầm lẫn ==========================
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title("Ma trận nhầm lẫn (Confusion Matrix)")
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.tight_layout()
plt.show()

# ========================== 8. Precision & Recall Visualization ==========================
report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0, output_dict=True)
precisions = [report[name]['precision'] for name in target_names]
recalls = [report[name]['recall'] for name in target_names]

x = np.arange(len(target_names))
width = 0.35
plt.figure(figsize=(8, 5))
plt.bar(x - width/2, precisions, width, label='Precision', color='skyblue')
plt.bar(x + width/2, recalls, width, label='Recall', color='salmon')
plt.xticks(x, target_names, rotation=15)
plt.ylim(0, 1)
plt.ylabel("Tỉ lệ")
plt.title("Precision & Recall theo từng nhóm tuổi")
plt.legend()
plt.tight_layout()
plt.show()

# ========================== 9. Dự đoán mẫu ngẫu nhiên ==========================
import random
filenames = test_gen.filenames
idxs = random.sample(range(len(filenames)), 9)

plt.figure(figsize=(10, 10))
for i, idx in enumerate(idxs):
    img_path = os.path.join(TEST_DIR, filenames[idx])
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    pred = np.argmax(model.predict(np.expand_dims(img_array, axis=0)))
    true_label = target_names[y_true[idx]]
    plt.subplot(3, 3, i + 1)
    plt.imshow(img_array)
    plt.title(f"Thực tế: {true_label}\nDự đoán: {target_names[pred]}")
    plt.axis("off")
plt.tight_layout()
plt.show()
