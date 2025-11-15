# ==========================================================
# Age Group Classification using MobileNetV2 (Fast + Balanced + Accurate + Visualization)
# ==========================================================
import os, numpy as np, matplotlib.pyplot as plt, seaborn as sns, tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ========================== 0. Cấu hình TensorFlow ==========================
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)
print("✅ TensorFlow đang chạy tối ưu đa luồng CPU.")

# ========================== 1. Đường dẫn dữ liệu ==========================
BASE_DIR = r"C:\Users\ADMIN\UTKFace_split"
TRAIN_DIR, VAL_DIR, TEST_DIR = [os.path.join(BASE_DIR, d) for d in ["train", "val", "test"]]
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

target_names = ["Tre em", "Thieu nhi", "Thanh nien", "Trung nien", "Nguoi gia"]
num_classes = len(target_names)

# ========================== 2. Data Augmentation ==========================
IMG_SIZE = (96, 96)
BATCH_SIZE = 16

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True
).flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

# ========================== 3. Cân bằng dữ liệu ==========================
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(train_gen.classes),
                                     y=train_gen.classes)
class_weights = dict(enumerate(class_weights))
print("⚖️ Trọng số lớp:", class_weights)

# ========================== 4. MobileNetV2 base model ==========================
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
for layer in base_model.layers[:-20]:
    layer.trainable = False
for layer in base_model.layers[-20:]:
    layer.trainable = True

# ========================== 5. Xây dựng mô hình ==========================
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=2e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# ========================== 6. Callback ==========================
checkpoint_path = os.path.join(MODEL_DIR, "mobilenetv2_balanced_best.h5")
callbacks = [
    EarlyStopping(patience=5, monitor='val_accuracy', restore_best_weights=True),
    ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
]

# ========================== 7. Huấn luyện ==========================
history = model.fit(train_gen,
                    validation_data=val_gen,
                    epochs=15,
                    callbacks=callbacks,
                    class_weight=class_weights,
                    verbose=1)

# ========================== 8. Đánh giá ==========================
loss, acc = model.evaluate(test_gen)
print(f"\n✅ Độ chính xác (Test Accuracy): {acc*100:.2f}%")

# ========================== 9. Biểu đồ huấn luyện ==========================
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Độ chính xác (Accuracy)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Hàm mất mát (Loss)")
plt.legend()
plt.tight_layout()
plt.show()

# ========================== 10. Báo cáo & Ma trận nhầm lẫn ==========================
y_pred = np.argmax(model.predict(test_gen), axis=1)
y_true = test_gen.classes

report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0, output_dict=True)
print("\n📊 Báo cáo chi tiết:")
print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title("Ma trận nhầm lẫn (Confusion Matrix)")
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.tight_layout()
cm_path = os.path.join(MODEL_DIR, "mobilenetv2_confusion_matrix.png")
plt.savefig(cm_path)
plt.close()
print(f"💾 Đã lưu Confusion Matrix tại: {cm_path}")

# Precision & Recall
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
chart_path = os.path.join(MODEL_DIR, "mobilenetv2_class_report.png")
plt.savefig(chart_path)
plt.close()
print(f"📊 Lưu biểu đồ Precision/Recall tại: {chart_path}")

# ========================== 11. Lưu mô hình ==========================
save_path = os.path.join(MODEL_DIR, "mobilenetv2_final.h5")
model.save(save_path)
print(f"💾 Mô hình đã được lưu tại: {save_path}")
