# ==========================
# Train SVM model (HOG + PCA + SVM, 5 lớp tuổi)
# ==========================
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from collections import Counter

# ==========================
# 1. Đường dẫn dataset & thư mục lưu model
# ==========================
TRAIN_DIR = r"C:\Users\ADMIN\UTKFace_split\train"
VAL_DIR   = r"C:\Users\ADMIN\UTKFace_split\val"
TEST_DIR  = r"C:\Users\ADMIN\UTKFace_split\test"
MODEL_DIR = r"C:\Users\ADMIN\UTKFace_split\models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================
# 2. Hàm gán nhãn tuổi (5 lớp)
# ==========================
def age_to_class(age):
    if age <= 10:
        return 0   # Trẻ em
    elif age <= 16:
        return 1   # Thiếu nhi
    elif age <= 35:
        return 2   # Thanh niên
    elif age <= 69:
        return 3   # Trung niên
    else:
        return 4   # Người già

target_names = ["Tre em", "Thieu nhi", "Thanh nien", "Trung nien", "Nguoi gia"]
labels = [0, 1, 2, 3, 4]

# ==========================
# 3. Hàm load ảnh + nhãn từ folder (sửa lại để đọc thư mục con)
# ==========================
def load_data_from_dir(folder):
    imgs, y = [], []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                try:
                    age = int(f.split("_")[0])
                    img = cv2.imread(os.path.join(root, f), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        imgs.append(img)
                        y.append(age_to_class(age))
                except Exception as e:
                    continue
    return imgs, y

# ==========================
# 4. Load dữ liệu
# ==========================
print("🔄 Loading dataset...")
X_train_imgs, y_train = load_data_from_dir(TRAIN_DIR)
X_val_imgs, y_val = load_data_from_dir(VAL_DIR)
X_test_imgs, y_test = load_data_from_dir(TEST_DIR)

print(f"✅ Loaded: Train={len(X_train_imgs)}, Val={len(X_val_imgs)}, Test={len(X_test_imgs)}")
if len(X_train_imgs) == 0:
    raise ValueError("❌ Không có ảnh trong thư mục TRAIN. Kiểm tra lại đường dẫn hoặc cấu trúc thư mục!")

# Thống kê phân bố lớp
print("\n📊 Phân bố lớp trong tập train:", Counter(y_train))

# ==========================
# 5. Trích xuất đặc trưng HOG
# ==========================
def extract_hog(images):
    X = []
    for img in images:
        img_resized = cv2.resize(img, (96,96))
        features = hog(img_resized,
                       orientations=9,
                       pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2),
                       block_norm="L2-Hys")
        X.append(features)
    return np.array(X)

print("🔄 Extracting HOG features...")
X_train = extract_hog(X_train_imgs)
X_val   = extract_hog(X_val_imgs)
X_test  = extract_hog(X_test_imgs)
print("✅ HOG shapes:", X_train.shape, X_val.shape, X_test.shape)

# ==========================
# 6. SMOTE cân bằng dữ liệu train
# ==========================
print("\n🔄 Đang oversample bằng SMOTE...")
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("✅ Train size sau SMOTE:", X_train_res.shape, len(y_train_res))
print("📊 Phân bố lớp sau SMOTE:", Counter(y_train_res))

# ==========================
# 7. PCA giảm chiều
# ==========================
print("\n🔄 Đang giảm chiều dữ liệu bằng PCA...")
pca = PCA(n_components=200)
X_train_pca = pca.fit_transform(X_train_res)
X_val_pca   = pca.transform(X_val)
X_test_pca  = pca.transform(X_test)
print("✅ Dữ liệu sau PCA:", X_train_pca.shape, X_val_pca.shape, X_test_pca.shape)

# ==========================
# 8. Train SVM
# ==========================
print("\n🚀 Training SVM model...")
model = SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42)
model.fit(X_train_pca, y_train_res)
print("✅ Huấn luyện hoàn tất!")

# ==========================
# 9. Đánh giá trên tập test
# ==========================
y_pred = model.predict(X_test_pca)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")

report = classification_report(
    y_test, y_pred,
    labels=labels,
    target_names=target_names,
    zero_division=0,
    output_dict=True
)
print("\n📊 Báo cáo chi tiết:")
print(classification_report(
    y_test, y_pred,
    labels=labels,
    target_names=target_names,
    zero_division=0
))

# ==========================
# 10. Ma trận nhầm lẫn
# ==========================
cm = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(7, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Ma trận nhầm lẫn (Confusion Matrix)")
plt.colorbar()
tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=45)
plt.yticks(tick_marks, target_names)

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], "d"),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()
print(f"💾 Đã lưu confusion matrix vào: {cm_path}")

# ==========================
# 11. Biểu đồ độ chính xác từng lớp
# ==========================
class_acc = [report[name]["precision"] for name in target_names]
plt.figure(figsize=(7, 5))
plt.bar(target_names, class_acc, color="skyblue")
plt.ylabel("Precision")
plt.title("Precision theo từng lớp tuổi")
plt.xticks(rotation=15)
plt.ylim(0, 1)
for i, v in enumerate(class_acc):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)

acc_bar_path = os.path.join(MODEL_DIR, "class_precision.png")
plt.tight_layout()
plt.savefig(acc_bar_path)
plt.close()
print(f"📊 Đã lưu biểu đồ precision theo lớp vào: {acc_bar_path}")

# ==========================
# 12. Lưu model + PCA
# ==========================
model_path = os.path.join(MODEL_DIR, "age_group_hog_pca_svm_5class1.joblib")
pca_path = os.path.join(MODEL_DIR, "pca_transformer.joblib")
joblib.dump(model, model_path)
joblib.dump(pca, pca_path)
print("\n💾 Model saved:", model_path)
print("💾 PCA saved:", pca_path)
