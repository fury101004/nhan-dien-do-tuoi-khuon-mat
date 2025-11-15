import os
import random
import cv2
import matplotlib.pyplot as plt

# 📂 Thay đường dẫn thành thư mục chứa UTKFace
DATASET_DIR = r"C:\Users\ADMIN\Downloads\UTKFace (1)\UTKFace"

# Lấy danh sách ảnh
files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".jpg")]

print(f"📊 Tổng số ảnh: {len(files)}")

# Lấy ngẫu nhiên 9 ảnh
sample_files = random.sample(files, 9)

plt.figure(figsize=(10,10))
for i, file in enumerate(sample_files):
    img_path = os.path.join(DATASET_DIR, file)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB

    age = file.split("_")[0]  # Lấy tuổi từ tên file

    plt.subplot(3, 3, i+1)
    plt.imshow(img)
    plt.title(f"Tuổi: {age}")
    plt.axis("off")

plt.tight_layout()
plt.show()
