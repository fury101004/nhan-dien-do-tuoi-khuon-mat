import os
import matplotlib.pyplot as plt
from collections import Counter

# 📂 Thay đường dẫn bằng thư mục chứa dataset UTKFace
DATASET_DIR = r"C:\Users\ADMIN\Downloads\UTKFace (1)\UTKFace"

# Danh sách file trong folder
files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".jpg")]

print(f"📊 Tổng số ảnh trong thư mục: {len(files)}")

# UTKFace đặt tên file theo cấu trúc: age_gender_race_date.jpg
# => Lấy tuổi từ tên file (số trước dấu "_")
ages = [int(f.split("_")[0]) for f in files]

# Chia nhóm tuổi (5 lớp)
def age_group(age):
    if age <= 10:
        return "Tre em"
    elif age <= 16:
        return "Thieu nhi"
    elif age <= 35:
        return "Thanh nien"
    elif age <= 69:
        return "Trung nien"
    else:
        return "Nguoi gia"

groups = [age_group(a) for a in ages]

# Đếm số lượng từng nhóm
counts = Counter(groups)

# Vẽ biểu đồ
plt.figure(figsize=(7,5))
plt.bar(counts.keys(), counts.values(),
        color=["lightblue","lightgreen","orange","gold","salmon"])
plt.title("📊 Số lượng ảnh theo nhóm tuổi trong UTKFace")
plt.xlabel("Nhóm tuổi")
plt.ylabel("Số lượng ảnh")
plt.show()
