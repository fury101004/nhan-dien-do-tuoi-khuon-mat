import os

# Đường dẫn đến thư mục chứa 3 tập dữ liệu
base_path = "UTKFace_split"

# Các thư mục con
folders = ["train", "val", "test"]

# Đếm ảnh trong từng thư mục
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                count += 1
    print(f"📁 {folder}: {count} ảnh")
