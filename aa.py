import os

base = r"C:\Users\ADMIN\UTKFace_split"
for subset in ["train", "val", "test"]:
    path = os.path.join(base, subset)
    total = sum(len(files) for _, _, files in os.walk(path))
    print(f"{subset}: {total} ảnh")
