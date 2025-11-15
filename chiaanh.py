import os, shutil, random

src_dir = r"C:\Users\ADMIN\Downloads\UTKFace (1)\UTKFace"
dst_dir = r"C:\Users\ADMIN\UTKFace_split"

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(dst_dir, split), exist_ok=True)

all_images = [f for f in os.listdir(src_dir) if ".jpg" in f.lower()]
random.shuffle(all_images)

n_total = len(all_images)
n_train = int(0.8 * n_total) 
n_val = int(0.1 * n_total)
train_files = all_images[:n_train]
val_files = all_images[n_train:n_train+n_val]
test_files = all_images[n_train+n_val:]

def copy_files(file_list, target_dir):
    for f in file_list:
        shutil.copy(os.path.join(src_dir, f), os.path.join(target_dir, f))

copy_files(train_files, os.path.join(dst_dir, "train"))
copy_files(val_files, os.path.join(dst_dir, "val"))
copy_files(test_files, os.path.join(dst_dir, "test"))

print("Train:", len(train_files), "Val:", len(val_files), "Test:", len(test_files))
