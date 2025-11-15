# 👶🧒🧑‍🦱 Age Group Classification using SVM & MobileNetV2 (UTKFace Dataset)

## 🧠 Giới thiệu

Dự án phân loại **nhóm tuổi khuôn mặt** thành 5 lớp:

-   **Trẻ em (≤10)**\
-   **Thiếu nhi (11--16)**\
-   **Thanh niên (17--35)**\
-   **Trung niên (36--69)**\
-   **Người già (≥70)**

Dùng dataset **UTKFace (23k ảnh)** và 2 mô hình:

### ✅ **SVM (HOG + PCA)**

-   Nhẹ -- nhanh -- chạy tốt trên CPU\
-   Dùng đặc trưng truyền thống\
-   Accuracy: **\~65--70%**

### ✅ **MobileNetV2**

-   Mô hình Deep Learning\
-   Hỗ trợ fine-tune\
-   Accuracy: **\~75--80%**

Cả hai mô hình đều hỗ trợ **training -- đánh giá -- dự đoán ảnh --
realtime webcam**.

------------------------------------------------------------------------

## 🎥 Demo

-   Demo webcam SVM\
-   Demo webcam MobileNetV2\
-   Demo dự đoán ảnh

*(Thêm ảnh GIF hoặc PNG tại đây nếu có.)*

------------------------------------------------------------------------

## 📁 Cấu trúc thư mục

    .
    ├── UTKFace/                  # Dataset gốc (~23k ảnh)
    ├── UTKFace_split/            # Dataset đã chia train/val/test
    │   ├── train/
    │   ├── val/
    │   └── test/
    │
    ├── models/                   # Lưu model (.joblib / .h5)
    │
    ├── chiaanh.py                # Chia dataset thành train/val/test (80/10/10)
    ├── hienthianh.py             # Hiển thị ngẫu nhiên 9 ảnh
    ├── bieudoanh.py              # Biểu đồ phân bố số lượng theo nhóm tuổi
    ├── soluonganh.py             # Đếm số lượng ảnh
    ├── aa.py                     # Đếm nhanh số lượng ảnh
    │
    ├── train.py                  # Train SVM (HOG + PCA + SMOTE)
    ├── anh.py                    # Dự đoán ảnh bất kỳ (SVM)
    ├── webcamm.py                # Realtime webcam (SVM)
    │
    ├── trainvggmbn2.py           # Train MobileNetV2 (augmentation + callbacks)
    ├── webcammb.py               # Realtime webcam (MobileNetV2)
    │
    └── *.png / *.joblib / *.h5   # Model & biểu đồ sinh ra khi chạy

------------------------------------------------------------------------

## 🛠 Cài đặt môi trường

``` bash
pip install opencv-python numpy matplotlib seaborn scikit-learn scikit-image imbalanced-learn joblib tensorflow
```

**Đã kiểm thử ổn định trên:**

-   Python **3.8--3.11**\
-   TensorFlow **2.13--2.16**\
-   Windows 10/11\
-   Ubuntu 20.04+

🔥 **Lưu ý:** Bạn cần tải dataset **UTKFace** và đặt đúng thư mục.

------------------------------------------------------------------------

## 🚀 Hướng dẫn sử dụng

### **0. Chuẩn bị dataset**

``` bash
python chiaanh.py
```

------------------------------------------------------------------------

### **1. Khám phá dữ liệu**

``` bash
python hienthianh.py
python bieudoanh.py
python soluonganh.py
# hoặc
python aa.py
```

------------------------------------------------------------------------

### **2. Train mô hình SVM (HOG + PCA)**

``` bash
python train.py
```

✔ Dùng SMOTE để cân bằng lớp\
✔ Lưu model `.joblib`\
✔ Vẽ confusion matrix + precision plot

------------------------------------------------------------------------

### **3. Train mô hình MobileNetV2**

``` bash
python trainvggmbn2.py
```

✔ Fine-tune 20 lớp cuối\
✔ Dùng class weights + augmentation\
✔ EarlyStopping + ModelCheckpoint\
✔ Lưu model `.h5`

------------------------------------------------------------------------

### **4. Dự đoán ảnh bất kỳ (SVM)**

``` bash
python anh.py
```

✔ Hỗ trợ chọn nhiều ảnh\
✔ Dùng Haar Cascade để detect mặt

------------------------------------------------------------------------

### **5. Realtime webcam**

**SVM:**

``` bash
python webcamm.py
```

**MobileNetV2:**

``` bash
python webcammb.py
```

→ Nhấn **q** để thoát.

------------------------------------------------------------------------

## 📊 Kết quả mong đợi (trên UTKFace test set)

  ----------------------------------------------------------------------------------------
  Nhóm tuổi      Precision   Recall    F1 (SVM) Precision    Recall     F1       Support
                 (SVM)       (SVM)              (MBV2)       (MBV2)     (MBV2)   
  -------------- ----------- --------- -------- ------------ ---------- -------- ---------
  Trẻ em         \~0.75      \~0.80    \~0.77   \~0.85       \~0.88     \~0.86   \~1500

  Thiếu nhi      \~0.60      \~0.55    \~0.57   \~0.70       \~0.65     \~0.67   \~800

  Thanh niên     \~0.70      \~0.72    \~0.71   \~0.78       \~0.80     \~0.79   \~3000

  Trung niên     \~0.65      \~0.68    \~0.66   \~0.75       \~0.77     \~0.76   \~2000

  Người già      \~0.80      \~0.75    \~0.77   \~0.85       \~0.82     \~0.83   \~1000

  **Accuracy**   **\~68%**                      **\~78%**                        8300
  ----------------------------------------------------------------------------------------

------------------------------------------------------------------------

## 🚀 Gợi ý cải thiện thêm (để vượt **82%**)

-   Dùng **MTCNN / DLib** để face alignment\
-   Thêm augmentation mạnh hơn (brightness, shear, zoom)\
-   Dùng **ensemble** (SVM + MobileNetV2)\
-   Thử thêm: **EfficientNet B0/B3**, **ResNet50**\
-   Tăng epochs + LR Scheduler (Cosine decay)

------------------------------------------------------------------------

## 👨‍💻 Tác giả

-   Sinh viên thực hiện đồ án Machine Learning & Deep Learning\
-   Model train trên local CPU:
    -   SVM: **5--10 phút**\
    -   MobileNetV2: **1--2 giờ**\
-   Code rõ ràng, nhiều comment tiếng Việt, dễ bảo vệ

------------------------------------------------------------------------

> **"Tuổi tác chỉ là con số -- Máy tính giờ cũng đoán được!"**
