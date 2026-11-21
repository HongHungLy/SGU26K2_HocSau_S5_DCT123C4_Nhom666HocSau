# NHẬT KÝ THỰC NGHIỆM (EXPERIMENT LOG) — BTLON02: FER2013 (PYTORCH)

- **Đề tài**: Nhận diện cảm xúc khuôn mặt bằng mạng CNN (Facial Expression Recognition — FER2013)
- **Môn học**: Học Sâu (Deep Learning)
- **Bộ dữ liệu**: Kaggle FER2013 (`Training`: 28.709 | `PublicTest/Val`: 3.589 | `PrivateTest/Test`: 3.589)
- **Framework**: PyTorch
- **Nhánh Git đồng bộ**: `develop`

---

## 1. NHÓM THỰC NGHIỆM 1 — BASELINE, DATA AUGMENTATION, ACTIVATION & CLASS WEIGHT
- **Phụ trách**: Thành viên 1 — **Lý Hồng Hưng** (`3123411132`)
- **Môi trường chạy**: Kaggle Notebook — GPU Tesla T4 (cập nhật ngày chạy sau khi hoàn tất)
- **Cấu hình cố định**: Optimizer = `Adam (lr=1e-3, weight_decay=1e-4)`, Scheduler = `ReduceLROnPlateau(factor=0.5, patience=4)`, `batch_size=64`, `seed=42`, `EarlyStopping(patience=10)`.

| Mã TN | Ngày chạy | Kiến trúc | Data Aug | Activation | Loss | Params | Best Ep | Val Acc (%) | Val Loss | Test Acc (%) | Test Macro F1 (%) | Disgust Recall (%) | Thời gian (s) | Nhận xét & Quyết định |
| :---: | :---: | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `Exp1_1` | *Chưa chạy* | SmallCNN | Không | ReLU | CrossEntropy | - | - | - | - | - | - | - | - | Mốc Baseline 0 (kỳ vọng overfit mạnh) |
| `Exp1_2` | *Chưa chạy* | SmallCNN | Có | ReLU | CrossEntropy | - | - | - | - | - | - | - | - | Đo mức giảm overfitting nhờ Augmentation |
| `Exp1_3` | *Chưa chạy* | MediumCNN | Có | ReLU | CrossEntropy | - | - | - | - | - | - | - | - | Đo tác động tăng độ sâu mạng |
| `Exp1_4` | *Chưa chạy* | MediumCNN | Có | LeakyReLU | CrossEntropy | - | - | - | - | - | - | - | - | So sánh LeakyReLU vs ReLU |
| `Exp1_5` | *Chưa chạy* | MediumCNN | Có | Sigmoid | CrossEntropy | - | - | - | - | - | - | - | - | Kiểm chứng hiện tượng triệt tiêu gradient |
| `Exp1_6` | *Chưa chạy* | MediumCNN | Có | ReLU | Weighted CE | - | - | - | - | - | - | - | - | Kiểm tra cải thiện Recall lớp thiểu số Disgust |

---

## 2. NHÓM THỰC NGHIỆM 2 — KIẾN TRÚC NÂNG CAO, TRANSFER LEARNING & TỐI ƯU SIÊU THAM SỐ
- **Phụ trách**: Thành viên 2 (*Điền tên & MSSV sau khi chốt*)

| Mã TN | Ngày chạy | Kiến trúc | Input Size | Optimizer & LR | Scheduler / Reg | Best Ep | Val Acc (%) | Test Acc (%) | Test Macro F1 (%) | Thời gian (s) | Nhận xét / Fail Case |
| :---: | :---: | :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `Exp2_1` | *Chưa chạy* | LargeCNN / ResNet-mini | 48×48 | - | - | - | - | - | - | - | - |
| `Exp2_2` | *Chưa chạy* | Pretrained ResNet18/34 | 224×224 | - | - | - | - | - | - | - | - |
| `Exp2_3` | *Chưa chạy* | Pretrained VGG16 (Fail Case / Fine-tune) | 48×48 / 224×224 | - | - | - | - | - | - | - | - |

---

## 3. ĐÁNH GIÁ CHUYÊN SÂU & TRỰC QUAN HÓA FINAL CONVOLUTION FILTERS
- **Phụ trách**: Thành viên 3 (*Điền tên & MSSV sau khi chốt*)

| Hạng mục | Mô hình áp dụng | Sản phẩm đầu ra (`figures/`) | Kết luận chính |
| :--- | :--- | :--- | :--- |
| Confusion Matrix (Public & Private) | Best Model | `confusion_matrix_private.png` | *Cập nhật sau* |
| True Cases & Wrong Cases (4×8) | Best Model | `true_cases.png`, `wrong_cases.png` | *Cập nhật sau* |
| Final Convolution Filters & Feature Maps | Best Model | `final_conv_filters.png`, `feature_maps.png` | *Cập nhật sau* |
| Grad-CAM Heatmap (7 lớp cảm xúc) | Best Model | `gradcam_7_emotions.png` | *Cập nhật sau* |
