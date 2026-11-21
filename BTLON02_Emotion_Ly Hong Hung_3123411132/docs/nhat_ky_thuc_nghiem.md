# NHẬT KÝ THỰC NGHIỆM (EXPERIMENT LOG) — BTLON02: FER2013 (PYTORCH)

- **Đề tài**: Nhận diện cảm xúc khuôn mặt bằng mạng CNN (Facial Expression Recognition — FER2013)
- **Môn học**: Học Sâu (Deep Learning)
- **Bộ dữ liệu**: Kaggle FER2013 (`Training`: 28.709 | `PublicTest/Val`: 3.589 | `PrivateTest/Test`: 3.589)
- **Framework**: PyTorch
- **Nhánh Git đồng bộ**: `develop`

---

## 1. NHÓM THỰC NGHIỆM 1 — BASELINE, DATA AUGMENTATION, ACTIVATION & CLASS WEIGHT
- **Phụ trách**: Thành viên 1 — **Lý Hồng Hưng** (`3123411132`)
- **Ngày chạy**: 22/11/2026
- **Môi trường chạy**: Kaggle Notebook — GPU NVIDIA Tesla T4 (PyTorch + AMP Mixed Precision)
- **Cấu hình cố định**: Optimizer = `Adam (lr=1e-3, weight_decay=1e-4)`, Scheduler = `ReduceLROnPlateau(mode='min', factor=0.5, patience=4, min_lr=1e-6)`, `batch_size=64`, `max_epochs=45`, `seed=42`, `EarlyStopping(patience=10, monitor='val_acc')`.
- **Nguồn số liệu thật**: `logs/exp1_summary.csv` và 6 file log từng epoch `logs/Exp1_*_epochs.csv`.

| Mã TN | Ngày chạy | Kiến trúc | Data Aug | Activation | Loss | Params | Epochs chạy / Best Ep | Train Acc (cuối) | Val Acc (%) | Min Val Loss | Test Acc (%) | Test Macro F1 (%) | Disgust Recall (%) | Thời gian (s) | Nhận xét & Quyết định |
| :---: | :---: | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `Exp1_1` | 22/11/2026 | SmallCNN | Không | ReLU | CrossEntropy | 1.469.095 | 36 / 26 | 91.16% | 64.86% | 1.0276 | 65.17% | 63.70% | 56.36% | 223.4 | **Overfitting nặng** (Gap Train-Val = +27.10%, Val Loss tăng ngược sau ep 12) → Bổ sung Data Augmentation ở `Exp1_2`. |
| `Exp1_2` | 22/11/2026 | SmallCNN | Có | ReLU | CrossEntropy | 1.469.095 | 45 / 44 | 61.83% | 65.23% | 0.9414 | 66.20% | 61.00% | 32.73% | 461.4 | **Triệt tiêu overfitting** (Gap = -2.33%), Test Acc tăng **+1.03%**, Val Loss giảm mạnh xuống 0.9414 → Tăng độ sâu mạng lên `MediumCNN`. |
| `Exp1_3` | 22/11/2026 | MediumCNN | Có | ReLU | CrossEntropy | 6.606.919 | 45 / 44 | 65.77% | **67.15%** | **0.9135** | 67.99% | 63.81% | 56.36% | 581.4 | Tăng độ sâu (8 tầng Conv) giúp Test Acc tăng **+1.79%** (67.99%), không overfit (Gap = -0.54%) → Dùng `MediumCNN` so sánh Activation & Loss. |
| `Exp1_4` | 22/11/2026 | MediumCNN | Có | LeakyReLU | CrossEntropy | 6.606.919 | 45 / 44 | 64.50% | 65.20% | 0.9204 | 67.40% | 63.97% | 56.36% | 587.4 | Hiệu năng ổn định, Test Macro F1 nhỉnh nhẹ (+0.16%) nhưng Val/Test Acc thấp hơn ReLU (-0.59%) → Giữ ReLU làm mặc định. |
| `Exp1_5` | 22/11/2026 | MediumCNN | Có | Sigmoid | CrossEntropy | 6.606.919 | 45 / 41 | 51.23% | 52.52% | 1.2335 | 53.13% | 40.18% | **0.00%** | 589.1 | **Sụp đổ do Vanishing Gradient**: Test Acc giảm -14.86% so với ReLU, lớp `Disgust` Recall = 0% → Loại bỏ Sigmoid cho mạng CNN sâu. |
| `Exp1_6` | 22/11/2026 | MediumCNN | Có | ReLU | **Weighted CE** (`sqrt_balanced`) | 6.606.919 | 45 / 44 | 66.81% | 67.09% | 0.9277 | **68.51%** | **65.63%** | **74.55%** | 581.1 | **Tốt nhất Nhóm 1**: Test Acc đạt **68.51%**, Macro F1 **65.63%** (+1.82%), **Disgust Recall tăng vọt +18.19%** (56.36% → 74.55%). Lưu checkpoint bàn giao TV3 & làm mốc cho TV2. |

### Tóm tắt phát hiện chính từ Nhóm 1 (chuyển giao cho TV2 & TV3)
1. **Data Augmentation bắt buộc phải bật**: Giúp giảm Gap Train-Val từ `+27.10%` xuống gần `0%` và tăng Test Acc `+1.03%`.
2. **Hàm kích hoạt**: `ReLU` vượt trội hoàn toàn `Sigmoid` (`+14.86%` Test Acc) và nhỉnh hơn `LeakyReLU` (`+0.59%`).
3. **Weighted CrossEntropy (`sqrt_balanced`) rất hiệu quả**: Tăng mạnh `Disgust Recall` lên `74.55%` đồng thời nâng cả `Test Acc` (`68.51%`) lẫn `Macro F1` (`65.63%`).
4. **Dư địa cải thiện cho TV2**: Tại epoch 45, `Exp1_3` và `Exp1_6` vẫn chưa bị overfitting (Val Loss vẫn đang giảm) → TV2 nên thử tăng số epoch (60–80), dùng `CosineAnnealingLR`, thử kiến trúc sâu hơn (ResNet-mini / LargeCNN) và Transfer Learning.
5. **Checkpoint bàn giao TV3**: `checkpoints/Exp1_6_MediumCNN_Aug_ReLU_WeightedCE_best.pth` (đã đẩy sẵn lên nhánh `develop`).

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
