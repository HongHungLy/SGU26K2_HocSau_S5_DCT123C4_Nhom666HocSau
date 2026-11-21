# BẢN THẢO BÁO CÁO — PHẦN THÀNH VIÊN 1 (VAI TRÒ A)
**Người phụ trách**: Lý Hồng Hưng — MSSV: `3123411132`  
**Phạm vi**: Phần 1 (Giới thiệu bài toán & Khám phá dữ liệu FER2013) + Phần 2 (Tiền xử lý, Data Augmentation, Pipeline PyTorch & Kết quả Thực nghiệm Nhóm 1)

---

## PHẦN 1: GIỚI THIỆU BÀI TOÁN VÀ KHÁM PHÁ DỮ LIỆU (EDA)

### 1.1. Đặt vấn đề
Nhận diện cảm xúc khuôn mặt (Facial Expression Recognition — FER) là bài toán phân loại ảnh tiêu biểu trong Thị giác máy tính và Học sâu. Đầu vào của bài toán là ảnh khuôn mặt mức xám kích thước `48 × 48` pixel, đầu ra là vector xác suất 7 chiều tương ứng với 7 trạng thái cảm xúc cơ bản:

- `0`: **Angry** (Tức giận)
- `1`: **Disgust** (Ghê tởm)
- `2`: **Fear** (Sợ hãi)
- `3`: **Happy** (Vui vẻ)
- `4`: **Sad** (Buồn bã)
- `5`: **Surprise** (Ngạc nhiên)
- `6`: **Neutral** (Trung tính)

Theo tài liệu đề bài, mốc độ chính xác tham chiếu cao nhất (world-best benchmark) trên bộ FER2013 đạt khoảng **69%** (Public Test) và **71%** (Private Test). Bài toán có độ khó đáng kể do ảnh độ phân giải thấp (`48×48`), nhiều góc chụp, điều kiện ánh sáng đa dạng, một số ảnh chứa watermark hoặc bị che khuất một phần, và có sự tương đồng cao giữa các biểu cảm gần nhau (như *Fear* vs. *Sad*, *Angry* vs. *Disgust*).

### 1.2. Mô tả và Khám phá bộ dữ liệu Kaggle FER2013
Bộ dữ liệu Kaggle FER2013 (nguồn `fer2013.csv`, kết xuất thống kê tại `logs/eda_class_distribution.csv`) gồm tổng cộng **35.887 ảnh**, chia sẵn thành 3 tập độc lập nhằm ngăn chặn hoàn toàn rò rỉ dữ liệu (*No Data Leakage*):
- **Training**: `28.709` ảnh (~80.0%) — dùng để cập nhật trọng số mô hình.
- **PublicTest (Validation)**: `3.589` ảnh (~10.0%) — dùng để giám sát quá trình hội tụ, điều chỉnh Learning Rate, chọn Checkpoint tốt nhất và dừng sớm (`EarlyStopping`).
- **PrivateTest (Test độc lập)**: `3.589` ảnh (~10.0%) — giữ cô lập hoàn toàn, chỉ dùng một lần cuối cùng để chấm điểm khách quan.

| Nhãn | Cảm xúc | Training | PublicTest (Val) | PrivateTest (Test) | Tổng cộng | Tỉ lệ tập Train |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | Angry | 3.995 | 467 | 491 | 4.953 | 13.92% |
| 1 | **Disgust** | **436** | **56** | **55** | **547** | **1.52%** |
| 2 | Fear | 4.097 | 496 | 528 | 5.121 | 14.27% |
| 3 | **Happy** | **7.215** | **895** | **879** | **8.989** | **25.13%** |
| 4 | Sad | 4.830 | 653 | 594 | 6.077 | 16.82% |
| 5 | Surprise | 3.171 | 415 | 416 | 4.002 | 11.05% |
| 6 | Neutral | 4.965 | 607 | 626 | 6.198 | 17.29% |
| — | **Tổng** | **28.709** | **3.589** | **3.589** | **35.887** | **100.00%** |

**Hình minh họa từ kết quả chạy EDA**:
- *Hình 1.1*: Biểu đồ phân bố 7 lớp cảm xúc trên 3 tập (`figures/eda_histogram_distribution.png`).
- *Hình 1.2*: Lưới ảnh mẫu 4×8 tập Training và PublicTest (`figures/eda_train_samples.png`, `figures/eda_val_samples.png`).

**Nhận xét chính từ EDA**:
1. **Mất cân bằng dữ liệu nghiêm trọng (Class Imbalance)**: Lớp *Happy* chiếm `25.13%` (7.215 ảnh train) trong khi lớp *Disgust* chỉ chiếm **`1.52%`** (436 ảnh train) — chênh lệch tới **16.55 lần**. Phân bố tỉ lệ giữa 3 tập tương đồng nhau, song nếu chỉ dùng Cross-Entropy tiêu chuẩn, mô hình rất dễ thiên vị các lớp đa số và bỏ sót lớp *Disgust*.
2. **Độ phân giải thấp và nhiễu ảnh**: Kích thước `48×48` làm mờ nhiều đặc trưng tinh tế ở khóe mắt, chân mày và miệng; đồng thời có nhiều ảnh nghiêng, bị che khuất hoặc chứa watermark. Điều này dẫn tới yêu cầu cấp thiết về **Data Augmentation** và **Regularization (BatchNorm, Dropout)** để tăng khả năng tổng quát hóa.

---

## PHẦN 2: TIỀN XỬ LÝ, DATA AUGMENTATION, PIPELINE PYTORCH & MÔ HÌNH BASELINE

### 2.1. Pipeline dữ liệu và Chiến lược Data Augmentation (`src/dataset.py`)
Toàn bộ mã nguồn được xây dựng hoàn toàn bằng **PyTorch** (`torch.utils.data.Dataset`, `DataLoader`, `torchvision.transforms`):
- **Chuẩn hóa (Normalization)**: Chuyển pixel từ `[0, 255]` sang tensor `[0, 1]` và chuẩn hóa về `[-1, 1]` (`mean=0.5, std=0.5`) cho ảnh grayscale 1 kênh (đồng thời hỗ trợ tham số `in_channels=3`, `img_size` linh hoạt với chuẩn hóa ImageNet để Thành viên 2 tái sử dụng cho Transfer Learning).
- **Data Augmentation (chỉ áp dụng trên tập Training)**:
  1. `RandomHorizontalFlip(p=0.5)`: Lật ngang ngẫu nhiên (bảo toàn nhãn biểu cảm khuôn mặt).
  2. `RandomRotation(degrees=10)`: Xoay nhẹ trong khoảng `[-10°, +10°]` mô phỏng góc nghiêng đầu.
  3. `RandomAffine(translate=(0.1, 0.1), scale=(0.9, 1.1))`: Dịch chuyển và co giãn tối đa 10% giúp mô hình bất biến với vị trí và tỉ lệ khuôn mặt.
  4. `RandomErasing(p=0.2, scale=(0.02, 0.12))`: Che khuất ngẫu nhiên vùng nhỏ mô phỏng tóc, kính hoặc tay che mặt.
- **Tập PublicTest và PrivateTest**: Chỉ áp dụng `ToTensor` và `Normalize` (không Augmentation) để đánh giá trung thực.

**Hình minh họa**:
- *Hình 2.1*: Minh họa 1 ảnh gốc và 6 biến thể Data Augmentation (`figures/eda_augmentation_demo.png`).
- *Hình 2.2*: Lưới ảnh Training Batch sau Augmentation (`figures/eda_augmented_batch.png`).

### 2.2. Khung huấn luyện chuẩn PyTorch (`src/trainer.py`)
Để đảm bảo tính công bằng và khả năng tái lập giữa mọi thực nghiệm:
- Cố định random seed (`set_seed(42)`).
- Phân tách rạch ròi `model.train()` khi huấn luyện và `model.eval()` + `torch.no_grad()` khi kiểm định/đánh giá.
- Tích hợp Mixed Precision (`AMP`) trên GPU Tesla T4, `Adam (lr=1e-3, weight_decay=1e-4)`, `ReduceLROnPlateau (factor=0.5, patience=4, min_lr=1e-6)`, `EarlyStopping (patience=10, monitor='val_acc')` và `ModelCheckpoint` lưu tự động trọng số tốt nhất để đánh giá trên `PrivateTest`.
- Tự động xuất nhật ký chi tiết từng epoch ra `logs/<exp_name>_epochs.csv`.

### 2.3. Thiết kế Mô hình CNN Baseline (`src/models_baseline.py`)
Hai kiến trúc Baseline được thiết kế theo khối cơ sở `Conv2d(3×3, padding=1, bias=False) → BatchNorm2d → Activation`:
1. **`SmallCNN` (1.469.095 tham số — 6 tầng Conv)**: 3 Block (`2×Conv 32 → 2×Conv 64 → 2×Conv 128`), sau mỗi block có `MaxPool2d(2×2)` và `Dropout2d(0.25)`; bộ phân loại gồm `AdaptiveAvgPool2d(6×6) → Flatten → Linear(4608, 256) → BatchNorm1d → Act → Dropout(0.5) → Linear(256, 7)`.
2. **`MediumCNN` (6.606.919 tham số — 8 tầng Conv VGG-style)**: 3 Block (`2×Conv 64 → 3×Conv 128 → 3×Conv 256`), sau mỗi block có `MaxPool2d(2×2)` và `Dropout2d(0.25)`; bộ phân loại gồm `AdaptiveAvgPool2d(6×6) → Flatten → Linear(9216, 512) → BatchNorm1d → Act → Dropout(0.5) → Linear(512, 7)`.
- Cả hai mô hình đều cung cấp thuộc tính `last_conv_layer` giúp Thành viên 3 trực tiếp trích xuất Final Convolution Filters, Feature Maps và Grad-CAM.

### 2.4. Kết quả Thực nghiệm Nhóm 1 (6 thực nghiệm có kiểm soát)
Toàn bộ 6 thực nghiệm được huấn luyện trên Kaggle GPU NVIDIA Tesla T4 với cùng siêu tham số cố định (`batch_size=64`, `max_epochs=45`, `Adam lr=1e-3`, `seed=42`). Số liệu tổng hợp từ `logs/exp1_summary.csv`:

| Mã TN | Mô hình | Augmentation | Activation | Loss | Params | Best Ep / Tổng Ep | Train Acc (cuối) | Val Acc (%) | Min Val Loss | Test Acc (%) | Test Macro F1 (%) | Disgust Recall (%) | Thời gian (s) |
| :---: | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Exp1_1` | SmallCNN | Không | ReLU | CrossEntropy | 1.469.095 | 26 / 36 | 91.16% | 64.86% | 1.0276 | 65.17% | 63.70% | 56.36% | 223.4 |
| `Exp1_2` | SmallCNN | Có | ReLU | CrossEntropy | 1.469.095 | 44 / 45 | 61.83% | 65.23% | 0.9414 | 66.20% | 61.00% | 32.73% | 461.4 |
| `Exp1_3` | MediumCNN | Có | ReLU | CrossEntropy | 6.606.919 | 44 / 45 | 65.77% | **67.15%** | **0.9135** | 67.99% | 63.81% | 56.36% | 581.4 |
| `Exp1_4` | MediumCNN | Có | LeakyReLU | CrossEntropy | 6.606.919 | 44 / 45 | 64.50% | 65.20% | 0.9204 | 67.40% | 63.97% | 56.36% | 587.4 |
| `Exp1_5` | MediumCNN | Có | Sigmoid | CrossEntropy | 6.606.919 | 41 / 45 | 51.23% | 52.52% | 1.2335 | 53.13% | 40.18% | **0.00%** | 589.1 |
| `Exp1_6` | MediumCNN | Có | ReLU | **Weighted CE** | 6.606.919 | 44 / 45 | 66.81% | 67.09% | 0.9277 | **68.51%** | **65.63%** | **74.55%** | 581.1 |

**Hình minh họa kết quả**:
- *Hình 2.3*: Biểu đồ cột so sánh 6 thực nghiệm Nhóm 1 (`figures/exp1_baseline_comparison.png`).
- *Hình 2.4*: Đường cong hội tụ Accuracy & Loss của các thực nghiệm (`figures/Exp1_1_SmallCNN_NoAug_ReLU_curves.png` đến `figures/Exp1_6_MediumCNN_Aug_ReLU_WeightedCE_curves.png`).

### 2.5. Phân tích và Giải thích Kết quả Thực nghiệm Nhóm 1
1. **Tác động quyết định của Data Augmentation (`Exp1_1` vs. `Exp1_2`)**:
   - Khi không có Data Augmentation (`Exp1_1`), mô hình `SmallCNN` nhanh chóng học thuộc tập huấn luyện: `Train Acc` vọt lên **91.16%** trong khi `Val Acc` chững quanh **64.06%** (khoảng cách **Overfitting Gap = +27.10%**), còn `Val Loss` tăng ngược từ `1.0276` (epoch 12) lên hơn `1.40`, buộc EarlyStopping dừng sớm tại epoch 36.
   - Khi bật Data Augmentation (`Exp1_2`), hiện tượng overfitting bị **triệt tiêu hoàn toàn** (Gap cuối `-2.33%`, tức `Val Acc` thậm chí cao hơn `Train Acc` do tập train chịu tác động đồng thời của biến đổi ảnh và Dropout). `Min Val Loss` giảm sâu từ `1.0276` xuống **`0.9414`** và `Test Acc` tăng từ `65.17%` lên **`66.20%` (+1.03%)**.
2. **Tác động của độ sâu và dung lượng biểu diễn (`Exp1_2` vs. `Exp1_3`)**:
   - Mở rộng từ `SmallCNN` (6 tầng Conv, 1.47M tham số) lên `MediumCNN` (8 tầng Conv, 6.61M tham số) giúp mạng trích xuất đặc trưng biểu cảm khuôn mặt ở mức trừu tượng cao hơn rõ rệt: `Val Acc` tăng **`+1.92%`** (`65.23% → 67.15%`), `Test Acc` tăng **`+1.79%`** (`66.20% → 67.99%`), `Test Macro F1` tăng **`+2.81%`** (`61.00% → 63.81%`), và `Min Val Loss` đạt mức thấp nhất Nhóm 1 (**`0.9135`**).
3. **So sánh hàm kích hoạt (`Exp1_3` vs. `Exp1_4` vs. `Exp1_5`)**:
   - **`ReLU` (`Exp1_3`) vs. `LeakyReLU` (`Exp1_4`)**: Cả hai đều hội tụ tốt và ổn định; `ReLU` vượt hơn về `Val Acc` (`67.15%` vs `65.20%`) và `Test Acc` (`67.99%` vs `67.40%`), trong khi `LeakyReLU` nhỉnh nhẹ về `Macro F1` (`63.97%` vs `63.81%`).
   - **`Sigmoid` (`Exp1_5`) — Minh chứng Vanishing Gradient**: Dù đã có `BatchNorm2d`, hàm `Sigmoid` có đạo hàm cực đại chỉ bằng `0.25` và dễ bão hòa ở hai đầu `(0, 1)`. Trên mạng CNN 8 tầng Conv, gradient bị suy giảm mạnh qua các tầng, khiến mô hình hội tụ rất kém: `Test Acc` tụt xuống **`53.13%`** (kém `ReLU` tới **`-14.86%`**), `Macro F1` chỉ **`40.18%`**, và lớp thiểu số `Disgust` có **Recall = `0.00%`** (không nhận diện đúng bất kỳ mẫu nào).
4. **Hiệu quả của Weighted CrossEntropy xử lý mất cân bằng lớp (`Exp1_3` vs. `Exp1_6`)**:
   - Sử dụng trọng số cân bằng mượt `sqrt_balanced` ($w_c \propto \sqrt{N / (7 \cdot N_c)}$) giúp tăng mức phạt khi phân loại sai lớp thiểu số `Disgust` mà không gây bất ổn gradient.
   - Kết quả cực kỳ ấn tượng: **`Disgust Recall` tăng vọt từ `56.36%` lên `74.55%` (`+18.19%`)**, đồng thời kéo **`Test Macro F1` tăng từ `63.81%` lên `65.63%` (`+1.82%`)** và **`Test Acc` đạt đỉnh Nhóm 1 là `68.51%` (`+0.52%`)** — vượt mốc ~67% của báo cáo tham khảo và tiến rất sát mốc benchmark `69% / 71%`.
5. **Đề xuất chuyển giao cho Thành viên 2**:
   - Tại epoch 45 của `Exp1_3` và `Exp1_6`, đường cong `Train Loss` và `Val Loss` vẫn bám sát nhau (Gap chỉ `+0.16%`) và chưa có dấu hiệu overfitting. Do đó, ở Nhóm thực nghiệm 2, Thành viên 2 có nhiều dư địa cải thiện thêm bằng cách tăng số epoch (60–80), kết hợp `CosineAnnealingLR`, thiết kế mạng sâu hơn có kết nối tắt (`Residual CNN` / `LargeCNN`) và thử nghiệm `Transfer Learning` (VGG16, ResNet18/34).
