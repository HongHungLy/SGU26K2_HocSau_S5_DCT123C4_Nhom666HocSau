# BẢN THẢO BÁO CÁO — PHẦN THÀNH VIÊN 1 (VAI TRÒ A)
**Người phụ trách**: Lý Hồng Hưng — MSSV: `3123411132`  
**Phạm vi**: Phần 1 (Giới thiệu bài toán & Khám phá dữ liệu FER2013) + Phần 2 (Tiền xử lý, Data Augmentation, Pipeline PyTorch & Kết quả Thực nghiệm Nhóm 1)

---

## PHẦN 1: GIỚI THIỆU BÀI TOÁN VÀ KHÁM PHÁ DỮ LIỆU (EDA)

### 1.1. Đặt vấn đề
Nhận diện cảm xúc khuôn mặt (Facial Expression Recognition — FER) là bài toán phân loại ảnh cốt lõi trong Thị giác máy tính và Học sâu. Đầu vào của bài toán là ảnh khuôn mặt mức xám kích thước `48 × 48` pixel, đầu ra là vector xác suất 7 chiều tương ứng với 7 trạng thái cảm xúc cơ bản:

- `0`: **Angry** (Tức giận)
- `1`: **Disgust** (Ghê tởm)
- `2`: **Fear** (Sợ hãi)
- `3`: **Happy** (Vui vẻ)
- `4`: **Sad** (Buồn bã)
- `5`: **Surprise** (Ngạc nhiên)
- `6`: **Neutral** (Trung tính)

Theo tài liệu đề bài, mốc độ chính xác tham chiếu cao nhất (world-best benchmark) trên bộ FER2013 đạt khoảng **69%** (Public Test) và **71%** (Private Test). Đây là bài toán có độ khó cao do ảnh có độ phân giải thấp (`48×48`), nhiều góc chụp, điều kiện ánh sáng khác nhau, bị che khuất một phần, và có sự giao thoa lớn giữa các biểu cảm gần nhau (như *Fear* vs. *Sad*, *Angry* vs. *Disgust*).

### 1.2. Mô tả và Khám phá bộ dữ liệu Kaggle FER2013
Bộ dữ liệu Kaggle FER2013 gồm tổng cộng **35.887 ảnh**, được chia sẵn thành 3 tập độc lập nhằm đảm bảo không xảy ra rò rỉ dữ liệu (*No Data Leakage*):
- **Training**: 28.709 ảnh (~80%) — dùng để cập nhật trọng số mô hình.
- **Public Testing (Validation)**: 3.589 ảnh (~10%) — dùng để theo dõi quá trình hội tụ, điều chỉnh Learning Rate, chọn Checkpoint tốt nhất và EarlyStopping.
- **Private Testing (Test độc lập)**: 3.589 ảnh (~10%) — giữ cô lập hoàn toàn, chỉ dùng một lần cuối cùng để chấm điểm khách quan.

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

**Hình chèn sau khi chạy `01_eda_fer2013.ipynb`**:
- *Hình 1.1*: Biểu đồ phân bố 7 lớp trên 3 tập (`figures/eda_histogram_distribution.png`).
- *Hình 1.2*: Lưới ảnh mẫu 4×8 tập Training & Validation (`figures/eda_train_samples.png`, `figures/eda_val_samples.png`).

**Nhận xét chính từ EDA**:
1. **Mất cân bằng dữ liệu nghiêm trọng (Class Imbalance)**: Lớp *Happy* chiếm 25.13% (7.215 ảnh train) trong khi lớp *Disgust* chỉ chiếm **1.52%** (436 ảnh train) — chênh lệch tới **16.55 lần**. Nếu chỉ tối ưu Cross-Entropy thông thường, mô hình dễ thiên vị lớp đa số và đạt Recall rất thấp trên lớp *Disgust*.
2. **Nhiễu và độ phân giải thấp**: Kích thước `48×48` làm mất nhiều chi tiết vi mô ở mắt/môi; một số ảnh chứa watermark, độ tương phản thấp, xoay nghiêng hoặc bị tay che mặt. Do đó cần kết hợp **Data Augmentation** và **Regularization (BatchNorm, Dropout)** để tăng khả năng tổng quát hóa.

---

## PHẦN 2: TIỀN XỬ LÝ, DATA AUGMENTATION, PIPELINE PYTORCH & MÔ HÌNH BASELINE

### 2.1. Pipeline dữ liệu và Chiến lược Data Augmentation (`src/dataset.py`)
Toàn bộ hệ thống được xây dựng 100% bằng **PyTorch** (`torch.utils.data.Dataset`, `DataLoader`, `torchvision.transforms`):
- **Chuẩn hóa (Normalization)**: Chuyển giá trị pixel từ `[0, 255]` sang tensor `[0, 1]` rồi chuẩn hóa về `[-1, 1]` (`mean=0.5, std=0.5`) cho ảnh grayscale 1 kênh (hoặc chuẩn ImageNet cho 3 kênh khi Thành viên 2 dùng Transfer Learning).
- **Data Augmentation (chỉ áp dụng cho tập Training)**:
  1. `RandomHorizontalFlip(p=0.5)`: Lật ngang ngẫu nhiên (khuôn mặt đối xứng trái-phải nên giữ nguyên nhãn cảm xúc).
  2. `RandomRotation(degrees=10)`: Xoay nhẹ trong khoảng `[-10°, +10°]` mô phỏng đầu nghiêng.
  3. `RandomAffine(translate=(0.1, 0.1), scale=(0.9, 1.1))`: Dịch chuyển và phóng to/thu nhỏ tối đa 10% giúp mô hình bất biến với vị trí khuôn mặt.
  4. `RandomErasing(p=0.2, scale=(0.02, 0.12))`: Che khuất ngẫu nhiên vùng nhỏ mô phỏng tóc, kính hoặc tay che mặt.
- **Tập PublicTest và PrivateTest**: Không áp dụng Augmentation để đảm bảo đánh giá trung thực.

**Hình chèn sau khi chạy `01_eda_fer2013.ipynb`**:
- *Hình 2.1*: Minh họa 1 ảnh gốc và 6 biến thể Data Augmentation (`figures/eda_augmentation_demo.png`).
- *Hình 2.2*: Lưới ảnh Training Batch sau Augmentation (`figures/eda_augmented_batch.png`).

### 2.2. Khung huấn luyện chuẩn PyTorch (`src/trainer.py`)
Để đảm bảo công bằng giữa mọi thực nghiệm của cả 3 thành viên, khung huấn luyện thống nhất các thành phần:
- Cố định random seed (`set_seed(42)`).
- Sử dụng `model.train()` khi huấn luyện, `model.eval()` + `torch.no_grad()` khi kiểm định/đánh giá.
- Tích hợp **Mixed Precision (AMP)** tăng tốc trên GPU, **ReduceLROnPlateau** giảm Learning Rate khi `val_loss` chững lại, **EarlyStopping (patience=10)** tránh overfitting, và **ModelCheckpoint** lưu tự động trọng số có `val_acc` tốt nhất để đánh giá trên `PrivateTest`.
- Tự động ghi log từng epoch ra file `.csv` (`logs/<exp_name>_epochs.csv`).

### 2.3. Thiết kế Mô hình CNN Baseline (`src/models_baseline.py`)
Hai kiến trúc Baseline được thiết kế với khối cơ sở `Conv2d(3×3, padding=1) -> BatchNorm2d -> Activation`:
1. **`SmallCNN` (3 Block — 6 tầng Conv)**: `32 → 64 → 128` kênh (mỗi block 2 tầng Conv + `MaxPool2d(2×2)` + `Dropout2d(0.25)`), kết thúc bằng `AdaptiveAvgPool2d(6×6) → Linear(4608, 256) → BN → Act → Dropout(0.5) → Linear(256, 7)`.
2. **`MediumCNN` (3 Block VGG-style — 8 tầng Conv)**: Block 1 (`2×Conv 64`), Block 2 (`3×Conv 128`), Block 3 (`3×Conv 256`), kết thúc bằng `Linear(9216, 512) → BN → Act → Dropout(0.5) → Linear(512, 7)`.

### 2.4. Kết quả Thực nghiệm Nhóm 1 (Baseline, Augmentation, Activation & Weighted Loss)
*(Sau khi chạy `02_train_baseline.ipynb` trên Kaggle GPU T4, điền số liệu thật từ `logs/exp1_summary.csv` vào bảng dưới đây)*

| Mã TN | Mô hình | Augmentation | Activation | Loss | Params | Best Epoch | Val Acc (%) | Test Acc (%) | Test Macro F1 (%) | Disgust Recall (%) |
| :---: | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Exp1_1` | SmallCNN | Không | ReLU | CrossEntropy | `[Điền]` | `[Điền]` | `[Điền]` | `[Điền]` | `[Điền]` | `[Điền]` |
| `Exp1_2` | SmallCNN | Có | ReLU | CrossEntropy | `[Điền]` | `[Điền]` | `[Điền]` | `[Điền]` | `[Điền]` | `[Điền]` |
| `Exp1_3` | MediumCNN | Có | ReLU | CrossEntropy | `[Điền]` | `[Điền]` | `[Điền]` | `[Điền]` | `[Điền]` | `[Điền]` |
| `Exp1_4` | MediumCNN | Có | LeakyReLU | CrossEntropy | `[Điền]` | `[Điền]` | `[Điền]` | `[Điền]` | `[Điền]` | `[Điền]` |
| `Exp1_5` | MediumCNN | Có | Sigmoid | CrossEntropy | `[Điền]` | `[Điền]` | `[Điền]` | `[Điền]` | `[Điền]` | `[Điền]` |
| `Exp1_6` | MediumCNN | Có | ReLU | Weighted CE | `[Điền]` | `[Điền]` | `[Điền]` | `[Điền]` | `[Điền]` | `[Điền]` |

**Hình chèn sau khi chạy `02_train_baseline.ipynb`**:
- *Hình 2.3*: Biểu đồ so sánh hiệu năng 6 thực nghiệm Nhóm 1 (`figures/exp1_baseline_comparison.png`).
- *Hình 2.4*: Đường cong Train/Val Accuracy & Loss của các thực nghiệm tiêu biểu (`figures/Exp1_*_curves.png`).

**Khung phân tích kết quả (cập nhật chi tiết sau khi có số liệu thật)**:
1. *Tác động của Data Augmentation (`Exp1_1` vs `Exp1_2`)*: So sánh khoảng cách `Train Acc - Val Acc` và mức tăng `Test Acc`.
2. *Tác động của độ sâu kiến trúc (`Exp1_2` vs `Exp1_3`)*: Đánh giá năng lực trích xuất đặc trưng của `MediumCNN` so với `SmallCNN`.
3. *So sánh hàm kích hoạt (`Exp1_3` vs `Exp1_4` vs `Exp1_5`)*: Giải thích vì sao `ReLU/LeakyReLU` hội tụ tốt hơn rõ rệt so với `Sigmoid` (do hiện tượng bão hòa và triệt tiêu đạo hàm khi mạng sâu 8 tầng Conv).
4. *Tác động của Class Weight (`Exp1_3` vs `Exp1_6`)*: Phân tích đánh đổi giữa `Disgust Recall` / `Macro F1` và `Overall Accuracy`.
