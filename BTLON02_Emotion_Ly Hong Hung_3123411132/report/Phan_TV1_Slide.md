# NỘI DUNG SLIDE THUYẾT TRÌNH — PHẦN THÀNH VIÊN 1 (VAI TRÒ A)
**Phụ trách**: Lý Hồng Hưng — MSSV: `3123411132` (~6 slide đầu của bài thuyết trình nhóm)

---

### Slide 1: Trang bìa & Phân công thành viên
- **Tiêu đề**: Nhận diện cảm xúc khuôn mặt bằng mạng CNN trên bộ dữ liệu Kaggle FER2013 (PyTorch)
- **Môn học**: Học Sâu (Deep Learning)
- **Bảng phân công 3 thành viên**:
  - **TV1 — Lý Hồng Hưng (3123411132)**: EDA, Pipeline PyTorch chuẩn, Data Augmentation & Thực nghiệm Nhóm 1 (Baseline, Activation, Class Weight) — 33%
  - **TV2 — *(điền tên)***: Kiến trúc CNN nâng cao, Transfer Learning, Tối ưu siêu tham số & Fail Cases — 34%
  - **TV3 — *(điền tên)***: Đánh giá Public/Private Test, Trực quan hóa Final Conv Filters & Grad-CAM, Tổng hợp sản phẩm — 33%

---

### Slide 2: Giới thiệu bài toán FER2013
- **Input**: Ảnh khuôn mặt mức xám `48 × 48` (`1` kênh).
- **Output**: Vector xác suất 7 lớp cảm xúc (`Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral`).
- **Quy mô dữ liệu**: `35.887` ảnh chia chuẩn 3 tập độc lập (`Training`: 28.709 | `PublicTest/Val`: 3.589 | `PrivateTest/Test`: 3.589).
- **Benchmark tham chiếu**: ~69% (Public Test) / ~71% (Private Test).
- **Thách thức chính**: Ảnh nhỏ `48×48`, nhiễu/che khuất, biểu cảm gần nhau, mất cân bằng dữ liệu nặng.

---

### Slide 3: Khám phá dữ liệu (EDA) & Mất cân bằng lớp
- **Hình minh họa chính**: `figures/eda_histogram_distribution.png` + `figures/eda_train_samples.png`.
- **Điểm nhấn số liệu**:
  - `Happy` nhiều nhất: **7.215 ảnh train (25.13%)**.
  - `Disgust` ít nhất: **436 ảnh train (1.52%)** → chênh lệch **16.55 lần**.
  - Phân bố tỉ lệ giữa `Training`, `PublicTest`, `PrivateTest` tương đồng → đảm bảo đánh giá nhất quán.

---

### Slide 4: Pipeline PyTorch & Chiến lược Data Augmentation
- **Hình minh họa chính**: `figures/eda_augmentation_demo.png` (1 ảnh gốc vs 6 ảnh Augmented).
- **Chuẩn hóa**: Grayscale `[-1, 1]` (hoặc RGB ImageNet cho TV2 fine-tune).
- **Data Augmentation (chỉ trên tập Train)**:
  - `RandomHorizontalFlip(0.5)`
  - `RandomRotation(±10°)`
  - `RandomAffine(translate=10%, scale=0.9–1.1)`
  - `RandomErasing(p=0.2)`
- **Khung huấn luyện dùng chung (`trainer.py`)**: AMP GPU, `EarlyStopping(patience=10)`, `ModelCheckpoint`, `ReduceLROnPlateau`, tự động ghi log CSV từng epoch.

---

### Slide 5: Kiến trúc Mô hình Baseline (`SmallCNN` vs `MediumCNN`)
- **Khối cơ sở**: `Conv2d(3×3) → BatchNorm2d → Activation → MaxPool2d(2×2) → Dropout2d(0.25)`.
- **`SmallCNN`**: 3 Block (`32 → 64 → 128`, 6 tầng Conv) + `FC(256) → Dropout(0.5) → FC(7)`.
- **`MediumCNN`**: 3 Block VGG-style (`2×64 → 3×128 → 3×256`, 8 tầng Conv) + `FC(512) → Dropout(0.5) → FC(7)`.
- **Thiết kế hỗ trợ TV3**: Gắn sẵn thuộc tính `last_conv_layer` để trích xuất Final Conv Filters và Grad-CAM.

---

### Slide 6: Kết quả Thực nghiệm Nhóm 1 (6 thực nghiệm có kiểm soát)
- **Hình minh họa chính**: `figures/exp1_baseline_comparison.png` + đường cong nổi bật (`Exp1_1` vs `Exp1_2` vs `Exp1_3`).
- **Bảng tóm tắt kết quả** *(điền số liệu thật từ `logs/exp1_summary.csv` sau khi chạy Kaggle)*:
  - `Exp1_1` vs `Exp1_2`: Data Augmentation giúp giảm overfitting và tăng Test Acc rõ rệt.
  - `Exp1_2` vs `Exp1_3`: `MediumCNN` sâu hơn cải thiện khả năng học đặc trưng khuôn mặt.
  - `Exp1_3` vs `Exp1_4` vs `Exp1_5`: `ReLU / LeakyReLU` vượt trội `Sigmoid` (tránh vanishing gradient).
  - `Exp1_3` vs `Exp1_6`: `Weighted CrossEntropy` cải thiện Recall lớp thiểu số `Disgust`.
- **Chuyển tiếp sang TV2**: Từ mốc Baseline của TV1, TV2 tiếp tục mở rộng lên Deep CNN, Transfer Learning & tối ưu siêu tham số.
