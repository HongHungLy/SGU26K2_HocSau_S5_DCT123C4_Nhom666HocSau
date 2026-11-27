# NỘI DUNG SLIDE THUYẾT TRÌNH — PHẦN THÀNH VIÊN 1 (VAI TRÒ A)
**Phụ trách**: Lý Hồng Hưng — MSSV: `3123411132` (~6 slide đầu của bài thuyết trình nhóm)

---

### Slide 1: Trang bìa & Phân công thành viên
- **Tiêu đề**: Nhận diện cảm xúc khuôn mặt bằng mạng CNN trên bộ dữ liệu Kaggle FER2013 (PyTorch)
- **Môn học**: Học Sâu (Deep Learning)
- **Bảng phân công 3 thành viên**:
  - **TV1 — Lý Hồng Hưng (3123411132)**: EDA, Pipeline PyTorch chuẩn, Data Augmentation & Thực nghiệm Nhóm 1 (Baseline, Activation, Class Weight) — 33%
  - **TV2 — Nguyễn Bá Khải (3123411138)**: Kiến trúc CNN nâng cao, Transfer Learning, Tối ưu siêu tham số & Fail Cases — 34%
  - **TV3 — Hà Trần Gia Nguyên (3123411199)**: Đánh giá Public/Private Test, Trực quan hóa Final Conv Filters & Grad-CAM, Tổng hợp sản phẩm — 33%

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
- **Điểm nhấn số liệu thật (`logs/eda_class_distribution.csv`)**:
  - `Happy` nhiều nhất: **7.215 ảnh train (25.13%)**.
  - `Disgust` ít nhất: **436 ảnh train (1.52%)** → chênh lệch **16.55 lần**.
  - Phân bố 7 lớp đồng đều giữa 3 tập `Training / PublicTest / PrivateTest` → đảm bảo đánh giá công bằng, không rò rỉ dữ liệu.

---

### Slide 4: Pipeline PyTorch & Chiến lược Data Augmentation
- **Hình minh họa chính**: `figures/eda_augmentation_demo.png` (1 ảnh gốc vs 6 ảnh Augmented).
- **Chuẩn hóa**: Grayscale `[-1, 1]` (và hỗ trợ RGB ImageNet cho TV2 fine-tune).
- **Data Augmentation (chỉ bật trên Train)**:
  - `RandomHorizontalFlip(0.5)`, `RandomRotation(±10°)`, `RandomAffine(translate=10%, scale=0.9–1.1)`, `RandomErasing(p=0.2)`.
- **Khung huấn luyện dùng chung (`src/trainer.py`)**: PyTorch + AMP GPU Tesla T4, `Adam (lr=1e-3, wd=1e-4)`, `ReduceLROnPlateau`, `EarlyStopping(patience=10)`, `ModelCheckpoint`, tự động ghi log CSV.

---

### Slide 5: Kiến trúc Mô hình Baseline (`SmallCNN` vs `MediumCNN`)
- **Khối cơ sở**: `Conv2d(3×3) → BatchNorm2d → Activation → MaxPool2d(2×2) → Dropout2d(0.25)`.
- **`SmallCNN` (1.47M params)**: 3 Block (`2×32 → 2×64 → 2×128`, 6 tầng Conv) + `FC(256) → Dropout(0.5) → FC(7)`.
- **`MediumCNN` (6.61M params)**: 3 Block VGG-style (`2×64 → 3×128 → 3×256`, 8 tầng Conv) + `FC(512) → Dropout(0.5) → FC(7)`.
- **Thiết kế hỗ trợ TV3**: Tích hợp thuộc tính `last_conv_layer` để trích xuất Final Conv Filters & Grad-CAM.

---

### Slide 6: Kết quả Thực nghiệm Nhóm 1 (6 thực nghiệm trên GPU T4)
- **Hình minh họa chính**: `figures/exp1_baseline_comparison.png` + cặp so sánh đường cong `Exp1_1` (Overfit) vs `Exp1_6` (Hội tụ chuẩn).
- **Bảng tóm tắt kết quả thật (`logs/exp1_summary.csv`)**:

| Mã TN | Cấu hình | Val Acc | Test Acc | Macro F1 | Disgust Recall |
| :---: | :--- | :---: | :---: | :---: | :---: |
| `Exp1_1` | SmallCNN — Không Aug, ReLU | 64.86% | 65.17% | 63.70% | 56.36% |
| `Exp1_2` | SmallCNN — Có Aug, ReLU | 65.23% | 66.20% | 61.00% | 32.73% |
| `Exp1_3` | MediumCNN — Có Aug, ReLU | **67.15%** | 67.99% | 63.81% | 56.36% |
| `Exp1_4` | MediumCNN — Có Aug, LeakyReLU | 65.20% | 67.40% | 63.97% | 56.36% |
| `Exp1_5` | MediumCNN — Có Aug, Sigmoid | 52.52% | 53.13% | 40.18% | **0.00%** |
| `Exp1_6` | **MediumCNN — Aug, ReLU, Weighted CE** | 67.09% | **68.51%** | **65.63%** | **74.55%** |

- **4 kết luận cốt lõi**:
  1. **Augmentation triệt tiêu overfitting**: Giảm Gap Train-Val từ `+27.10%` xuống `-2.33%`, Test Acc `+1.03%`.
  2. **Tăng độ sâu hiệu quả**: `MediumCNN` tăng `+1.79%` Test Acc so với `SmallCNN`.
  3. **Activation quyết định khả năng học**: `Sigmoid` bị triệt tiêu gradient (`-14.86%` Test Acc, `Disgust Recall = 0%`).
  4. **Weighted CE (`sqrt_balanced`) bứt phá**: Đạt **68.51% Test Acc**, **65.63% Macro F1**, kéo **Disgust Recall lên 74.55% (+18.19%)** → Bàn giao mốc & checkpoint cho TV2, TV3.
