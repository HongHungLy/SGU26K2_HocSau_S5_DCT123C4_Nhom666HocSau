# HƯỚNG DẪN BÀN GIAO CÔNG VIỆC CHO THÀNH VIÊN 2 & THÀNH VIÊN 3

- **Người bàn giao**: Thành viên 1 (Vai trò A) — **Lý Hồng Hưng** (`3123411132`)
- **Đề tài**: Nhận diện cảm xúc khuôn mặt bằng mạng CNN trên bộ dữ liệu Kaggle FER2013 (**100% PyTorch**)
- **Repository nhóm**: https://github.com/HongHungLy/SGU26K2_HocSau_S5_DCT123C4_Nhom666HocSau.git
- **Nhánh làm việc bắt buộc**: **`develop`** *(tuyệt đối đẩy code lên nhánh `develop`, luôn `git pull origin develop` trước khi làm & trước khi push)*

---

## 1. NHỮNG GÌ THÀNH VIÊN 1 ĐÃ HOÀN THÀNH & ĐẨY SẴN LÊN NHÁNH `develop`

1. **Khám phá dữ liệu (EDA) hoàn chỉnh**:
   - Script & Notebook: `eda_fer2013.py`, `notebooks/01_eda_fer2013.ipynb`
   - Bảng phân bố lớp thật: `logs/eda_class_distribution.csv` (`Training`: 28.709 | `PublicTest`: 3.589 | `PrivateTest`: 3.589; lớp `Disgust` chỉ chiếm 1.52% — ít hơn `Happy` 16.55 lần).
   - 5 hình 300 DPI trong `figures/`: `eda_histogram_distribution.png`, `eda_train_samples.png`, `eda_val_samples.png`, `eda_augmented_batch.png`, `eda_augmentation_demo.png`.
2. **Khung code PyTorch dùng chung (TV2 & TV3 import dùng trực tiếp, không cần viết lại)**:
   - `src/dataset.py`: Tự động nhận diện FER2013 trên Kaggle (`/kaggle/input`) hoặc local (`./data`), hỗ trợ cả `fer2013.csv` lẫn thư mục `train/test`, hỗ trợ cả `in_channels=1, img_size=48` (Custom CNN) và `in_channels=3, img_size=128/224` (Transfer Learning chuẩn ImageNet).
   - `src/trainer.py`: Vòng lặp `fit(...)` tích hợp AMP GPU, `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau` / `CosineAnnealingLR`, tự động xuất CSV từng epoch và đánh giá độc lập `PrivateTest`.
   - `src/utils.py`: `set_seed(42)`, `get_device()`, `compute_class_weights(..., method="sqrt_balanced")`, `plot_training_curves()`, `plot_experiment_comparison()`.
   - `src/models_baseline.py`: `SmallCNN` và `MediumCNN` (đều có thuộc tính `model.last_conv_layer` để TV3 trực quan hóa Filters & Grad-CAM).
3. **Kết quả Thực nghiệm Nhóm 1 (6 thực nghiệm trên Kaggle GPU Tesla T4)**:
   - Bảng tổng hợp: `logs/exp1_summary.csv` và 6 file log chi tiết `logs/Exp1_*_epochs.csv`.
   - 7 hình kết quả: `figures/exp1_baseline_comparison.png` và 6 đường cong `figures/Exp1_*_curves.png`.
   - **Mốc tốt nhất của TV1 (`Exp1_6_MediumCNN_Aug_ReLU_WeightedCE`)**:
     - `Val Acc (PublicTest)`: **67.09%** (`Exp1_3`: **67.15%**)
     - `Test Acc (PrivateTest)`: **68.51%**
     - `Test Macro F1`: **65.63%**
     - `Disgust Recall`: **74.55%** (tăng +18.19% nhờ Weighted CE `sqrt_balanced`)
   - **Checkpoint tốt nhất đã đẩy sẵn lên nhánh `develop`**:
     `checkpoints/Exp1_6_MediumCNN_Aug_ReLU_WeightedCE_best.pth`
4. **Tài liệu phần TV1 đã điền đủ số liệu thật**:
   - `docs/nhat_ky_thuc_nghiem.md` (đã hoàn tất Mục 1 — Nhóm thực nghiệm 1)
   - `report/Phan_TV1_Report.md` (Phần 1 & Phần 2 hoàn chỉnh)
   - `report/Phan_TV1_Slide.md` (6 slide đầu hoàn chỉnh)

---

## 2. HƯỚNG DẪN CHO THÀNH VIÊN 2 (VAI TRÒ B — KIẾN TRÚC NÂNG CAO & TỐI ƯU SIÊU THAM SỐ)

### Bước 1: Kéo code mới nhất từ nhánh `develop`
```bash
git clone -b develop https://github.com/HongHungLy/SGU26K2_HocSau_S5_DCT123C4_Nhom666HocSau.git
cd "SGU26K2_HocSau_S5_DCT123C4_Nhom666HocSau/BTLON02_Emotion_Ly Hong Hung_3123411132"
```
*(Nếu đã clone trước đó: chỉ cần chạy `git checkout develop && git pull origin develop`.)*

### Bước 2: Các file TV2 cần tạo & tái sử dụng pipeline của TV1
- Tạo **`src/models_advanced.py`** (100% PyTorch, nhớ gắn thuộc tính `last_conv_layer` trên mỗi model để TV3 trích xuất Filters/Grad-CAM dễ dàng):
  1. **Custom Deep CNN / Residual CNN (`LargeCNN` hoặc `ResNetMini`)** cho ảnh `48×48×1` (4 block Conv `64 → 128 → 256 → 512` + BatchNorm + Skip Connection + Dropout).
  2. **Transfer Learning / Fine-tuning (`torchvision.models`)**:
     - `ResNet18` / `ResNet34` (pretrained ImageNet, `in_channels=3`, `img_size=128` hoặc `224`).
     - `VGG16` (có thể ghi nhận cả trường hợp **Fail Case / Overfitting** khi chạy trên ảnh nhỏ `48×48` không freeze/không regularization như mục 5 của `demo_report.pdf` để có thêm điểm phân tích sâu!).
- Tạo **`exp2_advanced.py`** và notebook Kaggle **`notebooks/03_train_advanced.ipynb`** (copy cell 1 clone nhánh `develop` từ `02_train_baseline.ipynb`).
- **Cách gọi lại pipeline của TV1**:
  ```python
  from src.dataset import get_dataloaders
  from src.trainer import fit
  from src.utils import compute_class_weights, get_device, set_seed, plot_training_curves

  # Cho Custom CNN 48x48 (1 kênh):
  train_loader, val_loader, test_loader, info = get_dataloaders(batch_size=64, use_aug=True, img_size=48, in_channels=1)

  # Cho Transfer Learning VGG16/ResNet (3 kênh RGB chuẩn ImageNet):
  train_loader, val_loader, test_loader, info = get_dataloaders(batch_size=64, use_aug=True, img_size=224, in_channels=3)
  ```
- **Gợi ý từ kết quả Nhóm 1 để TV2 vượt mốc 68.51% (hướng tới 69%–71%)**:
  - Giữ **Data Augmentation = True**, **ReLU**, và dùng **Weighted CrossEntropy (`method="sqrt_balanced"`)** hoặc thử thêm `label_smoothing=0.05`.
  - Ở `Exp1_6`, epoch 45 vẫn chưa bị overfit → TV2 nên tăng lên **60–80 epochs** kết hợp **`CosineAnnealingLR`**, so sánh Optimizer **`SGD+Momentum` vs `Adam` vs `AdamW`**.

### Bước 3: Đầu ra TV2 cần đẩy lên nhánh `develop`
- Code: `src/models_advanced.py`, `exp2_advanced.py`, `notebooks/03_train_advanced.ipynb`.
- Kết quả: `logs/exp2_summary.csv`, `logs/Exp2_*_epochs.csv`, `figures/Exp2_*_curves.png`, `figures/exp2_comparison.png`.
- Trọng số tốt nhất toàn dự án (`best_model.pth`): Nếu `< 95 MB` thì mở ngoại lệ 1 dòng `!checkpoints/<ten_file>.pth` trong `.gitignore` rồi push lên `develop`; nếu `> 100 MB` thì lưu chỉ `model_state_dict` (giảm 3 lần dung lượng) hoặc gửi link Google Drive cho TV3.
- Tài liệu: Điền Mục 2 trong `docs/nhat_ky_thuc_nghiem.md`, viết `report/Phan_TV2_Report.md` (Phần 3 Report) và `report/Phan_TV2_Slide.md`.

---

## 3. HƯỚNG DẪN CHO THÀNH VIÊN 3 (VAI TRÒ C — ĐÁNH GIÁ SÂU, VISUALIZE FILTERS & TỔNG HỢP)

### Bước 1: Chạy thử ngay trên checkpoint của TV1 (không cần chờ TV2 xong!)
TV1 đã đẩy sẵn checkpoint `checkpoints/Exp1_6_MediumCNN_Aug_ReLU_WeightedCE_best.pth` (`Test Acc = 68.51%`) lên nhánh `develop`. TV3 có thể nạp ngay để viết và test trước toàn bộ code đánh giá & trực quan hóa:

```python
import torch
from src.dataset import get_dataloaders
from src.models_baseline import MediumCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MediumCNN(num_classes=7, in_channels=1, activation="relu").to(device)
ckpt = torch.load("checkpoints/Exp1_6_MediumCNN_Aug_ReLU_WeightedCE_best.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Lấy lớp tích chập cuối cùng để vẽ Filters / Feature Maps / Grad-CAM:
final_conv = model.last_conv_layer
print("Final Conv Layer:", final_conv, "Weight shape:", final_conv.weight.shape)
```

### Bước 2: Các file TV3 cần tạo
1. **`src/evaluate.py`** (và script/notebook `notebooks/04_evaluate_and_visualize.ipynb`):
   - Đánh giá mô hình tốt nhất (của TV1 & sau này là của TV2) trên cả **PublicTest (3.589 ảnh)** và **PrivateTest (3.589 ảnh)**.
   - Xuất Classification Report (Precision, Recall, F1-score cho đủ 7 lớp cảm xúc).
   - Vẽ **Confusion Matrix chuẩn hóa (%)** cho cả PublicTest và PrivateTest (`figures/confusion_matrix_public.png`, `figures/confusion_matrix_private.png`).
   - Vẽ lưới ảnh 4×8 **True Cases** (dự đoán đúng) và **Wrong Cases** (dự đoán sai kèm nhãn `Predict (True)`) giống mẫu trong `demo_report.pdf`.
2. **`src/visualize.py`** *(Mục tô đỏ bắt buộc trong đề bài: **Visualize the final convolution filters**)*:
   - Trực quan hóa **trọng số bộ lọc tích chập (Convolution Filters / Kernels)** và **Feature Maps** tại lớp tích chập cuối cùng (`model.last_conv_layer`) và đối chiếu với lớp Conv đầu.
   - Cài đặt **Grad-CAM Heatmap** cho đủ 7 lớp cảm xúc qua 3 tầng tích chập (đầu – giữa – cuối, tham khảo cấu trúc `code/valid_vgg16.html`), giải thích vùng khuôn mặt mô hình tập trung chú ý (khóe miệng, mắt, chân mày).
3. **Tổng hợp 4 sản phẩm cuối để nộp**:
   - Ghép `report/Phan_TV1_Report.md` + `Phan_TV2_Report.md` + Phần 4 (Đánh giá, Filters/Grad-CAM, Phân tích & Kết luận) + mục *Brief summary for source codes* → xuất bản báo cáo hoàn chỉnh (`.docx` / `.pdf`) trong `report/`.
   - Ghép slide của cả 3 thành viên (có bảng phân công) → xuất `.pptx` / `.pdf` trong `report/`.
   - Kiểm tra hoàn thiện `docs/nhat_ky_thuc_nghiem.md` (đủ 3 mục) và đẩy toàn bộ lên nhánh **`develop`**.

---

## 4. LƯU Ý KHI PUSH CODE LÊN NHÁNH `develop`
- **Không push thư mục `data/`** (`fer2013.csv` ~301 MB vượt giới hạn 100 MB của GitHub, đã có `.gitignore` chặn).
- **Không push nhiều file `.pth` nặng cùng lúc** — chỉ mở ngoại lệ cho duy nhất file checkpoint tốt nhất (`< 95 MB`).
- Quy ước commit message rõ thành viên:
  - TV2: `Emotion TV2 up <mo ta ngan>`
  - TV3: `Emotion TV3 up <mo ta ngan>`
