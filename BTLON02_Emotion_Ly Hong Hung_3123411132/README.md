# BÀI TẬP LỚN 02 (BTLON02): NHẬN DIỆN CẢM XÚC KHUÔN MẶT BẰNG MẠNG CNN (FER2013 - PYTORCH)

> **Học phần**: Học Sâu (Deep Learning)  
> **Repository nhóm**: https://github.com/HongHungLy/SGU26K2_HocSau_S5_DCT123C4_Nhom666HocSau.git  
> **Quy định Git**: Toàn bộ mã nguồn của các thành viên phải được đẩy lên nhánh (branch) **`develop`** trước khi tổng hợp nộp bài.  
> **Framework**: PyTorch (`torch`, `torchvision`).

---

## 1. PHÂN CÔNG THÀNH VIÊN NHÓM (3 THÀNH VIÊN)

| STT | Vai trò | Họ và tên | MSSV | Nhiệm vụ chính | Tỷ trọng |
| :---: | :---: | :--- | :---: | :--- | :---: |
| 1 | **Vai trò A (TV1)** | **Lý Hồng Hưng** | **3123411132** | EDA bộ FER2013, xây dựng pipeline PyTorch chuẩn (`dataset.py`, `trainer.py`, `utils.py`), cài đặt & chạy **Thực nghiệm Nhóm 1** (`models_baseline.py`: Baseline, Augmentation, Activation, Weighted Loss), viết Report Phần 1–2 & Slide tương ứng. | **33%** |
| 2 | **Vai trò B (TV2)** | *(Đang cập nhật)* | - | Phát triển kiến trúc CNN nâng cao & Transfer Learning (`models_advanced.py`), chạy **Thực nghiệm Nhóm 2** (Optimizer, LR Scheduler, Regularization, Fail Cases), bàn giao `best_model.pth`, viết Report Phần 3 & Slide tương ứng. | **34%** |
| 3 | **Vai trò C (TV3)** | *(Đang cập nhật)* | - | Đánh giá chuyên sâu Public/Private Test (`evaluate.py`: Confusion Matrix, True/Wrong Cases), **Trực quan hóa Final Convolution Filters, Feature Maps & Grad-CAM** (`visualize.py`), tổng hợp Report, Slide & Nhật ký thực nghiệm chung. | **33%** |

---

## 2. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
BTLON02_Emotion_Ly Hong Hung_3123411132/
├── src/
│   ├── __init__.py
│   ├── utils.py               # [TV1] set_seed, class weights, hàm vẽ biểu đồ chuẩn 300 DPI
│   ├── dataset.py             # [TV1] Auto-detect FER2013 (CSV/Folder), Augmentation, DataLoader (1ch & 3ch)
│   ├── trainer.py             # [TV1] Vòng lặp train/eval chuẩn PyTorch, EarlyStopping, Checkpoint, CSV log
│   ├── models_baseline.py     # [TV1] SmallCNN & MediumCNN (hỗ trợ ReLU, LeakyReLU, Sigmoid...)
│   ├── models_advanced.py     # [TV2] Deep CNN, ResNet/VGG Transfer Learning (sẽ bổ sung)
│   ├── evaluate.py            # [TV3] Confusion Matrix, Classification Report, True/Wrong Cases (sẽ bổ sung)
│   └── visualize.py           # [TV3] Final Conv Filters, Feature Maps & Grad-CAM (sẽ bổ sung)
├── notebooks/
│   ├── 01_eda_fer2013.ipynb        # [TV1] Kaggle Notebook: Khám phá dữ liệu & xuất hình EDA
│   └── 02_train_baseline.ipynb     # [TV1] Kaggle Notebook: Chạy Thực nghiệm Nhóm 1 trên GPU T4
├── docs/
│   └── nhat_ky_thuc_nghiem.md      # Nhật ký thực nghiệm chung của 3 thành viên
├── figures/                        # Biểu đồ, ảnh mẫu, confusion matrix, conv filters (DPI 300)
├── logs/                           # File CSV log từng epoch & bảng tổng hợp thực nghiệm
├── checkpoints/                    # Trọng số .pth (được .gitignore, chia sẻ qua Google Drive)
├── report/                         # Báo cáo Word/PDF & Slide PPTX/PDF nộp bài
├── data/                           # Chứa fer2013.csv khi chạy local (được .gitignore)
├── exp1_baseline.py                # [TV1] Script CLI chạy toàn bộ Thực nghiệm Nhóm 1
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 3. QUY TRÌNH LÀM VIỆC VỚI GIT (NHÁNH `develop`)

1. Clone repo và chuyển sang nhánh `develop`:
   ```bash
   git clone -b develop https://github.com/HongHungLy/SGU26K2_HocSau_S5_DCT123C4_Nhom666HocSau.git
   cd SGU26K2_HocSau_S5_DCT123C4_Nhom666HocSau
   ```
2. Trước mỗi lần làm việc hoặc push, luôn kéo code mới nhất từ `develop`:
   ```bash
   git pull origin develop
   ```
3. Commit rõ tên thành viên và push lên `develop`:
   ```bash
   git add "BTLON02_Emotion_Ly Hong Hung_3123411132"
   git commit -m "[TV1] Mo ta cong viec vua hoan thanh"
   git push origin develop
   ```

---

## 4. HƯỚNG DẪN CHẠY TRÊN KAGGLE NOTEBOOK (GPU T4)

1. Mở [Kaggle](https://www.kaggle.com/) → **Create → New Notebook**.
2. Bật **Accelerator: GPU T4 x2** (hoặc **GPU T4**) và bật **Internet: On** (để clone repo nhánh `develop`).
3. Bấm **Add Input** → tìm và thêm dataset **FER2013** (ưu tiên `deadskull7/fer2013` có file `fer2013.csv` chia đúng `28.709 / 3.589 / 3.589`; code cũng tự động tương thích `msambare/fer2013`).
4. Upload file `notebooks/01_eda_fer2013.ipynb` hoặc `notebooks/02_train_baseline.ipynb` rồi bấm **Run All**.
5. Cuối notebook sẽ tự nén toàn bộ `figures/`, `logs/` thành file `.zip` để tải về máy và push lên nhánh `develop`.
