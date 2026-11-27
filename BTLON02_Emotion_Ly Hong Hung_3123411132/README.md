# BÀI TẬP LỚN 02 (BTLON02): NHẬN DIỆN CẢM XÚC KHUÔN MẶT BẰNG MẠNG CNN (FER2013 — PYTORCH)

> **Học phần**: Học Sâu (Deep Learning) — **Nhóm 666**  
> **Repository nhóm**: https://github.com/HongHungLy/SGU26K2_HocSau_S5_DCT123C4_Nhom666HocSau.git  
> **Nhánh phát triển**: `develop`  
> **Framework**: 100% thuần **PyTorch** (`torch`, `torchvision`).

---

## 1. PHÂN CÔNG THÀNH VIÊN NHÓM 666

| STT | Vai trò | Họ và tên | MSSV | Nhiệm vụ chính đã hoàn thành | Tỷ trọng |
| :---: | :---: | :--- | :---: | :--- | :---: |
| 1 | **Vai trò A (TV1)** | **Lý Hồng Hưng** | **3123411132** | EDA bộ Kaggle FER2013, xây dựng pipeline PyTorch chuẩn dùng chung (`dataset.py`, `trainer.py`, `utils.py`), cài đặt & chạy **6 thực nghiệm Nhóm 1** (`models_baseline.py`: Baseline, Data Augmentation, Activation, Weighted Loss), viết Report Phần 1–2 & Slide phần TV1. | **33%** |
| 2 | **Vai trò B (TV2)** | **Nguyễn Bá Khải** | **3123411138** | Phát triển kiến trúc CNN nâng cao & Transfer Learning (`models_advanced.py`: LargeCNN, ResNetMini, ResNet18/34, VGG16), chạy **19 thực nghiệm Nhóm 2** (Optimizer, LR Scheduler, Regularization, Label Smoothing, Fail Cases), chọn Best Model (`ResNet34 TL 128×128`), viết Report Phần 3 & Slide phần TV2. | **34%** |
| 3 | **Vai trò C (TV3)** | **Hà Trần Gia Nguyên** | **3123411199** | Đánh giá độc lập Public/Private Test (`evaluate.py`: Confusion Matrix, Classification Report, True/Wrong Cases), **Trực quan hóa Final Convolution Filters, Feature Maps & Grad-CAM 7 lớp × 3 tầng** (`visualize.py`), tổng hợp Báo cáo (`BaoCao_DoAn_EmotionRecognition_Nhom666.docx/.pdf`), Slide (`Slide_DoAn_EmotionRecognition_Nhom666.pptx/.pdf`) & Nhật ký thực nghiệm. | **33%** |

---

## 2. TÓM TẮT KẾT QUẢ CHÍNH (25 THỰC NGHIỆM TRÊN KAGGLE GPU T4)

Mọi quyết định chọn mô hình đều căn cứ **Val Acc (PublicTest, 3.589 ảnh)**; **Test Acc (PrivateTest, 3.589 ảnh)** được giữ độc lập hoàn toàn chỉ để báo cáo cuối cùng (*No Data Leakage*).

| Giai đoạn | Mô hình tiêu biểu | PublicTest (Val Acc) | PrivateTest (Test Acc) | Test Macro F1 | Disgust Recall |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Benchmark đề bài | World-best FER2013 tham khảo | ~69.00% | ~71.00% | — | — |
| TV1 — Baseline 0 | `Exp1_1` SmallCNN (không Aug, ReLU) | 64.86% | 65.17% | 63.70% | 56.36% |
| TV1 — Vanishing Grad | `Exp1_5` MediumCNN (Aug, Sigmoid) | 52.52% | 53.13% | 40.18% | 0.00% |
| TV1 — Best Baseline | `Exp1_6` MediumCNN (Aug, ReLU, Weighted CE) | 67.09% | 68.51% | 65.63% | 74.55% |
| TV2 — Best Custom CNN | `Exp2_A2b` ResNetMini-w32 (48×48, 2.8M params) | 66.87% | 68.21% | 65.63% | 70.91% |
| TV2 — ResNet18 Final | `Exp2_G` ResNet18 TL (AdamW + Cosine + LS 0.1) | 70.24% | 71.16% | 71.29% | 78.18% |
| **TV2 + TV3 — BEST MODEL** | **`Exp2_A4` ResNet34 TL (128×128, Weighted CE)** | **70.55%** | **71.19%** | **70.83%** | **72.73%** |

---

## 3. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
BTLON02_Emotion_Ly Hong Hung_3123411132/
├── report/                                 # Sản phẩm nộp chính (Báo cáo & Slide hoàn chỉnh)
│   ├── BaoCao_DoAn_EmotionRecognition_Nhom666.docx  # Báo cáo đồ án 41 trang (5 phần + 25 hình nhúng + phân công)
│   ├── BaoCao_DoAn_EmotionRecognition_Nhom666.pdf   # Bản xuất PDF của Báo cáo đồ án
│   ├── Slide_DoAn_EmotionRecognition_Nhom666.pptx   # Bộ 20 slide thuyết trình PowerPoint
│   ├── Slide_DoAn_EmotionRecognition_Nhom666.pdf    # Bản xuất PDF của Slide thuyết trình
│   └── drafts/                             # Bản thảo riêng của từng thành viên (TV1, TV2, TV3)
│       ├── Phan_TV1_Report.md / Phan_TV1_Slide.md
│       ├── Phan_TV2_Report.md / Phan_TV2_Slide.md
│       └── Phan_TV3_Report.md / Phan_TV3_Slide.md
├── docs/
│   ├── nhat_ky_thuc_nghiem.md              # Nhật ký thực nghiệm tổng hợp đầy đủ cả 3 thành viên (25 runs)
│   └── HUONG_DAN_BAN_GIAO_TV2_TV3.md       # Tài liệu bàn giao kỹ thuật giữa các giai đoạn
├── src/                                    # Mã nguồn Python 100% thuần PyTorch
│   ├── __init__.py
│   ├── utils.py                            # [TV1] set_seed, class weights, vẽ biểu đồ 300 DPI
│   ├── dataset.py                          # [TV1] Auto-detect FER2013, Augmentation, DataLoader (1ch & 3ch)
│   ├── trainer.py                          # [TV1] Vòng lặp train/eval, AMP, EarlyStopping, Checkpoint, CSV log
│   ├── models_baseline.py                  # [TV1] SmallCNN & MediumCNN (hỗ trợ ReLU, LeakyReLU, Sigmoid)
│   ├── models_advanced.py                  # [TV2] LargeCNN, ResNetMini, ResNet18/34 TL, VGG16 TL
│   ├── evaluate.py                         # [TV3] Đánh giá Public/Private Test, Confusion Matrix, True/Wrong Cases
│   └── visualize.py                        # [TV3] Final Convolution Filters, Feature Maps & Grad-CAM 3 tầng
├── notebooks/
│   ├── 01_eda_fer2013.ipynb                # [TV1] Kaggle Notebook: Khám phá dữ liệu (EDA)
│   ├── 02_train_baseline.ipynb             # [TV1] Kaggle Notebook: Chạy 6 thực nghiệm Nhóm 1
│   └── 03_train_advanced.ipynb             # [TV2] Kaggle Notebook: Chạy 19 thực nghiệm Nhóm 2
├── figures/                                # 48 hình ảnh & biểu đồ độ phân giải cao (300 DPI)
├── logs/                                   # Toàn bộ file CSV log từng epoch, bảng tổng hợp & classification report
├── checkpoints/                            # Trọng số tiêu biểu: best_model.pth (TV2 ResNet34) & Exp1_6 (TV1)
├── data/
│   └── README.md                           # Hướng dẫn tải & đặt bộ dữ liệu Kaggle FER2013 (fer2013.csv)
├── eda_fer2013.py                          # [TV1] Script CLI chạy EDA
├── exp1_baseline.py                        # [TV1] Script CLI chạy Thực nghiệm Nhóm 1
├── exp2_advanced.py                        # [TV2] Script CLI chạy Thực nghiệm Nhóm 2
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 4. HƯỚNG DẪN CÀI ĐẶT & CHẠY TOÀN BỘ DỰ ÁN

### Cài đặt thư viện
```bash
pip install -r requirements.txt
```
Tải bộ dữ liệu **Kaggle FER2013** (`fer2013.csv`, VD: `deadskull7/fer2013` hoặc link trong đề bài) đặt vào thư mục `data/` (trên Kaggle chỉ cần bấm **Add Input** thêm dataset FER2013, code tự động nhận diện từ `/kaggle/input`).

### Chạy từng giai đoạn từ thư mục gốc dự án
1. **Giai đoạn 1 — Khám phá dữ liệu & Baseline (TV1)**:
   ```bash
   python eda_fer2013.py
   python exp1_baseline.py --epochs 45 --batch_size 64
   ```
2. **Giai đoạn 2 — Kiến trúc nâng cao, Transfer Learning & Tối ưu (TV2)**:
   ```bash
   python exp2_advanced.py --group A
   python exp2_advanced.py --group F
   python exp2_advanced.py --group B C D G
   python exp2_advanced.py --finalize
   ```
3. **Giai đoạn 3 — Đánh giá độc lập & Trực quan hóa Final Conv Filters / Grad-CAM (TV3)**:
   ```bash
   python -m src.evaluate --ckpt checkpoints/best_model.pth
   python -m src.visualize --ckpt checkpoints/best_model.pth --split private
   ```
   *(Có thể thay `--ckpt checkpoints/Exp1_6_MediumCNN_Aug_ReLU_WeightedCE_best.pth` để chạy kiểm thử ngay trên checkpoint Baseline của TV1 có sẵn trong repo.)*
