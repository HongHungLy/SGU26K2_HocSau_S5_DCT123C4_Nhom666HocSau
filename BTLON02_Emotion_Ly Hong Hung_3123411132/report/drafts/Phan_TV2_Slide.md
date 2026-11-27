# NỘI DUNG SLIDE THUYẾT TRÌNH — PHẦN THÀNH VIÊN 2 (VAI TRÒ B)
**Phụ trách**: Nguyễn Bá Khải — MSSV: `3123411138` (~7 slide, nối tiếp 6 slide của TV1)

> Số liệu lấy từ `logs/exp2_summary.csv` (1 lần chạy/cấu hình, seed 42, Tesla T4). Số tham số / GMACs đo thật từ `src/models_advanced.py`.

---

### Slide 7: Mục tiêu & Thiết kế thực nghiệm Nhóm 2
- **Xuất phát điểm (TV1)**: `Exp1_6` MediumCNN + Weighted CE → Val **67.09%**, Test **68.51%**, Macro F1 **65.63%**.
- **Mục tiêu**: tiến sát/vượt benchmark ~69% (Public) / ~71% (Private).
- **3 hướng**: Kiến trúc sâu & Transfer Learning • Optimizer/Scheduler • Regularization.
- **Nguyên tắc**: kiểm soát biến (seed 42, cùng pipeline TV1) • **chọn mô hình theo Val Acc, Test chỉ báo cáo** • Val Loss luôn là CE thuần (so sánh được).
- Sơ đồ quy trình: `A (kiến trúc) → B (optimizer) → C (LR/scheduler) → D (regularization) → G (final) → best_model.pth`.

---

### Slide 8: Kiến trúc tự thiết kế — LargeCNN & ResNetMini
- **Hình**: 2 sơ đồ khối cạnh nhau (vẽ từ bảng 3.2 của Report).
- **LargeCNN** (10.0M tham số, 0.64 GMACs): 4 block VGG `64→128→256→512` + BN + **Dropout2d tăng dần 0.1→0.4** + FC(512).
- **ResNetMini** (11.2M tham số, 1.25 GMACs; bản w32: 2.8M, 0.31 GMACs): stem 3×3 (không MaxPool) + 4 stage × 2 ResBlock, stride `1,2,2,2`, GAP + FC; zero-init residual.
- **Câu hỏi thực nghiệm**: skip connection có giúp không (A1 vs A2)? mạng nhẹ ×4 mất bao nhiêu (A2 vs A2b)?
- Cả hai có `last_conv_layer`/`gradcam_layer` → TV3 vẽ Filters & Grad-CAM.

---

### Slide 9: Transfer Learning / Fine-tuning
- **Mạng pretrained ImageNet**: ResNet18 (11.2M), ResNet34 (21.3M), EfficientNet-B0 (4.0M), VGG16 (134.3M).
- **Chuẩn bị dữ liệu**: ảnh xám → 3 kênh, chuẩn hóa ImageNet, resize 48 → **128**.
- **Fine-tune**: thay đầu phân loại `Dropout → Linear(7)`, **LR thân = 0.1× LR đầu**; so với **đóng băng thân** (chỉ huấn luyện 3.591 tham số).
- **Kết quả**: A3 ResNet18 — Val **69.32%** / Test **70.60%** (952 s) • A4 ResNet34 — Val **70.55%** / Test **71.19%** (1162 s) • A6 đóng băng thân — Val 44.47% / Test 42.85% • A5 EfficientNet-B0: không chạy.

---

### Slide 10: Kết quả Nhóm A — So sánh kiến trúc
- **Hình**: `figures/exp2_comparison.png` (lọc nhóm A) hoặc biểu đồ cột.
- **Bảng** (Val / Test / Gap / thời gian): `Exp1_6` 67.09 / 68.51 / −0.56 / 581 s • A1 LargeCNN 65.14 / 67.12 / +0.26 / 781 s • A2 ResNetMini 65.37 / 67.07 / +0.62 / 1405 s • A2b ResNetMini-w32 66.87 / 68.21 / +0.80 / 632 s • A3 ResNet18-TL 69.32 / 70.60 / +23.31 / 952 s • A4 ResNet34-TL **70.55 / 71.19** / +22.63 / 1162 s • A6 Frozen 44.47 / 42.85 / −5.05 / 405 s.
- **Kết luận**: (1) Transfer Learning vượt mạng tự thiết kế +2.2 đến +3.5 điểm Val so với `Exp1_6`, đạt mốc ~71% Private. (2) Mạng tự thiết kế chưa hội tụ trong 45 epoch (Best Ep = 45/45) nên không vượt mốc TV1; skip connection không cho lợi ích rõ; w32 tốt hơn w64. (3) Fine-tune toàn phần >> đóng băng. (4) TL overfit: Gap ≈ +23, Val Loss tăng sau epoch 5–7 ⇒ cần regularization (Nhóm D).
- **Hình**: `figures/exp2_comparison.png`.

---

### Slide 11: Tối ưu hóa — Optimizer, LR & Scheduler (Nhóm B, C)
- **Thiết lập**: ResNet18 pretrained 128×128, 30 epoch, LR thân = 0.1× LR đầu, Weighted CE + Aug. *(Giảm từ 60 xuống 30 epoch do giới hạn GPU.)*
- **Optimizer (Cosine)**: Val **AdamW 69.38%** • Adam 69.13% • SGD+Momentum 61.27% (chưa hội tụ: Train 62.3%, Best Ep 29/30; lr 0.03 chưa tinh chỉnh). AdamW ≈ Adam (chênh 0.25 = nhiễu).
- **Scheduler (AdamW)**: **Cosine lr 1e-3: 69.38%** • Cosine lr 3e-3: 69.04% (Val Loss cuối thấp hơn: 1.53 vs 1.73) • ReduceLROnPlateau: 68.90% (dừng sớm epoch 29). Cosine nhỉnh hơn +0.48 Val / +0.37 Test.
- **Hình**: `figures/exp2_optimizer_curves.png` và `figures/exp2_scheduler_curves.png`.
- **Kết luận**: chọn **AdamW + Cosine, lr 1e-3**. Chênh lệch giữa Adam/AdamW và giữa các scheduler nằm trong ngưỡng nhiễu (1 lần chạy/cấu hình); khác biệt lớn duy nhất là SGD chưa hội tụ trong 30 epoch.

---

### Slide 12: Regularization — Dropout, Weight Decay, Label Smoothing (Nhóm D)
- **Thiết lập**: tham chiếu = B-AdamW (Dropout 0.3, wd 1e-2, LS 0): Val 69.38%, Gap +19.10, Val Loss cuối 1.73. Mỗi lần đổi **một** yếu tố.
- **Bảng** (Val / Gap / Val Loss cuối): Không Dropout 68.85 / +23.8 / 1.78 • Dropout 0.5 68.85 / +16.1 / 1.66 • WD cao (0.1) 69.18 / +23.0 / 1.65 • **Label Smoothing 0.1: 69.77 / +22.4 / 1.06**.
- **Hình**: `figures/exp2_regularization.png`.
- **Kết luận**: Label Smoothing là yếu tố hiệu quả nhất — Val +0.39, Test +1.14 (71.61%) và **Val Loss cuối giảm 1.73 → 1.06** (hết quá tự tin). Dropout mạnh thu hẹp Gap nhưng không tăng Val Acc; Weight Decay cao không giảm Gap. **Gap Train–Val không thu hẹp nhiều** (Train vẫn ~92%). Chênh lệch Val Acc ≤ 0.5 điểm ≈ nhiễu. *(Không chạy: WD = 0, LS 0.05, AllReg.)*

---

### Slide 13: Fail Cases — Mạng lớn không regularization sẽ học thuộc
- **Hình**: `figures/exp2_fail_cases.png` (Train vs Val Acc/Loss: F1, F2, F4 và mô hình khỏe).
- **F1 — VGG16 pretrained, 48×48, không Aug/Dropout/WD**: Train **98.5%** vs Val **66.56%** → **Gap +32.05**; Val Loss thấp nhất **1.02 ở epoch 3**, rồi tăng lên **~2.0** (gần gấp đôi) trong khi Train Loss → 0.05. Accuracy không sụp (Test 67.15%) nhưng mô hình **quá tự tin**.
- **F2 — LargeCNN không regularization**: Train 98.6% vs Val 64.0% → **Gap +34.59**; Val Loss 1.07 (epoch 7) → **3.06**. So với A1 (có Aug + Dropout + WD): Gap +0.26, Val **65.14%**, Test **67.12%** (vs 64.31%).
- **Nguyên nhân**: 134M tham số vs 28.7K ảnh (riêng FC 4096 ≈ 119M) • không có regularization nào • *(giả thuyết)* ảnh 48×48 qua 5 MaxPool chỉ còn 1×1 và lệch miền ImageNet.
- **Khắc phục (F4)**: 128×128 + Aug + Dropout 0.5 + AdamW + LR thân ×0.01 + Cosine → **Gap +5.58** (giảm 26.5 điểm), Val Loss cuối 1.09, **nhưng Val Acc chỉ 63.47%** (−3.1 so với F1): hơi under-fit.
- **Bài học**: Gap và Val Loss là chỉ báo overfit đáng tin hơn Accuracy • regularization là bắt buộc • ResNet18 (11M, 952 s, Val 69.32%) vượt VGG16 đã khắc phục (134M, 2558 s, Val 63.47%).

---

### Slide 14: Best Model & Bàn giao
- **Best model (chọn theo Val Acc)**: `Exp2_A4_ResNet34_TL_128` — ResNet34 pretrained 128×128, Adam + Plateau — **Val 70.55% | Test 71.19% | Macro F1 70.83% | Disgust Recall 72.73%** (21.3M tham số, 85.3 MB).
- **So với TV1 (`Exp1_6`)**: Δ Val **+3.46** | Δ Test **+2.68** | Δ Macro F1 **+5.20**; Val cao hơn mốc ~69% (Public), Test ngang mốc ~71% (Private).
- **Chạy cuối `G` (ResNet18, AdamW + Cosine + LS 0.1, 50 epoch)**: Val 70.24% | Test 71.16% | Macro F1 71.29% | Disgust Recall 78.18% — kém A4 0.31 điểm Val (trong ngưỡng nhiễu), nhẹ hơn gần 2×.
- **Bàn giao TV3**: `checkpoints/best_model.pth` (tự mô tả kiến trúc, nạp bằng `load_model_from_checkpoint`), `logs/Exp2_*_epochs.csv`, `figures/Exp2_*_curves.png`.
- **Bài học chính**: (1) Transfer Learning (fine-tune toàn phần) là yếu tố cải thiện lớn nhất (+2.2 đến +3.5 Val so với mạng tự thiết kế). (2) Mạng lớn không regularization học thuộc (Gap > +32); Gap và Val Loss là chỉ báo đáng tin hơn Accuracy. (3) Label Smoothing là regularization hiệu quả nhất cho vấn đề quá tự tin; Optimizer/Scheduler chỉ khác nhau trong ngưỡng nhiễu (trừ SGD chưa hội tụ). (4) Mỗi cấu hình chỉ 1 lần chạy nên chênh lệch < 0.5 điểm không đủ kết luận.
