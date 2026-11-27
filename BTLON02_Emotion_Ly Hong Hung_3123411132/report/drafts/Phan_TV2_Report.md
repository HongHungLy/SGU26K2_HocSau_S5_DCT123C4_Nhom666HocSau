# BẢN THẢO BÁO CÁO — PHẦN THÀNH VIÊN 2 (VAI TRÒ B)
**Người phụ trách**: Nguyễn Bá Khải — MSSV: `3123411138`
**Phạm vi**: Phần 3 — Thiết kế kiến trúc mô hình nâng cao, Kỹ thuật huấn luyện & Siêu tham số, Bảng tổng hợp thực nghiệm Nhóm 2 và Phân tích Fail Cases.

> **Ghi chú số liệu**: mọi con số thực nghiệm lấy từ `logs/exp2_summary.csv` (mỗi cấu hình 1 lần chạy, seed 42, GPU Tesla T4); số tham số / GMACs đo trực tiếp từ `src/models_advanced.py`. Các nhận định chưa được kiểm chứng bằng thí nghiệm riêng được đánh dấu *(giả thuyết)*. **Phạm vi thực tế so với kế hoạch**: do giới hạn thời gian GPU, nhóm B, C, D chạy **30 epoch** với **ResNet18 pretrained** (không dùng ResNet34); bỏ các run `A5` (EfficientNet), `C3` (LR ÷ 3), `D` với WD = 0 / LS 0.05 / `AllReg`, `F3`, và nhóm G chỉ chạy ResNet18 (50 epoch).

---

## PHẦN 3: KIẾN TRÚC MÔ HÌNH NÂNG CAO, KỸ THUẬT HUẤN LUYỆN VÀ TỐI ƯU SIÊU THAM SỐ

### 3.1. Mục tiêu và nguyên tắc thiết kế thực nghiệm
Từ mốc tốt nhất của Nhóm 1 (`Exp1_6`, MediumCNN + Weighted CE: Val **67.09%**, Test **68.51%**, Macro F1 **65.63%**), Nhóm 2 hướng tới việc tiến sát/vượt benchmark tham chiếu (~69% Public / ~71% Private) bằng ba hướng: (i) kiến trúc sâu hơn có kết nối tắt và Transfer Learning, (ii) so sánh có hệ thống optimizer – scheduler, (iii) regularization để thu hẹp khoảng cách Train/Val.

Nguyên tắc để kết quả đáng tin cậy:
1. **Kiểm soát biến**: mỗi nhóm thực nghiệm chỉ thay đổi một yếu tố; seed cố định `42`, cùng `batch_size=64`, cùng pipeline dữ liệu của TV1 (`src/dataset.py`) và cùng vòng huấn luyện (`src/trainer.py`, AMP, EarlyStopping theo `val_acc`).
2. **Không rò rỉ dữ liệu**: mọi lựa chọn (kiến trúc, optimizer, scheduler, regularization, `best_model.pth`) dựa trên **Val Acc (PublicTest)**; **Test Acc (PrivateTest)** chỉ báo cáo, không dùng để chọn mô hình.
3. **Val Loss so sánh được**: khi bật Label Smoothing, loss khi train dùng smoothing nhưng Val/Test Loss vẫn là Cross-Entropy thuần (lớp `SmoothedWeightedCE` trong `exp2_advanced.py`), nên các đường cong Loss giữa các thí nghiệm có cùng thang đo.
4. **Công bằng với Nhóm 1**: Nhóm A dùng đúng công thức `Exp1_6` (Adam 1e-3, weight decay 1e-4, ReduceLROnPlateau, Weighted CE `sqrt_balanced`, 45 epoch) — chỉ khác kiến trúc (riêng mô hình pretrained dùng LR của thân bằng 0.1× LR của đầu phân loại).
5. **Tính tái lập & chịu lỗi**: log CSV từng epoch, gộp bảng tổng hợp theo tên thí nghiệm, bỏ qua thí nghiệm đã chạy xong khi chạy lại (an toàn khi Kaggle ngắt phiên).

### 3.2. Thiết kế kiến trúc mô hình (`src/models_advanced.py`)
Tất cả mô hình đều viết bằng PyTorch/torchvision và có thuộc tính `last_conv_layer` (lớp Conv cuối) và `gradcam_layer` để TV3 trực quan hóa Final Convolution Filters / Grad-CAM.

**(a) `LargeCNN` — VGG-style 4 block** (đầu vào `48×48×1`, **10.001.223 tham số**, ≈ **0.64 GMACs/ảnh**)

| Block | Cấu trúc | Kích thước ra |
| :---: | :--- | :---: |
| 1 | 2 × [Conv3×3(64) → BN → ReLU] → MaxPool → Dropout2d(0.1) | 24×24×64 |
| 2 | 2 × [Conv3×3(128) → BN → ReLU] → MaxPool → Dropout2d(0.2) | 12×12×128 |
| 3 | 3 × [Conv3×3(256) → BN → ReLU] → MaxPool → Dropout2d(0.3) | 6×6×256 |
| 4 | 3 × [Conv3×3(512) → BN → ReLU] → MaxPool → Dropout2d(0.4) | 3×3×512 |
| Head | Flatten(4608) → FC(512) → BN → ReLU → Dropout(0.5) → FC(7) | 7 |

Điểm thiết kế: (1) mở rộng `MediumCNN` của TV1 (3 block, 8 Conv, 6.6M tham số) thành 4 block, 10 Conv; (2) **Dropout2d tăng dần theo độ sâu** (0.1 → 0.4): block đầu giữ thông tin cạnh/góc, các block sau có nhiều tham số hơn nên được regularize mạnh hơn; (3) giữ đặc trưng không gian 3×3 trước FC (tùy chọn `head="gap"` giảm còn 7.90M tham số); (4) khởi tạo Kaiming.

**(b) `ResNetMini` — Residual CNN cho ảnh 48×48** (**11.171.271 tham số**, ≈ **1.25 GMACs/ảnh** ở `base_width=64`; biến thể `base_width=32`: **2.796.263 tham số**, ≈ **0.31 GMACs**)

- **Stem** `Conv3×3(base) → BN → ReLU` ở độ phân giải 48×48 (không dùng stem 7×7/stride 2 + MaxPool của ImageNet vì sẽ làm mất chi tiết khóe mắt, miệng ở ảnh chỉ 48×48).
- **4 stage** × 2 `BasicResBlock` với số kênh `base → 2base → 4base → 8base` (64-128-256-512), stride `1,2,2,2` ⇒ bản đồ đặc trưng `48 → 24 → 12 → 6`.
- Mỗi block: `Conv3×3 → BN → ReLU → [Dropout2d] → Conv3×3 → BN`, **cộng shortcut** rồi ReLU; shortcut dùng `Conv1×1(stride) + BN` khi đổi kích thước/số kênh.
- Head: `GlobalAvgPool → Dropout(0.3) → FC(7)`; `zero_init_residual` đặt γ của BN cuối mỗi block = 0 để mạng khởi đầu gần đồng nhất, ổn định khi huấn luyện sâu.
- Mục đích khoa học: so sánh `LargeCNN` (không có skip) với `ResNetMini` (có skip) ở số tham số tương đương (~10–11M) để cô lập **tác động của kết nối tắt**; so sánh `base_width=64` với `32` để đánh giá liệu mô hình nhẹ hơn ~4 lần có đủ dung lượng cho FER2013.

**(c) `TransferCNN` — Transfer Learning / Fine-tuning (torchvision, trọng số ImageNet)**

| Kiến trúc | Tham số (đầu 7 lớp) | Kích thước ảnh | GMACs/ảnh | Vai trò |
| :--- | :---: | :---: | :---: | :--- |
| ResNet18 | 11.180.103 | 128×128 | 0.59 | Fine-tune toàn phần (A3) và đóng băng thân (A6) |
| ResNet34 | 21.288.263 | 128×128 | 1.20 | Fine-tune toàn phần (A4) |
| EfficientNet-B0 | 4.016.515 | 128×128 | 0.13 | Mạng nhẹ (A5, tùy chọn) |
| VGG16 | 134.289.223 | 48×48 hoặc 128×128 | 0.82 / 5.13 | **Fail case** (F1) và bản khắc phục (F4) |

- Ảnh xám được nhân thành 3 kênh và chuẩn hóa theo thống kê ImageNet (`in_channels=3` trong `src/dataset.py`), resize 48 → 128.
- Thay đầu phân loại 1000 lớp bằng `Dropout → Linear(7)` (VGG: đặt lại Dropout của 2 lớp FC 4096).
- **Learning rate phân tầng**: thân = 0.1× LR đầu phân loại để không phá hủy đặc trưng pretrained; **freeze backbone** (kèm giữ BatchNorm ở eval) để so sánh *feature extraction* với *fine-tuning*.

### 3.3. Kỹ thuật huấn luyện và siêu tham số
| Yếu tố | Các giá trị khảo sát | Cơ sở lý thuyết ngắn gọn |
| :--- | :--- | :--- |
| Optimizer | **SGD + Momentum** (0.9, Nesterov, lr 0.03, wd 5e-4) • **Adam** (lr 1e-3, wd 1e-4 dạng L2) • **AdamW** (lr 1e-3, wd 1e-2) | SGD+Momentum thường hội tụ về cực tiểu phẳng, tổng quát hóa tốt nhưng nhạy LR; Adam thích nghi LR theo tham số nên hội tụ nhanh; AdamW **tách rời weight decay** khỏi cập nhật gradient nên regularization đúng nghĩa hơn Adam+L2. *(Giá trị LR là giá trị mặc định phổ biến, chưa được tinh chỉnh riêng — xem C2/C3.)* |
| LR & Scheduler | **ReduceLROnPlateau** (×0.5, patience 4) vs **CosineAnnealingLR** (T_max = số epoch, η_min = 1e-6); LR × 3 (`C2`); `C3` (LR ÷ 3) không chạy | Plateau phản ứng theo Val Loss (muộn, theo sự kiện); Cosine giảm LR mượt theo lịch cố định, thường giúp hội tụ vào vùng cực tiểu tốt hơn ở cuối. |
| Dropout | 0 / 0.3 (mặc định) / 0.5 — với mô hình pretrained Dropout chỉ nằm ở **đầu phân loại** | Ngẫu nhiên tắt đơn vị để giảm co-adaptation. |
| Weight Decay | 1e-2 (mặc định AdamW) / 1e-1 (cao); `WD = 0` không chạy | Phạt độ lớn trọng số, giảm độ phức tạp hiệu dụng. |
| Label Smoothing | 0 / 0.10; `0.05` không chạy | Mềm hóa nhãn one-hot, giảm quá tự tin (nhãn FER2013 vốn nhiễu). |
| Mất cân bằng lớp | Weighted CE `sqrt_balanced` (kế thừa từ TV1, dùng cho mọi thí nghiệm hợp lệ) | Tăng trọng số lớp `Disgust`. |
| Khác | AMP (mixed precision), EarlyStopping theo `val_acc` (patience 10 với Plateau; 20 với Cosine); nhóm B/C/D: 30 epoch, nhóm G: 50 epoch, ảnh 128×128, LR của thân = 0.1× LR đầu, lưu checkpoint theo Val Acc cao nhất | — |

### 3.4. Kết quả Thực nghiệm Nhóm 2
Số liệu lấy từ `logs/exp2_summary.csv` (cột `gap_at_best` = Train Acc − Val Acc tại epoch tốt nhất).

**Bảng 3.1 — Nhóm A: Kiến trúc (cùng công thức `Exp1_6`)**
| Mã TN | Kiến trúc | Input | Params | Best Ep | Train Acc | Val Acc | Test Acc | Macro F1 | Gap | Thời gian (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Exp1_6` (TV1) | MediumCNN | 48×48 | 6.606.919 | 44 | 66.53% | 67.09% | 68.51% | 65.63% | −0.56% | 581.1 |
| `Exp2_A1` | LargeCNN | 48×48 | 10.001.223 | 45 | 65.41% | 65.14% | 67.12% | 63.57% | +0.26% | 780.7 |
| `Exp2_A2` | ResNetMini (w64) | 48×48 | 11.171.271 | 45 | 65.99% | 65.37% | 67.07% | 64.14% | +0.62% | 1404.9 |
| `Exp2_A2b` | ResNetMini (w32) | 48×48 | 2.796.263 | 45 | 67.67% | 66.87% | 68.21% | 65.63% | +0.80% | 631.6 |
| `Exp2_A3` | ResNet18 pretrained | 128×128 | 11.180.103 | 38 | 92.64% | 69.32% | 70.60% | 70.28% | +23.31% | 952.0 |
| `Exp2_A4` | ResNet34 pretrained | 128×128 | 21.288.263 | 27 | 93.18% | 70.55% | 71.19% | 70.83% | +22.63% | 1161.6 |
| `Exp2_A5` | EfficientNet-B0 *(tùy chọn)* | 128×128 | 4.016.515 | — | — | — | — | — | — | *không chạy (bỏ bằng `--skip_optional`)* |
| `Exp2_A6` | ResNet18 frozen | 128×128 | 11.180.103 (huấn luyện 3.591) | 14 | 39.42% | 44.47% | 42.85% | 34.61% | -5.05% | 404.6 |

*(Gap của `Exp1_6` tính từ số liệu TV1: Train 66.53% − Val 67.09%.)*

**Bảng 3.2 — Nhóm B: Optimizer** (ResNet18 pretrained 128×128, Cosine, 30 epoch, Weighted CE, Aug)
| Mã TN | Cấu hình | Best Ep / Tổng | Train Acc | Val Acc | Test Acc | Macro F1 | Disgust Recall | Gap | Val Loss cuối | Thời gian (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Exp2_B_resnet18_SGD_Momentum` | SGD+Momentum 0.9 (Nesterov), lr 0.03, wd 5e-4 | 29 / 30 | 62.27% | **61.27%** | 62.36% | 58.07% | 61.82% | +1.00 | 1.0684 | 624 |
| `Exp2_B_resnet18_Adam` | Adam, lr 1e-3, wd 1e-4 | 23 / 30 | 89.08% | **69.13%** | 70.21% | 69.59% | 70.91% | +19.96 | 1.7196 | 636 |
| `Exp2_B_resnet18_AdamW` | AdamW, lr 1e-3, wd 1e-2 | 22 / 30 | 88.48% | **69.38%** | 70.47% | 69.91% | 74.55% | +19.10 | 1.7342 | 632 |

**Bảng 3.3 — Nhóm C: LR & Scheduler** (AdamW, ResNet18 pretrained, 30 epoch)
| Mã TN | Cấu hình | Best Ep / Tổng | Train Acc | Val Acc | Test Acc | Macro F1 | Disgust Recall | Gap | Val Loss cuối | Thời gian (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Exp2_B_resnet18_AdamW` | Cosine, lr 1e-3 *(= B-AdamW, tham chiếu)* | 22 / 30 | 88.48% | **69.38%** | 70.47% | 69.91% | 74.55% | +19.10 | 1.7342 | 632 |
| `Exp2_C1_resnet18_adamw_Plateau` | ReduceLROnPlateau (×0.5, patience 4), lr 1e-3 | 19 / 29 | 86.87% | **68.90%** | 70.10% | 70.07% | 72.73% | +17.96 | 1.7138 | 610 |
| `Exp2_C2_resnet18_adamw_Cosine_lrx3` | Cosine, lr 3e-3 | 25 / 30 | 87.67% | **69.04%** | 70.58% | 70.62% | 78.18% | +18.63 | 1.5261 | 634 |

**Bảng 3.4 — Nhóm D: Regularization** (AdamW, Cosine, lr 1e-3, ResNet18 pretrained, 30 epoch; thay đổi **một yếu tố** so với tham chiếu)
| Mã TN | Cấu hình | Best Ep / Tổng | Train Acc | Val Acc | Test Acc | Macro F1 | Disgust Recall | Gap | Val Loss cuối | Thời gian (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Exp2_B_resnet18_AdamW` | Mặc định: Dropout 0.3, wd 1e-2, LS 0 *(tham chiếu)* | 22 / 30 | 88.48% | **69.38%** | 70.47% | 69.91% | 74.55% | +19.10 | 1.7342 | 632 |
| `Exp2_D_resnet18_Drop_none` | Không Dropout (0.0) | 25 / 30 | 92.67% | **68.85%** | 69.96% | 70.03% | 70.91% | +23.83 | 1.7810 | 631 |
| `Exp2_D_resnet18_Drop_strong` | Dropout mạnh (0.5) | 20 / 30 | 84.96% | **68.85%** | 69.49% | 69.35% | 74.55% | +16.11 | 1.6553 | 630 |
| `Exp2_D_resnet18_WD_high` | Weight Decay cao (1e-1) | 30 / 30 | 92.15% | **69.18%** | 70.44% | 69.94% | 70.91% | +22.97 | 1.6519 | 632 |
| `Exp2_D_resnet18_LS_0.10` | Label Smoothing 0.10 | 27 / 30 | 92.21% | **69.77%** | 71.61% | 71.17% | 78.18% | +22.44 | 1.0551 | 643 |

**Bảng 3.5 — Nhóm G: Chạy cuối** (cấu hình chọn tự động theo Val Acc từ nhóm B, C, D: AdamW + Cosine + lr 1e-3 + Dropout 0.3 + wd 1e-2 + Label Smoothing 0.10)
| Mã TN | Cấu hình | Best Ep / Tổng | Train Acc | Val Acc | Test Acc | Macro F1 | Disgust Recall | Gap | Val Loss cuối | Thời gian (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Exp2_G_Final_resnet18_adamw` | AdamW cosine lr 1e-3, Dropout 0.3, wd 1e-2, LS 0.10, 50 epoch | 35 / 50 | 95.42% | **70.24%** | 71.16% | 71.29% | 78.18% | +25.18 | 1.0638 | 1079 |
| `Exp2_A4_ResNet34_TL_128` | *(đối chiếu)* ResNet34, Adam, Plateau, 45 epoch | 27 / 37 | 93.18% | **70.55%** | 71.19% | 70.83% | 72.73% | +22.63 | 1.9704 | 1162 |

*(Gap = Train Acc − Val Acc tại epoch có Val Acc cao nhất, đơn vị điểm %. Val Loss luôn là Cross-Entropy thuần nên so sánh được giữa các hàng, kể cả khi có Label Smoothing.)*

**Hình minh họa** *(sinh tự động bởi `exp2_advanced.py`)*:
- *Hình 3.1*: Biểu đồ so sánh toàn bộ thực nghiệm Nhóm 2 — `figures/exp2_comparison.png`.
- *Hình 3.2*: Đường cong Train/Val Loss & Accuracy từng thực nghiệm — `figures/Exp2_*_curves.png`.
- *Hình 3.3*: So sánh Fail Cases với mô hình khỏe — `figures/exp2_fail_cases.png`.
- *Hình 3.4*: Nhóm B (optimizer) — `figures/exp2_optimizer_curves.png`; *Hình 3.5*: Nhóm C (scheduler) — `figures/exp2_scheduler_curves.png`; *Hình 3.6*: Nhóm D (regularization; trục Val Acc bắt đầu từ 65%) — `figures/exp2_regularization.png`.

### 3.5. Phân tích kết quả

**Kết quả đã có — Nhóm A** *(số liệu từ `logs/exp2_summary.csv`, mỗi cấu hình 1 lần chạy với seed 42; chênh lệch dưới ~0.5 điểm có thể chỉ là nhiễu)*

1. **Transfer Learning vượt rõ rệt mạng tự thiết kế.** ResNet34 pretrained (`A4`) đạt Val **70.55%** / Test **71.19%**, ResNet18 pretrained (`A3`) đạt Val **69.32%** / Test **70.60%**; so với `Exp1_6` (Val 67.09%, Test 68.51%) là tăng **+3.46 / +2.68** (A4) và **+2.23 / +2.09** (A3) điểm. Test của `A4` ngang mốc tham chiếu ~71% (PrivateTest), Val của `A4` cao hơn mốc ~69% (PublicTest).
2. **Ba mạng tự thiết kế không vượt được `Exp1_6`.** `A1` (Val 65.14%), `A2` (65.37%), `A2b` (66.87%) đều thấp hơn 67.09%. Cả ba có **Best Epoch = 45/45**, Val Loss vẫn đang giảm ở epoch cuối và Gap chỉ +0.3 đến +0.8 điểm ⇒ biểu hiện **chưa hội tụ hết trong 45 epoch (under-trained)**, không phải overfitting. Công thức `Exp1_6` (Adam 1e-3, Plateau) có lẽ chưa phù hợp với mạng sâu hơn *(giả thuyết — cần kiểm tra bằng chạy dài hơn / scheduler Cosine)*.
3. **Skip connection không cho lợi ích rõ trong cấu hình này.** `A1` (không skip, 10.0M) so với `A2` (có skip, 11.2M): Val 65.14% vs 65.37% (+0.23, trong ngưỡng nhiễu). **Mạng rộng hơn không tốt hơn**: `A2b` (w32, 2.8M tham số) cao hơn `A2` (w64) **+1.50 điểm Val** và chạy nhanh hơn ×2.2 (631.6 s so với 1404.9 s).
4. **Fine-tune toàn phần >> đóng băng thân.** `A6` (chỉ huấn luyện 3.591 tham số) chỉ đạt Val **44.47%** / Test 42.85%, Recall lớp Disgust chỉ **3.64%** (so với 70.91% của `A3`): đặc trưng ImageNet cố định không đủ cho ảnh mặt xám 48×48 *(giả thuyết: lệch miền; cần fine-tune để thích nghi)*.
5. **Transfer Learning bị overfit rõ**: Train Acc 92.64% (`A3`) và 93.18% (`A4`) so với Val 69–70% ⇒ **Gap +23 điểm**. Val Loss thấp nhất ở epoch 7 (`A3`, 0.990) và epoch 5 (`A4`, 0.954) rồi **tăng lên 1.86 / 1.97**, trong khi Val Acc vẫn tăng đến epoch 38 / 27; Test Loss của `A3`/`A4` (1.63 / 1.66) cao hơn nhiều so với của `A1`–`A2b` (0.90 – 0.92) dù Accuracy cao hơn ⇒ mô hình **quá tự tin (over-confident)**. Đây là động lực cho Nhóm D (Dropout, Weight Decay, Label Smoothing) nhằm thu hẹp Gap.
6. **Chi phí / lợi ích** (T4): `A3` 952 s (≈21 s/epoch), `A4` 1161.6 s (≈31 s/epoch, dừng sớm ở epoch 37). `A4` hơn `A3` +1.23 điểm Val với chi phí/epoch ×1.5. Chạy lại `A4` lần 2 (để lấy checkpoint) cho kết quả **trùng khớp lần đầu** (Val 70.55%, Test 71.19%, Best Ep 27) ⇒ kết quả tái lập được với seed 42.

**Lựa chọn cho các nhóm sau (đề xuất):** dùng **ResNet18 pretrained** để quét optimizer / scheduler / regularization (Nhóm B, C, D) vì rẻ hơn ResNet34 ~32% mỗi epoch mà kém chỉ 1.2 điểm Val.

**Kết quả Nhóm B, C, D, G** *(ResNet18 pretrained; mỗi cấu hình 1 lần chạy — chênh lệch dưới ~0.5 điểm có thể chỉ là nhiễu)*

2. **Optimizer (Bảng 3.2).** AdamW (Val **69.38%**) và Adam (**69.13%**) gần như ngang nhau (chênh 0.25, trong ngưỡng nhiễu), cả hai vượt xa SGD+Momentum (**61.27%**). Tuy vậy SGD **chưa hội tụ** trong 30 epoch (Best Ep 29/30, Train Acc chỉ 62.27%, Gap +1.00) và LR 0.03 (thân 0.003) là giá trị mặc định chưa tinh chỉnh ⇒ kết quả chỉ cho thấy SGD cần nhiều epoch / LR lớn hơn trong ngân sách này, **không kết luận SGD kém hơn về bản chất**. Về tốc độ, AdamW đạt Val ≥ 65% ở epoch 6, Adam ở epoch 6, SGD không đạt. Cả Adam và AdamW đều có Val Loss thấp nhất ở epoch 7 rồi tăng lên ~1.72–1.73 (overfit / quá tự tin), còn SGD thì Val Loss giảm dần đến 1.07.
3. **LR & Scheduler (Bảng 3.3).** Cosine lr 1e-3 (Val **69.38%**) cao hơn ReduceLROnPlateau (**68.90%**, dừng sớm ở epoch 29 do EarlyStopping, Best Ep 19) 0.48 điểm Val và 0.37 điểm Test ⇒ Cosine nhỉnh hơn nhưng chênh lệch nhỏ. Tăng LR gấp 3 (3e-3) cho Val 69.04% (−0.34) nhưng **Val Loss cuối thấp hơn** (1.53 so với 1.73) và Recall lớp Disgust 78.18%. `C3` (LR ÷ 3) không chạy nên chưa khảo sát được chiều giảm LR.
4. **Regularization (Bảng 3.4, Hình 3.6).** So với tham chiếu (Val 69.38%, Gap +19.10): **Dropout mạnh 0.5** giảm Gap xuống +16.11 nhưng Val còn 68.85% (−0.53); **không Dropout** làm Gap tăng lên +23.83 và Val cũng 68.85%; **Weight Decay cao (0.1)** không giảm Gap (+22.97) và Val 69.18% (−0.20); **Label Smoothing 0.10** cho Val tốt nhất **69.77%** (+0.39), Test **71.61%** (+1.14) và đặc biệt kéo **Val Loss cuối từ 1.73 xuống 1.06** (Val Loss thấp nhất vẫn ở epoch ~7–9, nhưng không còn tăng vọt về sau). Như vậy Label Smoothing là yếu tố hiệu quả nhất để chống **quá tự tin**, còn **Gap Train–Val gần như không thu hẹp** (+22.44) vì Train Acc vẫn đạt ~92%. Các thay đổi Val Acc chỉ nằm trong khoảng −0.53 đến +0.39 điểm nên khó phân biệt với nhiễu; riêng cải thiện Val Loss của Label Smoothing (−0.68) là rõ ràng. Gap nhỏ hơn không đồng nghĩa Val Acc cao hơn (Dropout mạnh). `WD = 0`, `LS 0.05` và `AllReg` không chạy nên chưa kiểm tra *over-regularization*.
5. **Chạy cuối (Bảng 3.5).** Cấu hình tự chọn (AdamW + Cosine + LS 0.10, 50 epoch) đạt Val **70.24%** / Test **71.16%** / Macro F1 **71.29%** / Disgust Recall **78.18%** (+0.86 Val so với `B-AdamW`, +0.47 so với `D-LS_0.10`; nhờ thêm 20 epoch và Label Smoothing), Val Loss cuối chỉ 1.06. Tuy nhiên vẫn **kém `A4` (ResNet34) 0.31 điểm Val** (70.24% so với 70.55%) và ngang Test (71.16% so với 71.19%).
6. **So với mốc TV1 (`Exp1_6`: Val 67.09%, Test 68.51%, Macro F1 65.63%, Disgust Recall 74.55%).** Best model `A4`: Val **+3.46**, Test **+2.68**, Macro F1 **+5.20** điểm; Val 70.55% cao hơn mốc ~69% (PublicTest) và Test 71.19% ngang mốc ~71% (PrivateTest) của đề. Nhóm G ResNet18: Val +3.15, Test +2.65.
7. **Lớp thiểu số (Disgust).** Recall: `Exp1_6` 74.55% • `A4` 72.73% • `G` / `D-LS_0.10` / `C2` **78.18%** • `A6` (đóng băng) 3.64%. Best model `A4` thấp hơn `Exp1_6` 1.82 điểm ở lớp này dù Macro F1 cao hơn; chênh lệch tính trên 55 ảnh Disgust của tập Test nên 1 ảnh ≈ 1.8 điểm — cần thận trọng. Lớp nào bị nhầm nhiều nhất: xem confusion matrix của TV3.

### 3.6. Phân tích Fail Cases
*(Số liệu từ `logs/exp2_summary.csv` và `logs/Exp2_F*_epochs.csv`; Gap = Train Acc − Val Acc tại epoch có Val Acc cao nhất.)*

**Bảng 3.6 — Fail Cases so với các mô hình khỏe**
| Mã TN | Mô tả | Best Ep | Train Acc | Val Acc | Test Acc | Gap | Val Loss thấp nhất (epoch) | Val Loss cuối | Thời gian (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Exp2_F1` | VGG16 pretrained, 48×48, không Aug/Dropout/WD | 29 | 98.61% | 66.56% | 67.15% | **+32.05%** | 1.0185 (ep 3) | 1.9312 | 1661.4 |
| `Exp2_F2` | LargeCNN không Aug/Dropout/WD | 38 | 98.59% | 64.00% | 64.31% | **+34.59%** | 1.0683 (ep 7) | 3.0593 | 725.7 |
| `Exp2_F4` | VGG16 pretrained, 128×128, bản khắc phục F1 | 22 | 69.06% | 63.47% | 66.17% | +5.58% | 1.0604 (ep 19) | 1.0900 | 2558.4 |
| `Exp2_A1` (đối chứng F2) | LargeCNN có Aug/Dropout/WD/Weighted CE | 45 | 65.41% | 65.14% | 67.12% | +0.26% | 0.9522 (ep 42) | 0.9604 | 780.7 |
| `Exp2_A3` (đối chứng VGG16) | ResNet18 pretrained, 128×128 | 38 | 92.64% | 69.32% | 70.60% | +23.31% | 0.9897 (ep 7) | 1.8569 | 952.0 |
| `Exp2_F3` | VGG16 không BN, khởi tạo ngẫu nhiên | — | — | — | — | — | — | — | *không chạy (thí nghiệm tùy chọn, bỏ bằng `--skip_optional`)* |

**Fail Case 1 — `Exp2_F1_VGG16_TL_48_NoReg`: học thuộc tập huấn luyện (overfitting nặng)**
- *Cấu hình*: VGG16 pretrained, fine-tune **toàn phần**, ảnh **48×48**, **không** Augmentation, **không** Dropout, **không** Weight Decay, **không** Weighted CE, Adam 1e-4, không scheduler, 30 epoch (không dừng sớm để quan sát toàn bộ quá trình).
- *Hiện tượng đo được*: Train Acc tăng lên **98.50%** ở epoch cuối (Train Loss 0.046), trong khi Val Acc tốt nhất chỉ **66.56%** (epoch 29) ⇒ **Gap +32.05 điểm**. Val Loss chạm đáy **1.0185 ở epoch 3**, rồi tăng lên **2.05 ở epoch 10** và còn **1.93 ở epoch 30** (gần gấp đôi mức thấp nhất, trong khi Train Loss tiếp tục giảm).
- *Điểm cần lưu ý (trung thực)*: **Accuracy không sụp** — Val 66.56% / Test 67.15% ngang các mạng tự thiết kế (`A2b`: 66.87% / 68.21%). Dấu hiệu overfitting thể hiện rõ ở **Gap và Val Loss**, không ở Accuracy: mô hình học thuộc và trở nên **quá tự tin** ở những mẫu dự đoán sai. Epoch tốt nhất theo Val Acc (29) lệch xa epoch tốt nhất theo Val Loss (3).
- *Nguyên nhân*:
  1. **Dung lượng quá lớn so với dữ liệu** *(được số liệu ủng hộ)*: 134.289.223 tham số (riêng 2 lớp FC 4096 chiếm ~119M) so với 28.709 ảnh huấn luyện ⇒ đủ sức *học thuộc* từng ảnh (Train Loss → 0.046).
  2. **Không có bất kỳ regularization nào** (Aug, Dropout, WD) và fine-tune toàn phần *(được số liệu ủng hộ)*: không có lực nào cản việc ghi nhớ nhiễu nhãn của FER2013 (nhiều ảnh nhãn mơ hồ).
  3. **Đầu vào quá nhỏ** *(giả thuyết, chưa kiểm chứng riêng)*: ảnh 48×48 qua 5 lần MaxPool chỉ còn bản đồ đặc trưng 1×1; `AdaptiveAvgPool(7×7)` của VGG nhân bản giá trị này thành 49 vị trí nên phần FC 25.088 → 4096 học trên thông tin lặp lại. Thí nghiệm `F4` đổi độ phân giải cùng lúc với nhiều yếu tố khác nên không tách riêng được tác động này.
  4. **Lệch miền ImageNet** *(giả thuyết)*: ảnh màu vật thể so với ảnh mặt xám độ phân giải thấp.
- *Khắc phục & kiểm chứng* — `Exp2_F4_VGG16_TL_128_Fixed` (ảnh 128×128, Augmentation, Dropout 0.5, AdamW wd 1e-2, LR thân = 0.01× LR đầu, Cosine, Weighted CE, 25 epoch): **Gap giảm từ +32.05 xuống +5.58 điểm**, Val Loss cuối giảm từ 1.93 xuống 1.09. Tuy nhiên **Val Acc thấp hơn F1 (63.47% so với 66.56%, −3.09 điểm)** và Train Acc chỉ 69.06% ⇒ mô hình chuyển sang **hơi thiếu khớp (under-fit)**: LR thân 1e-5 rất nhỏ và chỉ 25 epoch có thể chưa đủ để thích nghi *(giả thuyết — chưa chạy lại để kiểm chứng)*. Vì F4 thay đổi nhiều yếu tố cùng lúc nên không thể quy riêng cho yếu tố nào. Quan trọng hơn: ở cùng 128×128, **ResNet18 (`A3`, 11.2M tham số) đạt Val 69.32% trong 952 s, còn VGG16 đã khắc phục (`F4`, 134.3M) chỉ 63.47% trong 2558 s** ⇒ VGG16 vừa kém hơn vừa chậm hơn ×2.7.

**Fail Case 2 — `Exp2_F2_LargeCNN_NoReg`: mạng tự thiết kế không regularization**
- *Cấu hình*: giống `Exp2_A1_LargeCNN` nhưng không Augmentation, Dropout = 0, Weight Decay = 0, không Weighted CE, không scheduler, 45 epoch.
- *Hiện tượng*: Train Acc **98.59%** vs Val Acc **64.00%** (epoch 38) ⇒ **Gap +34.59 điểm**; Val Loss thấp nhất **1.0683 ở epoch 7**, rồi tăng đều lên **3.06 ở epoch 45** (Train Loss còn 0.020).
- *So với `A1` (có đầy đủ Aug + Dropout + WD + Weighted CE + Plateau)*: Gap **+0.26**, Val Acc **65.14%** (+1.14), Test Acc **67.12%** (+2.81 so với 64.31%), Val Loss cuối 0.96 (so với 3.06). Như vậy tổ hợp regularization **xóa hầu hết khoảng cách Train/Val (≈34 điểm) mà không làm giảm độ chính xác**, thậm chí tăng. Lưu ý `F2` và `A1` khác nhau ở nhiều yếu tố cùng lúc nên chưa biết yếu tố nào đóng góp nhiều nhất — đó là nhiệm vụ của Nhóm D.
- *Liên hệ TV1*: `Exp1_1` (SmallCNN không Aug) ở epoch cuối có Train 91.16% / Val 64.06%, Gap +27.10 và Val Loss cuối 1.404; mức overfit của `F2` (Gap cuối +35.81, Val Loss cuối 3.06) còn nặng hơn, phù hợp với giả thuyết mạng có dung lượng lớn hơn thì học thuộc mạnh hơn (hai cấu hình khác nhau về nhiều mặt nên chỉ mang tính tham khảo).

**Fail Case 3 (tùy chọn) — `Exp2_F3_VGG16_Scratch_NoBN`**: *không chạy* (thí nghiệm tùy chọn, đã bỏ bằng `--skip_optional` do giới hạn thời gian GPU).

**Bài học rút ra**
1. **Gap Train–Val và Val Loss tăng ngược là chỉ báo overfitting đáng tin cậy hơn Accuracy**: `F1` vẫn đạt Val 66.56% trong khi Val Loss đã gấp đôi mức thấp nhất; chỉ nhìn Val Acc sẽ bỏ sót mô hình quá tự tin (cũng là lý do `A3`/`A4` có Test Loss 1.63/1.66 dù Accuracy cao).
2. **Mô hình lớn không regularization sẽ học thuộc**: cả VGG16 (134M) lẫn LargeCNN (10M) đều có Gap > +32 điểm; regularization là điều kiện bắt buộc với FER2013 (28.7K ảnh).
3. **Regularization giảm Gap hiệu quả nhưng phải đủ năng lực học**: `F2 → A1` giảm Gap ~34 điểm mà vẫn tăng Accuracy; `F1 → F4` giảm Gap ~27 điểm nhưng làm mất 3 điểm Val Acc (nghiêng sang under-fit).
4. **Kiến trúc phù hợp quan trọng hơn kích thước**: ResNet18 (11M, 952 s, Val 69.32%) vượt VGG16 (134M, 2558 s, Val 63.47%) ở cùng độ phân giải 128×128.

### 3.7. Chọn Best Model và bàn giao
- Chọn theo **Val Acc cao nhất** (hòa → Val Macro F1) trong các thí nghiệm hợp lệ (loại Fail Case) bằng `python exp2_advanced.py --finalize`; **không** dùng Test Acc để chọn.
- **Best model**: `Exp2_A4_ResNet34_TL_128` — ResNet34 pretrained (ImageNet), fine-tune toàn phần, ảnh 128×128, Adam lr 1e-3 (thân ×0.1), wd 1e-4, ReduceLROnPlateau, Weighted CE, Aug, Best Epoch 27/37: **Val 70.55% | Test 71.19% | Macro F1 70.83% | Disgust Recall 72.73% | 21.288.263 tham số**; file `best_model.pth` nặng **85.3 MB**.
- *Lưu ý khi diễn giải*: `A4` hơn mô hình chạy cuối `G` (ResNet18, Val 70.24%, Test 71.16%) chỉ 0.31 điểm Val và 0.03 điểm Test — nằm trong ngưỡng nhiễu của 1 lần chạy; `G` nhẹ hơn (11.2M tham số), có Macro F1 (71.29%) và Disgust Recall (78.18%) cao hơn, Val Loss cuối thấp hơn nhiều (1.06 so với 1.97). Theo quy ước chọn theo Val Acc, `A4` là best model; nhóm có thể cân nhắc `G` nếu ưu tiên lớp thiểu số và độ hiệu chuẩn.
- Bàn giao cho TV3: `checkpoints/best_model.pth` (chỉ `model_state_dict` + cấu hình dựng lại mô hình, nạp bằng `load_model_from_checkpoint`), `logs/best_model_info.json`, toàn bộ `figures/Exp2_*_curves.png` và `logs/Exp2_*_epochs.csv`.
