# NHẬT KÝ THỰC NGHIỆM (EXPERIMENT LOG) — BTLON02: FER2013 (PYTORCH)

- **Đề tài**: Nhận diện cảm xúc khuôn mặt bằng mạng CNN (Facial Expression Recognition — FER2013)
- **Môn học**: Học Sâu (Deep Learning) — **Nhóm 666**
- **Bộ dữ liệu**: Kaggle FER2013 (`Training`: 28.709 | `PublicTest/Val`: 3.589 | `PrivateTest/Test`: 3.589)
- **Framework**: 100% thuần PyTorch (`torch`, `torchvision`)
- **Nhánh Git đồng bộ**: `develop`
- **Thành viên thực hiện**:
  - **TV1 (Vai trò A)**: Lý Hồng Hưng (`3123411132`) — EDA, Pipeline PyTorch & Thực nghiệm Nhóm 1 (6 lần chạy)
  - **TV2 (Vai trò B)**: Nguyễn Bá Khải (`3123411138`) — Kiến trúc nâng cao, Transfer Learning, Tối ưu & Fail Cases (19 lần chạy)
  - **TV3 (Vai trò C)**: Hà Trần Gia Nguyên (`3123411199`) — Đánh giá độc lập, Trực quan hóa Filters/Feature Maps/Grad-CAM & Tổng hợp Báo cáo/Slide

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
4. **Dư địa cải thiện cho TV2**: Tại epoch 45, `Exp1_3` và `Exp1_6` vẫn chưa bị overfitting → TV2 kế thừa công thức Augmentation + Weighted CE để phát triển Deep CNN & Transfer Learning.

---

## 2. NHÓM THỰC NGHIỆM 2 — KIẾN TRÚC NÂNG CAO, TRANSFER LEARNING & TỐI ƯU SIÊU THAM SỐ
- **Phụ trách**: Thành viên 2 — **Nguyễn Bá Khải** (`3123411138`)
- **Ngày chạy**: 22–23/11/2026 (theo `logs/exp2_summary.csv`) | **Môi trường**: Kaggle Notebook — GPU Tesla T4 (PyTorch + AMP)
- **Script / Notebook**: `exp2_advanced.py`, `notebooks/03_train_advanced.ipynb`; mã nguồn mô hình: `src/models_advanced.py`.
- **Cấu hình cố định**: `batch_size=64`, `seed=42`, Data Augmentation theo `src/dataset.py` của TV1 (trừ các Fail Case ghi rõ), `EarlyStopping(monitor='val_acc')`.
  Nhóm A dùng đúng công thức **`Exp1_6`** của TV1 (Adam 1e-3, wd 1e-4, ReduceLROnPlateau, Weighted CE `sqrt_balanced`, 45 epoch) để so sánh công bằng; nhóm B/C/D chạy **30 epoch** trên **ResNet18 pretrained 128×128** (LR thân = 0.1× LR đầu), nhóm G chạy 50 epoch.
- **Nguyên tắc chọn mô hình**: Mọi quyết định chọn cấu hình và `best_model.pth` căn cứ **Val Acc (PublicTest)**; **Test Acc (PrivateTest)** chỉ dùng báo cáo độc lập.
- **Nguồn số liệu thật**: `logs/exp2_summary.csv`, 19 file `logs/Exp2_*_epochs.csv`, `logs/best_model_info.json`.

### 2.1. Kết quả 19 thực nghiệm Nhóm 2 (sinh từ `logs/exp2_summary.csv`)
| Mã TN | Ngày chạy | Kiến trúc | Input | Optimizer & LR | Scheduler / Reg | Params | Best Ep / Tổng Ep | Train Acc @best | Gap @best | Val Acc (%) | Min Val Loss | Test Acc (%) | Macro F1 (%) | Thời gian (s) | Nhận xét / Fail Case |
| :--- | :---: | :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `Exp2_A1_LargeCNN` | 22/11/2026 | large_cnn | 48×48 | adam lr=0.001 | plateau; wd=0.0001; ls=0; aug=có | 10.001.223 | 45 / 45 | 65.41% | +0.26% | **65.14** | 0.9522 | 67.12 | 63.57 | 781 | Chưa hội tụ hết (Best Ep = 45/45, Val Loss còn giảm); Gap ≈ 0 nên không overfit. Không vượt `Exp1_6` (Val 67.09%). |
| `Exp2_A2_ResNetMini` | 22/11/2026 | resnet_mini | 48×48 | adam lr=0.001 | plateau; wd=0.0001; ls=0; aug=có | 11.171.271 | 45 / 45 | 65.99% | +0.62% | **65.37** | 0.9671 | 67.07 | 64.14 | 1405 | Tương đương A1 (+0.23 Val): skip connection chưa cho lợi ích rõ; rộng 64 chậm gấp 2.2 lần A2b. |
| `Exp2_A2b_ResNetMini_w32` | 22/11/2026 | resnet_mini | 48×48 | adam lr=0.001 | plateau; wd=0.0001; ls=0; aug=có | 2.796.263 | 45 / 45 | 67.67% | +0.80% | **66.87** | 0.9524 | 68.21 | 65.63 | 632 | Tốt nhất trong các mạng tự thiết kế (Val 66.87%), nhẹ nhất (2.8M), nhưng vẫn thấp hơn `Exp1_6` 0.22 Val. |
| `Exp2_A3_ResNet18_TL_128` | 22/11/2026 | resnet18 | 128×128 | adam lr=0.001 (thân ×0.1) | plateau; wd=0.0001; ls=0; aug=có | 11.180.103 | 38 / 45 | 92.64% | +23.31% | **69.32** | 0.9897 | 70.60 | 70.28 | 952 | Transfer Learning: +2.23 Val so với `Exp1_6`. Overfit: Train 92.6% vs Val 69.3% (Gap +23.3); Val Loss min ở epoch 7 rồi tăng lên 1.86. |
| `Exp2_A4_ResNet34_TL_128` | 23/11/2026 | resnet34 | 128×128 | adam lr=0.001 (thân ×0.1) | plateau; wd=0.0001; ls=0; aug=có | 21.288.263 | 27 / 37 | 93.18% | +22.63% | **70.55** | 0.9543 | **71.19** | **70.83** | 1162 | **BEST MODEL (theo Val Acc)**: Val 70.55% / Test 71.19% (vượt mốc benchmark 69%/71%). Dừng sớm epoch 37; chạy lại lần 2 cho kết quả trùng khớp. |
| `Exp2_A6_ResNet18_Frozen_128` | 22/11/2026 | resnet18 | 128×128 | adam lr=0.001 | plateau; wd=0.0001; ls=0; aug=có | 11.180.103 | 14 / 24 | 39.42% | -5.05% | **44.47** | 1.5650 | 42.85 | 34.61 | 405 | Thân đóng băng chỉ đạt Val 44.47%, Disgust Recall 3.64% ⇒ đặc trưng ImageNet cố định không đủ cho ảnh mặt xám. |
| `Exp2_F1_VGG16_TL_48_NoReg` | 23/11/2026 | vgg16 | 48×48 | adam lr=0.0001 | none; wd=0; ls=0; aug=không | 134.289.223 | 29 / 30 | 98.61% | +32.05% | **66.56** | 1.0185 | 67.15 | 66.11 | 1661 | **FAIL CASE**: Gap +32.05, Val Loss min 1.0185 (ep 3) → 1.93; quá tự tin do 134M tham số không regularization. |
| `Exp2_F2_LargeCNN_NoReg` | 23/11/2026 | large_cnn | 48×48 | adam lr=0.001 | none; wd=0; ls=0; aug=không | 10.001.223 | 38 / 45 | 98.59% | +34.59% | **64.00** | 1.0683 | 64.31 | 61.80 | 726 | **FAIL CASE**: Gap +34.59, Val Loss min 1.0683 (ep 7) → 3.06; kém A1 1.14 Val và 2.81 Test. |
| `Exp2_F4_VGG16_TL_128_Fixed` | 23/11/2026 | vgg16 | 128×128 | adamw lr=0.001 (thân ×0.01) | cosine; wd=0.01; ls=0; aug=có | 134.289.223 | 22 / 25 | 69.06% | +5.58% | **63.47** | 1.0604 | 66.17 | 64.13 | 2558 | Bản khắc phục F1: Gap giảm mạnh còn +5.58 nhưng Val 63.47% (hơi under-fit); chậm gấp 2.7 lần A3. |
| `Exp2_B_resnet18_SGD_Momentum` | 23/11/2026 | resnet18 | 128×128 | sgd lr=0.03 (thân ×0.1) | cosine; wd=0.0005; ls=0; aug=có | 11.180.103 | 29 / 30 | 62.27% | +1.00% | **61.27** | 1.0670 | 62.36 | 58.07 | 624 | Chưa hội tụ trong 30 epoch (Best Ep 29/30, Train 62.3%, Gap +1.0). |
| `Exp2_B_resnet18_Adam` | 23/11/2026 | resnet18 | 128×128 | adam lr=0.001 (thân ×0.1) | cosine; wd=0.0001; ls=0; aug=có | 11.180.103 | 23 / 30 | 89.08% | +19.96% | **69.13** | 0.9810 | 70.21 | 69.59 | 636 | Gap +19.96; Val Loss min ở epoch 7 rồi tăng lên 1.72 (quá tự tin). |
| `Exp2_B_resnet18_AdamW` | 23/11/2026 | resnet18 | 128×128 | adamw lr=0.001 (thân ×0.1) | cosine; wd=0.01; ls=0; aug=có | 11.180.103 | 22 / 30 | 88.48% | +19.10% | **69.38** | 0.9749 | 70.47 | 69.91 | 632 | Tốt nhất Nhóm B theo Val (69.38%); làm tham chiếu cho Nhóm C, D. |
| `Exp2_C1_resnet18_adamw_Plateau` | 23/11/2026 | resnet18 | 128×128 | adamw lr=0.001 (thân ×0.1) | plateau; wd=0.01; ls=0; aug=có | 11.180.103 | 19 / 29 | 86.87% | +17.96% | **68.90** | 0.9621 | 70.10 | 70.07 | 610 | Dừng sớm ở epoch 29 (Best Ep 19); kém Cosine 0.48 Val / 0.37 Test. |
| `Exp2_C2_resnet18_adamw_Cosine_lrx3` | 23/11/2026 | resnet18 | 128×128 | adamw lr=0.003 (thân ×0.1) | cosine; wd=0.01; ls=0; aug=có | 11.180.103 | 25 / 30 | 87.67% | +18.63% | **69.04** | 0.9749 | 70.58 | 70.62 | 634 | lr 3e-3: Val 69.04%, Test 70.58%, Disgust Recall 78.18%; Val Loss cuối thấp hơn (1.53 vs 1.73). |
| `Exp2_D_resnet18_Drop_none` | 23/11/2026 | resnet18 | 128×128 | adamw lr=0.001 (thân ×0.1) | cosine; wd=0.01; ls=0; aug=có | 11.180.103 | 25 / 30 | 92.67% | +23.83% | **68.85** | 0.9712 | 69.96 | 70.03 | 631 | Không Dropout: Gap +23.83 (+4.7 so tham chiếu), Val −0.53. |
| `Exp2_D_resnet18_Drop_strong` | 23/11/2026 | resnet18 | 128×128 | adamw lr=0.001 (thân ×0.1) | cosine; wd=0.01; ls=0; aug=có | 11.180.103 | 20 / 30 | 84.96% | +16.11% | **68.85** | 0.9597 | 69.49 | 69.35 | 630 | Dropout 0.5: Gap giảm còn +16.11 nhưng Val −0.53. |
| `Exp2_D_resnet18_WD_high` | 23/11/2026 | resnet18 | 128×128 | adamw lr=0.001 (thân ×0.1) | cosine; wd=0.1; ls=0; aug=có | 11.180.103 | 30 / 30 | 92.15% | +22.97% | **69.18** | 0.9816 | 70.44 | 69.94 | 632 | wd 0.1: Gap +22.97, Val −0.20; Best Ep 30/30. |
| `Exp2_D_resnet18_LS_0.10` | 23/11/2026 | resnet18 | 128×128 | adamw lr=0.001 (thân ×0.1) | cosine; wd=0.01; ls=0.1; aug=có | 11.180.103 | 27 / 30 | 92.21% | +22.44% | **69.77** | 0.9764 | 71.61 | 71.17 | 643 | Val 69.77% (+0.39), Test 71.61% (+1.14); **Val Loss cuối giảm 1.73 → 1.06** (hết quá tự tin). |
| `Exp2_G_Final_resnet18_adamw` | 23/11/2026 | resnet18 | 128×128 | adamw lr=0.001 (thân ×0.1) | cosine; wd=0.01; ls=0.1; aug=có | 11.180.103 | 35 / 50 | 95.42% | +25.18% | **70.24** | 0.9865 | 71.16 | 71.29 | 1079 | Chạy cuối 50 epoch (AdamW + Cosine + LS 0.10): Val 70.24%, Test 71.16%, Macro F1 71.29%, Disgust Recall 78.18%; kém A4 0.31 Val. |

### 2.2. Best Model bàn giao cho Thành viên 3
- **Mô hình được chọn (theo Val Acc)**: **`Exp2_A4_ResNet34_TL_128`** — ResNet34 pretrained ImageNet (`128×128×3`, 21.288.263 tham số, Adam `lr=1e-3`, LR thân ×0.1, `wd=1e-4`, `ReduceLROnPlateau`, Weighted CE `sqrt_balanced`, Best Epoch 27).
- **Hiệu năng**: `PublicTest (Val) = 70.55%` (Macro F1 `69.85%`) | `PrivateTest (Test) = 71.19%` (Macro F1 `70.83%`) | `Disgust Recall = 72.73%`.
- Thông tin cấu hình lưu tại `logs/best_model_info.json` và file trọng số lưu trực tiếp tại `checkpoints/best_model.pth` (81.34 MB, kèm hướng dẫn nạp tại `checkpoints/README.txt`).

---

## 3. ĐÁNH GIÁ CHUYÊN SÂU & TRỰC QUAN HÓA FINAL CONVOLUTION FILTERS
- **Phụ trách**: Thành viên 3 — **Hà Trần Gia Nguyên** (`3123411199`)
- **Mã nguồn**: `src/evaluate.py`, `src/visualize.py`
- **Nguồn số liệu thật**: `logs/tv3_eval_summary.csv`, `logs/classification_report_public.txt`, `logs/classification_report_private.txt`.

| Hạng mục | Mô hình áp dụng | Sản phẩm đầu ra (`figures/` & `logs/`) | Kết luận chính |
| :--- | :--- | :--- | :--- |
| Đánh giá độc lập Public & Private Test | `Exp2_A4_ResNet34_TL_128` | `logs/tv3_eval_summary.csv`, `logs/classification_report_{public,private}.txt` | Khớp 100% log TV2: **PublicTest 70.55%** (Macro F1 69.85%), **PrivateTest 71.19%** (Macro F1 70.83%). Vượt mốc 69% Public và đạt mốc 71% Private; chênh lệch hai tập chỉ 0.64% (*no selection bias*). |
| Confusion Matrix (Public & Private) | `Exp2_A4_ResNet34_TL_128` | `figures/confusion_matrix_public.png`, `figures/confusion_matrix_private.png` | **Dễ nhất**: *Happy* (F1 90.12%, Recall 91.81%) & *Surprise* (F1 82.71%). **Lớp thiểu số *Disgust***: F1 **79.21%** (Precision 86.96%, Recall 72.73%) khẳng định hiệu quả Weighted CE. **Khó nhất**: *Fear* (Recall 50.38%) & *Sad* (F1 55.58%). Top 5 cặp nhầm lẫn đều chứa lớp *Sad* (Sad→Neutral: 112, Fear→Sad: 103, Neutral→Sad: 78, Angry→Sad: 77, Sad→Fear: 64). |
| Lưới 4×8 True Cases & Wrong Cases | `Exp2_A4_ResNet34_TL_128` | `figures/cases_true_{public,private}.png`, `figures/cases_wrong_{public,private}.png` | Ca sai chủ yếu do nhãn gốc FER2013 mơ hồ/nhiễu, biểu cảm tinh tế gần nhau, ảnh `48×48` mờ, có watermark đè lên mặt, tay che mặt, góc nghiêng hoặc ảnh dạng tranh vẽ. |
| Final Convolution Filters & Feature Maps *(yêu cầu trọng tâm)* | `Exp2_A4_ResNet34_TL_128` | `figures/filters_first_conv.png`, `figures/filters_last_conv.png`, `figures/feature_maps_first_conv.png`, `figures/feature_maps_last_conv.png` | Conv đầu (`conv1`, 7×7 RGB) học bộ lọc cạnh/màu rõ ràng và feature map `64×64` còn thấy rõ mặt; Final Conv (`layer4`, 3×3, 512 kênh) mang đặc trưng trừu tượng hóa, feature map `4×4` không còn hình dạng mặt trực quan nên cần Grad-CAM để giải thích. |
| Grad-CAM Heatmap (7 lớp × 3 tầng & Ca sai) | `Exp2_A4_ResNet34_TL_128` | `figures/gradcam_7classes_3layers.png`, `figures/gradcam_wrong_cases.png` | Tầng đầu bám cạnh cục bộ → tầng giữa tập trung bộ phận (mắt, chân mày, mũi, má) → tầng cuối hội tụ vùng mũi–miệng trên khuôn mặt (không nhìn nhầm nền/tóc). *Happy* & *Surprise* nhấn rõ vùng miệng; các ca sai vẫn nhìn đúng vùng mặt nên nguyên nhân chính đến từ nhiễu dữ liệu đầu vào. |
| Tổng hợp sản phẩm nộp cuối cùng | Cả 3 thành viên | `report/BaoCao_DoAn_EmotionRecognition_Nhom666.{docx,pdf}`, `report/Slide_DoAn_EmotionRecognition_Nhom666.{pptx,pdf}` | Hoàn tất báo cáo 41 trang (25 hình nhúng, đủ 5 phần + phân công) và bộ 20 slide thuyết trình đồng bộ số liệu toàn nhóm. |
