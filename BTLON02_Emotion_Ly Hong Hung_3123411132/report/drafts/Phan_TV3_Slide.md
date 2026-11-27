# NỘI DUNG SLIDE THUYẾT TRÌNH — PHẦN THÀNH VIÊN 3 (VAI TRÒ C)
**Phụ trách**: *Hà Trần Gia Nguyên* — MSSV: `3123411199` (~6 slide cuối của bài thuyết trình nhóm; đánh số lại khi ghép với TV1, TV2)

---

### Slide 1: Mô hình tốt nhất & Kết quả đánh giá độc lập
- **Mô hình đánh giá**: `Exp2_A4_ResNet34_TL_128` (TV2) — ResNet34 pretrained ImageNet, ảnh `128×128×3`, 21.3M tham số, chọn theo **Val Acc**.
- **Kết quả đánh giá lại bằng `src/evaluate.py`** (trùng khớp log của TV2):

| Tập | Accuracy | Macro F1 |
| :--- | :---: | :---: |
| PublicTest (3.589) | **70.55%** | 69.85% |
| PrivateTest (3.589) | **71.19%** | 70.83% |

- **So với benchmark** ~69% / ~71%: vượt Public (+1.55%), đạt Private.
- **So với baseline tốt nhất của TV1 (`Exp1_6`)**: Val **+3.46%**, Test **+2.68%**, Macro F1 **+5.20%**.
- Private cao hơn Public 0.64% → chọn checkpoint theo Val **không gây thiên lệch**.

---

### Slide 2: Phân tích theo từng lớp cảm xúc (PrivateTest)
- **Hình minh họa chính**: `figures/confusion_matrix_private.png` (chuẩn hóa % theo hàng).
- **Kết quả theo lớp (`logs/classification_report_private.txt`)**:

| Lớp | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: |
| Happy | 0.885 | **0.918** | **0.901** |
| Surprise | 0.832 | 0.822 | 0.827 |
| Disgust | 0.870 | 0.727 | 0.792 |
| Neutral | 0.668 | 0.716 | 0.691 |
| Angry | 0.636 | 0.644 | 0.640 |
| Sad | **0.546** | 0.566 | 0.556 |
| Fear | 0.609 | **0.504** | 0.551 |

- **Dễ nhất**: *Happy*, *Surprise*. **Khó nhất**: *Fear*, *Sad*.
- ***Disgust*** chỉ 55 ảnh test vẫn đạt Recall **72.73%** → Weighted CE của TV1 hiệu quả.

---

### Slide 3: Các cặp nhầm lẫn chính & Ca dự đoán sai
- **Hình minh họa chính**: `figures/cases_wrong_private.png` (+ `cases_true_private.png`).
- **Top nhầm lẫn (Thật → Đoán)**: Sad→Neutral (**112**), Fear→Sad (**103**), Neutral→Sad (78), Angry→Sad (77), Sad→Fear (64). Cùng 5 cặp ở cả Public và Private; lớp *Sad* nằm trong cả 5 cặp.
- **Nguyên nhân quan sát được ở ca sai**:
  - Nhãn gốc mơ hồ/đáng ngờ (FER2013 có nhãn nhiễu).
  - Biểu cảm tinh tế, gần nhau (cau mày nhẹ, miệng mím).
  - Watermark chèn lên mặt, tay che mặt, ảnh nghiêng, ảnh dạng tranh vẽ.
  - Ảnh `48×48` mất chi tiết ở mắt, chân mày.

---

### Slide 4: Final Convolution Filters & Feature Maps
- **Hình minh họa chính**: `figures/filters_last_conv.png` + `figures/feature_maps_last_conv.png`.
- **Conv đầu** (`filters_first_conv.png`, `64×3×7×7`): bộ lọc cạnh có hướng và bộ lọc màu rõ ràng; feature map Conv đầu (`64×64×64`) vẫn nhận ra khuôn mặt.
- **Final Conv Filters** (lớp Conv cuối, shape `512×512×3×3`, hiển thị 64 filter): kernel `3×3` **không có cấu trúc trực quan rõ ràng** — bình thường ở tầng sâu vì tác động lên đặc trưng đã trừu tượng hóa.
- **Feature Maps lớp Conv cuối** (output `512×4×4`): chỉ còn lưới `4×4`, **không còn nhận ra khuôn mặt**; các kênh kích hoạt ở những ô khác nhau.
- **Ý nghĩa**: trọng số tầng sâu khó diễn giải trực tiếp → dùng **Grad-CAM** để biết mô hình "nhìn vào đâu".

---

### Slide 5: Grad-CAM — Mô hình chú ý vào đâu?
- **Hình minh họa chính**: `figures/gradcam_7classes_3layers.png` (7 cảm xúc × 3 tầng: `conv1` → `layer3.0.conv2` → `layer4`).
- **Theo tầng**:
  - **Đầu**: bám theo cạnh (viền mặt, gọng kính, mép mắt, viền môi).
  - **Giữa**: điểm nóng ở từng bộ phận (mắt, trán/chân mày, mũi, má).
  - **Cuối**: mảng lớn ở vùng giữa mặt, quanh mũi–miệng (độ phân giải `4×4`).
- **Theo lớp**: *Happy*, *Surprise* nhấn rõ vào **miệng**; các lớp còn lại ít khác biệt → phù hợp với việc hay nhầm lẫn nhau.
- Mô hình chú ý vào **vùng mặt**, không phải nền/tóc (trên 7 ảnh minh họa).
- **Ca sai** (`figures/gradcam_wrong_cases.png`): vùng chú ý vẫn ở mũi–miệng như ca đúng → lỗi do **ảnh đầu vào** (mặt nghiêng, tay che mặt, watermark, ảnh tranh vẽ), không phải do nhìn nhầm vùng.

---

### Slide 6: Kết luận & Hướng phát triển
- **Kết quả**: ResNet34 Transfer Learning đạt **70.55% (Public) / 71.19% (Private)**, đạt và vượt mốc tham chiếu 69%/71%, tốt hơn baseline CNN tự thiết kế (68.51%).
- **Điểm mạnh**: *Happy*, *Surprise* rất tốt; xử lý được lớp thiểu số *Disgust*.
- **Điểm yếu**: nhầm lẫn quanh lớp *Sad*; Recall *Fear* chỉ 50.38%.
- **Hạn chế**: overfit (Gap +22.6%), nhãn nhiễu, ảnh `48×48`; kết quả từ một seed.
- **Hướng phát triển**: MixUp/CutMix, ensemble nhiều mô hình/seed (ResNet34 + ResNet18), làm sạch nhãn, dùng dữ liệu sạch/lớn hơn (FER+, AffectNet).
- **Mã nguồn TV3**: `src/evaluate.py`, `src/visualize.py` — chạy `python -m src.evaluate` và `python -m src.visualize`.
