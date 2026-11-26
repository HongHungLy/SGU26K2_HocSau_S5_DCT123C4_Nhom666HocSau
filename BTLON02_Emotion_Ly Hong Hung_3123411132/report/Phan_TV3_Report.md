# BẢN THẢO BÁO CÁO — PHẦN THÀNH VIÊN 3 (VAI TRÒ C)
**Người phụ trách**: *Hà Trần Gia Nguyên* — MSSV: `3123411199`  
**Phạm vi**: Phần 4 (Đánh giá chuyên sâu trên Public/Private Test, Trực quan hóa Final Convolution Filters, Feature Maps, Grad-CAM, Phân tích & Kết luận)

---

## PHẦN 4: ĐÁNH GIÁ CHUYÊN SÂU VÀ TRỰC QUAN HÓA MÔ HÌNH TỐT NHẤT

### 4.1. Mô hình được đánh giá
Mô hình tốt nhất của dự án là **`Exp2_A4_ResNet34_TL_128`** (Thành viên 2), được chọn **theo Val Acc (PublicTest)** và bàn giao tại `checkpoints/best_model.pth`:
- Kiến trúc: **ResNet34 pretrained ImageNet**, fine-tune toàn phần, ảnh đầu vào `128×128×3`, 21.288.263 tham số.
- Huấn luyện: Adam (`lr=1e-3`, LR thân ×0.1), `weight_decay=1e-4`, `ReduceLROnPlateau`, Weighted CE `sqrt_balanced`, Data Augmentation; Best Epoch 27 (dừng sớm ở epoch 37).
- Đánh giá lại độc lập bằng `src/evaluate.py` trên cả **PublicTest (3.589 ảnh)** và **PrivateTest (3.589 ảnh)**, kết quả **trùng khớp** với số liệu TV2 ghi trong `logs/exp2_summary.csv` (Val 70.55%, Test 71.19%).

### 4.2. Kết quả định lượng

**Bảng 4.1. Kết quả tổng hợp của mô hình tốt nhất**

| Tập | Accuracy | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: |
| PublicTest (3.589 ảnh) | **70.55%** | 69.85% | 70.36% |
| PrivateTest (3.589 ảnh) | **71.19%** | 70.83% | 70.98% |

So với mốc tham chiếu của đề bài (~**69%** Public, ~**71%** Private), mô hình vượt mốc Public (+1.55%) và đạt mốc Private (+0.19%).  
Chênh lệch giữa hai tập chỉ **0.64%** (Private cao hơn Public), cho thấy việc chọn checkpoint theo Val Acc không gây thiên lệch (*no selection bias*): kết quả trên tập Private độc lập vẫn giữ nguyên.

**Bảng 4.2. Classification Report theo từng lớp trên PrivateTest**

| Lớp | Precision | Recall | F1-score | Số mẫu |
| :--- | :---: | :---: | :---: | :---: |
| Angry | 0.6358 | 0.6436 | 0.6397 | 491 |
| Disgust | 0.8696 | 0.7273 | 0.7921 | 55 |
| Fear | 0.6087 | **0.5038** | 0.5513 | 528 |
| Happy | 0.8849 | **0.9181** | **0.9012** | 879 |
| Sad | **0.5463** | 0.5657 | 0.5558 | 594 |
| Surprise | 0.8321 | 0.8221 | 0.8271 | 416 |
| Neutral | 0.6677 | 0.7157 | 0.6908 | 626 |
| **Macro avg** | 0.7207 | 0.6995 | 0.7083 | 3.589 |

**Hình 4.1**: Confusion Matrix chuẩn hóa theo hàng (%) — `figures/confusion_matrix_public.png`, `figures/confusion_matrix_private.png`.

**Nhận xét theo lớp (PrivateTest):**
1. **Lớp dễ nhất — *Happy* (F1 = 0.90) và *Surprise* (F1 = 0.83)**: biểu cảm có đặc trưng hình học rõ ràng (khóe miệng nhếch lên, miệng/mắt mở to) nên mô hình nhận diện ổn định. *Happy* còn là lớp đa số (25.13% tập train).
2. **Lớp khó nhất — *Fear* (Recall = 50.38%) và *Sad* (F1 = 0.556)**: gần một nửa ảnh *Fear* bị dự đoán sang lớp khác; *Sad* có Precision thấp nhất (54.63%), tức mô hình hay "đoán *Sad*" cho các ảnh thuộc lớp khác.
3. **Lớp thiểu số — *Disgust***: chỉ có 55 ảnh test (1.52% tập train) nhưng đạt Recall **72.73%** và Precision **86.96%**, F1 = 0.79 — cao hơn nhiều lớp đa số. Điều này khẳng định hiệu quả của **Weighted CrossEntropy (`sqrt_balanced`)** do TV1 đề xuất ở `Exp1_6` (Disgust Recall từ 56.36% lên 74.55%) và được TV2 kế thừa. Lưu ý Recall *Disgust* tính trên chỉ 55 mẫu nên mỗi mẫu đúng/sai làm lệch ~1.8%.

**Bảng 4.3. Các cặp nhầm lẫn nhiều nhất (Thật → Dự đoán)**

| Hạng | PublicTest | PrivateTest | Tỉ lệ so với số mẫu lớp thật (Private) |
| :---: | :--- | :--- | :---: |
| 1 | Sad → Neutral (101) | Sad → Neutral (**112**) | 18.9% |
| 2 | Fear → Sad (91) | Fear → Sad (**103**) | 19.5% |
| 3 | Neutral → Sad (79) | Neutral → Sad (78) | 12.5% |
| 4 | Sad → Fear (65) | Angry → Sad (77) | 15.7% |
| 5 | Angry → Sad (60) | Sad → Fear (64) | 10.8% |

Kết quả ổn định giữa hai tập: cả 5 cặp đều xuất hiện ở cả Public và Private, và **lớp *Sad* nằm trong cả 5 cặp** (đóng vai trò vừa là nguồn vừa là đích nhầm lẫn). Đây là điểm yếu cốt lõi của mô hình.

**Giải thích nguyên nhân (nhận định dựa trên đặc điểm dữ liệu):**
- Ảnh xám `48×48` làm mờ các chi tiết nhỏ (nếp nhăn chân mày, khóe mắt) vốn phân biệt *Sad*, *Fear*, *Neutral* và *Angry*; các biểu cảm này đều có cơ mặt căng ít, miệng đóng và chân mày thay đổi nhẹ.
- *Sad ↔ Neutral* và *Fear ↔ Sad* là các cặp biểu cảm tinh tế, ngay cả người gán nhãn cũng dễ bất đồng; FER2013 vốn có nhãn nhiễu.
- Mô hình có hiện tượng overfit trên train (Train Acc 93.18% vs Val 70.55%, Gap +22.63%) và Val Loss thấp nhất ở epoch 5 rồi tăng, cho thấy mô hình đã bão hòa khả năng tổng quát hóa với dữ liệu hiện có, thay vì thiếu dung lượng.

**Hình 4.2**: Lưới 4×8 các ca dự đoán **đúng** (True Cases) và **sai** (Wrong Cases, nhãn dạng `Dự đoán (Thực tế)`) — `figures/cases_true_private.png`, `figures/cases_wrong_private.png` (bản Public: `cases_true_public.png`, `cases_wrong_public.png`).

**Nhận xét ca sai (quan sát lưới 32 ảnh sai ngẫu nhiên của PrivateTest):**
- **Nhãn gốc mơ hồ hoặc đáng ngờ**: nhiều ảnh mà dự đoán của mô hình hợp lý hơn nhãn thật. Ví dụ ảnh người cười tươi, miệng mở có nhãn *Surprise* nhưng mô hình dự đoán *Happy*; ảnh người há miệng lớn có nhãn *Happy* nhưng mô hình dự đoán *Fear*; ảnh em bé đang cười có nhãn *Happy* nhưng mô hình dự đoán *Neutral*. FER2013 vốn được gán nhãn bán tự động nên chứa nhãn nhiễu, đây là một phần của "trần" hiệu năng.
- **Biểu cảm tinh tế, gần nhau**: các ca *Sad ↔ Neutral/Fear*, *Angry ↔ Fear/Sad* chủ yếu là mặt cau mày nhẹ, miệng mím hoặc mắt nhắm; khác biệt giữa các lớp rất nhỏ, đúng với các cặp nhầm lẫn ở Bảng 4.3.
- **Watermark chèn lên mặt**: một số ảnh có chữ mờ đè lên vùng mắt/mũi/má (kiểu logo ảnh stock), làm nhiễu đặc trưng.
- **Che khuất và tư thế**: tay che miệng/má, đầu cúi hoặc nghiêng, mặt nhìn nghiêng.
- **Ảnh không phải ảnh chụp thật**: có vài ảnh có vẻ là tranh vẽ/đồ họa, khác miền dữ liệu so với phần lớn ảnh khuôn mặt thật.

### 4.3. So sánh với các mô hình khác của nhóm

| Mô hình | Val Acc | Test Acc | Macro F1 | Disgust Recall |
| :--- | :---: | :---: | :---: | :---: |
| `Exp1_6` MediumCNN (TV1, baseline tốt nhất) | 67.09% | 68.51% | 65.63% | 74.55% |
| `Exp2_A2b` ResNetMini w32 (TV2, mạng tự thiết kế tốt nhất) | 66.87% | 68.21% | 65.63% | — |
| **`Exp2_A4` ResNet34 TL (TV2, Best Model)** | **70.55%** | **71.19%** | **70.83%** | 72.73% |

Transfer Learning với ResNet34 cải thiện so với baseline tốt nhất của TV1: **Val +3.46%, Test +2.68%, Macro F1 +5.20%**. Lợi ích đến từ đặc trưng thị giác học sẵn từ ImageNet và độ phân giải đầu vào lớn hơn (`128×128` thay vì `48×48`). Đánh đổi: Disgust Recall giảm nhẹ (−1.82%, tương đương 1 ảnh trên 55 mẫu) và mô hình lớn hơn ~3.2 lần (21.3M so với 6.6M tham số).

### 4.4. Trực quan hóa Final Convolution Filters
Phương pháp (`src/visualize.py`): lấy trọng số lớp `nn.Conv2d` cuối cùng của mô hình (`model.last_conv_layer`, dạng `[512, 512, 3, 3]` với ResNet34), với mỗi filter lấy **trung bình theo 512 kênh đầu vào** để thu được kernel `3×3`, chuẩn hóa min–max và hiển thị 64/512 filter đầu tiên; đối chiếu với lớp Conv đầu `backbone.conv1` (`[64, 3, 7, 7]`, hiển thị dạng màu RGB).

**Hình 4.3**: `figures/filters_last_conv.png` (**Final Convolution Filters**) và `figures/filters_first_conv.png`.

**Nhận xét:**
- Các filter của lớp Conv cuối (kernel `3×3`) **không có cấu trúc trực quan rõ ràng**: không thấy các mẫu bộ lọc cạnh/hướng như ở tầng nông; giá trị trọng số phân bố khá tản mạn giữa các ô.
- Điều này là **bình thường với mạng sâu**: kernel `3×3` ở tầng sâu tác động lên feature map đã được trừu tượng hóa qua hàng chục lớp (đầu vào 512 kênh), và việc lấy trung bình theo kênh đầu vào càng làm mất thông tin. Vì vậy trọng số riêng lẻ khó diễn giải, và để hiểu mô hình "nhìn vào đâu" cần dùng Feature Maps và Grad-CAM (các mục 4.5, 4.6).
- **Lớp Conv đầu** (`backbone.conv1`, `[64, 3, 7, 7]`, hiển thị dạng màu RGB) có filter **cấu trúc rõ ràng**: nhiều bộ lọc **cạnh có hướng** (sọc dọc, sọc chéo) và **bộ lọc màu / vệt sáng–tối** (cam–xanh, xanh lá, tím), đúng đặc trưng lớp đầu của mạng pretrained ImageNet. Một vài filter hiển thị đen do trọng số gần như đồng nhất (gần bằng 0). Tương phản với lớp Conv cuối: tầng nông học đặc trưng cạnh/màu dễ diễn giải, còn tầng sâu học đặc trưng trừu tượng khó diễn giải trực tiếp.

### 4.5. Trực quan hóa Feature Maps
Phương pháp: đăng ký `forward hook` tại lớp Conv đầu và lớp Conv cuối, cho một ảnh test được dự đoán đúng đi qua mạng và hiển thị 32 kênh đầu tiên.

**Hình 4.4**: `figures/feature_maps_first_conv.png`, `figures/feature_maps_last_conv.png`.

**Nhận xét:**
- **Lớp Conv cuối** có đầu ra kích thước `(512, 4, 4)` (ảnh `128×128` bị giảm 32 lần qua các tầng stride), nên mỗi feature map chỉ là lưới `4×4` ô: **không còn nhận ra hình dạng khuôn mặt**, đặc trưng đã được mã hóa thành các mẫu kích hoạt trừu tượng.
- Mỗi kênh kích hoạt mạnh ở **những ô khác nhau** (ô giữa, ô dưới, ô góc...), cho thấy các kênh chuyên biệt hóa theo các vùng/đặc trưng khác nhau của khuôn mặt, và đầu ra này được gộp (global average pooling) rồi đưa vào lớp phân loại.
- Độ phân giải không gian rất thấp (`4×4`) cũng giải thích vì sao bản đồ Grad-CAM ở tầng cuối (mục 4.6) trông thô, dạng mảng lớn.
- **Lớp Conv đầu** có đầu ra `(64, 64, 64)` (ảnh `128×128` giảm 2 lần): feature map **vẫn nhận ra rõ khuôn mặt**. Các kênh khác nhau làm nổi bật các thành phần khác nhau: một số giữ độ sáng tổng thể, một số chỉ giữ **cạnh theo hướng** (viền mặt, gọng kính, sống mũi, nếp nhăn quanh miệng), một số bắt **kết cấu mịn** (kể cả hoa văn watermark trên kính/trán), và vài kênh gần như không kích hoạt. Kết quả khớp với cột "Đầu" của Hình 4.5: kích hoạt tầng đầu bám theo **cạnh và chi tiết cục bộ**, chưa mang ý nghĩa cảm xúc.

### 4.6. Trực quan hóa Grad-CAM
Phương pháp: Grad-CAM tính trọng số kênh là trung bình gradient của điểm số lớp dự đoán đối với feature map, $\alpha_k = \frac{1}{HW}\sum_{i,j}\frac{\partial y^c}{\partial A^k_{ij}}$, rồi $L = \mathrm{ReLU}\left(\sum_k \alpha_k A^k\right)$, nội suy về kích thước ảnh và phủ lên ảnh gốc (colormap `jet`: **đỏ = vùng ảnh hưởng nhiều nhất** tới quyết định). Thực hiện cho **3 tầng tích chập** — đầu (`backbone.conv1`), giữa (`backbone.layer3.0.conv2`), cuối (`layer4`, `gradcam_layer`) — trên **7 lớp cảm xúc** (mỗi lớp lấy 1 ảnh được dự đoán đúng), và riêng một lưới các ca dự đoán sai.

**Hình 4.5**: `figures/gradcam_7classes_3layers.png`. **Hình 4.6**: `figures/gradcam_wrong_cases.png`.

**Nhận xét theo tầng:**
- **Tầng đầu (`conv1`)**: bản đồ kích hoạt bám theo **cạnh và chi tiết cục bộ** — viền khuôn mặt, gọng kính, mép mắt, nếp nhăn, viền môi (rõ nhất ở ảnh *Disgust*, *Happy*, *Surprise*). Đặc trưng ở mức thấp, chưa mang ý nghĩa cảm xúc.
- **Tầng giữa (`layer3.0.conv2`)**: xuất hiện các **điểm nóng tại từng bộ phận** như mắt, vùng trán/chân mày, mũi, má; vị trí khá phân tán và khác nhau giữa các ảnh (ví dụ *Surprise* nhấn mạnh dải trán–chân mày, *Disgust* và *Sad* nhấn mạnh vùng quanh mắt/má). Đây là đặc trưng bộ phận khuôn mặt.
- **Tầng cuối (`layer4`)**: bản đồ tập trung thành **mảng lớn, mượt ở vùng trung tâm khuôn mặt, quanh mũi–miệng**, đúng với việc tầng này chỉ có độ phân giải `4×4`. Mô hình dùng thông tin tổng hợp từ các tầng trước, nên vùng nhấn mạnh nhất không phải một chi tiết nhỏ.

**Nhận xét theo lớp cảm xúc (tầng cuối):**
- ***Happy*** và ***Surprise***: vùng đỏ nằm ở **miệng và phần mặt dưới** (nụ cười / miệng mở) — phù hợp với đặc trưng nhận diện của hai lớp này và giải thích vì sao đây là hai lớp có F1 cao nhất.
- ***Angry, Disgust, Fear, Sad, Neutral***: vùng đỏ tập trung ở **giữa mặt (mũi, môi trên, má)**, ít khác biệt giữa các lớp; nhấn mạnh vùng mắt/chân mày chủ yếu thể hiện ở tầng giữa chứ không rõ ở tầng cuối. Sự thiếu phân biệt về vùng chú ý giữa các biểu cảm "khó" này phù hợp với kết quả định lượng: các lớp này hay bị nhầm lẫn với nhau (mục 4.2).
- Ở tất cả 7 ảnh, mô hình **chú ý vào vùng mặt chứ không phải nền, tóc hay quần áo**, cho thấy quyết định dựa trên khuôn mặt chứ không phải tín hiệu nhiễu rõ rệt trên các ảnh này.

**Hạn chế của phân tích định tính:** mỗi lớp chỉ minh họa 1 ảnh được dự đoán đúng nên đây là quan sát minh họa, chưa đại diện thống kê cho toàn bộ tập test; Grad-CAM ở tầng cuối thô do độ phân giải `4×4`.

**Nhận xét ca dự đoán sai (Hình 4.6):** trong 8 ca sai minh họa, bản đồ Grad-CAM tầng cuối **vẫn tập trung vào vùng giữa–dưới khuôn mặt (mũi, miệng)**, giống như ở các ca đúng; mô hình **không chú ý nhầm sang nền hay tóc**. Nguyên nhân sai nằm ở **chính ảnh đầu vào**: mặt nghiêng/nhìn ngang (dự đoán Sad, thực tế Fear), tay che mặt (dự đoán Fear, thực tế Sad), kính che mắt, watermark đè lên mặt, và ảnh dạng **tranh vẽ/hoạt họa** (dự đoán Disgust, thực tế Surprise) khác miền với ảnh chụp thật. Vì vùng chú ý ở ca đúng và ca sai gần như giống nhau, Grad-CAM tầng cuối (độ phân giải `4×4`) **không đủ để phân biệt** mô hình đúng hay sai; cần kết hợp với phân tích nhãn và ảnh đầu vào (mục 4.2).

### 4.7. Kết luận & Hướng phát triển
1. **Kết quả đạt được**: mô hình ResNet34 Transfer Learning đạt **70.55% (Public) / 71.19% (Private)**, đạt và vượt mốc tham chiếu ~69%/71% của đề bài, cải thiện rõ so với baseline CNN tự thiết kế (68.51%).
2. **Điểm mạnh**: nhận diện rất tốt *Happy* và *Surprise*; xử lý được lớp thiểu số *Disgust* (Recall 72.73%) nhờ Weighted CE.
3. **Điểm yếu**: nhầm lẫn tập trung quanh lớp *Sad* (*Sad↔Neutral*, *Fear↔Sad*, *Angry→Sad*); Recall *Fear* chỉ 50.38%.
4. **Hạn chế**: overfit đáng kể (Gap +22.6%); FER2013 chứa nhãn nhiễu và ảnh độ phân giải thấp nên trần hiệu năng bị giới hạn; mô hình chọn theo một seed duy nhất (chênh 0.31% giữa A4 và G nằm trong ngưỡng nhiễu).
5. **Hướng phát triển**: tăng cường Augmentation mạnh hơn (MixUp/CutMix), Label Smoothing kết hợp Weighted CE, ensemble nhiều mô hình/nhiều seed (ResNet34 + ResNet18 của nhóm G), làm sạch nhãn nhiễu, hoặc dùng tập dữ liệu lớn/sạch hơn (FER+, AffectNet) để huấn luyện trước.

---

## BRIEF SUMMARY FOR SOURCE CODES (phần TV3)
| File | Chức năng |
| :--- | :--- |
| `src/evaluate.py` | Nạp `best_model.pth` (hoặc checkpoint baseline), dự đoán trên Public/Private Test, xuất Classification Report, Confusion Matrix (%), lưới ca đúng/sai, `logs/tv3_eval_summary.csv` |
| `src/visualize.py` | Vẽ Final Conv Filters, Feature Maps (forward hook), Grad-CAM 3 tầng × 7 lớp và ca sai |
| Lệnh chạy | `python -m src.evaluate --ckpt checkpoints/best_model.pth` và `python -m src.visualize --ckpt checkpoints/best_model.pth` (chạy từ thư mục gốc project) |
