# BÀI TẬP LỚN 01 (BTL01): XẤP XỈ HÀM MẬT ĐỘ GAUSSIAN 2 CHIỀU BẰNG MẠNG MLP (PYTORCH)

> **Trường Đại học Sài Gòn (SGU) – Khoa Công nghệ Thông tin**  
> **Học phần**: Học sâu (Deep Learning) – Học kỳ 1, Năm học 2026–2027  
> **Giảng viên hướng dẫn**: TS. Đỗ Như Tài  
> **Lớp**: DCT123C4  
> **Nhóm thực hiện**: Nhóm 666 (Tỷ lệ đóng góp: 100% – 100% – 100%)  
> **Công nghệ triển khai**: 100% thuần **PyTorch** (`torch.nn.Module`, `DataLoader`, `MSELoss`, `Adam/SGD`). Tuyệt đối không dùng Keras / TensorFlow.

---

## 👥 THÀNH VIÊN NHÓM VÀ PHÂN CÔNG NHIỆM VỤ

| STT | Họ và tên | MSSV | Vai trò chính | Nhiệm vụ chính đảm nhiệm | Đóng góp |
| :---: | :--- | :---: | :--- | :--- | :---: |
| 1 | **Lý Hồng Hưng** *(Trưởng nhóm)* | **3123411132** | Data & Model Architect | Thiết kế `dataset.py`, xây dựng kiến trúc `FlexibleMLP` (`model.py`), chủ trì **Thực nghiệm A** (Quy mô dữ liệu), xây dựng Jupyter Notebook demo. | **100%** |
| 2 | **Hà Trần Gia Nguyên** | **3123411199** | Training Engine & Optimization | Xây dựng động cơ huấn luyện `trainer.py` (EarlyStopping, checkpoints), chủ trì **Thực nghiệm B** (Độ rộng mạng), phân tích Overfitting khi thiếu mẫu ($N=200$). | **100%** |
| 3 | **Nguyễn Bá Khải** | **3123411138** | Evaluation & In-depth Analysis | Xây dựng module đánh giá `evaluate.py`, chủ trì **Thực nghiệm C** (Độ sâu mạng), nghiên cứu hiện tượng Triệt tiêu đạo hàm (Vanishing Gradient), so sánh đối đầu ABC. | **100%** |

---

## 1. GIỚI THIỆU BÀI TOÁN

Xấp xỉ hàm mật độ xác suất Gaussian hai chiều liên tục:
$$f(\mathbf{x}) = \frac{1}{2\pi \sqrt{\det(C)}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T C^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

* **Vector kỳ vọng**: $\boldsymbol{\mu} = [0, 0]^T$
* **Ma trận hiệp phương sai**: $C = \begin{bmatrix} 0.3 & -0.5 \\ -0.5 & 2.0 \end{bmatrix}$ với $\det(C) = 0.35 > 0$
* **Giá trị cực đại lý thuyết**: $f(0, 0) = \frac{1}{2\pi \sqrt{0.35}} \approx 0.269002$
* **Miền xác định dữ liệu**: $x_1 \in [-2.0, 2.0]$, $x_2 \in [-4.0, 4.0]$
* **Mục tiêu mô hình**: Mạng MLP nhận đầu vào 2 chiều $(x_1, x_2)$ và dự đoán 1 giá trị đầu ra liên tục $\hat{y} \approx y = f(x_1, x_2)$.

---

## 2. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
BTL01_Nhom666_Ly Hong Hung_3123411132/
├── report/                                  # Toàn bộ tài liệu báo cáo và slide nộp bài
│   ├── Bao_Cao_BTL01_Nhom666.docx           # Báo cáo kỹ thuật Word chuẩn (đầy đủ công thức, hình 4, footer)
│   ├── Bao_Cao_BTL01_Nhom666.pdf            # Bản xuất bản PDF của Báo cáo
│   ├── BTL01_Nhom666_Slide.pptx             # Slide thuyết trình 14 trang tỉ lệ 16:9 chuẩn khoa học
│   └── BTL01_Nhom666_Slide.pdf              # Bản xuất bản PDF của Slide thuyết trình
├── data/                                    # Dữ liệu CSV phân tách độc lập (No Data Leakage)
│   ├── train_pool_2000.csv                  # Kho dữ liệu 2000 mẫu train (seed = 342)
│   ├── val.csv                              # 400 mẫu validation chọn checkpoint & early stop (seed = 242)
│   ├── test.csv                             # 100 mẫu test độc lập cô lập để chấm điểm (seed = 142)
│   └── data_statistics.csv                  # Thống kê mô tả dữ liệu huấn luyện
├── checkpoints/                             # Lưu vết trọng số mô hình tốt nhất (.pt)
│   ├── expA_size_*.pt                       # Checkpoints các mô hình Thực nghiệm A
│   └── expC_layers_*.pt                     # Checkpoints các mô hình Thực nghiệm C
├── logs/                                    # Nhật ký huấn luyện và kết quả thực nghiệm chi tiết (.csv)
│   ├── exp_A_runs_detailed.csv              # Kết quả chi tiết từng lần chạy Thực nghiệm A
│   ├── exp_A_summary.csv                    # Bảng tổng hợp Mean ± Std Thực nghiệm A
│   ├── exp_B_runs_detailed.csv              # Kết quả chi tiết Thực nghiệm B (train = 1000 mẫu)
│   ├── exp_B_summary.csv                    # Bảng tổng hợp Thực nghiệm B
│   ├── exp_B_runs_detailed_n200.csv         # Kết quả chi tiết khảo sát Overfitting (train = 200 mẫu)
│   ├── exp_B_summary_n200.csv               # Bảng tổng hợp khảo sát Overfitting
│   ├── exp_B_epochs/                        # Log chi tiết train/val loss qua từng epoch của Thực nghiệm B
│   ├── exp_C_summary.csv                    # Bảng tổng hợp Thực nghiệm C (Độ sâu 1-5 tầng)
│   ├── exp_C_gradient_norms.csv             # Dữ liệu chuẩn gradient từng tầng lúc khởi tạo
│   ├── exp_C_gradient_summary.csv           # Bảng tỉ số chuẩn gradient Sigmoid, Tanh, ReLU
│   ├── exp_C_epochs/                        # Log chi tiết train/val loss qua từng epoch của Thực nghiệm C
│   ├── exp_ABC_best_configs.csv             # Bảng so sánh các cấu hình tiêu biểu nhất
│   └── exp_ABC_ranking_1000samples.csv      # Bảng xếp hạng hiệu năng trên 1000 mẫu train
├── figures/                                 # 15 hình ảnh, biểu đồ trực quan hóa độ nét cao (DPI 300)
│   ├── sgu_logo.png                         # Logo Đại học Sài Gòn
│   ├── mlp_architecture.png                 # Hình 4: Sơ đồ kiến trúc mạng nơ-ron FlexibleMLP
│   ├── true_gaussian_surface.png            # Bề mặt giải tích lý thuyết 3D
│   ├── data_3d_scatter.png                  # Scatter plot 3D dữ liệu huấn luyện (500 mẫu đại diện)
│   ├── data_distributions.png               # Phân phối tần suất của x1, x2 và nhãn y
│   ├── exp_A_error_vs_size.png              # Thực nghiệm A: Sai số MSE/MAE theo số mẫu huấn luyện A
│   ├── exp_A_best_loss_curve.png            # Đường cong hội tụ Loss Train vs Val (A = 2000, seed = 123)
│   ├── exp_A_best_pred_vs_true_3d.png       # Đánh giá độ bám khít trên Test Set 100 mẫu độc lập
│   ├── exp_A_surface_comparison.png         # So sánh trực quan bề mặt thực tế vs bề mặt xấp xỉ MLP
│   ├── exp_B_error_vs_nodes.png             # Thực nghiệm B: Sai số theo số nơ-ron tầng ẩn B
│   ├── exp_B_ratio_val_train.png            # Tỉ số Val MSE / Train MSE (N = 1000 mẫu)
│   ├── exp_B_ratio_val_train_n200.png       # Minh chứng Overfitting khi thiếu dữ liệu (N = 200 mẫu)
│   ├── exp_C_error_and_time_vs_depth.png    # Thực nghiệm C: Đánh giá Sai số và Thời gian theo độ sâu C
│   ├── exp_C_gradient_norms.png             # Phân tích Triệt tiêu đạo hàm: Sigmoid vs Tanh vs ReLU
│   └── exp_ABC_comparison.png               # Đồ thị tổng hợp đối chiếu 4 chiều Thực nghiệm A - B - C
├── src/                                     # Module mã nguồn Python chuẩn hướng đối tượng
│   ├── __init__.py
│   ├── dataset.py                           # Tính toán hàm Gaussian 2D, sinh và chia dữ liệu độc lập
│   ├── model.py                             # Kiến trúc FlexibleMLP tùy biến linh hoạt số tầng và nơ-ron
│   ├── trainer.py                           # Động cơ huấn luyện chuẩn PyTorch, EarlyStopping, Checkpoints
│   ├── evaluate.py                          # Tính toán các chỉ số kiểm thử: MSE, RMSE, MAE, R² Score
│   └── plot_utils.py                        # Trực quan hóa khoa học đồ thị 2D và bề mặt 3D
├── generate_data.py                         # Script sinh dữ liệu và vẽ phân phối dữ liệu ban đầu
├── exp_A_datasize.py                        # Script thực hiện Thực nghiệm A (Quy mô dữ liệu)
├── exp_B_width.py                           # Script thực hiện Thực nghiệm B (Độ rộng nơ-ron & Overfitting)
├── exp_C_depth.py                           # Script thực hiện Thực nghiệm C (Độ sâu số tầng ẩn)
├── exp_C_gradient_analysis.py               # Script phân tích hiện tượng Triệt tiêu gradient
├── make_comparison_ABC.py                   # Script tổng hợp so sánh đối đầu ABC và xếp hạng
├── BTL01_Thuc_Nghiem_A.ipynb                # Jupyter Notebook demo tương tác trực quan
├── requirements.txt                         # Danh sách thư viện phụ thuộc
└── README.md                                # Tài liệu hướng dẫn sử dụng toàn diện
```

---

## 3. CÀI ĐẶT MÔI TRƯỜNG

Dự án yêu cầu **Python >= 3.9** và **PyTorch >= 2.0**. Cài đặt các thư viện cần thiết bằng lệnh:

```bash
pip install -r requirements.txt
```

Các thư viện chính được sử dụng:
* `torch`, `torchvision`: Xây dựng mô hình MLP, bộ tối ưu hóa Adam, tính toán gradient tự động.
* `numpy`, `scipy`: Lấy mẫu phân bố đều, tính định thức và ma trận hiệp phương sai nghịch đảo.
* `pandas`: Xử lý dữ liệu bảng, xuất log thực nghiệm chi tiết.
* `matplotlib`: Vẽ biểu đồ khoa học 2D và biểu diễn không gian 3D.
* `tqdm`: Hiển thị thanh tiến trình huấn luyện.

---

## 4. HƯỚNG DẪN THỰC THI (CHẠY TOÀN BỘ DỰ ÁN)

### Bước 1: Khởi tạo Dữ liệu & Trực quan hóa ban đầu
Sinh 3 tập dữ liệu độc lập (Train Pool 2000 mẫu, Validation 400 mẫu, Test 100 mẫu) với random seed kiểm soát chặt chẽ nhằm chống rò rỉ dữ liệu:
```bash
python generate_data.py
```
*Kết quả:* Tạo ra các file CSV trong `data/` và 3 hình ảnh phân bố trong `figures/` (`true_gaussian_surface.png`, `data_3d_scatter.png`, `data_distributions.png`).

---

### Bước 2: Chạy Thực nghiệm A – Quy mô Dữ liệu Huấn luyện ($A \in [200, 2000]$)
Khảo sát tác động của số lượng mẫu train trên kiến trúc chuẩn (1 tầng ẩn × 128 nơ-ron) lặp lại qua 3 seeds (42, 123, 999):
```bash
python exp_A_datasize.py
```
*(Chạy kiểm tra nhanh code: `python exp_A_datasize.py --fast_test`)*  
*Kết quả:* Xuất bảng số liệu `logs/exp_A_summary.csv`, đồ thị sai số `figures/exp_A_error_vs_size.png`, đường cong hội tụ `figures/exp_A_best_loss_curve.png`, so sánh dự đoán trên test set `figures/exp_A_best_pred_vs_true_3d.png` và bề mặt 3D `figures/exp_A_surface_comparison.png`.

---

### Bước 3: Chạy Thực nghiệm B – Độ rộng tầng ẩn ($B \in [1, 64]$) & Khảo sát Overfitting
Khảo sát tác động của số lượng nơ-ron trên 1000 mẫu train, đồng thời thực hiện khảo sát Overfitting khi thiếu dữ liệu ($N = 200$ mẫu):
```bash
python exp_B_width.py
```
*Kết quả:* Xuất các file tổng hợp `logs/exp_B_summary.csv`, `logs/exp_B_summary_n200.csv`, đồ thị `figures/exp_B_error_vs_nodes.png`, `figures/exp_B_ratio_val_train_n200.png` và toàn bộ log epoch trong `logs/exp_B_epochs/`.

---

### Bước 4: Chạy Thực nghiệm C – Độ sâu số tầng ẩn ($C \in [1, 5]$)
Khảo sát tác động của số lượng tầng ẩn (mỗi tầng cố định 16 nơ-ron) trên 1000 mẫu train qua 3 seeds:
```bash
python exp_C_depth.py
```
*Kết quả:* Xuất `logs/exp_C_summary.csv`, đồ thị sai số & thời gian `figures/exp_C_error_and_time_vs_depth.png`, các file checkpoint trong `checkpoints/` và log epoch trong `logs/exp_C_epochs/`.

---

### Bước 5: Phân tích Triệt tiêu Đạo hàm (Vanishing Gradient Analysis)
Đo lường định lượng tỉ số chuẩn Frobenius gradient tầng đầu so với tầng ra giữa 3 hàm kích hoạt Sigmoid, Tanh và ReLU qua các độ sâu $C \in [1, 5]$:
```bash
python exp_C_gradient_analysis.py
```
*Kết quả:* Xuất bảng số liệu `logs/exp_C_gradient_summary.csv` và đồ thị `figures/exp_C_gradient_norms.png`.

---

### Bước 6: Tổng hợp Đối chiếu ABC & Xếp hạng Mô hình
Tổng hợp kết quả của cả 3 nhóm thực nghiệm A, B, C trên cùng điều kiện 1000 mẫu train để so sánh đối đầu và xếp hạng:
```bash
python make_comparison_ABC.py
```
*Kết quả:* Xuất bảng xếp hạng `logs/exp_ABC_ranking_1000samples.csv` và đồ thị đối chiếu tổng hợp 4 ô `figures/exp_ABC_comparison.png`.

---

### Bước 7: Chạy Demo trực quan trên Jupyter Notebook
Để trình diễn trực quan cho Giảng viên và Hội đồng chấm bài:
```bash
jupyter notebook BTL01_Thuc_Nghiem_A.ipynb
```
Notebook tích hợp đầy đủ quy trình từ sinh dữ liệu, nạp checkpoint đã huấn luyện, dự đoán và tương tác xoay đồ thị 3D trực tiếp.

---

## 5. TÓM TẮT KẾT QUẢ THỰC NGHIỆM CHÍNH

1. **Thực nghiệm A (Quy mô dữ liệu):**
   * Tăng mẫu từ 200 lên 2000 giúp Test MSE giảm mạnh **5.6 lần** (từ $0.002610 \rightarrow 0.000465$), $R^2$ tăng từ $0.2681 \rightarrow 0.8696$.
   * Sai số bão hòa dần quanh 1600 – 2000 mẫu do chạm trần dung lượng biểu diễn của mạng 1 tầng ẩn.
2. **Thực nghiệm B (Độ rộng tầng ẩn):**
   * $B = 1$ nơ-ron bị thiếu khớp nghiêm trọng (Underfitting, $\text{MSE} = 0.003573$).
   * $B = 32$ nơ-ron đạt điểm tối ưu ngọt (Sweet spot, $\text{MSE} = 0.000308 \pm 0.000066$).
   * Khảo sát $N = 200$ mẫu: Tỉ số $\text{Val MSE} / \text{Train MSE} > 1.40$ ở $B \ge 16$, minh chứng rõ hiện tượng quá khớp (Overfitting) khi mạng rộng học vẹt dữ liệu ít.
3. **Thực nghiệm C (Độ sâu mạng – Kỷ lục toàn đồ án):**
   * Tăng từ 1 lên 2 tầng tạo bước nhảy vọt: Test MSE giảm đột ngột hơn **31 lần** (từ $0.000437 \rightarrow 0.000014$).
   * **Cấu hình Vô địch (Top 1 toàn bài)**: $C = 5$ tầng $\times$ 16 nơ-ron đạt Test MSE kỷ lục:
     $$\mathbf{3.89 \times 10^{-6} \pm 0.000001} \quad (\text{sai số chỉ 4 phần triệu}), \quad R^2 = 0.9989$$
   * Thời gian huấn luyện 5 tầng ($5.78s$) hầu như không chênh lệch so với 1 tầng ($5.14s$), khẳng định ưu thế áp đảo của mạng sâu phân cấp so với việc chỉ mở rộng bề ngang.
4. **Phân tích Gradient:**
   * **Sigmoid**: Đạo hàm cực đại $\le 0.25$, gradient tầng 1 bị triệt tiêu hoàn toàn về **$0.0000$** ở độ sâu 5 tầng (mạng bị đóng băng).
   * **Tanh & ReLU**: Duy trì gradient tầng đầu ổn định ($\text{Ratio} \approx 0.8 - 1.4$), giải thích vì sao mạng 5 tầng Tanh hội tụ xuất sắc.
