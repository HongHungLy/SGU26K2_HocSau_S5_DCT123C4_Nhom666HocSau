# BTL01: XẤP XỈ HÀM MẬT ĐỘ GAUSSIAN HAI CHIỀU BẰNG MẠNG MLP (PYTORCH)

> **Môn học**: Deep Learning (Học sâu)  
> **Thành viên nhóm**: 3 người  
> **Triển khai**: Hoàn toàn bằng **PyTorch** (`torch.nn.Module`, `DataLoader`, `MSELoss`, `Adam/SGD`).

---

## 1. Giới thiệu Bài toán

Xấp xỉ hàm mật độ xác suất Gaussian hai chiều:
$$f(\mathbf{x}) = \frac{1}{2\pi \sqrt{\det(C)}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T C^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

với:
* Kỳ vọng: $\boldsymbol{\mu} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$
* Ma trận hiệp phương sai: $C = \begin{bmatrix} 0.3 & -0.5 \\ -0.5 & 2.0 \end{bmatrix}$
* Miền giá trị đầu vào: $x_1 \in [-2, 2]$, $x_2 \in [-4, 4]$
* Mục tiêu: $y = f(x_1, x_2)$

Mạng MLP nhận 2 đầu vào $(x_1, x_2)$ và dự đoán giá trị đầu ra $\hat{y} \approx y$.

---

## 2. Cấu trúc Thư mục

```text
Deep Learning - BTL01/
├── data/                         # Chứa các file dữ liệu CSV độc lập
│   ├── train_pool_2000.csv       # Pool 2000 mẫu train
│   ├── val.csv                   # 400 mẫu validation (chọn checkpoint)
│   ├── test.csv                  # 100 mẫu test độc lập (đánh giá cuối cùng)
│   └── data_statistics.csv       # Bảng thống kê mô tả (mean, std, min, max)
├── checkpoints/                  # Lưu các trọng số mô hình tốt nhất (.pt)
├── logs/                         # Lưu log chi tiết quá trình huấn luyện (.csv)
│   ├── exp_A_runs_detailed.csv   # Kết quả từng lần chạy
│   ├── exp_A_summary.csv         # Bảng tổng hợp Mean ± Std
│   └── exp_A_nhan_xet_xu_huong.txt # Nhận xét xu hướng khoa học
├── figures/                      # Chứa toàn bộ đồ thị, biểu đồ xuất ra
│   ├── data_3d_scatter.png       # Scatter plot 3D dữ liệu huấn luyện
│   ├── true_gaussian_surface.png # Bề mặt hàm Gaussian thật
│   ├── data_distributions.png    # Histogram phân bố x1, x2, y
│   ├── exp_A_error_vs_size.png   # Đồ thị sai số MSE/MAE theo quy mô A
│   ├── exp_A_best_loss_curve.png # Đường cong hội tụ train vs val loss
│   ├── exp_A_best_pred_vs_true_3d.png # So sánh y thật vs y_hat dự đoán
│   └── exp_A_surface_comparison.png   # Bề mặt thật vs MLP xấp xỉ
├── src/                          # Mã nguồn cốt lõi (Module PyTorch)
│   ├── dataset.py                # Hàm Gaussian 2D, sinh & chia tập dữ liệu
│   ├── model.py                  # Class FlexibleMLP tùy biến số tầng & nơ-ron
│   ├── trainer.py                # Vòng lặp train/val tường minh, early stopping
│   ├── evaluate.py               # Tính MSE, MAE, RMSE, R2
│   └── plot_utils.py             # Trực quan hóa 2D/3D khoa học
├── generate_data.py              # Script sinh dữ liệu và vẽ EDA ban đầu
├── exp_A_datasize.py             # Script chạy Thực nghiệm A (Quy mô dữ liệu)
├── requirements.txt              # Thư viện phụ thuộc
└── README.md                     # Tài liệu hướng dẫn sử dụng
```

---

## 3. Cài đặt Môi trường

Dự án yêu cầu **Python >= 3.9** và **PyTorch >= 2.0**. Cài đặt các gói phụ thuộc:

```bash
pip install -r requirements.txt
```

---

## 4. Hướng dẫn Chạy (Dành cho Thành viên 1)

### Bước 1: Khởi tạo Dữ liệu và Trực quan hóa Ban đầu
Chạy script sinh 3 tập dữ liệu độc lập (train pool, val, test) và vẽ các đồ thị khảo sát:
```bash
python generate_data.py
```
*Kết quả:* Tạo ra các file CSV trong `data/` và 3 hình ảnh trực quan hóa trong `figures/`.

### Bước 2: Chạy Thực nghiệm A (Quy mô dữ liệu huấn luyện)
Chạy toàn bộ thực nghiệm A trên 10 mức quy mô $A \in [200, 2000]$, lặp lại 3 seeds ngẫu nhiên:
```bash
python exp_A_datasize.py
```
*(Nếu muốn chạy nhanh để kiểm tra code: `python exp_A_datasize.py --fast_test`)*

### Bước 3: Thu thập Kết quả đưa vào Báo cáo & Slide
Sau khi chạy xong:
1. Mở file `logs/exp_A_summary.csv` để lấy bảng số liệu Mean ± Std của Test MSE, MAE, $R^2$.
2. Mở file `logs/exp_A_nhan_xet_xu_huong.txt` để lấy nhận xét học thuật đã được soạn sẵn.
3. Lấy các hình ảnh trong thư mục `figures/` để chèn vào báo cáo Word và Slide thuyết trình.
