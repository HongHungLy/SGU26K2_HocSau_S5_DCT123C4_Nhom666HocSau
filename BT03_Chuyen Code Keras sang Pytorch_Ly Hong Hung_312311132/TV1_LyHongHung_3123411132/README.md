![OS](https://img.shields.io/badge/OS-Windows%20%7C%20Linux%20%7C%20Colab-green.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)
![TorchVision](https://img.shields.io/badge/TorchVision-0.17%2B-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8%20%2F%2012.1%20(optional)-yellowgreen.svg)

# Deep Learning Example Programs (PyTorch Edition)

> Dự án chuyển đổi toàn bộ mã nguồn thực hành Deep Learning từ **TensorFlow/Keras** sang **PyTorch** (BTL03).  
> Phụ trách tài liệu môi trường & Nhóm 1: **Thành viên 1 — Lý Hồng Hưng (MSSV: 3123411132)**.

---

## 1. Cấu trúc thư mục dự án

```text
├── README.md                                 # Hướng dẫn cài đặt môi trường & chạy PyTorch
├── requirements.txt                          # Danh sách thư viện Python cần thiết
└── code_pytorch/
    ├── colab_getting_started.ipynb           # Hướng dẫn sử dụng GPU & Google Drive trên Colab
    ├── 0. Getting Started/
    │   ├── getting_started.ipynb             # Kiểm tra phiên bản PyTorch & GPU CUDA
    │   └── Iris_Bayesian_decision_making.ipynb # Phân loại Iris bằng GaussianNB (scikit-learn)
    ├── 1. MLP/
    │   ├── Func_apxm_MLP.ipynb               # Xấp xỉ hàm Gaussian 2D bằng MLP (3 thí nghiệm A, B, C)
    │   └── MNIST_MLP.ipynb                   # Phân loại MNIST bằng MLP + trích xuất đặc trưng fc2
    ├── 2. CNN + MLP/
    │   ├── 1_MNIST_CNN.ipynb                 # CNN+MLP phân loại MNIST + trích xuất fc2/GAP + PCA
    │   ├── 2_Human_Gender_CNN.ipynb          # (TV2) CNN phân loại giới tính (WIKI 64x64)
    │   └── 3_Emotion_Reg_CNN.ipynb           # (TV2) CNN nhận diện cảm xúc 7 lớp (FER2013 48x48)
    ├── 3. CNN + LSTM/
    │   └── MNIST_CNN_LSTM.ipynb              # (TV3) CNN+LSTM tính tổng chuỗi ảnh MNIST
    ├── 4. Training Utilities/
    │   ├── 2_Human_Gender_CNN.ipynb          # (TV2) Checkpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
    │   └── 3_Emotion_Reg_CNN.ipynb           # (TV2) Data Augmentation + EarlyStopping
    └── 5. Others/
        ├── 1_segnet.ipynb                    # (TV3) SegNet phân đoạn ảnh độ sâu bàn tay (depth_hand.zip)
        └── 2_obj_detection.ipynb             # (TV3) FCN phát hiện & định vị xe hơi (car_data.zip)
```

---

## 2. Hướng dẫn cài đặt môi trường

### Cách 1: Chạy trực tiếp trên Google Colab với GPU miễn phí (Khuyên dùng)
1. Truy cập [Google Colaboratory](https://colab.research.google.com/) và đăng nhập tài khoản Google.
2. Tải thư mục `code_pytorch/` lên Google Drive hoặc mở trực tiếp file `.ipynb` qua `File` → `Upload notebook`.
3. Bật GPU: chọn menu **`Runtime` → `Change runtime type` → `T4 GPU` → `Save`**.
4. Chạy file `colab_getting_started.ipynb` để kiểm tra GPU, cài thêm `torchinfo` (`!pip install torchinfo`) và kết nối Google Drive.
5. Chọn **`Runtime` → `Run all`** để chạy toàn bộ notebook.

### Cách 2: Cài đặt trên máy cá nhân có GPU NVIDIA (Windows / Linux)
1. Kiểm tra GPU NVIDIA hỗ trợ CUDA (Compute Capability ≥ 5.0) và cập nhật NVIDIA Driver mới nhất.
2. Cài đặt **Python 3.10+** (qua [python.org](https://www.python.org/downloads/) hoặc [Miniconda/Anaconda](https://docs.conda.io/en/latest/miniconda.html)).
3. Tạo môi trường ảo:
   ```bash
   python -m venv venv_pytorch
   # Windows (PowerShell):
   .\venv_pytorch\Scripts\Activate.ps1
   # Linux / macOS:
   source venv_pytorch/bin/activate
   ```
4. Cài đặt PyTorch hỗ trợ CUDA (ví dụ CUDA 12.1 — xem thêm tại [pytorch.org/get-started](https://pytorch.org/get-started/locally/)):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
5. Cài đặt các thư viện phụ trợ còn lại:
   ```bash
   pip install -r requirements.txt
   ```

### Cách 3: Cài đặt trên máy cá nhân chỉ có CPU
1. Cài đặt **Python 3.10+** và kích hoạt môi trường ảo (như Bước 2–3 ở Cách 2).
2. Cài đặt PyTorch bản CPU và các thư viện phụ trợ:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```

---

## 3. Hướng dẫn chạy chương trình (Local Jupyter)

1. Mở Terminal / PowerShell và di chuyển vào thư mục chứa dự án:
   ```bash
   cd path/to/code_pytorch
   ```
2. Khởi chạy Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Trình duyệt sẽ mở giao diện Jupyter. Mở notebook cần chạy (bắt đầu từ `0. Getting Started/getting_started.ipynb` để kiểm tra môi trường).
4. Chọn **`Kernel` → `Restart & Run All`** để thực thi toàn bộ mã nguồn.

---

## 4. Lưu ý về dữ liệu & khác biệt chính Keras → PyTorch

- **Dữ liệu MNIST & Iris**: Tự động tải về qua `torchvision.datasets.MNIST` và `sklearn.datasets.load_iris` (không cần tải thủ công).
- **Định dạng ảnh**: Keras dùng `NHWC (N, H, W, C)`, còn PyTorch dùng `NCHW (N, C, H, W)`. `transforms.ToTensor()` tự động chuyển ảnh sang `(C, H, W)` và chuẩn hóa về `[0.0, 1.0]`.
- **Hàm mất mát phân loại**: Thay cho `one-hot` + `Softmax` + `categorical_crossentropy` ở Keras, PyTorch dùng trực tiếp nhãn số nguyên (`torch.long`) + logits thô (không qua Softmax) đưa vào `nn.CrossEntropyLoss()`.
- **Thiết bị tính toán**: Tất cả notebook đều tự động nhận diện thiết bị bằng `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
