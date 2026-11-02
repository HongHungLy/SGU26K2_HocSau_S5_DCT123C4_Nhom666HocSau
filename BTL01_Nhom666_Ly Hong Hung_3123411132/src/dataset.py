"""
Module: dataset.py
Nhiệm vụ (Thành viên 1):
- Cài đặt chính xác hàm mật độ Gaussian 2 chiều theo công thức đề bài:
    f(x) = (1 / (2 * pi * sqrt(det(C)))) * exp(-0.5 * (x - mu)^T * C^(-1) * (x - mu))
    với mu = [0, 0]^T, C = [[0.3, -0.5], [-0.5, 2.0]]
- Sinh dữ liệu ngẫu nhiên độc lập trong miền x1 in [-2, 2], x2 in [-4, 4]
- Chia tập Train / Validation / Test độc lập, kiểm soát random seed để chống rò rỉ dữ liệu (data leakage)
- Đóng gói PyTorch Dataset / DataLoader phục vụ huấn luyện
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Thông số chuẩn của bài toán
DEFAULT_MU = np.array([0.0, 0.0], dtype=np.float64)
DEFAULT_COV = np.array([[0.3, -0.5], [-0.5, 2.0]], dtype=np.float64)
RANGE_X1 = (-2.0, 2.0)
RANGE_X2 = (-4.0, 4.0)


class Gaussian2D:
    """Lớp tính toán giá trị mật độ xác suất Gaussian hai chiều chính xác."""

    def __init__(self, mu: np.ndarray = None, cov: np.ndarray = None):
        self.mu = np.array(mu if mu is not None else DEFAULT_MU, dtype=np.float64)
        self.cov = np.array(cov if cov is not None else DEFAULT_COV, dtype=np.float64)

        # Tính định thức và ma trận nghịch đảo
        self.det_cov = np.linalg.det(self.cov)
        if self.det_cov <= 0:
            raise ValueError(f"Ma trận hiệp phương sai C phải xác định dương. det(C) = {self.det_cov}")
        
        self.inv_cov = np.linalg.inv(self.cov)
        # Hệ số chuẩn hóa: 1 / (2 * pi * sqrt(det(C)))
        self.norm_factor = 1.0 / (2.0 * np.pi * np.sqrt(self.det_cov))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Tính f(x) cho mảng điểm x.
        x: ndarray kích thước (N, 2) hoặc (2,)
        Trả về: y có kích thước (N, 1) hoặc (1,)
        """
        x_arr = np.asarray(x, dtype=np.float64)
        single_point = False
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
            single_point = True

        if x_arr.shape[1] != 2:
            raise ValueError(f"Dữ liệu đầu vào phải có 2 chiều, nhận được shape {x_arr.shape}")

        diff = x_arr - self.mu  # (N, 2)
        # Khoảng cách Mahalanobis d^2 = (x - mu)^T * C^(-1) * (x - mu)
        # (N, 2) @ (2, 2) = (N, 2)
        maha = np.sum((diff @ self.inv_cov) * diff, axis=1)  # (N,)
        y = self.norm_factor * np.exp(-0.5 * maha)

        y = y.reshape(-1, 1)
        return y[0, 0] if single_point else y


def generate_dataset(
    n_samples: int,
    seed: int = None,
    x1_range: tuple = RANGE_X1,
    x2_range: tuple = RANGE_X2,
    gaussian_fn: Gaussian2D = None,
) -> tuple:
    """
    Sinh ngẫu nhiên n_samples điểm (x1, x2) phân bố đều (uniform) trong miền và tính y.
    Trả về:
        X: ndarray shape (n_samples, 2)
        y: ndarray shape (n_samples, 1)
    """
    rng = np.random.default_rng(seed)
    if gaussian_fn is None:
        gaussian_fn = Gaussian2D()

    # Sinh uniform trong [x1_min, x1_max] và [x2_min, x2_max]
    x1 = rng.uniform(x1_range[0], x1_range[1], size=(n_samples, 1))
    x2 = rng.uniform(x2_range[0], x2_range[1], size=(n_samples, 1))
    X = np.hstack([x1, x2])
    y = gaussian_fn(X)

    return X.astype(np.float32), y.astype(np.float32)


def save_dataset_to_csv(X: np.ndarray, y: np.ndarray, filepath: str) -> None:
    """Lưu tập dữ liệu vào file CSV với tiêu đề x1, x2, y."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    df = pd.DataFrame({
        "x1": X[:, 0],
        "x2": X[:, 1],
        "y": y[:, 0]
    })
    df.to_csv(filepath, index=False)
    print(f"Đã lưu {len(df)} mẫu vào: {filepath}")


def load_dataset_from_csv(filepath: str) -> tuple:
    """Tải tập dữ liệu từ file CSV, trả về (X, y) dạng numpy float32."""
    df = pd.read_csv(filepath)
    X = df[["x1", "x2"]].to_numpy(dtype=np.float32)
    y = df[["y"]].to_numpy(dtype=np.float32)
    return X, y


def create_data_splits(
    train_pool_size: int = 2000,
    val_size: int = 400,
    test_size: int = 100,
    base_seed: int = 42,
    data_dir: str = "data",
) -> dict:
    """
    Sinh 3 tập dữ liệu Train Pool, Validation, Test hoàn toàn độc lập với seed riêng biệt
    để đảm bảo không bị rò rỉ dữ liệu (data leakage) giữa các tập.
    """
    gaussian_fn = Gaussian2D()
    os.makedirs(data_dir, exist_ok=True)

    # 1. Sinh tập Test độc lập (100 mẫu như đề bài yêu cầu)
    X_test, y_test = generate_dataset(
        test_size, seed=base_seed + 100, gaussian_fn=gaussian_fn
    )
    test_path = os.path.join(data_dir, "test.csv")
    save_dataset_to_csv(X_test, y_test, test_path)

    # 2. Sinh tập Validation độc lập (dùng để chọn checkpoint, tune tham số)
    X_val, y_val = generate_dataset(
        val_size, seed=base_seed + 200, gaussian_fn=gaussian_fn
    )
    val_path = os.path.join(data_dir, "val.csv")
    save_dataset_to_csv(X_val, y_val, val_path)

    # 3. Sinh Training Pool (2000 mẫu để cắt lấy A mẫu cho Thực nghiệm A)
    X_train_pool, y_train_pool = generate_dataset(
        train_pool_size, seed=base_seed + 300, gaussian_fn=gaussian_fn
    )
    train_path = os.path.join(data_dir, "train_pool_2000.csv")
    save_dataset_to_csv(X_train_pool, y_train_pool, train_path)

    return {
        "test": (X_test, y_test),
        "val": (X_val, y_val),
        "train_pool": (X_train_pool, y_train_pool),
    }


class GaussianDataset(Dataset):
    """PyTorch Dataset bọc dữ liệu (X, y)."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.y[idx]


def get_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """Tạo DataLoader tiện lợi cho việc huấn luyện và đánh giá."""
    dataset = GaussianDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
