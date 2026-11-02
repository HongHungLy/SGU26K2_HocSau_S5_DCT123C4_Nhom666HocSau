"""
Module: evaluate.py
Nhiệm vụ:
- Đánh giá mô hình trên tập Test độc lập
- Tính toán đầy đủ các chỉ số theo yêu cầu:
    + MSE (Mean Squared Error) - chỉ số bắt buộc
    + MAE (Mean Absolute Error) & RMSE (Root Mean Squared Error) - chỉ số bổ sung
    + R^2 Score (Hệ số xác định)
"""

import numpy as np
import torch
import torch.nn as nn


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Tính các độ đo thống kê giữa nhãn thật y và dự đoán y_hat."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    diff = y_true - y_pred
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))

    # Hệ số R2
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def evaluate_model(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = "cpu",
) -> tuple:
    """
    Đưa dữ liệu test qua mô hình để dự đoán và tính metrics.
    Trả về:
        metrics: dict gồm mse, rmse, mae, r2
        y_pred: ndarray kích thước (N, 1)
    """
    model.eval()
    X_tensor = torch.from_numpy(X_test.astype(np.float32)).to(device)

    with torch.no_grad():
        y_pred_tensor = model(X_tensor)
        y_pred = y_pred_tensor.cpu().numpy()

    metrics = compute_metrics(y_test, y_pred)
    return metrics, y_pred
