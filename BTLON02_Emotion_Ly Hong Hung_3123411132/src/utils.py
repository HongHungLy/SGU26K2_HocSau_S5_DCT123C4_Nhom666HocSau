"""
Module tiện ích dùng chung cho toàn bộ dự án FER2013 (PyTorch).
Phụ trách: Thành viên 1 - Lý Hồng Hưng (3123411132).
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Thứ tự chuẩn 7 lớp cảm xúc của Kaggle FER2013 (Emotion_Recognition.pdf, trang 2)
EMOTION_LABELS = [
    "Angry",     # 0
    "Disgust",   # 1
    "Fear",      # 2
    "Happy",     # 3
    "Sad",       # 4
    "Surprise",  # 5
    "Neutral",   # 6
]


def set_seed(seed: int = 42) -> None:
    """Cố định random seed để đảm bảo tính tái lập kết quả thực nghiệm."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Tự động chọn GPU (CUDA) nếu có, ngược lại dùng CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[Device] Đang sử dụng GPU: {gpu_name}")
    else:
        device = torch.device("cpu")
        print("[Device] Không phát hiện CUDA GPU, đang chạy trên CPU.")
    return device


def compute_class_weights(
    labels: Sequence[int],
    num_classes: int = 7,
    method: str = "balanced",
) -> torch.Tensor:
    """
    Tính trọng số cho từng lớp nhằm xử lý mất cân bằng dữ liệu (VD: lớp Disgust).

    Args:
        labels: Danh sách nhãn số nguyên (0..num_classes-1) của tập Training.
        num_classes: Số lượng lớp (mặc định 7).
        method: 'balanced' -> N / (num_classes * count_c),
                'sqrt_balanced' -> căn bậc hai của balanced (mượt hơn, ít làm giảm overall acc).

    Returns:
        torch.Tensor kích thước (num_classes,) kiểu float32.
    """
    labels_arr = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels_arr, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    total = float(len(labels_arr))

    weights = total / (num_classes * counts)
    if method == "sqrt_balanced":
        weights = np.sqrt(weights)
        weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Đếm tổng số tham số và số tham số huấn luyện được của mô hình."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def ensure_dir(path: str | Path) -> Path:
    """Tạo thư mục nếu chưa tồn tại và trả về đối tượng Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_training_curves(
    csv_path: str | Path,
    save_path: Optional[str | Path] = None,
    title: Optional[str] = None,
) -> None:
    """
    Vẽ cặp đồ thị Accuracy và Loss (Train vs Validation) từ file log CSV từng epoch.
    """
    df = pd.read_csv(csv_path)
    epochs = df["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 1. Accuracy
    axes[0].plot(epochs, df["train_acc"] * 100.0, label="Train Acc", linewidth=2.0, color="#1B365D")
    axes[0].plot(epochs, df["val_acc"] * 100.0, label="Val Acc (PublicTest)", linewidth=2.0, color="#E07A5F")
    best_idx = df["val_acc"].idxmax()
    best_ep = df.loc[best_idx, "epoch"]
    best_acc = df.loc[best_idx, "val_acc"] * 100.0
    axes[0].axvline(best_ep, linestyle="--", color="gray", alpha=0.6)
    axes[0].scatter([best_ep], [best_acc], color="#C00000", zorder=5, label=f"Best Val: {best_acc:.2f}% (Ep {int(best_ep)})")
    axes[0].set_title("Model Accuracy (%)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 2. Loss
    axes[1].plot(epochs, df["train_loss"], label="Train Loss", linewidth=2.0, color="#1B365D")
    axes[1].plot(epochs, df["val_loss"], label="Val Loss (PublicTest)", linewidth=2.0, color="#E07A5F")
    min_loss_idx = df["val_loss"].idxmin()
    min_ep = df.loc[min_loss_idx, "epoch"]
    min_loss = df.loc[min_loss_idx, "val_loss"]
    axes[1].axvline(min_ep, linestyle="--", color="gray", alpha=0.6)
    axes[1].scatter([min_ep], [min_loss], color="#C00000", zorder=5, label=f"Min Val Loss: {min_loss:.4f} (Ep {int(min_ep)})")
    axes[1].set_title("Model Loss", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-Entropy Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    if save_path is not None:
        ensure_dir(Path(save_path).parent)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_experiment_comparison(
    summary_csv: str | Path,
    save_path: Optional[str | Path] = None,
    title: str = "So sánh hiệu năng các thực nghiệm Nhóm 1 (Baseline CNN)",
) -> None:
    """
    Vẽ biểu đồ cột so sánh Val Accuracy, Test Accuracy, Macro F1 và Disgust Recall
    giữa các thực nghiệm từ file summary CSV.
    """
    df = pd.read_csv(summary_csv)
    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5.5))
    b1 = ax.bar(x - width, df["best_val_acc"] * 100.0, width, label="Val Acc (PublicTest)", color="#1B365D")
    b2 = ax.bar(x, df["test_acc"] * 100.0, width, label="Test Acc (PrivateTest)", color="#2B5B84")
    b3 = ax.bar(x + width, df["test_macro_f1"] * 100.0, width, label="Test Macro F1", color="#E07A5F")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(df["exp_name"], rotation=15, ha="right")
    ax.set_ylabel("Phần trăm (%)")
    ax.set_ylim(0, 85)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    if save_path is not None:
        ensure_dir(Path(save_path).parent)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
