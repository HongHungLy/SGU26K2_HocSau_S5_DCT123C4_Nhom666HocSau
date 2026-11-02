"""
Module: plot_utils.py
Nhiệm vụ:
- Trực quan hóa dữ liệu 3D, bề mặt hàm thật Gaussian hai chiều
- Vẽ đồ thị phân bố của x1, x2, y
- Vẽ đường cong huấn luyện (Loss curves: Train vs Validation)
- So sánh bề mặt hàm thật vs Bề mặt xấp xỉ do mạng MLP dự đoán
- Vẽ đồ thị kết quả các thực nghiệm (A, B, C)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn

# Cấu hình phong cách biểu đồ khoa học
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8


def plot_3d_scatter(
    X: np.ndarray,
    y: np.ndarray,
    title: str = "3D Scatter Plot của Dữ liệu Gaussian",
    color: str = "#1f77b4",
    save_path: str = None,
):
    """Vẽ scatter 3D của tập dữ liệu (x1, x2, y)."""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        X[:, 0], X[:, 1], y[:, 0],
        c=y[:, 0], cmap="viridis", edgecolors="k", linewidth=0.3, alpha=0.85, s=25
    )
    ax.set_xlabel(r"$x_1$", fontsize=11, labelpad=8)
    ax.set_ylabel(r"$x_2$", fontsize=11, labelpad=8)
    ax.set_zlabel(r"$y = f(x_1, x_2)$", fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=15, pad=0.1, label="Mật độ Gaussian")

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Đã lưu biểu đồ: {save_path}")
    plt.close()


def plot_true_surface(
    gaussian_fn,
    x1_range: tuple = (-2.0, 2.0),
    x2_range: tuple = (-4.0, 4.0),
    resolution: int = 60,
    title: str = "Bề mặt Hàm Mật độ Gaussian 2 Chiều Thật",
    save_path: str = None,
):
    """Vẽ bề mặt 3D hàm Gaussian thật."""
    x1_grid = np.linspace(x1_range[0], x1_range[1], resolution)
    x2_grid = np.linspace(x2_range[0], x2_range[1], resolution)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)

    points = np.vstack([X1.ravel(), X2.ravel()]).T
    Y_grid = gaussian_fn(points).reshape(resolution, resolution)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X1, X2, Y_grid, cmap="coolwarm", edgecolor="none", alpha=0.9, antialiased=True
    )
    ax.set_xlabel(r"$x_1$", fontsize=11, labelpad=8)
    ax.set_ylabel(r"$x_2$", fontsize=11, labelpad=8)
    ax.set_zlabel(r"$y = f(x_1, x_2)$", fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, pad=0.1)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Đã lưu biểu đồ: {save_path}")
    plt.close()


def plot_data_distributions(
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Phân bố Xác suất của x1, x2 và Giá trị Mục tiêu y",
    save_path: str = None,
):
    """Vẽ biểu đồ histogram phân bố của các biến."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].hist(X[:, 0], bins=25, color="#3498db", edgecolor="black", alpha=0.7)
    axes[0].set_title(r"Phân bố $x_1 \in [-2, 2]$", fontweight="bold")
    axes[0].set_xlabel(r"$x_1$")
    axes[0].set_ylabel("Tần suất")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].hist(X[:, 1], bins=25, color="#2ecc71", edgecolor="black", alpha=0.7)
    axes[1].set_title(r"Phân bố $x_2 \in [-4, 4]$", fontweight="bold")
    axes[1].set_xlabel(r"$x_2$")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].hist(y[:, 0], bins=25, color="#e74c3c", edgecolor="black", alpha=0.7)
    axes[2].set_title(r"Phân bố Mục tiêu $y$", fontweight="bold")
    axes[2].set_xlabel(r"$y$")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Đã lưu biểu đồ: {save_path}")
    plt.close()


def plot_loss_curves(
    train_losses: list,
    val_losses: list,
    title: str = "Đường cong Hội tụ Loss (Train vs Validation)",
    save_path: str = None,
):
    """Vẽ biểu đồ biến thiên train loss và validation loss theo epoch."""
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss (MSE)", color="#e74c3c", linewidth=2.0)
    plt.plot(epochs, val_losses, label="Validation Loss (MSE)", color="#2980b9", linewidth=2.0, linestyle="--")

    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("MSE Loss", fontsize=11)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Đã lưu biểu đồ: {save_path}")
    plt.close()


def plot_pred_vs_true_3d(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    title: str = "So sánh Giá trị Thật (y) và Dự đoán (y_hat) trên Tập Test",
    save_path: str = None,
):
    """Vẽ scatter 3D so sánh trực tiếp điểm thực tế và điểm mô hình dự đoán."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        X_test[:, 0], X_test[:, 1], y_test[:, 0],
        color="#2980b9", label="Giá trị thật (y)", s=35, alpha=0.8, edgecolors="k", linewidth=0.5
    )
    ax.scatter(
        X_test[:, 0], X_test[:, 1], y_pred[:, 0],
        color="#e74c3c", label=r"Dự đoán ($\hat{y}$)", s=35, alpha=0.8, marker="^", edgecolors="k", linewidth=0.5
    )

    ax.set_xlabel(r"$x_1$", fontsize=11, labelpad=8)
    ax.set_ylabel(r"$x_2$", fontsize=11, labelpad=8)
    ax.set_zlabel(r"$y$", fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
    ax.legend(frameon=True, fontsize=10)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Đã lưu biểu đồ: {save_path}")
    plt.close()


def plot_comparison_surface(
    model: nn.Module,
    gaussian_fn,
    x1_range: tuple = (-2.0, 2.0),
    x2_range: tuple = (-4.0, 4.0),
    resolution: int = 50,
    save_path: str = None,
):
    """Vẽ 2 bề mặt cạnh nhau: Hàm thật Gaussian vs Bề mặt mô hình MLP xấp xỉ."""
    x1_grid = np.linspace(x1_range[0], x1_range[1], resolution)
    x2_grid = np.linspace(x2_range[0], x2_range[1], resolution)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    points = np.vstack([X1.ravel(), X2.ravel()]).T

    # Giá trị thật
    Y_true = gaussian_fn(points).reshape(resolution, resolution)

    # Dự đoán từ mô hình MLP
    model.eval()
    with torch.no_grad():
        pt_input = torch.from_numpy(points.astype(np.float32))
        Y_pred = model(pt_input).cpu().numpy().reshape(resolution, resolution)

    fig = plt.figure(figsize=(16, 7))

    # Subplot 1: Hàm thật
    ax1 = fig.add_subplot(121, projection="3d")
    surf1 = ax1.plot_surface(X1, X2, Y_true, cmap="viridis", alpha=0.85, edgecolor="none")
    ax1.set_title("Hàm Mật độ Gaussian Thật", fontsize=12, fontweight="bold")
    ax1.set_xlabel(r"$x_1$")
    ax1.set_ylabel(r"$x_2$")
    ax1.set_zlabel(r"$y$")
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=12, pad=0.1)

    # Subplot 2: MLP dự đoán
    ax2 = fig.add_subplot(122, projection="3d")
    surf2 = ax2.plot_surface(X1, X2, Y_pred, cmap="plasma", alpha=0.85, edgecolor="none")
    ax2.set_title(r"Bề mặt Xấp xỉ của Mạng MLP ($\hat{y}$)", fontsize=12, fontweight="bold")
    ax2.set_xlabel(r"$x_1$")
    ax2.set_ylabel(r"$x_2$")
    ax2.set_zlabel(r"$\hat{y}$")
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=12, pad=0.1)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Đã lưu biểu đồ: {save_path}")
    plt.close()


def plot_experiment_a(
    sizes: list,
    mse_means: list,
    mse_stds: list,
    mae_means: list,
    mae_stds: list,
    save_path: str = None,
):
    """Vẽ biểu đồ sai số (MSE & MAE) theo số lượng mẫu huấn luyện A (kèm khoảng tin cậy std)."""
    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    sizes = np.array(sizes)
    mse_means = np.array(mse_means)
    mse_stds = np.array(mse_stds)
    mae_means = np.array(mae_means)
    mae_stds = np.array(mae_stds)

    color_mse = "#d63031"
    ax1.set_xlabel("Số lượng mẫu huấn luyện (A)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Test MSE", color=color_mse, fontsize=11, fontweight="bold")
    line1 = ax1.plot(sizes, mse_means, "o-", color=color_mse, linewidth=2.2, label="Test MSE (Mean)")
    ax1.fill_between(sizes, mse_means - mse_stds, mse_means + mse_stds, color=color_mse, alpha=0.18, label="MSE ± Std")
    ax1.tick_params(axis="y", labelcolor=color_mse)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Trục phụ cho MAE
    ax2 = ax1.twinx()
    color_mae = "#0984e3"
    ax2.set_ylabel("Test MAE", color=color_mae, fontsize=11, fontweight="bold")
    line2 = ax2.plot(sizes, mae_means, "s--", color=color_mae, linewidth=2.0, label="Test MAE (Mean)")
    ax2.fill_between(sizes, mae_means - mae_stds, mae_means + mae_stds, color=color_mae, alpha=0.15, label="MAE ± Std")
    ax2.tick_params(axis="y", labelcolor=color_mae)

    # Gộp legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=True)

    plt.title("Thực nghiệm A: Tác động của Quy mô Dữ liệu Huấn luyện (A) đến Sai số Mô hình", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Đã lưu biểu đồ: {save_path}")
    plt.close()
