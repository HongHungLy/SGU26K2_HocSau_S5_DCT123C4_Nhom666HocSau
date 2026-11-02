"""
Script: exp_C_depth.py
Phụ trách: Thành viên 3 (Kỹ sư Đánh giá & Độ sâu)
Thực nghiệm C: Khảo sát ảnh hưởng của số tầng ẩn C (Depth) đến hiệu năng mạng MLP.
- Biến độc lập: C in [1, 2, 3, 4, 5] (mỗi tầng ẩn cố định 16 nơ-ron hoặc 32 nơ-ron)
- Ghi nhận bắt buộc theo đề bài:
    + Test Error (MSE, MAE, RMSE)
    + Loss hội tụ
    + Thời gian huấn luyện (Training time)
- Phân tích hiện tượng triệt tiêu đạo hàm (vanishing gradient) và độ phức tạp tính toán
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.dataset import load_dataset_from_csv, GaussianDataset
from src.model import FlexibleMLP
from src.trainer import Trainer
from src.evaluate import evaluate_model


def run_experiment_c(
    data_dir="data",
    figures_dir="figures",
    logs_dir="logs",
    checkpoints_dir="checkpoints",
    layers_list=None,
    nodes_per_layer=16,
    seeds=None,
    train_size=1000,
    lr=0.005,
    batch_size=32,
    epochs=100,
):
    if layers_list is None:
        layers_list = [1, 2, 3, 4, 5]
    if seeds is None:
        seeds = [42, 123, 999]

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    print("=" * 75)
    print(" THỰC NGHIỆM C: KHẢO SÁT ẢNH HƯỞNG CỦA ĐỘ SÂU MẠNG (SỐ TẦNG ẨN C) ")
    print("=" * 75)
    print(f"[*] Số nơ-ron mỗi tầng ẩn: {nodes_per_layer}")
    print(f"[*] Danh sách số tầng C  : {layers_list}")
    print(f"[*] Các seeds lặp lại    : {seeds}")
    print("-" * 75)

    X_val, y_val = load_dataset_from_csv(os.path.join(data_dir, "val.csv"))
    X_test, y_test = load_dataset_from_csv(os.path.join(data_dir, "test.csv"))
    X_pool, y_pool = load_dataset_from_csv(os.path.join(data_dir, "train_pool_2000.csv"))

    val_dataset = GaussianDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    summary_records = []

    for C in layers_list:
        hidden_dims = [nodes_per_layer] * C
        test_mse_list, time_list = [], []

        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)

            rng = np.random.default_rng(seed)
            indices = rng.choice(len(X_pool), size=train_size, replace=False)
            train_loader = DataLoader(
                GaussianDataset(X_pool[indices], y_pool[indices]),
                batch_size=batch_size,
                shuffle=True,
            )

            model = FlexibleMLP(input_dim=2, hidden_dims=hidden_dims, output_dim=1, activation="tanh")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                checkpoint_dir=checkpoints_dir,
                log_dir=os.path.join(logs_dir, "exp_C_epochs"),
            )

            start = time.time()
            res = trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                checkpoint_name=f"expC_layers_{C}_seed_{seed}.pt",
                log_csv_name=f"expC_log_layers_{C}_seed_{seed}.csv",
                verbose=False,
            )
            elapsed = time.time() - start

            test_metrics, _ = evaluate_model(trainer.model, X_test, y_test)
            test_mse_list.append(test_metrics["mse"])
            time_list.append(elapsed)

        mse_mean, mse_std = np.mean(test_mse_list), np.std(test_mse_list)
        time_mean = np.mean(time_list)

        summary_records.append({
            "layers_C": C,
            "test_mse_mean": mse_mean,
            "test_mse_std": mse_std,
            "training_time_s": time_mean,
            "total_params": model.count_parameters(),
        })
        print(f"C = {C} tầng | Params: {model.count_parameters():4d} | Test MSE: {mse_mean:.6f} +/- {mse_std:.6f} | Time: {time_mean:.2f}s")

    df_summary = pd.DataFrame(summary_records)
    summary_path = os.path.join(logs_dir, "exp_C_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"\n[*] Đã lưu kết quả Thực nghiệm C vào: {summary_path}")

    # Vẽ biểu đồ 2 trục: Test Error vs Training Time theo độ sâu C
    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    color_mse = "#c0392b"
    ax1.set_xlabel("Số lượng tầng ẩn (C)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Test MSE", color=color_mse, fontsize=11, fontweight="bold")
    line1 = ax1.plot(df_summary["layers_C"], df_summary["test_mse_mean"], "o-", color=color_mse, linewidth=2.2, label="Test MSE")
    ax1.fill_between(
        df_summary["layers_C"],
        df_summary["test_mse_mean"] - df_summary["test_mse_std"],
        df_summary["test_mse_mean"] + df_summary["test_mse_std"],
        color=color_mse, alpha=0.15
    )
    ax1.tick_params(axis="y", labelcolor=color_mse)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color_time = "#2980b9"
    ax2.set_ylabel("Thời gian huấn luyện (giây)", color=color_time, fontsize=11, fontweight="bold")
    line2 = ax2.plot(df_summary["layers_C"], df_summary["training_time_s"], "s--", color=color_time, linewidth=2.0, label="Training Time")
    ax2.tick_params(axis="y", labelcolor=color_time)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.title("Thực nghiệm C: Đánh giá Sai số và Chi phí Thời gian theo Độ sâu C", fontsize=11, fontweight="bold")
    plt.tight_layout()

    fig_path = os.path.join(figures_dir, "exp_C_error_and_time_vs_depth.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[*] Đã lưu biểu đồ: {fig_path}")


if __name__ == "__main__":
    run_experiment_c()
