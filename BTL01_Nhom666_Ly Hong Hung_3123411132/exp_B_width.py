"""
Script: exp_B_width.py
Phụ trách: Thành viên 2 (Kỹ sư Huấn luyện & Độ rộng)
Thực nghiệm B: Khảo sát ảnh hưởng của số nơ-ron tầng ẩn B (Width) với mạng MLP 1 tầng ẩn.
- Biến độc lập: B in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64]
- Cố định quy mô tập train A = 1000 mẫu
- Lặp lại 3 seeds (42, 123, 999) để tính Mean +/- Std
- Bắt buộc phân tích: Hiện tượng thiếu khớp (Underfitting khi B nhỏ) và quá khớp (Overfitting khi B lớn)
"""

import os
import sys
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


def run_experiment_b(
    data_dir="data",
    figures_dir="figures",
    logs_dir="logs",
    checkpoints_dir="checkpoints",
    hidden_units=None,
    seeds=None,
    train_size=1000,
    lr=0.005,
    batch_size=32,
    epochs=100,
):
    if hidden_units is None:
        hidden_units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64]
    if seeds is None:
        seeds = [42, 123, 999]

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    print("=" * 75)
    print(" THỰC NGHIỆM B: KHẢO SÁT ẢNH HƯỞNG CỦA ĐỘ RỘNG MẠNG (SỐ NƠ-RON B) ")
    print("=" * 75)
    print(f"[*] Cố định quy mô mẫu: {train_size} mẫu")
    print(f"[*] Danh sách số nơ-ron B : {hidden_units}")
    print(f"[*] Các seeds lặp lại     : {seeds}")
    print("-" * 75)

    X_val, y_val = load_dataset_from_csv(os.path.join(data_dir, "val.csv"))
    X_test, y_test = load_dataset_from_csv(os.path.join(data_dir, "test.csv"))
    X_pool, y_pool = load_dataset_from_csv(os.path.join(data_dir, "train_pool_2000.csv"))

    val_dataset = GaussianDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    summary_records = []

    for B in hidden_units:
        test_mse_list, train_loss_list = [], []

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

            model = FlexibleMLP(input_dim=2, hidden_dims=[B], output_dim=1, activation="tanh")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                checkpoint_dir=checkpoints_dir,
                log_dir=os.path.join(logs_dir, "exp_B_epochs"),
            )

            res = trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                checkpoint_name=f"expB_nodes_{B}_seed_{seed}.pt",
                log_csv_name=f"expB_log_nodes_{B}_seed_{seed}.csv",
                verbose=False,
            )

            test_metrics, _ = evaluate_model(trainer.model, X_test, y_test)
            test_mse_list.append(test_metrics["mse"])
            train_loss_list.append(res["history"]["train_loss"][-1])

        mse_mean, mse_std = np.mean(test_mse_list), np.std(test_mse_list)
        train_mean = np.mean(train_loss_list)

        summary_records.append({
            "nodes_B": B,
            "train_mse_mean": train_mean,
            "test_mse_mean": mse_mean,
            "test_mse_std": mse_std,
        })
        print(f"B = {B:2d} | Train MSE: {train_mean:.6f} | Test MSE: {mse_mean:.6f} +/- {mse_std:.6f}")

    df_summary = pd.DataFrame(summary_records)
    summary_path = os.path.join(logs_dir, "exp_B_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"\n[*] Đã lưu kết quả Thực nghiệm B vào: {summary_path}")

    # Vẽ biểu đồ Train vs Test MSE để thấy rõ Underfitting & Overfitting
    plt.figure(figsize=(9, 5.5))
    plt.plot(df_summary["nodes_B"], df_summary["train_mse_mean"], "o-", color="#27ae60", label="Train Loss (MSE)")
    plt.plot(df_summary["nodes_B"], df_summary["test_mse_mean"], "s--", color="#e74c3c", label="Test Loss (MSE)")
    plt.fill_between(
        df_summary["nodes_B"],
        df_summary["test_mse_mean"] - df_summary["test_mse_std"],
        df_summary["test_mse_mean"] + df_summary["test_mse_std"],
        color="#e74c3c", alpha=0.15
    )
    plt.xlabel("Số lượng nơ-ron tầng ẩn (B)", fontsize=11, fontweight="bold")
    plt.ylabel("MSE Loss", fontsize=11, fontweight="bold")
    plt.title("Thực nghiệm B: Phân tích Thiếu khớp (Underfitting) và Quá khớp (Overfitting) theo Độ rộng B", fontsize=11, fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    fig_path = os.path.join(figures_dir, "exp_B_error_vs_nodes.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[*] Đã lưu biểu đồ: {fig_path}")


if __name__ == "__main__":
    run_experiment_b()
