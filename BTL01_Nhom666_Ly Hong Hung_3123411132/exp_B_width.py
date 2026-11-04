"""
Script: exp_B_width.py  (bản nâng cấp - Thành viên 2)
Thực nghiệm B: ảnh hưởng của số nơ-ron tầng ẩn B (MLP 1 tầng ẩn).

Dùng chung hạ tầng của nhóm (src/dataset, src/model, src/trainer, src/evaluate).
Khác bản gốc:
  - Báo cáo thêm Val MSE, Test MAE/RMSE, số tham số, best_epoch, tỉ số val/train.
  - Train loss đo lại ở chế độ eval tại checkpoint tốt nhất (không phải trung bình epoch cuối).
  - Lưu kết quả từng lần chạy (exp_B_runs_detailed.csv).
  - Có tham số dòng lệnh --train_size / --epochs để chạy khảo sát bổ sung (vd 200 mẫu).

Chạy:
    python exp_B_width.py                      # baseline: 1000 mẫu
    python exp_B_width.py --train_size 200     # khảo sát bổ sung (file kết quả có hậu tố _n200)
    python exp_B_width.py --fast_test          # chạy thử nhanh
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
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

DEFAULT_WIDTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64]
DEFAULT_SEEDS = [42, 123, 999]


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
    hidden_units = hidden_units or DEFAULT_WIDTHS
    seeds = seeds or DEFAULT_SEEDS
    suffix = "" if train_size == 1000 else f"_n{train_size}"

    for d in (figures_dir, logs_dir, checkpoints_dir):
        os.makedirs(d, exist_ok=True)

    print("=" * 75)
    print(" THỰC NGHIỆM B: KHẢO SÁT ẢNH HƯỞNG CỦA ĐỘ RỘNG MẠNG (SỐ NƠ-RON B) ")
    print("=" * 75)
    print(f"[*] Quy mô tập train : {train_size} | epochs={epochs} | lr={lr} | batch={batch_size}")
    print(f"[*] B                : {hidden_units}")
    print(f"[*] Seeds            : {seeds}")
    print("-" * 75)

    X_val, y_val = load_dataset_from_csv(os.path.join(data_dir, "val.csv"))
    X_test, y_test = load_dataset_from_csv(os.path.join(data_dir, "test.csv"))
    X_pool, y_pool = load_dataset_from_csv(os.path.join(data_dir, "train_pool_2000.csv"))

    val_loader = DataLoader(GaussianDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    run_records, summary_records = [], []

    for B in hidden_units:
        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)

            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X_pool), size=train_size, replace=False)
            Xtr, ytr = X_pool[idx], y_pool[idx]
            train_loader = DataLoader(GaussianDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
            train_eval_loader = DataLoader(GaussianDataset(Xtr, ytr), batch_size=256, shuffle=False)

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
                checkpoint_name=f"expB{suffix}_nodes_{B}_seed_{seed}.pt",
                log_csv_name=f"expB{suffix}_log_nodes_{B}_seed_{seed}.csv",
                verbose=False,
            )

            # Model đã được nạp lại trọng số tốt nhất theo VAL. Test chỉ đo một lần ở đây.
            train_mse = trainer.evaluate(train_eval_loader)
            test_metrics, _ = evaluate_model(trainer.model, X_test, y_test, device=trainer.device)

            run_records.append({
                "nodes_B": B,
                "seed": seed,
                "n_params": model.count_parameters(),
                "best_epoch": res["best_epoch"],
                "train_mse": train_mse,
                "val_mse": res["best_val_loss"],
                "test_mse": test_metrics["mse"],
                "test_rmse": test_metrics["rmse"],
                "test_mae": test_metrics["mae"],
                "train_time_s": res["training_time"],
            })

        sub = pd.DataFrame([r for r in run_records if r["nodes_B"] == B])
        rec = {
            "nodes_B": B,
            "n_params": int(sub["n_params"].iloc[0]),
            "train_mse_mean": sub["train_mse"].mean(), "train_mse_std": sub["train_mse"].std(ddof=0),
            "val_mse_mean": sub["val_mse"].mean(), "val_mse_std": sub["val_mse"].std(ddof=0),
            "test_mse_mean": sub["test_mse"].mean(), "test_mse_std": sub["test_mse"].std(ddof=0),
            "test_mae_mean": sub["test_mae"].mean(), "test_mae_std": sub["test_mae"].std(ddof=0),
            "best_epoch_mean": sub["best_epoch"].mean(),
        }
        rec["val_train_ratio"] = rec["val_mse_mean"] / rec["train_mse_mean"]
        summary_records.append(rec)
        print(f"B = {B:2d} | Train {rec['train_mse_mean']:.3e} | Val {rec['val_mse_mean']:.3e} | "
              f"Test {rec['test_mse_mean']:.3e} +/- {rec['test_mse_std']:.2e} | "
              f"val/train = {rec['val_train_ratio']:.2f}")

    df_runs = pd.DataFrame(run_records)
    df_sum = pd.DataFrame(summary_records)
    runs_path = os.path.join(logs_dir, f"exp_B_runs_detailed{suffix}.csv")
    sum_path = os.path.join(logs_dir, f"exp_B_summary{suffix}.csv")
    df_runs.to_csv(runs_path, index=False)
    df_sum.to_csv(sum_path, index=False)
    print(f"\n[*] Đã lưu: {runs_path}\n[*] Đã lưu: {sum_path}")

    # ---- Hình 1: Train / Val / Test MSE theo B ----
    plt.figure(figsize=(9, 5.5))
    for name, col, style, color in [
        ("Train", "train_mse", "o-", "#27ae60"),
        ("Validation", "val_mse", "^-", "#2980b9"),
        ("Test", "test_mse", "s--", "#e74c3c"),
    ]:
        m, s = df_sum[f"{col}_mean"], df_sum[f"{col}_std"]
        plt.plot(df_sum["nodes_B"], m, style, color=color, label=f"{name} MSE")
        plt.fill_between(df_sum["nodes_B"], np.maximum(m - s, 1e-12), m + s, color=color, alpha=0.12)
    plt.xscale("log"); plt.yscale("log")
    plt.xticks(df_sum["nodes_B"], [str(b) for b in df_sum["nodes_B"]])
    plt.minorticks_off()
    plt.xlabel("Số nơ-ron tầng ẩn (B)", fontsize=11, fontweight="bold")
    plt.ylabel(f"MSE (trung bình ± độ lệch chuẩn, {len(seeds)} seeds)", fontsize=11, fontweight="bold")
    plt.title(f"Thực nghiệm B: ảnh hưởng độ rộng mạng (train = {train_size} mẫu)", fontsize=11, fontweight="bold")
    plt.legend(); plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, f"exp_B_error_vs_nodes{suffix}.png")
    plt.savefig(fig_path, dpi=300); plt.close()
    print(f"[*] Đã lưu biểu đồ: {fig_path}")

    # ---- Hình 2: tỉ số val/train ----
    plt.figure(figsize=(8, 4.5))
    plt.bar([str(b) for b in df_sum["nodes_B"]], df_sum["val_train_ratio"], color="#e67e22", alpha=0.85)
    plt.axhline(1.0, color="k", ls="--", lw=0.8)
    plt.xlabel("Số nơ-ron tầng ẩn (B)", fontsize=11, fontweight="bold")
    plt.ylabel("Val MSE / Train MSE", fontsize=11, fontweight="bold")
    plt.title(f"Tỉ số tổng quát hoá theo B (train = {train_size} mẫu)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    ratio_path = os.path.join(figures_dir, f"exp_B_ratio_val_train{suffix}.png")
    plt.savefig(ratio_path, dpi=300); plt.close()
    print(f"[*] Đã lưu biểu đồ: {ratio_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_size", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--fast_test", action="store_true", help="Chạy thử nhanh (B=1,4,16; 1 seed; 20 epochs)")
    a = p.parse_args()

    if a.fast_test:
        run_experiment_b(hidden_units=[1, 4, 16], seeds=[42], epochs=20,
                         train_size=a.train_size, lr=a.lr, batch_size=a.batch_size)
    else:
        run_experiment_b(train_size=a.train_size, epochs=a.epochs, lr=a.lr, batch_size=a.batch_size)
