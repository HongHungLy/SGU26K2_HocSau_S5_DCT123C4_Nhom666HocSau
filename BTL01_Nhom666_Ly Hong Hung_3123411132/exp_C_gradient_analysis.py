"""
Script: exp_C_gradient_analysis.py
Phụ trách: Thành viên 3 (Kỹ sư Đánh giá & Độ sâu)

Mục đích: cung cấp BẰNG CHỨNG ĐỊNH LƯỢNG cho phân tích hiện tượng triệt tiêu đạo hàm
(vanishing gradient) trong Thực nghiệm C, thay vì chỉ lập luận lý thuyết.

Cách đo:
- Với mỗi số tầng ẩn C in [1, 2, 3, 4, 5] (mỗi tầng ẩn 16 nơ-ron, giống exp_C_depth.py),
  tính gradient FULL-BATCH của hàm mất mát MSE trên tập train (1000 mẫu, cùng cách chọn mẫu
  theo seed như exp_C_depth.py), rồi lấy chuẩn L2 của gradient trọng số từng tầng nn.Linear.
- Giai đoạn "init"   : ngay sau khởi tạo (chưa huấn luyện), so sánh tanh / sigmoid / relu.
- Giai đoạn "trained": sau khi huấn luyện đúng cấu hình Thực nghiệm C (tanh, Adam lr=0.005,
                       100 epochs, nạp checkpoint tốt nhất theo val loss).
- Chỉ số chính: tỉ số  ||grad tầng 1|| / ||grad tầng đầu ra||.
  Tỉ số << 1 và giảm mạnh theo C  =>  gradient bị suy giảm khi lan truyền ngược về các tầng đầu.

Đầu ra (đường dẫn tương đối):
    logs/exp_C_gradient_norms.csv     chuẩn gradient từng tầng, từng seed
    logs/exp_C_gradient_summary.csv   tổng hợp Mean ± Std theo (giai đoạn, hàm kích hoạt, C)
    figures/exp_C_gradient_norms.png  đồ thị

Chạy:
    python exp_C_gradient_analysis.py               # đầy đủ (vài phút trên CPU)
    python exp_C_gradient_analysis.py --skip_train  # chỉ đo lúc khởi tạo (vài giây)
"""

import os
import sys
import argparse
import tempfile
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# 1. ĐO GRADIENT (cần PyTorch; import trong hàm để phần tổng hợp/vẽ chạy độc lập)
# --------------------------------------------------------------------------- #
def layer_grad_norms(model, X, y):
    """Trả về (list chuẩn L2 gradient trọng số từng tầng Linear, giá trị loss) trên batch (X, y)."""
    import torch.nn as nn

    model.train()
    model.zero_grad()
    loss = nn.functional.mse_loss(model(X), y)
    loss.backward()
    norms = [
        m.weight.grad.norm(2).item()
        for m in model.network
        if isinstance(m, nn.Linear)
    ]
    model.zero_grad()
    return norms, loss.item()


def run_measurements(
    data_dir, layers_list, seeds, width, train_size,
    activations, lr, batch_size, epochs, skip_train,
):
    import torch
    from torch.utils.data import DataLoader
    from src.dataset import load_dataset_from_csv, GaussianDataset
    from src.model import FlexibleMLP
    from src.trainer import Trainer

    X_pool, y_pool = load_dataset_from_csv(os.path.join(data_dir, "train_pool_2000.csv"))
    X_val, y_val = load_dataset_from_csv(os.path.join(data_dir, "val.csv"))
    val_loader = DataLoader(GaussianDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    records = []

    def add(stage, act, C, seed, norms, loss):
        for k, g in enumerate(norms, start=1):
            records.append({
                "stage": stage, "activation": act, "layers_C": C, "seed": seed,
                "layer_idx": k, "n_linear": len(norms), "grad_norm": g, "loss": loss,
            })

    for seed in seeds:
        # Chọn mẫu train giống hệt exp_C_depth.py
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_pool), size=train_size, replace=False)
        X_sub, y_sub = X_pool[idx], y_pool[idx]

        # --- Giai đoạn 1: ngay sau khởi tạo ---
        for act in activations:
            for C in layers_list:
                torch.manual_seed(seed)
                model = FlexibleMLP(input_dim=2, hidden_dims=[width] * C, output_dim=1, activation=act)
                Xt, yt = torch.from_numpy(X_sub), torch.from_numpy(y_sub)
                norms, loss = layer_grad_norms(model, Xt, yt)
                add("init", act, C, seed, norms, loss)
        print(f"[seed {seed}] đã đo gradient lúc khởi tạo ({', '.join(activations)})")

        # --- Giai đoạn 2: sau huấn luyện (tanh, đúng cấu hình exp_C_depth.py) ---
        if not skip_train:
            for C in layers_list:
                torch.manual_seed(seed)
                train_loader = DataLoader(
                    GaussianDataset(X_sub, y_sub), batch_size=batch_size, shuffle=True
                )
                model = FlexibleMLP(input_dim=2, hidden_dims=[width] * C, output_dim=1, activation="tanh")
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                with tempfile.TemporaryDirectory() as tmp:
                    trainer = Trainer(
                        model=model, optimizer=optimizer,
                        checkpoint_dir=os.path.join(tmp, "ckpt"),
                        log_dir=os.path.join(tmp, "log"),
                    )
                    trainer.fit(
                        train_loader=train_loader, val_loader=val_loader, epochs=epochs,
                        checkpoint_name="tmp.pt", log_csv_name="tmp.csv", verbose=False,
                    )
                dev = next(trainer.model.parameters()).device
                Xt = torch.from_numpy(X_sub).to(dev)
                yt = torch.from_numpy(y_sub).to(dev)
                norms, loss = layer_grad_norms(trainer.model, Xt, yt)
                add("trained", "tanh", C, seed, norms, loss)
            print(f"[seed {seed}] đã đo gradient sau huấn luyện (tanh, C = {layers_list})")

    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# 2. TỔNG HỢP & VẼ (không cần PyTorch)
# --------------------------------------------------------------------------- #
def summarize(df_norms):
    """Mean ± Std qua các seed của: gradient tầng 1, gradient tầng đầu ra, tỉ số tầng 1 / đầu ra."""
    rows = []
    keys = ["stage", "activation", "layers_C", "seed"]
    per_seed = []
    for (stage, act, C, seed), g in df_norms.groupby(keys):
        g = g.sort_values("layer_idx")
        first, last = g["grad_norm"].iloc[0], g["grad_norm"].iloc[-1]
        per_seed.append({
            "stage": stage, "activation": act, "layers_C": C, "seed": seed,
            "grad_first": first, "grad_output": last,
            "ratio_first_over_output": first / last if last > 0 else np.nan,
            "loss": g["loss"].iloc[0],
        })
    df_seed = pd.DataFrame(per_seed)
    for (stage, act, C), g in df_seed.groupby(["stage", "activation", "layers_C"]):
        rows.append({
            "stage": stage, "activation": act, "layers_C": C, "n_seeds": len(g),
            "grad_first_mean": g["grad_first"].mean(), "grad_first_std": g["grad_first"].std(ddof=0),
            "grad_output_mean": g["grad_output"].mean(), "grad_output_std": g["grad_output"].std(ddof=0),
            "ratio_mean": g["ratio_first_over_output"].mean(),
            "ratio_std": g["ratio_first_over_output"].std(ddof=0),
            "loss_mean": g["loss"].mean(),
        })
    return pd.DataFrame(rows)


def plot_gradient(df_norms, df_sum, fig_path):
    has_trained = (df_norms["stage"] == "trained").any()
    n_panels = 3 if has_trained else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.4 * n_panels, 4.8))
    cmap = plt.get_cmap("viridis")

    def per_layer_panel(ax, stage, title):
        sub = df_norms[(df_norms["stage"] == stage) & (df_norms["activation"] == "tanh")]
        Cs = sorted(sub["layers_C"].unique())
        for i, C in enumerate(Cs):
            g = sub[sub["layers_C"] == C].groupby("layer_idx")["grad_norm"].mean()
            ax.plot(g.index, g.values, "o-", color=cmap(i / max(len(Cs) - 1, 1)), label=f"C = {C}")
        ax.set_yscale("log")
        ax.set_xlabel("Tầng Linear (1 = gần đầu vào, cuối = tầng đầu ra)", fontsize=9)
        ax.set_ylabel("‖∇W‖₂ (trung bình 3 seeds)", fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5, which="both")
        ax.legend(fontsize=8)

    per_layer_panel(axes[0], "init", "(a) Gradient từng tầng lúc khởi tạo (tanh)")

    ax = axes[1]
    markers = {"tanh": "o-", "sigmoid": "s--", "relu": "^:"}
    for act in sorted(df_sum[df_sum["stage"] == "init"]["activation"].unique()):
        s = df_sum[(df_sum["stage"] == "init") & (df_sum["activation"] == act)].sort_values("layers_C")
        ax.errorbar(s["layers_C"], s["ratio_mean"], yerr=s["ratio_std"],
                    fmt=markers.get(act, "o-"), capsize=3, linewidth=1.8, label=act)
    ax.set_yscale("log")
    ax.set_xlabel("Số tầng ẩn C", fontsize=10)
    ax.set_ylabel("‖∇W tầng 1‖ / ‖∇W tầng đầu ra‖", fontsize=10)
    ax.set_title("(b) Tỉ số gradient tầng đầu / tầng cuối lúc khởi tạo", fontsize=10, fontweight="bold")
    ax.set_xticks(sorted(df_sum["layers_C"].unique()))
    ax.grid(True, linestyle="--", alpha=0.5, which="both")
    ax.legend(fontsize=9, title="Hàm kích hoạt")

    if has_trained:
        per_layer_panel(axes[2], "trained", "(c) Gradient từng tầng sau huấn luyện (tanh)")

    plt.suptitle("Thực nghiệm C: khảo sát triệt tiêu đạo hàm theo độ sâu", fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
    plt.savefig(fig_path, dpi=300)
    plt.close()


def print_report(df_sum):
    print("\n" + "=" * 78)
    print(" TỈ SỐ  ||grad tầng 1|| / ||grad tầng đầu ra||  (Mean ± Std qua seeds)")
    print("=" * 78)
    for (stage, act), g in df_sum.groupby(["stage", "activation"]):
        print(f"\n[{stage} | {act}]")
        for _, r in g.sort_values("layers_C").iterrows():
            print(f"  C = {int(r['layers_C'])} | ratio = {r['ratio_mean']:.4f} ± {r['ratio_std']:.4f}"
                  f" | grad tầng 1 = {r['grad_first_mean']:.3e} | grad đầu ra = {r['grad_output_mean']:.3e}")


def main():
    p = argparse.ArgumentParser(description="Phân tích gradient theo độ sâu (Thực nghiệm C)")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--logs_dir", default="logs")
    p.add_argument("--figures_dir", default="figures")
    p.add_argument("--width", type=int, default=16, help="Số nơ-ron mỗi tầng ẩn (giống exp_C_depth.py)")
    p.add_argument("--train_size", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--skip_train", action="store_true", help="Chỉ đo lúc khởi tạo")
    a = p.parse_args()

    os.makedirs(a.logs_dir, exist_ok=True)
    os.makedirs(a.figures_dir, exist_ok=True)

    df_norms = run_measurements(
        data_dir=a.data_dir, layers_list=[1, 2, 3, 4, 5], seeds=[42, 123, 999],
        width=a.width, train_size=a.train_size, activations=["tanh", "sigmoid", "relu"],
        lr=a.lr, batch_size=a.batch_size, epochs=a.epochs, skip_train=a.skip_train,
    )
    norms_path = os.path.join(a.logs_dir, "exp_C_gradient_norms.csv")
    df_norms.to_csv(norms_path, index=False)

    df_sum = summarize(df_norms)
    sum_path = os.path.join(a.logs_dir, "exp_C_gradient_summary.csv")
    df_sum.to_csv(sum_path, index=False)

    fig_path = os.path.join(a.figures_dir, "exp_C_gradient_norms.png")
    plot_gradient(df_norms, df_sum, fig_path)

    print_report(df_sum)
    print(f"\n[*] Đã lưu: {norms_path}\n[*] Đã lưu: {sum_path}\n[*] Đã lưu: {fig_path}")


if __name__ == "__main__":
    main()
