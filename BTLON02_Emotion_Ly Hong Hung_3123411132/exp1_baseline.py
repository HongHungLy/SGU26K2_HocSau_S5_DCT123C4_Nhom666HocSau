"""
Script chạy Thực nghiệm Nhóm 1 (Thành viên 1 - Lý Hồng Hưng - 3123411132):
Khảo sát tác động của:
  - Exp1_1: SmallCNN (Không Augmentation, ReLU, Standard CE)   -> Baseline 0
  - Exp1_2: SmallCNN + Data Augmentation (ReLU, Standard CE)   -> Tác động Data Augmentation
  - Exp1_3: MediumCNN + Data Augmentation (ReLU, Standard CE)  -> So sánh độ sâu mạng
  - Exp1_4: MediumCNN + Data Augmentation (LeakyReLU, Std CE)  -> So sánh hàm kích hoạt
  - Exp1_5: MediumCNN + Data Augmentation (Sigmoid, Std CE)    -> Minh chứng Vanishing Gradient
  - Exp1_6: MediumCNN + Data Augmentation (ReLU, Weighted CE)  -> Xử lý mất cân bằng lớp (Disgust)

Cách chạy:
  python exp1_baseline.py --epochs 45 --batch_size 64
  python exp1_baseline.py --exp Exp1_1 Exp1_2 --epochs 30
  python exp1_baseline.py --fast_test
"""

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn

from src.dataset import get_dataloaders
from src.models_baseline import build_baseline_model
from src.trainer import fit
from src.utils import (
    compute_class_weights,
    ensure_dir,
    get_device,
    plot_experiment_comparison,
    plot_training_curves,
    set_seed,
)

BATCH1_EXPERIMENTS: List[Dict] = [
    {
        "exp_name": "Exp1_1_SmallCNN_NoAug_ReLU",
        "arch": "small_cnn",
        "use_aug": False,
        "activation": "relu",
        "weighted_loss": False,
        "note": "Baseline 0 (không Augmentation)",
    },
    {
        "exp_name": "Exp1_2_SmallCNN_Aug_ReLU",
        "arch": "small_cnn",
        "use_aug": True,
        "activation": "relu",
        "weighted_loss": False,
        "note": "Đánh giá hiệu quả Data Augmentation",
    },
    {
        "exp_name": "Exp1_3_MediumCNN_Aug_ReLU",
        "arch": "medium_cnn",
        "use_aug": True,
        "activation": "relu",
        "weighted_loss": False,
        "note": "So sánh độ sâu mạng (Small vs Medium)",
    },
    {
        "exp_name": "Exp1_4_MediumCNN_Aug_LeakyReLU",
        "arch": "medium_cnn",
        "use_aug": True,
        "activation": "leaky_relu",
        "weighted_loss": False,
        "note": "So sánh hàm kích hoạt LeakyReLU",
    },
    {
        "exp_name": "Exp1_5_MediumCNN_Aug_Sigmoid",
        "arch": "medium_cnn",
        "use_aug": True,
        "activation": "sigmoid",
        "weighted_loss": False,
        "note": "So sánh hàm kích hoạt Sigmoid (vanishing gradient)",
    },
    {
        "exp_name": "Exp1_6_MediumCNN_Aug_ReLU_WeightedCE",
        "arch": "medium_cnn",
        "use_aug": True,
        "activation": "relu",
        "weighted_loss": True,
        "note": "Xử lý mất cân bằng dữ liệu bằng Weighted CE (sqrt_balanced)",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chạy Thực nghiệm Nhóm 1 (TV1 - FER2013 PyTorch)")
    parser.add_argument("--data_dir", type=str, default=None, help="Đường dẫn tới fer2013.csv hoặc thư mục dữ liệu (mặc định tự dò ./data hoặc /kaggle/input)")
    parser.add_argument("--epochs", type=int, default=45, help="Số epoch tối đa cho mỗi thực nghiệm")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate khởi tạo cho Adam")
    parser.add_argument("--patience", type=int, default=10, help="EarlyStopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=2, help="Số worker của DataLoader")
    parser.add_argument("--exp", nargs="+", default=None, help="Chỉ chạy các mã thực nghiệm khớp tiền tố (VD: Exp1_1 Exp1_2)")
    parser.add_argument("--fast_test", action="store_true", help="Chạy nhanh 2 epoch để kiểm tra code")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fast_test:
        args.epochs = 2
        args.patience = 2

    device = get_device()
    ensure_dir("logs")
    ensure_dir("figures")
    ensure_dir("checkpoints")

    selected = BATCH1_EXPERIMENTS
    if args.exp:
        selected = [
            cfg for cfg in BATCH1_EXPERIMENTS
            if any(cfg["exp_name"].startswith(prefix) for prefix in args.exp)
        ]
        if not selected:
            raise ValueError(f"Không tìm thấy thực nghiệm khớp với {args.exp}")

    all_summaries = []
    summary_csv = Path("logs/exp1_summary.csv")

    for cfg in selected:
        print("\n" + "=" * 72)
        print(f"ĐANG CHẠY: {cfg['exp_name']} | {cfg['note']}")
        print("=" * 72)

        set_seed(args.seed)
        train_loader, val_loader, test_loader, info = get_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            use_aug=cfg["use_aug"],
            img_size=48,
            in_channels=1,
            num_workers=args.num_workers,
            seed=args.seed,
        )

        model = build_baseline_model(
            arch=cfg["arch"],
            num_classes=info["num_classes"],
            in_channels=1,
            activation=cfg["activation"],
        )

        if cfg["weighted_loss"]:
            weights = compute_class_weights(info["train_labels"], num_classes=7, method="sqrt_balanced").to(device)
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6
        )

        summary = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            monitor="val_acc",
            exp_name=cfg["exp_name"],
            logs_dir="logs",
            checkpoints_dir="checkpoints",
        )
        summary.update({
            "arch": cfg["arch"],
            "use_aug": cfg["use_aug"],
            "activation": cfg["activation"],
            "weighted_loss": cfg["weighted_loss"],
            "note": cfg["note"],
        })
        all_summaries.append(summary)

        # Vẽ đồ thị Accuracy & Loss cho từng thực nghiệm
        plot_training_curves(
            csv_path=summary["log_csv"],
            save_path=Path("figures") / f"{cfg['exp_name']}_curves.png",
            title=f"{cfg['exp_name']} (Best Val: {summary['best_val_acc']*100:.2f}% | Test: {summary['test_acc']*100:.2f}%)",
        )

        # Lưu bảng tổng hợp sau mỗi thực nghiệm để tránh mất dữ liệu giữa chừng
        pd.DataFrame(all_summaries).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    # Vẽ đồ thị so sánh tổng hợp Nhóm 1
    if len(all_summaries) >= 2:
        plot_experiment_comparison(
            summary_csv=summary_csv,
            save_path="figures/exp1_baseline_comparison.png",
        )

    print("\nHoàn thành Thực nghiệm Nhóm 1! Bảng tổng hợp:")
    df_sum = pd.read_csv(summary_csv)
    cols = ["exp_name", "best_epoch", "best_val_acc", "test_acc", "test_macro_f1", "test_disgust_recall", "train_time_sec"]
    print(df_sum[cols].to_string(index=False))


if __name__ == "__main__":
    main()
