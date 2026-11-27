"""
Script CLI Khám phá dữ liệu (EDA) & Minh họa Data Augmentation cho bộ Kaggle FER2013.
Phụ trách: Thành viên 1 - Lý Hồng Hưng (3123411132).

Cách chạy:
  python eda_fer2013.py
  python eda_fer2013.py --data_dir data/fer2013.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.dataset import discover_fer2013_source, get_dataloaders, get_transforms
from src.utils import EMOTION_LABELS, ensure_dir, set_seed


def count_per_class(labels: np.ndarray) -> list[int]:
    return [int((labels == i).sum()) for i in range(len(EMOTION_LABELS))]


def plot_image_grid(loader, title: str, save_path: Path, rows: int = 4, cols: int = 8) -> None:
    images, labels = next(iter(loader))
    n = min(rows * cols, len(images))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.8))
    for i, ax in enumerate(axes.flat):
        if i < n:
            img = images[i].squeeze().numpy() * 0.5 + 0.5
            ax.imshow(np.clip(img, 0, 1), cmap="gray")
            ax.set_title(EMOTION_LABELS[int(labels[i])], fontsize=9, fontweight="bold")
        ax.axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy EDA cho bộ dữ liệu Kaggle FER2013 (TV1)")
    parser.add_argument("--data_dir", type=str, default=None, help="Đường dẫn tới fer2013.csv hoặc thư mục ảnh")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir("figures")
    ensure_dir("logs")

    source = discover_fer2013_source(args.data_dir)
    print("Nguồn dữ liệu phát hiện:", source)

    train_loader_raw, val_loader, _, info = get_dataloaders(
        data_dir=args.data_dir, batch_size=64, use_aug=False, img_size=48, in_channels=1, num_workers=0, seed=args.seed
    )
    train_loader_aug, _, _, _ = get_dataloaders(
        data_dir=args.data_dir, batch_size=64, use_aug=True, img_size=48, in_channels=1, num_workers=0, seed=args.seed
    )

    tr_cnt = count_per_class(info["train_labels"])
    va_cnt = count_per_class(info["val_labels"])
    te_cnt = count_per_class(info["test_labels"])

    df_dist = pd.DataFrame({
        "Label_ID": list(range(7)),
        "Emotion": EMOTION_LABELS,
        "Training": tr_cnt,
        "PublicTest_Val": va_cnt,
        "PrivateTest_Test": te_cnt,
    })
    df_dist["Total"] = df_dist["Training"] + df_dist["PublicTest_Val"] + df_dist["PrivateTest_Test"]
    df_dist["Train_Ratio_%"] = (df_dist["Training"] / df_dist["Training"].sum() * 100).round(2)
    df_dist.to_csv("logs/eda_class_distribution.csv", index=False, encoding="utf-8-sig")
    print("\nBảng phân bố 7 lớp cảm xúc:")
    print(df_dist.to_string(index=False))

    # 1. Biểu đồ Histogram 3 tập
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    splits = [
        ("Training Dataset", tr_cnt, "#1B365D"),
        ("PublicTest (Validation)", va_cnt, "#2B5B84"),
        ("PrivateTest (Testing)", te_cnt, "#4A7C9B"),
    ]
    for ax, (title, counts, color) in zip(axes, splits):
        colors = ["#C00000" if i == 1 else color for i in range(7)]
        bars = ax.bar(EMOTION_LABELS, counts, color=colors, edgecolor="black", alpha=0.9)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{int(h)}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_title(f"{title} (N = {sum(counts):,})", fontsize=12, fontweight="bold")
        ax.set_ylabel("Số lượng ảnh")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Phân bố 7 lớp cảm xúc trên bộ dữ liệu Kaggle FER2013 (Đỏ: lớp thiểu số Disgust)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/eda_histogram_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Lưới ảnh mẫu 4x8
    plot_image_grid(train_loader_raw, "Ảnh mẫu tập Training (48x48 Grayscale)", Path("figures/eda_train_samples.png"))
    plot_image_grid(val_loader, "Ảnh mẫu tập PublicTest / Validation", Path("figures/eda_val_samples.png"))
    plot_image_grid(train_loader_aug, "Ảnh mẫu tập Training sau Data Augmentation", Path("figures/eda_augmented_batch.png"))

    # 3. Minh họa 1 ảnh gốc vs 6 ảnh Augmented
    raw_ds = train_loader_raw.dataset
    aug_tf = get_transforms(is_train=True, use_aug=True, img_size=48, in_channels=1)
    sample_idx = int(np.where(info["train_labels"] == 3)[0][0])
    img_tensor, lbl = raw_ds[sample_idx]
    orig_np = np.clip((img_tensor.squeeze().numpy() * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(orig_np, mode="L")

    fig, axes = plt.subplots(1, 7, figsize=(15, 2.6))
    axes[0].imshow(orig_np, cmap="gray")
    axes[0].set_title("Ảnh gốc\n(Original)", fontsize=10, fontweight="bold", color="#1B365D")
    axes[0].axis("off")
    for k in range(1, 7):
        aug_t = aug_tf(pil_img)
        aug_np = np.clip(aug_t.squeeze().numpy() * 0.5 + 0.5, 0, 1)
        axes[k].imshow(aug_np, cmap="gray")
        axes[k].set_title(f"Aug #{k}", fontsize=10)
        axes[k].axis("off")
    fig.suptitle(f"Minh họa Data Augmentation trên cùng 1 mẫu (Nhãn: {EMOTION_LABELS[lbl]})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/eda_augmentation_demo.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nĐã hoàn tất EDA! Toàn bộ hình lưu tại figures/ và bảng lưu tại logs/eda_class_distribution.csv")


if __name__ == "__main__":
    main()
