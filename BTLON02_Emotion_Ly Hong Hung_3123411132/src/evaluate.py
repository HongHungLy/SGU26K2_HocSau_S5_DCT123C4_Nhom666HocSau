"""
Đánh giá chuyên sâu mô hình tốt nhất trên PublicTest & PrivateTest của FER2013.

Phụ trách: Thành viên 3 (Vai trò C).

Chạy từ THƯ MỤC GỐC của project (nơi có src/, checkpoints/, data/):
    python -m src.evaluate                                   # tự tìm checkpoints/best_model.pth
    python -m src.evaluate --ckpt checkpoints/Exp1_6_MediumCNN_Aug_ReLU_WeightedCE_best.pth

Đầu ra:
    figures/confusion_matrix_public.png, figures/confusion_matrix_private.png
    figures/cases_true_{public,private}.png, figures/cases_wrong_{public,private}.png
    logs/classification_report_{public,private}.txt
    logs/tv3_eval_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .dataset import get_dataloaders
from .models_advanced import load_model_from_checkpoint
from .models_baseline import MediumCNN
from .utils import EMOTION_LABELS, get_device, set_seed


# ----------------------------------------------------------------------------------------------
# Nạp mô hình (hỗ trợ cả best_model.pth của TV2 và checkpoint baseline của TV1)
# ----------------------------------------------------------------------------------------------
def resolve_ckpt(path: str | None) -> Path:
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Không thấy checkpoint: {p}")
        return p
    for cand in [Path("checkpoints/best_model.pth"), Path("best_model.pth")]:
        if cand.exists():
            return cand
    found = sorted(Path("checkpoints").glob("*.pth")) if Path("checkpoints").exists() else []
    if found:
        print(f"[!] Không có best_model.pth, dùng tạm: {found[0]}")
        return found[0]
    raise FileNotFoundError("Không tìm thấy file .pth nào trong ./checkpoints. Dùng --ckpt <đường dẫn>.")


def disable_inplace(model: nn.Module) -> None:
    """Tắt inplace (ReLU...) để hook / Grad-CAM không bị lỗi autograd."""
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = False


def load_any_model(path: Path, device: torch.device) -> Tuple[nn.Module, int, int, str]:
    """Trả về (model, img_size, in_channels, tên)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "arch" in ckpt:          # best_model.pth "tự mô tả" của TV2
        model, ckpt = load_model_from_checkpoint(path, device)
        name, img_size, in_ch = ckpt["arch"], int(ckpt.get("img_size", 48)), int(ckpt.get("in_channels", 1))
    else:                                                   # checkpoint baseline của TV1
        model = MediumCNN(num_classes=7, in_channels=1, activation="relu").to(device)
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(sd)
        name, img_size, in_ch = "MediumCNN_baseline", 48, 1
    model.eval()
    disable_inplace(model)
    return model, img_size, in_ch, name


def get_eval_loaders(img_size: int, in_channels: int, batch_size: int = 128):
    """Trả về {'public': loader, 'private': loader} (không augmentation, không shuffle)."""
    _, val_loader, test_loader, _ = get_dataloaders(
        batch_size=batch_size, use_aug=False, img_size=img_size, in_channels=in_channels, num_workers=0
    )
    return {"public": val_loader, "private": test_loader}


@torch.no_grad()
def predict(model: nn.Module, loader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, ps, probs = [], [], []
    for x, y in loader:
        out = torch.softmax(model(x.to(device)), dim=1).cpu()
        probs.append(out)
        ps.append(out.argmax(1))
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(probs).numpy()


def to_display(x: torch.Tensor, in_channels: int) -> np.ndarray:
    """Tensor đã chuẩn hóa (C,H,W) -> ảnh xám [0,1] để hiển thị."""
    if in_channels == 3:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (x.cpu() * std + mean)[0]
    else:
        img = x.cpu()[0] * 0.5 + 0.5
    return img.clamp(0, 1).numpy()


# ----------------------------------------------------------------------------------------------
# Vẽ hình
# ----------------------------------------------------------------------------------------------
def plot_confusion(y, p, title: str, path: Path) -> np.ndarray:
    cm = confusion_matrix(y, p, labels=range(len(EMOTION_LABELS)))
    cm_pct = cm / cm.sum(1, keepdims=True).clip(min=1) * 100
    n = len(EMOTION_LABELS)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(EMOTION_LABELS, rotation=45, ha="right"); ax.set_yticklabels(EMOTION_LABELS)
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế"); ax.set_title(title)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm_pct[i, j]:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if cm_pct[i, j] > 50 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="% theo hàng")
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()
    return cm


def plot_cases(dataset, idxs, y, p, in_channels, title, path, rows=4, cols=8) -> None:
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.2))
    for a in axes.ravel():
        a.axis("off")
    for a, i in zip(axes.ravel(), idxs[: rows * cols]):
        a.imshow(to_display(dataset[i][0], in_channels), cmap="gray")
        ok = y[i] == p[i]
        a.set_title(f"{EMOTION_LABELS[p[i]]} ({EMOTION_LABELS[y[i]]})", fontsize=8,
                    color="green" if ok else "red")
    plt.suptitle(title + "   [nhãn: Dự đoán (Thực tế)]", fontsize=12)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


# ----------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--fig-dir", default="figures")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--tag", default="", help="hậu tố tên file, vd: _tv2 để không ghi đè kết quả cũ")
    args = ap.parse_args()

    set_seed(42)
    fig_dir, log_dir = Path(args.fig_dir), Path(args.log_dir)
    fig_dir.mkdir(exist_ok=True, parents=True); log_dir.mkdir(exist_ok=True, parents=True)

    device = get_device()
    ckpt_path = resolve_ckpt(args.ckpt)
    model, img_size, in_ch, name = load_any_model(ckpt_path, device)
    print(f"Model: {name} | img_size={img_size} | in_channels={in_ch} | ckpt={ckpt_path}")

    loaders = get_eval_loaders(img_size, in_ch, args.batch_size)
    rng = np.random.default_rng(42)
    rows: list[Dict[str, Any]] = []

    for split, loader in loaders.items():
        label = "PublicTest" if split == "public" else "PrivateTest"
        y, p, probs = predict(model, loader, device)
        acc = float((y == p).mean())
        macro_f1 = float(f1_score(y, p, average="macro"))
        report = classification_report(y, p, target_names=EMOTION_LABELS, digits=4, zero_division=0)
        print(f"\n===== {label}: Acc={acc:.4f}  MacroF1={macro_f1:.4f}  (N={len(y)}) =====\n{report}")
        (log_dir / f"classification_report_{split}{args.tag}.txt").write_text(
            f"Model: {name} | ckpt: {ckpt_path}\n{label}: Acc={acc:.4f} MacroF1={macro_f1:.4f}\n\n{report}",
            encoding="utf-8",
        )

        cm = plot_confusion(y, p, f"Confusion Matrix (%) - {label} | {name}",
                            fig_dir / f"confusion_matrix_{split}{args.tag}.png")

        off = cm.copy(); np.fill_diagonal(off, 0)
        top = sorted(((int(off[i, j]), EMOTION_LABELS[i], EMOTION_LABELS[j])
                      for i in range(len(off)) for j in range(len(off))), reverse=True)[:5]
        print("Top 5 cặp nhầm lẫn (số ảnh, Thật -> Đoán):", top)

        ds = loader.dataset
        right, wrong = np.where(y == p)[0], np.where(y != p)[0]
        plot_cases(ds, rng.choice(right, min(32, len(right)), replace=False), y, p, in_ch,
                   f"True Cases - {label}", fig_dir / f"cases_true_{split}{args.tag}.png")
        plot_cases(ds, rng.choice(wrong, min(32, len(wrong)), replace=False), y, p, in_ch,
                   f"Wrong Cases - {label}", fig_dir / f"cases_wrong_{split}{args.tag}.png")

        recalls = cm.diagonal() / cm.sum(1).clip(min=1)
        rows.append({"model": name, "split": label, "acc": acc, "macro_f1": macro_f1,
                     **{f"recall_{c}": float(r) for c, r in zip(EMOTION_LABELS, recalls)},
                     "top_confusions": str(top)})

    pd.DataFrame(rows).to_csv(log_dir / f"tv3_eval_summary{args.tag}.csv", index=False, encoding="utf-8-sig")
    print(f"\nĐã lưu hình vào {fig_dir}/ và log vào {log_dir}/")


if __name__ == "__main__":
    main()
