"""
Script chạy Thực nghiệm Nhóm 2 (Vai trò B - Thành viên 2): Kiến trúc nâng cao, Transfer Learning,
Tối ưu hóa & Regularization, Fail Cases. Tái sử dụng nguyên khung của Thành viên 1
(src/dataset.py, src/trainer.py, src/utils.py) - KHÔNG sửa các file đó.

CÁC NHÓM THỰC NGHIỆM (chọn bằng --group A B C D F G hoặc --exp <tiền tố tên>):
  A  Kiến trúc    : LargeCNN, ResNetMini, ResNet18/34 & EfficientNet-B0 (pretrained), ResNet18 frozen.
                    Dùng đúng công thức huấn luyện của Exp1_6 (Adam 1e-3, ReduceLROnPlateau, Weighted CE,
                    45 epoch) để so sánh công bằng với mốc 68.51% của TV1.
  B  Optimizer    : SGD+Momentum vs Adam vs AdamW        (cố định kiến trúc --hp_arch, scheduler Cosine).
  C  LR/Scheduler : ReduceLROnPlateau vs CosineAnnealingLR, quét LR x3 / ÷3 (theo --hp_opt).
  D  Regularization: Dropout, Weight Decay, Label Smoothing (một yếu tố mỗi lần, theo --hp_*).
  F  Fail cases   : VGG16 pretrained fine-tune 48x48 không regularization (overfit nặng), LargeCNN không
                    regularization, VGG16 from-scratch (không hội tụ) + bản VGG16 đã khắc phục.
  G  Final        : Chạy dài (80 epoch) với cấu hình tốt nhất tự chọn qua --final_*.

QUY TRÌNH KHUYẾN NGHỊ (mỗi bước xem kết quả rồi mới chọn tham số cho bước sau):
  python exp2_advanced.py --group A F                       # Bước 1: kiến trúc + fail cases
  python exp2_advanced.py --group B --hp_arch resnet_mini   # Bước 2: so sánh optimizer
  python exp2_advanced.py --group C --hp_arch resnet_mini --hp_opt adamw
  python exp2_advanced.py --group D --hp_arch resnet_mini --hp_opt adamw --hp_sched cosine
  python exp2_advanced.py --group G --hp_arch resnet_mini --hp_opt adamw --final_dropout default \
                          --final_wd default --final_ls 0.05
  python exp2_advanced.py --finalize                        # Chọn best model theo Val Acc -> best_model.pth

Kiểm tra nhanh code:  python exp2_advanced.py --group A --fast_test --data_dir <csv>
Bảng tổng hợp được GỘP (merge) theo tên thí nghiệm vào logs/exp2_summary.csv nên có thể chạy từng nhóm ở
các phiên Kaggle khác nhau; thí nghiệm đã chạy xong sẽ được bỏ qua (dùng --force để chạy lại).
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset import get_dataloaders
from src.models_advanced import build_model, save_deployable_checkpoint
from src.trainer import fit
from src.utils import (
    compute_class_weights,
    count_parameters,
    ensure_dir,
    get_device,
    plot_training_curves,
    set_seed,
)

SUMMARY_CSV = Path("logs/exp2_summary.csv")

# ----------------------------------------------------------------------------------------------
# Loss: Weighted CE + Label Smoothing, nhưng val/test loss luôn tính KHÔNG smoothing
# ----------------------------------------------------------------------------------------------
class SmoothedWeightedCE(nn.Module):
    """
    CrossEntropy có trọng số lớp (sqrt_balanced) và Label Smoothing.

    Lưu ý quan trọng: `src/trainer.py` dùng CHUNG một criterion cho cả train lẫn val. Nếu bật label smoothing
    cho cả val thì Val Loss bị cộng thêm một hằng số và KHÔNG so sánh được với các thực nghiệm khác
    (kể cả Nhóm 1). Vì vậy label smoothing chỉ áp dụng khi autograd đang bật (tức đang train);
    `evaluate_epoch` chạy dưới `torch.no_grad()` nên val/test loss là Cross-Entropy thuần (vẫn giữ trọng số lớp
    như Exp1_6) -> các đường cong Loss giữa các thực nghiệm so sánh trực tiếp được.
    """

    def __init__(self, weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ls = self.label_smoothing if torch.is_grad_enabled() else 0.0
        return F.cross_entropy(logits.float(), target, weight=self.weight, label_smoothing=ls)


# ----------------------------------------------------------------------------------------------
# Cấu hình thí nghiệm
# ----------------------------------------------------------------------------------------------
DEFAULT_LR = {"sgd": 0.03, "adam": 1e-3, "adamw": 1e-3}
DEFAULT_WD = {"sgd": 5e-4, "adam": 1e-4, "adamw": 1e-2}
HIGH_WD = {"sgd": 5e-3, "adam": 1e-3, "adamw": 1e-1}


def dropout_kwargs(arch: str, level: str) -> Dict[str, Any]:
    """Quy đổi mức dropout 'none' | 'default' | 'strong' sang tham số của từng kiến trúc."""
    table = {
        "large_cnn": {
            "none": dict(block_dropouts=(0, 0, 0, 0), fc_dropout=0.0),
            "default": dict(block_dropouts=(0.1, 0.2, 0.3, 0.4), fc_dropout=0.5),
            "strong": dict(block_dropouts=(0.2, 0.3, 0.4, 0.5), fc_dropout=0.6),
        },
        "resnet_mini": {
            "none": dict(block_dropout=0.0, fc_dropout=0.0),
            "default": dict(block_dropout=0.1, fc_dropout=0.3),
            "strong": dict(block_dropout=0.2, fc_dropout=0.5),
        },
    }
    if arch in table:
        return dict(table[arch][level])
    return {"dropout": {"none": 0.0, "default": 0.3, "strong": 0.5}[level]}  # transfer learning


def arch_setup(arch: str, level: str, tl_size: int, width: int = 64) -> Dict[str, Any]:
    """model_kwargs + kích thước/số kênh đầu vào phù hợp với kiến trúc (scratch: 48x48x1, pretrained: tl_size x3)."""
    mk = dropout_kwargs(arch, level)
    if arch == "resnet_mini" and width != 64:
        mk["base_width"] = width
    if arch in ("large_cnn", "resnet_mini"):
        return dict(model_kwargs=mk, in_channels=1, img_size=48)
    mk["pretrained"] = True
    return dict(model_kwargs=mk, in_channels=3, img_size=tl_size, backbone_lr_mult=0.1)


def make_cfg(**kw: Any) -> Dict[str, Any]:
    """Cấu hình mặc định = công thức Exp1_6 của TV1; mỗi thí nghiệm chỉ ghi đè phần khác biệt."""
    cfg: Dict[str, Any] = dict(
        exp_name="", group="", note="",
        arch="resnet_mini", model_kwargs={}, img_size=48, in_channels=1,
        use_aug=True, weighted_loss=True, label_smoothing=0.0,
        opt="adam", lr=1e-3, weight_decay=1e-4, momentum=0.9, backbone_lr_mult=1.0,
        sched="plateau", epochs=45, patience=10, batch_size=64,
        fail_case=False, keep_ckpt=True, optional=False,
    )
    cfg.update(kw)
    return cfg


def build_experiments(args: argparse.Namespace) -> List[Dict[str, Any]]:
    exps: List[Dict[str, Any]] = []
    E = args.epochs  # số epoch cho nhóm B, C, D

    # ---------------- Nhóm A: Kiến trúc (công thức Exp1_6, 45 epoch) ----------------
    tl_common = dict(in_channels=3, img_size=args.tl_size, backbone_lr_mult=0.1, group="A")
    exps += [
        make_cfg(exp_name="Exp2_A1_LargeCNN", group="A", arch="large_cnn",
                 model_kwargs=dropout_kwargs("large_cnn", "default"),
                 note="VGG-style 4 block 64-128-256-512, Dropout2d tăng dần"),
        make_cfg(exp_name="Exp2_A2_ResNetMini", group="A", arch="resnet_mini",
                 model_kwargs=dropout_kwargs("resnet_mini", "default"),
                 note="Residual CNN 4 stage 64-128-256-512 cho ảnh 48x48"),
        make_cfg(exp_name="Exp2_A2b_ResNetMini_w32", group="A", arch="resnet_mini",
                 model_kwargs=dict(base_width=32, **dropout_kwargs("resnet_mini", "default")),
                 note="ResNetMini rộng 32 (2.8M tham số, ~4 lần nhẹ hơn A2) - dùng để quyết định cấu hình quét siêu tham số"),
        make_cfg(exp_name=f"Exp2_A3_ResNet18_TL_{args.tl_size}", arch="resnet18",
                 model_kwargs=dict(pretrained=True, dropout=0.3), **tl_common,
                 note="ResNet18 pretrained ImageNet, fine-tune toàn phần, LR thân = 0.1 x LR đầu"),
        make_cfg(exp_name=f"Exp2_A4_ResNet34_TL_{args.tl_size}", arch="resnet34",
                 model_kwargs=dict(pretrained=True, dropout=0.3), **tl_common,
                 note="ResNet34 pretrained ImageNet, fine-tune toàn phần"),
        make_cfg(exp_name=f"Exp2_A5_EfficientNetB0_TL_{args.tl_size}", arch="efficientnet_b0",
                 model_kwargs=dict(pretrained=True, dropout=0.3), optional=True, **tl_common,
                 note="EfficientNet-B0 pretrained (tùy chọn)"),
        make_cfg(exp_name=f"Exp2_A6_ResNet18_Frozen_{args.tl_size}", arch="resnet18",
                 model_kwargs=dict(pretrained=True, dropout=0.3, freeze_backbone=True),
                 in_channels=3, img_size=args.tl_size, lr=1e-3, backbone_lr_mult=1.0, group="A",
                 note="ResNet18 đóng băng thân - chỉ huấn luyện đầu phân loại (feature extraction)"),
    ]

    # ---------------- Nhóm F: Fail cases ----------------
    exps += [
        make_cfg(exp_name="Exp2_F1_VGG16_TL_48_NoReg", group="F", arch="vgg16",
                 model_kwargs=dict(pretrained=True, dropout=0.0), in_channels=3, img_size=48,
                 use_aug=False, weighted_loss=False, opt="adam", lr=1e-4, weight_decay=0.0,
                 sched="none", epochs=30, patience=30, fail_case=True, keep_ckpt=False,
                 note="FAIL CASE: VGG16 (134M tham số) fine-tune toàn phần trên ảnh 48x48, không Aug/Dropout/WD"),
        make_cfg(exp_name="Exp2_F2_LargeCNN_NoReg", group="F", arch="large_cnn",
                 model_kwargs=dropout_kwargs("large_cnn", "none"), use_aug=False, weighted_loss=False,
                 weight_decay=0.0, sched="none", epochs=45, patience=45, fail_case=True,
                 note="FAIL CASE: LargeCNN không Aug/Dropout/WD -> học thuộc tập train"),
        make_cfg(exp_name="Exp2_F3_VGG16_Scratch_NoBN", group="F", arch="vgg16",
                 model_kwargs=dict(pretrained=False, dropout=0.5), in_channels=3, img_size=48,
                 use_aug=True, weighted_loss=False, opt="adam", lr=1e-3, weight_decay=0.0,
                 sched="none", epochs=20, patience=20, fail_case=True, keep_ckpt=False, optional=True,
                 note="FAIL CASE (tùy chọn): VGG16 không BN, khởi tạo ngẫu nhiên, Adam 1e-3 -> khó hội tụ"),
        make_cfg(exp_name=f"Exp2_F4_VGG16_TL_{args.tl_size}_Fixed", group="F", arch="vgg16",
                 model_kwargs=dict(pretrained=True, dropout=0.5), in_channels=3, img_size=args.tl_size,
                 use_aug=True, weighted_loss=True, opt="adamw", lr=1e-3, weight_decay=1e-2,
                 backbone_lr_mult=0.01, sched="cosine", epochs=25, patience=25, keep_ckpt=False,
                 note="BẢN KHẮC PHỤC F1: tăng độ phân giải, Aug, Dropout 0.5, WD, LR thân x0.01, Cosine"),
    ]

    # ---------------- Nhóm B: Optimizer (scheduler Cosine) ----------------
    arch = args.hp_arch
    W = args.hp_width
    tag = arch + (f"_w{W}" if arch == "resnet_mini" and W != 64 else "")
    base = dict(arch=arch, group="B", epochs=E, patience=args.cosine_patience, sched="cosine",
                **arch_setup(arch, "default", args.tl_size, W))
    for key, label in (("sgd", "SGD_Momentum"), ("adam", "Adam"), ("adamw", "AdamW")):
        exps.append(make_cfg(exp_name=f"Exp2_B_{tag}_{label}", opt=key, lr=DEFAULT_LR[key],
                             weight_decay=DEFAULT_WD[key],
                             note=f"So sánh optimizer: {label} (lr={DEFAULT_LR[key]}, wd={DEFAULT_WD[key]}), Cosine",
                             **base))

    # ---------------- Nhóm C: LR & Scheduler (optimizer = --hp_opt) ----------------
    opt = args.hp_opt
    lr0 = args.hp_lr if args.hp_lr else DEFAULT_LR[opt]
    wd0 = DEFAULT_WD[opt]
    cbase = dict(arch=arch, group="C", epochs=E, opt=opt, weight_decay=wd0,
                 **arch_setup(arch, "default", args.tl_size, W))
    exps += [
        make_cfg(exp_name=f"Exp2_C1_{tag}_{opt}_Plateau", lr=lr0, sched="plateau", patience=10,
                 note="ReduceLROnPlateau(factor 0.5, patience 4)", **cbase),
        make_cfg(exp_name=f"Exp2_C2_{tag}_{opt}_Cosine_lrx3", lr=lr0 * 3, sched="cosine",
                 patience=args.cosine_patience, note="Cosine, LR gấp 3 lần mốc", **cbase),
        make_cfg(exp_name=f"Exp2_C3_{tag}_{opt}_Cosine_lrdiv3", lr=lr0 / 3, sched="cosine",
                 patience=args.cosine_patience, note="Cosine, LR chia 3 so với mốc", **cbase),
    ]

    # ---------------- Nhóm D: Regularization (một yếu tố mỗi lần) ----------------
    sched = args.hp_sched
    dbase = dict(arch=arch, group="D", epochs=E, opt=opt, lr=lr0, sched=sched,
                 patience=args.cosine_patience if sched == "cosine" else 10)

    def dcfg(name: str, drop: str = "default", wd: Optional[float] = None, ls: float = 0.0, note: str = ""):
        return make_cfg(exp_name=f"Exp2_D_{tag}_{name}", **arch_setup(arch, drop, args.tl_size, W),
                        weight_decay=wd0 if wd is None else wd, label_smoothing=ls, note=note, **dbase)

    exps += [
        dcfg("Drop_none", drop="none", note="Không Dropout"),
        dcfg("Drop_strong", drop="strong", note="Dropout mạnh"),
        dcfg("WD_0", wd=0.0, note="Không Weight Decay"),
        dcfg("WD_high", wd=HIGH_WD[opt], note=f"Weight Decay cao ({HIGH_WD[opt]})"),
        dcfg("LS_0.05", ls=0.05, note="Label Smoothing 0.05"),
        dcfg("LS_0.10", ls=0.10, note="Label Smoothing 0.10"),
        dcfg("AllReg", drop="strong", wd=HIGH_WD[opt], ls=0.10,
             note="Regularization tối đa (Dropout mạnh + WD cao + LS 0.1) - kiểm tra over-regularization"),
    ]

    # ---------------- Nhóm G: Final run ----------------
    fwd = {"default": wd0, "high": HIGH_WD[opt], "zero": 0.0}.get(str(args.final_wd), None)
    if fwd is None:
        fwd = float(args.final_wd)
    exps.append(make_cfg(
        exp_name=f"Exp2_G_Final_{tag}_{opt}", group="G", arch=arch, epochs=args.final_epochs,
        opt=opt, lr=lr0, weight_decay=fwd, sched=sched, label_smoothing=args.final_ls,
        patience=args.final_epochs if sched == "cosine" else 15,
        **arch_setup(arch, args.final_dropout, args.tl_size, W),
        note=f"Chạy cuối: dropout={args.final_dropout}, wd={fwd}, ls={args.final_ls}, {args.final_epochs} epoch"))
    return exps


# ----------------------------------------------------------------------------------------------
# Dựng optimizer / scheduler
# ----------------------------------------------------------------------------------------------
def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    mult = cfg["backbone_lr_mult"]
    if hasattr(model, "param_groups") and mult != 1.0:
        params: Any = model.param_groups(lr_backbone=cfg["lr"] * mult, lr_head=cfg["lr"])
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    wd, lr = cfg["weight_decay"], cfg["lr"]
    if cfg["opt"] == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=cfg["momentum"], nesterov=True, weight_decay=wd)
    if cfg["opt"] == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if cfg["opt"] == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    raise ValueError(cfg["opt"])


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict[str, Any]):
    if cfg["sched"] == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6)
    if cfg["sched"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-6)
    return None


# ----------------------------------------------------------------------------------------------
# Chạy một thí nghiệm
# ----------------------------------------------------------------------------------------------
def gap_metrics(csv_path: str | Path, best_epoch: int) -> Dict[str, float]:
    """Chỉ số overfitting đọc từ log từng epoch (Train Acc - Val Acc, đơn vị %)."""
    df = pd.read_csv(csv_path)
    last = df.iloc[-1]
    best = df[df["epoch"] == best_epoch].iloc[0]
    min_idx = df["val_loss"].idxmin()
    return {
        "train_acc_at_best": round(float(best["train_acc"]) * 100, 2),
        "gap_at_best": round(float(best["train_acc"] - best["val_acc"]) * 100, 2),
        "final_train_acc": round(float(last["train_acc"]) * 100, 2),
        "final_gap": round(float(last["train_acc"] - last["val_acc"]) * 100, 2),
        "min_val_loss": round(float(df.loc[min_idx, "val_loss"]), 4),
        "min_val_loss_epoch": int(df.loc[min_idx, "epoch"]),
        "final_val_loss": round(float(last["val_loss"]), 4),
    }


def run_experiment(cfg: Dict[str, Any], args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    name = cfg["exp_name"]
    print("\n" + "=" * 78)
    print(f"ĐANG CHẠY: {name} | {cfg['note']}")
    print("=" * 78)

    set_seed(args.seed)
    train_loader, val_loader, test_loader, info = get_dataloaders(
        data_dir=args.data_dir, batch_size=cfg["batch_size"], use_aug=cfg["use_aug"],
        img_size=cfg["img_size"], in_channels=cfg["in_channels"],
        num_workers=args.num_workers, seed=args.seed,
    )

    mkw = copy.deepcopy(cfg["model_kwargs"])
    if cfg["arch"] in ("large_cnn", "resnet_mini"):
        mkw["in_channels"] = cfg["in_channels"]
    model = build_model(cfg["arch"], num_classes=info["num_classes"], **mkw)
    params = count_parameters(model)

    weights = None
    if cfg["weighted_loss"]:
        weights = compute_class_weights(info["train_labels"], num_classes=7, method="sqrt_balanced").to(device)
    criterion = SmoothedWeightedCE(weight=weights, label_smoothing=cfg["label_smoothing"]).to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    summary = fit(
        model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler, device=device,
        epochs=cfg["epochs"], patience=cfg["patience"], monitor="val_acc", exp_name=name,
        logs_dir="logs", checkpoints_dir="checkpoints",
    )
    summary.update(gap_metrics(summary["log_csv"], summary["best_epoch"]))
    summary.update({
        "group": cfg["group"], "arch": cfg["arch"], "img_size": cfg["img_size"],
        "in_channels": cfg["in_channels"], "trainable_params": params["trainable"],
        "optimizer": cfg["opt"], "lr": cfg["lr"], "backbone_lr_mult": cfg["backbone_lr_mult"],
        "weight_decay": cfg["weight_decay"], "scheduler": cfg["sched"],
        "label_smoothing": cfg["label_smoothing"], "use_aug": cfg["use_aug"],
        "weighted_loss": cfg["weighted_loss"], "epochs_cfg": cfg["epochs"], "fail_case": cfg["fail_case"],
        "model_kwargs": json.dumps(cfg["model_kwargs"]), "note": cfg["note"],
        "date": time.strftime("%d/%m/%Y"),
    })

    plot_training_curves(
        csv_path=summary["log_csv"], save_path=Path("figures") / f"{name}_curves.png",
        title=f"{name} (Best Val: {summary['best_val_acc']*100:.2f}% | Test: {summary['test_acc']*100:.2f}%"
              f" | Gap@best: {summary['gap_at_best']:+.1f}%)",
    )

    if not cfg["keep_ckpt"]:  # VGG16 ~540 MB mỗi file: xóa sau khi đã đánh giá xong để khỏi đầy ổ đĩa
        Path(summary["checkpoint_path"]).unlink(missing_ok=True)
        summary["checkpoint_path"] = "(đã xóa)"
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


# ----------------------------------------------------------------------------------------------
# Tổng hợp, biểu đồ, chọn best model
# ----------------------------------------------------------------------------------------------
def load_summary() -> pd.DataFrame:
    return pd.read_csv(SUMMARY_CSV) if SUMMARY_CSV.exists() else pd.DataFrame()


def merge_summary(new_row: Dict[str, Any]) -> pd.DataFrame:
    df = load_summary()
    if len(df):
        df = df[df["exp_name"] != new_row["exp_name"]]
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    return df


def plot_exp2_overview(df: pd.DataFrame, save_path: str | Path) -> None:
    """Biểu đồ ngang: Val Acc / Test Acc của mọi thực nghiệm (xếp theo Val Acc) + mốc Exp1_6 & benchmark."""
    d = df.sort_values("best_val_acc").reset_index(drop=True)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.45 * len(d) + 1.5)))
    ax.barh(y - 0.2, d["best_val_acc"] * 100, 0.4, label="Val Acc (PublicTest)", color="#1B365D")
    ax.barh(y + 0.2, d["test_acc"] * 100, 0.4, label="Test Acc (PrivateTest)", color="#E07A5F")
    for i, r in d.iterrows():
        ax.text(r["test_acc"] * 100 + 0.4, i + 0.2, f"{r['test_acc']*100:.1f}", va="center", fontsize=8)
        ax.text(r["best_val_acc"] * 100 + 0.4, i - 0.2, f"{r['best_val_acc']*100:.1f}", va="center", fontsize=8)
    ax.axvline(68.51, color="green", ls="--", lw=1.2, label="Exp1_6 (TV1): Test 68.51%")
    ax.axvline(71.0, color="gray", ls=":", lw=1.2, label="Benchmark ~71% (Private)")
    ax.set_yticks(y)
    ax.set_yticklabels(d["exp_name"], fontsize=8)
    lo = 0 if min(d["best_val_acc"].min(), d["test_acc"].min()) * 100 < 25 else 20
    ax.set_xlim(lo, 80)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("So sánh hiệu năng các thực nghiệm Nhóm 2", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    ensure_dir(Path(save_path).parent)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_overfit_comparison(exp_names: List[str], save_path: str | Path, logs_dir: str = "logs") -> None:
    """Mỗi cột = một thực nghiệm: hàng trên Train/Val Acc, hàng dưới Train/Val Loss (minh họa Fail Case)."""
    names = [n for n in exp_names if (Path(logs_dir) / f"{n}_epochs.csv").exists()]
    if not names:
        return
    fig, axes = plt.subplots(2, len(names), figsize=(4.6 * len(names), 7.5), squeeze=False)
    for j, n in enumerate(names):
        d = pd.read_csv(Path(logs_dir) / f"{n}_epochs.csv")
        a, l = axes[0, j], axes[1, j]
        a.plot(d["epoch"], d["train_acc"] * 100, label="Train", color="#1B365D", lw=2)
        a.plot(d["epoch"], d["val_acc"] * 100, label="Val", color="#E07A5F", lw=2)
        a.fill_between(d["epoch"], d["val_acc"] * 100, d["train_acc"] * 100, alpha=0.12, color="red")
        a.set_title(n, fontsize=9, fontweight="bold")
        a.set_ylabel("Accuracy (%)")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
        l.plot(d["epoch"], d["train_loss"], label="Train", color="#1B365D", lw=2)
        l.plot(d["epoch"], d["val_loss"], label="Val", color="#E07A5F", lw=2)
        l.set_xlabel("Epoch")
        l.set_ylabel("Loss")
        l.grid(alpha=0.3)
        l.legend(fontsize=8)
    plt.tight_layout()
    ensure_dir(Path(save_path).parent)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_markdown_rows(df: pd.DataFrame, path: str | Path = "logs/exp2_log_rows.md") -> None:
    """Sinh sẵn các dòng bảng Markdown cho Mục 2 của docs/nhat_ky_thuc_nghiem.md (cột nhận xét để trống)."""
    lines = [
        "| Mã TN | Ngày chạy | Kiến trúc | Input | Optimizer & LR | Scheduler / Reg | Params | Best Ep / Tổng Ep | "
        "Train Acc @best | Gap @best | Val Acc (%) | Min Val Loss | Test Acc (%) | Macro F1 (%) | Thời gian (s) | Nhận xét / Fail Case |",
        "| :--- | :---: | :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    for _, r in df.sort_values("exp_name").iterrows():
        opt = f"{r['optimizer']} lr={r['lr']:g}" + (f" (thân x{r['backbone_lr_mult']:g})" if r["backbone_lr_mult"] != 1 else "")
        reg = (f"{r['scheduler']}; wd={r['weight_decay']:g}; ls={r['label_smoothing']:g}; "
               f"aug={'có' if r['use_aug'] else 'không'}; {r['model_kwargs']}")
        lines.append(
            f"| `{r['exp_name']}` | {r.get('date', '')} | {r['arch']} | {int(r['img_size'])}×{int(r['img_size'])} | {opt} | {reg} | "
            f"{int(r['total_params']):,} | {int(r['best_epoch'])} / {int(r['epochs_ran'])} | {r['train_acc_at_best']:.2f}% | "
            f"{r['gap_at_best']:+.2f}% | **{r['best_val_acc']*100:.2f}** | {r['min_val_loss']:.4f} | "
            f"{r['test_acc']*100:.2f} | {r['test_macro_f1']*100:.2f} | {r['train_time_sec']:.0f} | *(điền nhận xét)* |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def refresh_outputs(df: pd.DataFrame) -> None:
    if len(df) >= 1:
        plot_exp2_overview(df, "figures/exp2_comparison.png")
        write_markdown_rows(df)
    fails = [n for n in df["exp_name"] if "_F1_" in n or "_F2_" in n or "_F3_" in n or "_F4_" in n]
    ref = df[(df["group"] == "A") & (~df["fail_case"])].sort_values("best_val_acc", ascending=False)
    names = sorted(fails) + ([ref.iloc[0]["exp_name"]] if len(ref) else [])
    plot_overfit_comparison(names, "figures/exp2_fail_cases.png")


def finalize_best_model(device: torch.device, out_path: str = "checkpoints/best_model.pth") -> None:
    """
    Chọn best model THEO VAL ACC (PublicTest) - không chọn theo Test để tránh rò rỉ thông tin từ PrivateTest -
    trong số các thí nghiệm hợp lệ (không phải fail case, còn file checkpoint). Hòa -> chọn Val Macro F1 cao hơn.
    Ghi best_model.pth (chỉ model_state_dict + cấu hình dựng lại mô hình) và logs/best_model_info.json.
    """
    df = load_summary()
    if df.empty:
        raise SystemExit("Chưa có logs/exp2_summary.csv - hãy chạy thực nghiệm trước.")
    ok = df[(~df["fail_case"].astype(bool)) & df["checkpoint_path"].apply(lambda p: Path(str(p)).exists())]
    if ok.empty:
        raise SystemExit("Không có checkpoint hợp lệ nào (cần file .pth trong checkpoints/).")
    best = ok.sort_values(["best_val_acc", "best_val_macro_f1"], ascending=False).iloc[0]
    print(f"Best model theo Val Acc: {best['exp_name']} (Val {best['best_val_acc']*100:.2f}% | Test {best['test_acc']*100:.2f}%)")

    raw = torch.load(best["checkpoint_path"], map_location="cpu", weights_only=False)
    kwargs = json.loads(best["model_kwargs"])
    mkw = copy.deepcopy(kwargs)
    if best["arch"] in ("large_cnn", "resnet_mini"):
        mkw["in_channels"] = int(best["in_channels"])
    model = build_model(best["arch"], num_classes=7, **mkw)
    model.load_state_dict(raw["model_state_dict"])
    meta = {k: (v.item() if hasattr(v, "item") else v) for k, v in best.to_dict().items()
            if k in ("exp_name", "best_epoch", "best_val_acc", "best_val_macro_f1", "test_acc", "test_macro_f1",
                     "test_disgust_recall", "optimizer", "lr", "weight_decay", "scheduler", "label_smoothing")}
    save_kwargs = kwargs
    path = save_deployable_checkpoint(out_path, model, best["arch"], save_kwargs, int(best["img_size"]),
                                      int(best["in_channels"]), meta)
    size_mb = path.stat().st_size / 1e6
    Path("logs/best_model_info.json").write_text(json.dumps(meta | {"size_mb": round(size_mb, 1)}, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
    print(f"Đã lưu {path} ({size_mb:.1f} MB)")
    if size_mb > 95:
        print("CẢNH BÁO: > 95 MB - không push lên GitHub được, hãy gửi TV3 qua Google Drive.")
    else:
        print("Gợi ý: mở ngoại lệ trong .gitignore:  !checkpoints/best_model.pth  rồi push lên nhánh develop.")


# ----------------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Thực nghiệm Nhóm 2 (TV2 - FER2013 PyTorch)")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--group", nargs="+", default=None, help="Nhóm cần chạy: A B C D F G")
    p.add_argument("--exp", nargs="+", default=None, help="Chỉ chạy thí nghiệm có tên bắt đầu bằng tiền tố này")
    p.add_argument("--epochs", type=int, default=60, help="Số epoch cho nhóm B, C, D")
    p.add_argument("--final_epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--tl_size", type=int, default=128, help="Kích thước ảnh cho Transfer Learning (128 hoặc 224)")
    p.add_argument("--cosine_patience", type=int, default=20, help="EarlyStopping patience khi dùng Cosine")
    p.add_argument("--hp_arch", type=str, default="resnet_mini", help="Kiến trúc dùng cho nhóm B, C, D, G")
    p.add_argument("--hp_opt", type=str, default="adamw", choices=["sgd", "adam", "adamw"])
    p.add_argument("--hp_sched", type=str, default="cosine", choices=["plateau", "cosine"])
    p.add_argument("--hp_width", type=int, default=64, help="base_width của ResNetMini cho nhóm B,C,D,G (64 hoặc 32)")
    p.add_argument("--hp_lr", type=float, default=None, help="LR mốc (mặc định theo optimizer)")
    p.add_argument("--final_dropout", type=str, default="default", choices=["none", "default", "strong"])
    p.add_argument("--final_wd", type=str, default="default", help="default | high | zero | <số>")
    p.add_argument("--final_ls", type=float, default=0.05)
    p.add_argument("--skip_optional", action="store_true", help="Bỏ qua các thí nghiệm tùy chọn (A5, F3)")
    p.add_argument("--force", action="store_true", help="Chạy lại cả thí nghiệm đã có trong summary")
    p.add_argument("--fast_test", action="store_true", help="Chạy 2 epoch để kiểm tra code")
    p.add_argument("--finalize", action="store_true", help="Chọn best model theo Val Acc -> checkpoints/best_model.pth")
    p.add_argument("--refresh", action="store_true", help="Chỉ vẽ lại biểu đồ & bảng Markdown từ summary hiện có")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    for d in ("logs", "figures", "checkpoints"):
        ensure_dir(d)

    if args.finalize:
        refresh_outputs(load_summary())
        finalize_best_model(device)
        return
    if args.refresh:
        refresh_outputs(load_summary())
        return

    if args.fast_test:  # không làm bẩn bảng kết quả thật
        global SUMMARY_CSV
        SUMMARY_CSV = Path("logs/exp2_fast_test_summary.csv")

    exps = build_experiments(args)
    if args.group:
        exps = [c for c in exps if c["group"] in {g.upper() for g in args.group}]
    if args.exp:
        exps = [c for c in exps if any(c["exp_name"].startswith(pref) for pref in args.exp)]
    if args.skip_optional:
        exps = [c for c in exps if not c["optional"]]
    if not exps:
        raise SystemExit("Không có thí nghiệm nào khớp bộ lọc --group/--exp.")

    done = set(load_summary()["exp_name"]) if SUMMARY_CSV.exists() else set()
    print(f"Sẽ chạy {len(exps)} thí nghiệm: " + ", ".join(c["exp_name"] for c in exps))

    for cfg in exps:
        if args.fast_test:
            cfg["epochs"], cfg["patience"] = 2, 2
            cfg["exp_name"] = "FAST_" + cfg["exp_name"]
        elif cfg["exp_name"] in done and not args.force:
            print(f"[BỎ QUA] {cfg['exp_name']} đã có trong {SUMMARY_CSV} (dùng --force để chạy lại).")
            continue
        try:
            cfg["batch_size"] = args.batch_size
            summary = run_experiment(cfg, args, device)
        except Exception:  # noqa: BLE001  (một thí nghiệm lỗi không làm hỏng cả chuỗi)
            print(f"[LỖI] {cfg['exp_name']}:\n{traceback.format_exc()}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        df = merge_summary(summary)
        if not args.fast_test:
            refresh_outputs(df)

    df = load_summary()
    if len(df):
        cols = ["exp_name", "best_epoch", "best_val_acc", "test_acc", "test_macro_f1", "gap_at_best", "train_time_sec"]
        print("\nBẢNG TỔNG HỢP NHÓM 2:")
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
