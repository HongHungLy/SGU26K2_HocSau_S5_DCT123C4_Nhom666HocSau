"""
Động cơ huấn luyện chuẩn PyTorch dùng chung cho cả 3 thành viên.
Hỗ trợ: EarlyStopping, ModelCheckpoint (best val_loss / val_acc), LR Scheduler,
ghi log từng epoch ra CSV và đánh giá cuối cùng trên PrivateTest.

Phụ trách: Thành viên 1 - Lý Hồng Hưng (3123411132).
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .utils import EMOTION_LABELS, count_parameters, ensure_dir


class EarlyStopping:
    """Dừng huấn luyện sớm khi chỉ số giám sát (val_loss hoặc val_acc) không cải thiện."""

    def __init__(self, patience: int = 10, mode: str = "min", min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Trả về True nếu đạt kỷ lục mới (cần lưu checkpoint)."""
        if self.best_score is None:
            self.best_score = value
            self.counter = 0
            return True

        improved = (
            (value < self.best_score - self.min_delta)
            if self.mode == "min"
            else (value > self.best_score + self.min_delta)
        )
        if improved:
            self.best_score = value
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = True,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Tuple[float, float]:
    """Huấn luyện 1 epoch với model.train()."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    amp_enabled = use_amp and (device.type == "cuda")

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if amp_enabled and scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    """Đánh giá mô hình với model.eval() và torch.no_grad()."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * labels.size(0)
        all_preds.append(outputs.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    total = len(y_true)

    avg_loss = running_loss / max(total, 1)
    acc = float((y_pred == y_true).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    per_class_recall = recall_score(
        y_true,
        y_pred,
        labels=list(range(len(EMOTION_LABELS))),
        average=None,
        zero_division=0,
    )

    return {
        "loss": avg_loss,
        "acc": acc,
        "macro_f1": macro_f1,
        "per_class_recall": per_class_recall,
        "disgust_recall": float(per_class_recall[1]),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 40,
    scheduler: Optional[Any] = None,
    patience: int = 10,
    monitor: str = "val_acc",
    exp_name: str = "baseline",
    logs_dir: str | Path = "logs",
    checkpoints_dir: str | Path = "checkpoints",
    test_loader: Optional[DataLoader] = None,
    use_amp: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Vòng lặp huấn luyện hoàn chỉnh chuẩn PyTorch:
      - Lưu CSV log từng epoch vào logs/<exp_name>_epochs.csv
      - Lưu checkpoint tốt nhất vào checkpoints/<exp_name>_best.pth
      - EarlyStopping theo val_acc (mode='max') hoặc val_loss (mode='min')
      - Nếu truyền test_loader (PrivateTest), tự động nạp lại checkpoint tốt nhất
        để đánh giá độc lập cuối cùng (No Data Leakage).
    """
    logs_path = ensure_dir(logs_dir)
    ckpt_path = ensure_dir(checkpoints_dir)

    csv_file = logs_path / f"{exp_name}_epochs.csv"
    best_model_file = ckpt_path / f"{exp_name}_best.pth"

    mode = "max" if monitor == "val_acc" else "min"
    early_stopper = EarlyStopping(patience=patience, mode=mode)
    if use_amp and device.type == "cuda":
        scaler = (
            torch.amp.GradScaler("cuda")
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler")
            else torch.cuda.amp.GradScaler()
        )
    else:
        scaler = None

    model = model.to(device)
    param_stats = count_parameters(model)

    best_epoch = 0
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_val_f1 = 0.0
    start_time = time.time()

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "lr", "epoch_sec"])

        pbar = tqdm(range(1, epochs + 1), desc=exp_name, disable=not verbose)
        for epoch in pbar:
            ep_start = time.time()

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, use_amp=use_amp, scaler=scaler
            )
            val_metrics = evaluate_epoch(model, val_loader, criterion, device)
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["acc"]
            val_f1 = val_metrics["macro_f1"]

            current_lr = optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            ep_sec = time.time() - ep_start
            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{train_acc:.6f}",
                f"{val_loss:.6f}",
                f"{val_acc:.6f}",
                f"{val_f1:.6f}",
                f"{current_lr:.8f}",
                f"{ep_sec:.2f}",
            ])
            f.flush()

            monitor_val = val_acc if monitor == "val_acc" else val_loss
            is_best = early_stopper.step(monitor_val)
            if is_best:
                best_epoch = epoch
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_val_f1 = val_f1
                torch.save(
                    {
                        "exp_name": exp_name,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "val_macro_f1": val_f1,
                    },
                    best_model_file,
                )

            pbar.set_postfix({
                "tr_acc": f"{train_acc*100:.1f}%",
                "val_acc": f"{val_acc*100:.1f}%",
                "best": f"{best_val_acc*100:.1f}%",
                "lr": f"{current_lr:.1e}",
            })

            if early_stopper.should_stop:
                if verbose:
                    print(f"[{exp_name}] Early stopping tại epoch {epoch} (best epoch: {best_epoch}).")
                break

    total_time_sec = time.time() - start_time

    summary: Dict[str, Any] = {
        "exp_name": exp_name,
        "total_params": param_stats["total"],
        "epochs_ran": epoch,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_acc": round(best_val_acc, 4),
        "best_val_macro_f1": round(best_val_f1, 4),
        "train_time_sec": round(total_time_sec, 1),
        "log_csv": str(csv_file),
        "checkpoint_path": str(best_model_file),
    }

    # Đánh giá độc lập trên PrivateTest bằng checkpoint tốt nhất
    if test_loader is not None and best_model_file.exists():
        ckpt = torch.load(best_model_file, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        test_metrics = evaluate_epoch(model, test_loader, criterion, device)
        summary["test_loss"] = round(test_metrics["loss"], 4)
        summary["test_acc"] = round(test_metrics["acc"], 4)
        summary["test_macro_f1"] = round(test_metrics["macro_f1"], 4)
        summary["test_disgust_recall"] = round(test_metrics["disgust_recall"], 4)

    return summary
