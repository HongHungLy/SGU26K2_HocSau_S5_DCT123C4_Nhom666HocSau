"""
Module: trainer.py
Nhiệm vụ:
- Xây dựng vòng lặp huấn luyện chuẩn PyTorch:
    forward -> loss -> zero_grad -> backward -> optimizer.step()
- Sử dụng đúng ngữ cảnh model.train(), model.eval(), torch.no_grad()
- Tự động lưu checkpoint mô hình có validation loss tốt nhất (best_model.pt)
- Xuất log tiến trình ra file CSV (epoch, train_loss, val_loss)
- Hỗ trợ cả CPU và GPU (nếu khả dụng)
"""

import os
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class EarlyStopping:
    """Cơ chế dừng sớm nếu validation loss không cải thiện sau patience epochs."""

    def __init__(self, patience: int = 30, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


class Trainer:
    """Bộ điều khiển huấn luyện mạng MLP bằng PyTorch."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module = None,
        device: str = None,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
    ):
        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion if criterion is not None else nn.MSELoss()

        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Huấn luyện 1 epoch trên tập train."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred, y_batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(X_batch)
            total_samples += len(X_batch)

        return total_loss / max(total_samples, 1)

    def evaluate(self, data_loader: DataLoader) -> float:
        """Đánh giá loss trên một DataLoader (Validation hoặc Test)."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)

                total_loss += loss.item() * len(X_batch)
                total_samples += len(X_batch)

        return total_loss / max(total_samples, 1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 200,
        checkpoint_name: str = "best_model.pt",
        log_csv_name: str = "train_log.csv",
        patience: int = None,
        verbose: bool = False,
    ) -> dict:
        """
        Thực hiện toàn bộ quá trình huấn luyện qua các epoch.
        Trả về lịch sử loss và thời gian huấn luyện.
        """
        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
        }

        best_val_loss = float("inf")
        best_epoch = 0
        best_model_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        log_path = os.path.join(self.log_dir, log_csv_name)

        early_stopping = EarlyStopping(patience=patience) if patience else None
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            # Checkpoint: Lưu mô hình tốt nhất theo val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                    },
                    best_model_path,
                )

            if verbose and (epoch % 50 == 0 or epoch == epochs):
                print(
                    f"Epoch [{epoch:3d}/{epochs:3d}] - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}"
                )

            if early_stopping and early_stopping(val_loss):
                if verbose:
                    print(f"Early stopping tại epoch {epoch}!")
                break

        training_time = time.time() - start_time

        # Lưu lịch sử loss ra file CSV
        df_log = pd.DataFrame(history)
        df_log.to_csv(log_path, index=False)

        # Nạp lại trọng số tốt nhất vào mô hình
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        return {
            "history": history,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "training_time": training_time,
            "log_path": log_path,
            "best_model_path": best_model_path,
        }
