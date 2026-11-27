"""Tiện ích dùng chung cho nhóm BTL03 (Keras -> PyTorch). Do TV2 viết."""
import random, time
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    """Tương đương keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)."""
    def __init__(self, patience=5, min_delta=0.0):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.counter, self.early_stop = None, 0, False

    def __call__(self, val_loss):
        if self.best is None or val_loss < self.best - self.min_delta:
            self.best, self.counter = val_loss, 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total_loss += criterion(out, y).item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


def fit(model, train_loader, val_loader, criterion, optimizer, device, epochs,
        scheduler=None, early_stopping=None, ckpt_path=None, writer=None, verbose=True):
    """Vong lap train/eval mau: ho tro Checkpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard."""
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    best = float("inf")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl, va = evaluate(model, val_loader, criterion, device)
        for k, v in zip(history, (tl, ta, vl, va)):
            history[k].append(v)
        if writer is not None:
            writer.add_scalars("Loss", {"train": tl, "val": vl}, epoch)
            writer.add_scalars("Accuracy", {"train": ta, "val": va}, epoch)
            writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)
        if ckpt_path and vl < best:
            best = vl
            torch.save(model.state_dict(), ckpt_path)
        if scheduler is not None:
            scheduler.step(vl)
        if verbose:
            print(f"Epoch {epoch:03d} - {time.time()-t0:.1f}s - loss: {tl:.4f} - acc: {ta:.4f} - val_loss: {vl:.4f} - val_acc: {va:.4f}")
        if early_stopping is not None and early_stopping(vl):
            print(f"Early stopping tai epoch {epoch}")
            break
    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return history
