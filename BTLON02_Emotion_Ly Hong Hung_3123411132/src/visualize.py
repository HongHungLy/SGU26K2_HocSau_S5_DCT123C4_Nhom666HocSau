"""
Trực quan hóa: Final Convolution Filters, Feature Maps, Grad-CAM (3 tầng đầu - giữa - cuối).

Phụ trách: Thành viên 3 (Vai trò C).

Chạy từ THƯ MỤC GỐC của project:
    python -m src.visualize
    python -m src.visualize --ckpt checkpoints/best_model.pth --split private

Đầu ra (figures/):
    filters_first_conv.png, filters_last_conv.png
    feature_maps_first_conv.png, feature_maps_last_conv.png
    gradcam_7classes_3layers.png   (7 cảm xúc x 3 tầng tích chập)
    gradcam_wrong_cases.png        (các ca dự đoán sai)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import cm as mcm

from .evaluate import get_eval_loaders, load_any_model, predict, resolve_ckpt, to_display
from .utils import EMOTION_LABELS, get_device, set_seed


def norm01(a: np.ndarray) -> np.ndarray:
    a = a - a.min()
    return a / (a.max() + 1e-8)


def pick_conv_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """3 điểm hook: [tầng đầu, tầng giữa, tầng cuối]. Bỏ qua Conv 1x1 (shortcut) khi chọn tầng đầu/giữa."""
    convs = [(n, m) for n, m in model.named_modules()
             if isinstance(m, nn.Conv2d) and m.kernel_size[0] > 1]
    if not convs:
        convs = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    last = getattr(model, "gradcam_layer", convs[-1][1])
    return [("Đầu: " + convs[0][0], convs[0][1]),
            ("Giữa: " + convs[len(convs) // 2][0], convs[len(convs) // 2][1]),
            ("Cuối: gradcam_layer", last)]


# ----------------------------------------------------------------------------------------------
# 1. Convolution filters
# ----------------------------------------------------------------------------------------------
def plot_filters(conv: nn.Conv2d, title: str, path: Path, max_n: int = 64, cols: int = 8) -> None:
    w = conv.weight.detach().cpu().numpy()                 # [out, in, k, k]
    n = min(max_n, w.shape[0])
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * 1.45))
    for a in np.atleast_1d(axes).ravel():
        a.axis("off")
    for a, f in zip(np.atleast_1d(axes).ravel(), w[:n]):
        if f.shape[0] == 3:                                # RGB kernel (conv đầu của transfer learning)
            a.imshow(norm01(f.transpose(1, 2, 0)), interpolation="nearest")
        else:                                              # trung bình theo các kênh đầu vào
            a.imshow(norm01(f.mean(0)), cmap="viridis", interpolation="nearest")
    plt.suptitle(f"{title}  |  weight shape = {tuple(conv.weight.shape)}  (hiển thị {n}/{w.shape[0]} filter)",
                 fontsize=11)
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()


# ----------------------------------------------------------------------------------------------
# 2. Feature maps
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def plot_feature_maps(model, layer, x, title, path, n_ch: int = 32, cols: int = 8) -> None:
    store = {}
    h = layer.register_forward_hook(lambda m, i, o: store.__setitem__("f", o.detach().cpu()))
    model(x)
    h.remove()
    f = store["f"][0]
    n = min(n_ch, f.shape[0])
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.5))
    for a in np.atleast_1d(axes).ravel():
        a.axis("off")
    for a, ch in zip(np.atleast_1d(axes).ravel(), f[:n]):
        a.imshow(ch.numpy(), cmap="magma")
    plt.suptitle(f"{title}  |  output {tuple(f.shape)}", fontsize=11)
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()


# ----------------------------------------------------------------------------------------------
# 3. Grad-CAM
# ----------------------------------------------------------------------------------------------
def grad_cam(model: nn.Module, layer: nn.Module, x: torch.Tensor, target: int | None = None):
    """Trả về (heatmap [H,W] trong [0,1], lớp được dùng)."""
    store = {}

    def fwd_hook(_m, _i, out):
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))

    h = layer.register_forward_hook(fwd_hook)
    model.zero_grad()
    x = x.clone().requires_grad_(True)
    logits = model(x)
    cls = int(logits.argmax(1)) if target is None else int(target)
    logits[0, cls].backward()
    h.remove()

    w = store["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * store["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    return norm01(cam.detach().cpu().numpy()), cls


def overlay(img: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    if cam.shape != img.shape:
        cam = np.array(F.interpolate(torch.tensor(cam)[None, None], size=img.shape, mode="bilinear",
                                     align_corners=False)[0, 0])
    heat = mcm.jet(cam)[..., :3]
    return (1 - alpha) * np.stack([img] * 3, -1) + alpha * heat


def plot_gradcam_7classes(model, layers, ds, y, p, in_ch, device, path) -> None:
    """Mỗi lớp cảm xúc 1 ảnh (ưu tiên ảnh đoán đúng) x [ảnh gốc + 3 tầng]."""
    fig, axes = plt.subplots(len(EMOTION_LABELS), 1 + len(layers), figsize=(2.3 * (1 + len(layers)), 2.3 * 7))
    for c, name in enumerate(EMOTION_LABELS):
        good = np.where((y == c) & (p == c))[0]
        pool = good if len(good) else np.where(y == c)[0]
        i = int(pool[0])
        x = ds[i][0].unsqueeze(0).to(device)
        img = to_display(ds[i][0], in_ch)
        axes[c, 0].imshow(img, cmap="gray")
        axes[c, 0].set_ylabel(name, fontsize=11, rotation=0, labelpad=38, va="center")
        axes[c, 0].set_title("Ảnh gốc" if c == 0 else "", fontsize=10)
        for k, (lname, layer) in enumerate(layers):
            cam, _ = grad_cam(model, layer, x, target=int(p[i]))
            axes[c, k + 1].imshow(overlay(img, cam))
            if c == 0:
                axes[c, k + 1].set_title(lname, fontsize=9)
        for a in axes[c]:
            a.set_xticks([]); a.set_yticks([])
    plt.suptitle("Grad-CAM theo 7 lớp cảm xúc qua 3 tầng tích chập", fontsize=13)
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()


def plot_gradcam_wrong(model, layer, ds, y, p, in_ch, device, path, n=8, seed=42) -> None:
    wrong = np.where(y != p)[0]
    idxs = np.random.default_rng(seed).choice(wrong, min(n, len(wrong)), replace=False)
    fig, axes = plt.subplots(2, len(idxs), figsize=(2.3 * len(idxs), 5))
    axes = np.array(axes).reshape(2, -1)
    for k, i in enumerate(idxs):
        x = ds[i][0].unsqueeze(0).to(device)
        img = to_display(ds[i][0], in_ch)
        cam, _ = grad_cam(model, layer, x, target=int(p[i]))
        axes[0, k].imshow(img, cmap="gray")
        axes[0, k].set_title(f"{EMOTION_LABELS[p[i]]} ({EMOTION_LABELS[y[i]]})", fontsize=9, color="red")
        axes[1, k].imshow(overlay(img, cam))
        for a in axes[:, k]:
            a.set_xticks([]); a.set_yticks([])
    axes[0, 0].set_ylabel("Ảnh gốc"); axes[1, 0].set_ylabel("Grad-CAM")
    plt.suptitle("Grad-CAM các ca dự đoán SAI  [Dự đoán (Thực tế)]", fontsize=12)
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()


# ----------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", choices=["public", "private"], default="private")
    ap.add_argument("--fig-dir", default="figures")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    set_seed(42)
    fig_dir = Path(args.fig_dir); fig_dir.mkdir(exist_ok=True, parents=True)
    device = get_device()
    model, img_size, in_ch, name = load_any_model(resolve_ckpt(args.ckpt), device)
    print(f"Model: {name} | img_size={img_size} | in_channels={in_ch}")

    loader = get_eval_loaders(img_size, in_ch)[args.split]
    ds = loader.dataset
    y, p, _ = predict(model, loader, device)
    t = args.tag

    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    first_conv, last_conv = convs[0], model.last_conv_layer
    plot_filters(first_conv, f"Filters lớp Conv ĐẦU ({name})", fig_dir / f"filters_first_conv{t}.png")
    plot_filters(last_conv, f"Final Convolution Filters - lớp Conv CUỐI ({name})",
                 fig_dir / f"filters_last_conv{t}.png")

    x0 = ds[int(np.where(y == p)[0][0])][0].unsqueeze(0).to(device)
    plot_feature_maps(model, first_conv, x0, "Feature Maps - lớp Conv đầu", fig_dir / f"feature_maps_first_conv{t}.png")
    plot_feature_maps(model, last_conv, x0, "Feature Maps - lớp Conv cuối", fig_dir / f"feature_maps_last_conv{t}.png")

    layers = pick_conv_layers(model)
    plot_gradcam_7classes(model, layers, ds, y, p, in_ch, device, fig_dir / f"gradcam_7classes_3layers{t}.png")
    plot_gradcam_wrong(model, layers[-1][1], ds, y, p, in_ch, device, fig_dir / f"gradcam_wrong_cases{t}.png")
    print(f"Xong! Xem hình trong {fig_dir}/")


if __name__ == "__main__":
    main()
