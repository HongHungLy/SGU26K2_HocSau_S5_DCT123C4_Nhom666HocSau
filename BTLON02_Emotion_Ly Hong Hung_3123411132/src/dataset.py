"""
Module nạp dữ liệu & Data Augmentation chuẩn PyTorch cho bộ Kaggle FER2013.
Hỗ trợ tự động nhận diện cả 2 định dạng phổ biến trên Kaggle & Local:
  1) File fer2013.csv (cột emotion, pixels, Usage: Training / PublicTest / PrivateTest)
  2) Thư mục ảnh (Training/PublicTest/PrivateTest hoặc train/test)

Hỗ trợ cả in_channels=1 (CNN tự thiết kế) và in_channels=3 + img_size linh hoạt
(dùng chung cho Thành viên 2 khi fine-tune VGG16 / ResNet / EfficientNet).

Phụ trách: Thành viên 1 - Lý Hồng Hưng (3123411132).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .utils import EMOTION_LABELS

# Ánh xạ tên thư mục ảnh (không phân biệt hoa/thường) -> nhãn chuẩn 0..6
LABEL_NAME_TO_IDX = {name.lower(): idx for idx, name in enumerate(EMOTION_LABELS)}


def discover_fer2013_source(data_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    """
    Tự động dò tìm nguồn dữ liệu FER2013 ở thư mục local (./data) hoặc /kaggle/input.

    Returns:
        dict với khóa 'mode' in {'csv', 'folder_3split', 'folder_2split'} và đường dẫn tương ứng.
    """
    search_roots: List[Path] = []
    if data_dir is not None:
        p = Path(data_dir)
        if p.is_file() and p.suffix.lower() == ".csv":
            return {"mode": "csv", "csv_path": p}
        search_roots.append(p)
    else:
        search_roots.extend([Path("data"), Path("../data"), Path("/kaggle/input")])

    # 1. Ưu tiên tìm file fer2013.csv (có sẵn đúng 3 tập Training / PublicTest / PrivateTest)
    for root in search_roots:
        if not root.exists():
            continue
        for csv_candidate in sorted(root.rglob("*.csv")):
            if "fer2013" in csv_candidate.name.lower() or "icml_face_data" in csv_candidate.name.lower():
                try:
                    cols = [c.strip().lower() for c in pd.read_csv(csv_candidate, nrows=1).columns]
                    if "pixels" in cols and "emotion" in cols:
                        return {"mode": "csv", "csv_path": csv_candidate}
                except Exception:
                    continue

    # 2. Tìm dạng thư mục 3 tập (Training / PublicTest / PrivateTest)
    for root in search_roots:
        if not root.exists():
            continue
        for d in [root, *sorted(root.rglob("*"))]:
            if not d.is_dir():
                continue
            if (d / "Training").is_dir() and (d / "PublicTest").is_dir() and (d / "PrivateTest").is_dir():
                return {
                    "mode": "folder_3split",
                    "train_dir": d / "Training",
                    "val_dir": d / "PublicTest",
                    "test_dir": d / "PrivateTest",
                }

    # 3. Tìm dạng thư mục 2 tập phổ biến trên Kaggle (msambare/fer2013: train/ và test/)
    for root in search_roots:
        if not root.exists():
            continue
        for d in [root, *sorted(root.rglob("*"))]:
            if not d.is_dir():
                continue
            if (d / "train").is_dir() and (d / "test").is_dir():
                sub_names = {p.name.lower() for p in (d / "train").iterdir() if p.is_dir()}
                if {"angry", "happy", "sad"}.issubset(sub_names):
                    return {
                        "mode": "folder_2split",
                        "train_dir": d / "train",
                        "test_dir": d / "test",
                    }

    raise FileNotFoundError(
        "Không tìm thấy dữ liệu FER2013! Hãy đặt fer2013.csv vào thư mục ./data/ "
        "hoặc trên Kaggle bấm 'Add Input' và thêm dataset FER2013 (VD: deadskull7/fer2013 hoặc msambare/fer2013)."
    )


def get_transforms(
    is_train: bool = True,
    use_aug: bool = True,
    img_size: int = 48,
    in_channels: int = 1,
) -> transforms.Compose:
    """
    Tạo pipeline tiền xử lý & Data Augmentation chuẩn PyTorch.

    Args:
        is_train: True cho tập Training, False cho PublicTest (Val) & PrivateTest (Test).
        use_aug: Có bật Data Augmentation hay không (chỉ áp dụng khi is_train=True).
        img_size: Kích thước ảnh đầu vào (48 cho CNN tự thiết kế; 128/224 cho Transfer Learning).
        in_channels: 1 (grayscale) hoặc 3 (RGB cho mạng pretrained torchvision).
    """
    ops: List[Any] = []

    if img_size != 48:
        ops.append(transforms.Resize((img_size, img_size)))

    if in_channels == 3:
        ops.append(transforms.Grayscale(num_output_channels=3))
    else:
        ops.append(transforms.Grayscale(num_output_channels=1))

    if is_train and use_aug:
        ops.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
            ),
        ])

    ops.append(transforms.ToTensor())

    if in_channels == 3:
        # Chuẩn ImageNet cho Transfer Learning (TV2)
        ops.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    else:
        # Chuẩn hóa ảnh grayscale về [-1, 1]
        ops.append(transforms.Normalize(mean=[0.5], std=[0.5]))

    if is_train and use_aug:
        ops.append(transforms.RandomErasing(p=0.2, scale=(0.02, 0.12), ratio=(0.3, 3.3)))

    return transforms.Compose(ops)


class FER2013ArrayDataset(Dataset):
    """Dataset từ mảng numpy (N, 48, 48) uint8 - đọc trực tiếp từ fer2013.csv."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.images = images
        self.labels = labels.astype(np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.fromarray(self.images[idx], mode="L")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])


class FER2013FolderDataset(Dataset):
    """Dataset từ danh sách đường dẫn ảnh (dạng thư mục theo tên lớp)."""

    def __init__(
        self,
        file_paths: List[Path],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.file_paths = file_paths
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        with Image.open(self.file_paths[idx]) as im:
            img = im.convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])


def _scan_class_folder(folder: Path) -> Tuple[List[Path], List[int]]:
    """Quét thư mục chứa 7 thư mục con theo tên cảm xúc."""
    paths: List[Path] = []
    labels: List[int] = []
    exts = {".png", ".jpg", ".jpeg", ".bmp"}

    for sub in sorted(folder.iterdir()):
        if not sub.is_dir():
            continue
        key = sub.name.strip().lower()
        if key not in LABEL_NAME_TO_IDX:
            continue
        label_idx = LABEL_NAME_TO_IDX[key]
        for f in sorted(sub.iterdir()):
            if f.is_file() and f.suffix.lower() in exts:
                paths.append(f)
                labels.append(label_idx)
    return paths, labels


def _parse_fer2013_csv(csv_path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Đọc file fer2013.csv và tách thành 3 tập Training / PublicTest / PrivateTest."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    usage_col = next((c for c in df.columns if c.lower() == "usage"), None)
    emotion_col = next(c for c in df.columns if c.lower() == "emotion")
    pixels_col = next(c for c in df.columns if c.lower() == "pixels")

    def to_arrays(sub_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        imgs = np.vstack([
            np.fromstring(p, dtype=np.uint8, sep=" ") for p in sub_df[pixels_col].values
        ]).reshape(-1, 48, 48)
        lbls = sub_df[emotion_col].values.astype(np.int64)
        return imgs, lbls

    if usage_col is not None:
        usage_vals = df[usage_col].astype(str).str.strip()
        train_df = df[usage_vals == "Training"]
        val_df = df[usage_vals == "PublicTest"]
        test_df = df[usage_vals == "PrivateTest"]
        return {
            "train": to_arrays(train_df),
            "val": to_arrays(val_df),
            "test": to_arrays(test_df),
        }

    # Trường hợp CSV không có cột Usage -> chia stratified 80/10/10
    all_imgs, all_lbls = to_arrays(df)
    idx = np.arange(len(all_lbls))
    tr_idx, tmp_idx = train_test_split(idx, test_size=0.2, stratify=all_lbls, random_state=42)
    va_idx, te_idx = train_test_split(tmp_idx, test_size=0.5, stratify=all_lbls[tmp_idx], random_state=42)
    return {
        "train": (all_imgs[tr_idx], all_lbls[tr_idx]),
        "val": (all_imgs[va_idx], all_lbls[va_idx]),
        "test": (all_imgs[te_idx], all_lbls[te_idx]),
    }


def get_dataloaders(
    data_dir: Optional[str | Path] = None,
    batch_size: int = 64,
    use_aug: bool = True,
    img_size: int = 48,
    in_channels: int = 1,
    num_workers: int = 2,
    pin_memory: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Hàm chuẩn dùng chung cho cả 3 thành viên để tạo DataLoader cho FER2013.

    Returns:
        train_loader, val_loader (PublicTest), test_loader (PrivateTest), info dict
    """
    source = discover_fer2013_source(data_dir)
    mode = source["mode"]

    train_tf = get_transforms(is_train=True, use_aug=use_aug, img_size=img_size, in_channels=in_channels)
    eval_tf = get_transforms(is_train=False, use_aug=False, img_size=img_size, in_channels=in_channels)

    if mode == "csv":
        splits = _parse_fer2013_csv(source["csv_path"])
        train_ds = FER2013ArrayDataset(*splits["train"], transform=train_tf)
        val_ds = FER2013ArrayDataset(*splits["val"], transform=eval_tf)
        test_ds = FER2013ArrayDataset(*splits["test"], transform=eval_tf)
        train_labels, val_labels, test_labels = splits["train"][1], splits["val"][1], splits["test"][1]
    elif mode == "folder_3split":
        tr_p, tr_y = _scan_class_folder(source["train_dir"])
        va_p, va_y = _scan_class_folder(source["val_dir"])
        te_p, te_y = _scan_class_folder(source["test_dir"])
        train_ds = FER2013FolderDataset(tr_p, tr_y, transform=train_tf)
        val_ds = FER2013FolderDataset(va_p, va_y, transform=eval_tf)
        test_ds = FER2013FolderDataset(te_p, te_y, transform=eval_tf)
        train_labels, val_labels, test_labels = np.array(tr_y), np.array(va_y), np.array(te_y)
    else:  # folder_2split (VD: msambare/fer2013 chỉ có train/ và test/)
        tr_p, tr_y = _scan_class_folder(source["train_dir"])
        eval_p, eval_y = _scan_class_folder(source["test_dir"])
        idx = np.arange(len(eval_y))
        va_idx, te_idx = train_test_split(idx, test_size=0.5, stratify=eval_y, random_state=seed)
        va_p = [eval_p[i] for i in va_idx]
        va_y = [eval_y[i] for i in va_idx]
        te_p = [eval_p[i] for i in te_idx]
        te_y = [eval_y[i] for i in te_idx]
        train_ds = FER2013FolderDataset(tr_p, tr_y, transform=train_tf)
        val_ds = FER2013FolderDataset(va_p, va_y, transform=eval_tf)
        test_ds = FER2013FolderDataset(te_p, te_y, transform=eval_tf)
        train_labels, val_labels, test_labels = np.array(tr_y), np.array(va_y), np.array(te_y)

    g = torch.Generator()
    g.manual_seed(seed)

    use_pin = pin_memory and torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin,
    )

    info = {
        "source": source,
        "num_classes": len(EMOTION_LABELS),
        "class_names": EMOTION_LABELS,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "train_labels": train_labels,
        "val_labels": val_labels,
        "test_labels": test_labels,
    }
    return train_loader, val_loader, test_loader, info
