"""
Các kiến trúc CNN nâng cao cho bài toán FER2013 (100% PyTorch / torchvision).

Gồm 3 nhóm mô hình (Vai trò B - Thành viên 2):
  (a) LargeCNN     : VGG-style 4 block (64 -> 128 -> 256 -> 512) + BatchNorm + Dropout2d tăng dần theo độ sâu.
  (b) ResNetMini   : Residual CNN tự thiết kế cho ảnh 48x48 (stem 3x3, 4 stage, kết nối tắt).
  (c) TransferCNN  : Bọc các mạng pretrained của torchvision (ResNet18/34, VGG16, EfficientNet-B0, MobileNet)
                     hỗ trợ fine-tune toàn phần, freeze backbone và learning rate phân tầng.

Mọi mô hình đều cung cấp:
  - `model.last_conv_layer`  : lớp Conv2d cuối cùng (để Thành viên 3 vẽ Final Conv Filters).
  - `model.gradcam_layer`    : module nên gắn hook khi vẽ Grad-CAM / Feature Maps.

Ngoài ra có `build_model(...)`, `save_deployable_checkpoint(...)` và `load_model_from_checkpoint(...)`
để file `best_model.pth` tự mô tả kiến trúc (Thành viên 3 nạp được mà không cần đoán cấu hình).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .models_baseline import ConvBNAct, get_activation


def _last_conv2d(module: nn.Module) -> nn.Conv2d:
    """Trả về lớp nn.Conv2d xuất hiện sau cùng trong module (theo thứ tự đăng ký)."""
    convs = [m for m in module.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise RuntimeError("Mô hình không chứa lớp Conv2d nào.")
    return convs[-1]


def _init_weights(model: nn.Module) -> None:
    """Khởi tạo Kaiming cho Conv/Linear, BatchNorm = (1, 0)."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)


# ----------------------------------------------------------------------------------------------
# (a) LargeCNN - VGG-style 4 block
# ----------------------------------------------------------------------------------------------
class LargeCNN(nn.Module):
    """
    VGG-style CNN 4 block cho ảnh 48x48x1:
      Block 1: 2 x Conv(64)  -> MaxPool -> Dropout2d(p1)   [48 -> 24]
      Block 2: 2 x Conv(128) -> MaxPool -> Dropout2d(p2)   [24 -> 12]
      Block 3: 3 x Conv(256) -> MaxPool -> Dropout2d(p3)   [12 -> 6]
      Block 4: 3 x Conv(512) -> MaxPool -> Dropout2d(p4)   [6  -> 3]
      Head   : AdaptiveAvgPool(3x3) -> Flatten -> FC(512) -> BN -> Act -> Dropout(fc) -> FC(7)
               (head="gap": AdaptiveAvgPool(1x1) thay cho Flatten để giảm tham số)

    Dropout2d tăng dần theo độ sâu (0.1 -> 0.4): các block đầu giữ nhiều thông tin cạnh/góc,
    các block sau (nhiều tham số, dễ overfit) được regularize mạnh hơn.
    """

    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 1,
        activation: str = "relu",
        channels: Sequence[int] = (64, 128, 256, 512),
        convs_per_block: Sequence[int] = (2, 2, 3, 3),
        block_dropouts: Sequence[float] = (0.1, 0.2, 0.3, 0.4),
        fc_hidden: int = 512,
        fc_dropout: float = 0.5,
        head: str = "flatten",
    ) -> None:
        super().__init__()
        assert len(channels) == len(convs_per_block) == len(block_dropouts)

        blocks: List[nn.Module] = []
        prev = in_channels
        for ch, n_conv, p in zip(channels, convs_per_block, block_dropouts):
            layers: List[nn.Module] = []
            for i in range(n_conv):
                layers.append(ConvBNAct(prev if i == 0 else ch, ch, activation))
            layers.append(nn.MaxPool2d(2, 2))
            if p > 0:
                layers.append(nn.Dropout2d(p))
            blocks.append(nn.Sequential(*layers))
            prev = ch
        self.features = nn.Sequential(*blocks)

        if head == "gap":
            self.pool = nn.AdaptiveAvgPool2d(1)
            flat_dim = prev
        elif head == "flatten":
            self.pool = nn.AdaptiveAvgPool2d((3, 3))
            flat_dim = prev * 3 * 3
        else:
            raise ValueError("head phải là 'flatten' hoặc 'gap'")

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            get_activation(activation),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden, num_classes),
        )
        _init_weights(self)

    @property
    def last_conv_layer(self) -> nn.Conv2d:
        return _last_conv2d(self.features)

    @property
    def gradcam_layer(self) -> nn.Module:
        return self.last_conv_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


# ----------------------------------------------------------------------------------------------
# (b) ResNetMini - Residual CNN cho 48x48
# ----------------------------------------------------------------------------------------------
class BasicResBlock(nn.Module):
    """
    Khối residual cơ bản: Conv3x3-BN-Act-[Dropout2d]-Conv3x3-BN, cộng shortcut rồi Act.
    Shortcut dùng Conv1x1(stride)+BN khi đổi kích thước/số kênh (projection shortcut).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = get_activation(activation)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = get_activation(activation)

        if stride != 1 or in_ch != out_ch:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return self.act2(out + self.shortcut(x))


class ResNetMini(nn.Module):
    """
    Residual CNN thiết kế riêng cho ảnh 48x48 (không dùng stem 7x7/stride 2 + MaxPool của ImageNet
    vì sẽ làm mất chi tiết khóe mắt/miệng ở ảnh nhỏ):

      Stem   : Conv3x3(base) + BN + Act                                   [48x48]
      Stage1 : layers[0] x ResBlock(base,   stride 1)                     [48x48]
      Stage2 : layers[1] x ResBlock(2*base, stride 2)                     [24x24]
      Stage3 : layers[2] x ResBlock(4*base, stride 2)                     [12x12]
      Stage4 : layers[3] x ResBlock(8*base, stride 2)                     [6x6]
      Head   : GlobalAvgPool -> Dropout(fc) -> FC(7)

    base_width=64, layers=(2,2,2,2) cho các kênh 64 -> 128 -> 256 -> 512.
    base_width=32 cho phiên bản "mini" nhẹ hơn ~4 lần về tham số/FLOPs.
    `zero_init_residual=True` khởi tạo gamma của BN cuối mỗi block = 0 để mạng bắt đầu gần như đồng nhất
    (ổn định hơn khi huấn luyện sâu).
    """

    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 1,
        activation: str = "relu",
        base_width: int = 64,
        layers: Sequence[int] = (2, 2, 2, 2),
        block_dropout: float = 0.1,
        fc_dropout: float = 0.3,
        zero_init_residual: bool = True,
    ) -> None:
        super().__init__()
        self.stem = ConvBNAct(in_channels, base_width, activation)

        widths = [base_width * m for m in (1, 2, 4, 8)][: len(layers)]
        strides = [1, 2, 2, 2][: len(layers)]
        stages: List[nn.Module] = []
        prev = base_width
        for w, n_blocks, s in zip(widths, layers, strides):
            blocks = []
            for i in range(n_blocks):
                blocks.append(
                    BasicResBlock(prev if i == 0 else w, w, stride=s if i == 0 else 1,
                                  activation=activation, dropout=block_dropout)
                )
            stages.append(nn.Sequential(*blocks))
            prev = w
        self.stages = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(fc_dropout),
            nn.Linear(prev, num_classes),
        )
        _init_weights(self)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicResBlock):
                    nn.init.zeros_(m.bn2.weight)

    @property
    def last_conv_layer(self) -> nn.Conv2d:
        return _last_conv2d(self.stages)

    @property
    def gradcam_layer(self) -> nn.Module:
        """Đầu ra của stage cuối (sau phép cộng residual) - điểm hook chuẩn cho Grad-CAM."""
        return self.stages[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        return self.classifier(self.pool(x))


# ----------------------------------------------------------------------------------------------
# (c) Transfer Learning / Fine-tuning (torchvision)
# ----------------------------------------------------------------------------------------------
SUPPORTED_TRANSFER = (
    "resnet18", "resnet34", "resnet50",
    "vgg16", "vgg16_bn",
    "efficientnet_b0", "mobilenet_v2", "mobilenet_v3_small",
)


class TransferCNN(nn.Module):
    """
    Bọc mạng pretrained của torchvision, thay đầu phân loại 1000 lớp ImageNet -> `num_classes`.
    Đầu vào luôn là ảnh 3 kênh (dataset.py nhân ảnh xám thành 3 kênh + chuẩn hóa ImageNet).

    Args:
        arch: một trong SUPPORTED_TRANSFER.
        pretrained: True -> nạp trọng số ImageNet (cần Internet lần đầu); False -> khởi tạo ngẫu nhiên.
        dropout: xác suất Dropout của đầu phân loại (với VGG áp dụng cho cả 2 lớp FC 4096).
        freeze_backbone: True -> chỉ huấn luyện đầu phân loại (feature extraction).
    """

    def __init__(
        self,
        arch: str = "resnet18",
        num_classes: int = 7,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        import torchvision.models as tvm

        key = arch.strip().lower()
        if key not in SUPPORTED_TRANSFER:
            raise ValueError(f"arch không hỗ trợ: {arch}. Chọn một trong {SUPPORTED_TRANSFER}")
        self.arch = key
        self.dropout_p = dropout

        weights = None
        if pretrained:
            try:
                weights = tvm.get_model_weights(key).DEFAULT
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Không lấy được weights pretrained cho {key}: {exc}") from exc
        try:
            self.backbone = tvm.get_model(key, weights=weights)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Không tải được trọng số pretrained cho '{key}'. Trên Kaggle hãy bật Internet = On "
                f"(hoặc đặt pretrained=False). Lỗi gốc: {exc}"
            ) from exc

        self._head_names = self._replace_head(num_classes, dropout)
        self._frozen = False
        if freeze_backbone:
            self.set_backbone_frozen(True)

    # -- thay đầu phân loại ------------------------------------------------------------------
    def _replace_head(self, num_classes: int, dropout: float) -> Tuple[str, ...]:
        m = self.backbone
        k = self.arch
        if k.startswith("resnet"):
            m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))
            return ("fc",)
        if k.startswith("vgg"):
            for layer in m.classifier:
                if isinstance(layer, nn.Dropout):
                    layer.p = dropout
            in_f = m.classifier[-1].in_features
            m.classifier[-1] = nn.Linear(in_f, num_classes)
            return ("classifier",)
        if k in ("efficientnet_b0", "mobilenet_v2"):
            in_f = m.classifier[-1].in_features
            m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
            return ("classifier",)
        if k == "mobilenet_v3_small":
            for layer in m.classifier:
                if isinstance(layer, nn.Dropout):
                    layer.p = dropout
            in_f = m.classifier[-1].in_features
            m.classifier[-1] = nn.Linear(in_f, num_classes)
            return ("classifier",)
        raise ValueError(k)

    # -- freeze / learning rate phân tầng ---------------------------------------------------
    def _backbone_body_modules(self) -> List[nn.Module]:
        return [mod for name, mod in self.backbone.named_children() if name not in self._head_names]

    def _head_modules(self) -> List[nn.Module]:
        return [mod for name, mod in self.backbone.named_children() if name in self._head_names]

    def set_backbone_frozen(self, frozen: bool = True) -> None:
        """Đóng/mở băng phần thân (mọi thứ trừ đầu phân loại)."""
        for mod in self._backbone_body_modules():
            for p in mod.parameters():
                p.requires_grad = not frozen
        self._frozen = frozen
        self.train(self.training)

    def train(self, mode: bool = True) -> "TransferCNN":
        super().train(mode)
        if self._frozen and mode:
            # Thân bị đóng băng: giữ BatchNorm ở eval để không làm lệch running stats ImageNet
            for mod in self._backbone_body_modules():
                for sub in mod.modules():
                    if isinstance(sub, nn.modules.batchnorm._BatchNorm):
                        sub.eval()
        return self

    def param_groups(self, lr_backbone: float, lr_head: float) -> List[Dict[str, Any]]:
        """2 nhóm tham số: thân (LR nhỏ - giữ đặc trưng pretrained) và đầu (LR lớn - học mới)."""
        body = [p for mod in self._backbone_body_modules() for p in mod.parameters() if p.requires_grad]
        head = [p for mod in self._head_modules() for p in mod.parameters() if p.requires_grad]
        groups: List[Dict[str, Any]] = []
        if body:
            groups.append({"params": body, "lr": lr_backbone})
        if head:
            groups.append({"params": head, "lr": lr_head})
        return groups

    # -- thuộc tính phục vụ TV3 --------------------------------------------------------------
    @property
    def last_conv_layer(self) -> nn.Conv2d:
        return _last_conv2d(self.backbone)

    @property
    def gradcam_layer(self) -> nn.Module:
        if self.arch.startswith("resnet"):
            return self.backbone.layer4
        return self.last_conv_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ----------------------------------------------------------------------------------------------
# Factory & checkpoint "tự mô tả"
# ----------------------------------------------------------------------------------------------
def build_model(arch: str, num_classes: int = 7, **kwargs: Any) -> nn.Module:
    """
    Tạo mô hình theo tên: 'large_cnn' | 'resnet_mini' | một trong SUPPORTED_TRANSFER.
    kwargs được truyền thẳng vào constructor tương ứng.
    """
    key = arch.strip().lower()
    if key == "large_cnn":
        return LargeCNN(num_classes=num_classes, **kwargs)
    if key == "resnet_mini":
        return ResNetMini(num_classes=num_classes, **kwargs)
    if key in SUPPORTED_TRANSFER:
        kwargs.pop("in_channels", None)  # TransferCNN luôn dùng 3 kênh
        return TransferCNN(arch=key, num_classes=num_classes, **kwargs)
    raise ValueError(f"Kiến trúc không hợp lệ: {arch}")


def save_deployable_checkpoint(
    path: str | Path,
    model: nn.Module,
    arch: str,
    model_kwargs: Dict[str, Any],
    img_size: int,
    in_channels: int,
    meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Lưu checkpoint gọn (chỉ model_state_dict, không có optimizer) kèm toàn bộ thông tin để dựng lại mô hình:
    {arch, model_kwargs, img_size, in_channels, num_classes, meta...}.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arch": arch,
        "model_kwargs": model_kwargs,
        "img_size": img_size,
        "in_channels": in_channels,
        "num_classes": 7,
        "model_state_dict": model.state_dict(),
        "meta": meta or {},
    }
    torch.save(payload, path)
    return path


def load_model_from_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Dựng lại mô hình từ checkpoint tạo bởi `save_deployable_checkpoint` và nạp trọng số.
    Trả về (model.eval(), ckpt_dict).  VD cho Thành viên 3:
        model, ckpt = load_model_from_checkpoint("checkpoints/best_model.pth", device)
        print(ckpt["arch"], ckpt["img_size"], ckpt["in_channels"]); layer = model.last_conv_layer
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    kwargs = dict(ckpt["model_kwargs"])
    if "pretrained" in kwargs:
        kwargs["pretrained"] = False  # trọng số sẽ được nạp từ checkpoint, không cần tải ImageNet
    model = build_model(ckpt["arch"], num_classes=ckpt.get("num_classes", 7), **kwargs)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt
