"""
Các kiến trúc CNN Baseline viết 100% thuần PyTorch cho bài toán FER2013 (48x48).
Hỗ trợ linh hoạt hàm kích hoạt (ReLU, LeakyReLU, Sigmoid, ELU, GELU) để phục vụ
Thực nghiệm Nhóm 1 theo yêu cầu Slide 4 (Emotion_Recognition.pdf).

Cung cấp thuộc tính `last_conv_layer` giúp Thành viên 3 dễ dàng trích xuất
Final Convolution Filters, Feature Maps và Grad-CAM Heatmap.

Phụ trách: Thành viên 1 - Lý Hồng Hưng (3123411132).
"""

import torch
import torch.nn as nn


def get_activation(name: str = "relu") -> nn.Module:
    """Trả về module hàm kích hoạt theo tên."""
    key = name.strip().lower()
    if key == "relu":
        return nn.ReLU(inplace=True)
    if key == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    if key == "sigmoid":
        return nn.Sigmoid()
    if key == "elu":
        return nn.ELU(inplace=True)
    if key == "gelu":
        return nn.GELU()
    raise ValueError(f"Hàm kích hoạt không được hỗ trợ: {name}")


class ConvBNAct(nn.Module):
    """Khối cơ sở Conv2d(3x3) -> BatchNorm2d -> Activation."""

    def __init__(self, in_ch: int, out_ch: int, activation: str = "relu") -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SmallCNN(nn.Module):
    """
    Mô hình Baseline 1 (SmallCNN - 3 Block):
      - Block 1: Conv(32) -> Pool(2x2) -> Dropout(0.25)   [48 -> 24]
      - Block 2: Conv(64) -> Pool(2x2) -> Dropout(0.25)   [24 -> 12]
      - Block 3: Conv(128) -> Pool(2x2) -> Dropout(0.25)  [12 -> 6]
      - Classifier: Flatten -> FC(256) -> Dropout(0.5) -> FC(7)
    """

    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 1,
        activation: str = "relu",
        dropout: float = 0.25,
        fc_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            ConvBNAct(in_channels, 32, activation),
            ConvBNAct(32, 32, activation),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
        )
        self.block2 = nn.Sequential(
            ConvBNAct(32, 64, activation),
            ConvBNAct(64, 64, activation),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
        )
        self.block3 = nn.Sequential(
            ConvBNAct(64, 128, activation),
            ConvBNAct(128, 128, activation),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.BatchNorm1d(256),
            get_activation(activation),
            nn.Dropout(fc_dropout),
            nn.Linear(256, num_classes),
        )

    @property
    def last_conv_layer(self) -> nn.Conv2d:
        """Lớp Conv2d cuối cùng để Thành viên 3 trực quan hóa Filters / Grad-CAM."""
        return self.block3[1].conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        return self.classifier(x)


class MediumCNN(nn.Module):
    """
    Mô hình Baseline 2 (MediumCNN - VGG-style 3 Block cải tiến với BatchNorm):
      - Block 1: 2 x Conv(64, 3x3)  -> Pool(2x2) -> Dropout(0.25)  [48 -> 24]
      - Block 2: 3 x Conv(128, 3x3) -> Pool(2x2) -> Dropout(0.25)  [24 -> 12]
      - Block 3: 3 x Conv(256, 3x3) -> Pool(2x2) -> Dropout(0.25)  [12 -> 6]
      - Classifier: Flatten -> FC(512) -> BN -> Act -> Dropout(0.5) -> FC(7)
    """

    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 1,
        activation: str = "relu",
        dropout: float = 0.25,
        fc_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            ConvBNAct(in_channels, 64, activation),
            ConvBNAct(64, 64, activation),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
        )
        self.block2 = nn.Sequential(
            ConvBNAct(64, 128, activation),
            ConvBNAct(128, 128, activation),
            ConvBNAct(128, 128, activation),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
        )
        self.block3 = nn.Sequential(
            ConvBNAct(128, 256, activation),
            ConvBNAct(256, 256, activation),
            ConvBNAct(256, 256, activation),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 512),
            nn.BatchNorm1d(512),
            get_activation(activation),
            nn.Dropout(fc_dropout),
            nn.Linear(512, num_classes),
        )

    @property
    def last_conv_layer(self) -> nn.Conv2d:
        """Lớp Conv2d cuối cùng để Thành viên 3 trực quan hóa Filters / Grad-CAM."""
        return self.block3[2].conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        return self.classifier(x)


def build_baseline_model(
    arch: str = "small_cnn",
    num_classes: int = 7,
    in_channels: int = 1,
    activation: str = "relu",
    dropout: float = 0.25,
    fc_dropout: float = 0.5,
) -> nn.Module:
    """Factory tạo mô hình baseline theo tên kiến trúc."""
    key = arch.strip().lower()
    if key == "small_cnn":
        return SmallCNN(
            num_classes=num_classes,
            in_channels=in_channels,
            activation=activation,
            dropout=dropout,
            fc_dropout=fc_dropout,
        )
    if key == "medium_cnn":
        return MediumCNN(
            num_classes=num_classes,
            in_channels=in_channels,
            activation=activation,
            dropout=dropout,
            fc_dropout=fc_dropout,
        )
    raise ValueError(f"Kiến trúc baseline không hợp lệ: {arch}. Chọn 'small_cnn' hoặc 'medium_cnn'.")
