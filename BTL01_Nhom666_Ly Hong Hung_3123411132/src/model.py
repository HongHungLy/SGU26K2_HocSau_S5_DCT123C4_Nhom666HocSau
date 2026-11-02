"""
Module: model.py
Nhiệm vụ (Thành viên 1):
- Thiết kế kiến trúc mạng Perceptron đa tầng (MLP) tổng quát bằng PyTorch (torch.nn.Module, nn.Linear).
- Tùy biến linh hoạt:
    + Số tầng ẩn C (Depth)
    + Số nơ-ron mỗi tầng B (Width)
    + Hàm kích hoạt đa dạng (Sigmoid, Tanh, ReLU, GELU, LeakyReLU)
- Đầu vào 2 chiều (x1, x2), đầu ra 1 chiều (y_hat).
"""

from typing import List, Union
import torch
import torch.nn as nn


class FlexibleMLP(nn.Module):
    """
    Mạng Perceptron Đa Tầng (MLP) tùy biến linh hoạt số tầng và số nơ-ron.
    
    Tham số:
        input_dim: Số chiều đầu vào (mặc định = 2 cho x1, x2).
        hidden_dims: Danh sách số lượng nơ-ron cho từng tầng ẩn.
                     Ví dụ: [128] là 1 tầng ẩn 128 nodes.
                            [64, 64] là 2 tầng ẩn 64 nodes.
        output_dim: Số chiều đầu ra (mặc định = 1 cho y).
        activation: Tên hàm kích hoạt ('sigmoid', 'tanh', 'relu', 'leaky_relu', 'gelu').
        use_output_activation: Có sử dụng hàm kích hoạt ở tầng output hay không.
                               Với hồi quy xấp xỉ hàm Gaussian có y in [0, 0.27],
                               đầu ra tuyến tính (Linear) hoặc Sigmoid đều có thể áp dụng.
                               Mặc định None (tuyến tính) để tránh bão hòa gradient.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Union[List[int], int] = 128,
        output_dim: int = 1,
        activation: str = "sigmoid",
        use_output_activation: bool = False,
    ):
        super(FlexibleMLP, self).__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        self.hidden_dims = hidden_dims
        self.activation_name = activation.lower()
        self.use_output_activation = use_output_activation

        # Ánh xạ hàm kích hoạt
        activation_dict = {
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(negative_slope=0.01),
            "gelu": nn.GELU(),
        }

        if self.activation_name not in activation_dict:
            raise ValueError(
                f"Hàm kích hoạt '{activation}' không hỗ trợ. Hãy chọn một trong {list(activation_dict.keys())}"
            )
        act_fn = activation_dict[self.activation_name]

        layers = []
        prev_dim = input_dim

        # Xây dựng các tầng ẩn
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(act_fn)
            prev_dim = h_dim

        # Tầng đầu ra (Linear)
        layers.append(nn.Linear(prev_dim, output_dim))
        if use_output_activation:
            layers.append(act_fn)

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Khởi tạo trọng số Xavier cho hàm Sigmoid/Tanh hoặc He cho ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.activation_name in ["sigmoid", "tanh"]:
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lan truyền xuôi (Forward pass): x -> y_hat."""
        return self.network(x)

    def count_parameters(self) -> int:
        """Đếm tổng số tham số có thể huấn luyện (trainable parameters)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"FlexibleMLP(hidden_dims={self.hidden_dims}, "
            f"activation='{self.activation_name}', "
            f"total_params={self.count_parameters()})"
        )
