"""
BTL01: Xấp xỉ hàm mật độ Gaussian 2 chiều sử dụng mạng MLP bằng PyTorch.
Thành viên 1: Quản lý Dữ liệu, Kiến trúc Mô hình & Thực nghiệm A.
"""

from .dataset import Gaussian2D, generate_dataset, create_data_splits, GaussianDataset
from .model import FlexibleMLP
from .trainer import Trainer, EarlyStopping
from .evaluate import evaluate_model, compute_metrics
from .plot_utils import plot_3d_scatter, plot_true_surface, plot_loss_curves, plot_pred_vs_true_3d
