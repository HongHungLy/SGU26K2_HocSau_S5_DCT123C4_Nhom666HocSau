"""
Script: generate_data.py
Nhiệm vụ (Thành viên 1):
1. Khởi tạo dữ liệu Gaussian 2D với thông số chuẩn theo đề bài:
   - mu = [0, 0]^T, C = [[0.3, -0.5], [-0.5, 2.0]]
   - Miền x1 in [-2, 2], x2 in [-4, 4]
2. Tạo 3 tập dữ liệu độc lập:
   - Test set: 100 mẫu (cố định để đánh giá cuối cùng, không rò rỉ)
   - Validation set: 400 mẫu (dùng để chọn checkpoint mô hình)
   - Training pool: 2000 mẫu (dùng để trích xuất tập con cho Thực nghiệm A)
3. Xuất file CSV vào thư mục data/
4. Vẽ đầy đủ các biểu đồ trực quan hóa dữ liệu (Mục 3.1 & Tiêu chí 1 trong Rubric):
   - figures/data_3d_scatter.png
   - figures/true_gaussian_surface.png
   - figures/data_distributions.png
"""

import os
import sys
import argparse
import pandas as pd

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi bảng mã cp1252
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.dataset import Gaussian2D, create_data_splits, load_dataset_from_csv
from src.plot_utils import plot_3d_scatter, plot_true_surface, plot_data_distributions


def main():
    parser = argparse.ArgumentParser(description="Sinh dữ liệu và trực quan hóa EDA cho BTL01")
    parser.add_argument("--data_dir", type=str, default="data", help="Thư mục lưu trữ CSV")
    parser.add_argument("--figures_dir", type=str, default="figures", help="Thư mục lưu trữ hình ảnh")
    parser.add_argument("--base_seed", type=int, default=42, help="Seed gốc để sinh dữ liệu")
    args = parser.parse_args()

    print("=" * 65)
    print(" BTL01: KHỞI TẠO VÀ TRỰC QUAN HÓA DỮ LIỆU GAUSSIAN 2 CHIỀU ")
    print("=" * 65)

    # 1. Sinh các tập dữ liệu độc lập
    print("\n[Bước 1/3] Đang sinh các tập dữ liệu độc lập...")
    splits = create_data_splits(
        train_pool_size=2000,
        val_size=400,
        test_size=100,
        base_seed=args.base_seed,
        data_dir=args.data_dir,
    )

    X_train, y_train = splits["train_pool"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    print(f" -> Training Pool : {len(X_train)} mẫu (đã lưu data/train_pool_2000.csv)")
    print(f" -> Validation    : {len(X_val)} mẫu (đã lưu data/val.csv)")
    print(f" -> Test          : {len(X_test)} mẫu (đã lưu data/test.csv)")

    # 2. Thống kê mô tả dữ liệu
    print("\n[Bước 2/3] Bảng thống kê mô tả (Descriptive Statistics):")
    df_train = pd.DataFrame({"x1": X_train[:, 0], "x2": X_train[:, 1], "y": y_train[:, 0]})
    stats_df = df_train.describe().T[["count", "mean", "std", "min", "50%", "max"]]
    stats_df.rename(columns={"50%": "median"}, inplace=True)
    print(stats_df.to_string())

    # Lưu bảng thống kê ra CSV để đưa vào báo cáo
    stats_csv_path = os.path.join(args.data_dir, "data_statistics.csv")
    stats_df.to_csv(stats_csv_path)
    print(f" -> Đã lưu bảng thống kê vào: {stats_csv_path}")

    # 3. Vẽ biểu đồ trực quan hóa dữ liệu theo yêu cầu rubric
    print("\n[Bước 3/3] Đang vẽ các biểu đồ trực quan hóa dữ liệu (figures/)...")
    gaussian_fn = Gaussian2D()

    # Hình 1: 3D Scatter plot dữ liệu huấn luyện
    scatter_path = os.path.join(args.figures_dir, "data_3d_scatter.png")
    plot_3d_scatter(
        X_train[:500], y_train[:500],
        title="3D Scatter Plot của Dữ liệu Huấn luyện Gaussian (500 mẫu đại diện)",
        save_path=scatter_path,
    )

    # Hình 2: Bề mặt hàm thật lý thuyết 3D
    surface_path = os.path.join(args.figures_dir, "true_gaussian_surface.png")
    plot_true_surface(
        gaussian_fn,
        x1_range=(-2.0, 2.0),
        x2_range=(-4.0, 4.0),
        resolution=60,
        title=r"Bề mặt Hàm Mật độ Gaussian 2 Chiều: $\mu = [0, 0]^T$, $C = [[0.3, -0.5], [-0.5, 2.0]]$",
        save_path=surface_path,
    )

    # Hình 3: Phân bố tần suất histogram
    dist_path = os.path.join(args.figures_dir, "data_distributions.png")
    plot_data_distributions(
        X_train, y_train,
        title="Phân bố Dữ liệu Huấn luyện: x1 in [-2, 2], x2 in [-4, 4] và y = f(x1, x2)",
        save_path=dist_path,
    )

    print("\n Hoàn thành xuất sắc toàn bộ bước Chuẩn bị Dữ liệu & Trực quan hóa!")
    print(f"Các hình ảnh đã được lưu tại thư mục: '{args.figures_dir}/'")


if __name__ == "__main__":
    main()
