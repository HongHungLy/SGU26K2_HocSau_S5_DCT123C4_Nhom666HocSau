"""
Script: make_comparison_ABC.py
Phụ trách: Thành viên 3 (Kỹ sư Đánh giá & Độ sâu) - phục vụ Mục 5 (Thảo luận & Phân tích chuyên sâu)

Đọc kết quả TỰ SINH của 3 thực nghiệm (không nhập tay số liệu) và tạo:
    figures/exp_ABC_comparison.png        4 đồ thị so sánh A / B / C và Test MSE theo số tham số
    logs/exp_ABC_ranking_1000samples.csv  bảng xếp hạng mọi cấu hình có cùng 1000 mẫu train
    logs/exp_ABC_best_configs.csv         cấu hình tốt nhất của từng thực nghiệm

Điều kiện: đã chạy xong  python exp_C_depth.py  (tạo logs/exp_C_summary.csv).
Chạy:
    python make_comparison_ABC.py

LƯU Ý khi diễn giải (đã ghi trong báo cáo): A chạy 80 epochs với 1 tầng ẩn 128 nơ-ron;
B và C chạy 100 epochs; cả ba dùng chung tập test 100 mẫu, tanh, Adam lr = 0.005, batch 32.
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

A_HIDDEN = 128                              # baseline Thực nghiệm A: MLP 2 -> 128 -> 1
A_PARAMS = 2 * A_HIDDEN + A_HIDDEN + A_HIDDEN * 1 + 1   # = 513
RANK_SIZE = 1000                            # B và C đều cố định 1000 mẫu train


def first_existing(paths, what):
    for p in paths:
        if os.path.exists(p):
            return p
    raise SystemExit(
        f"[LỖI] Không tìm thấy file {what}. Đã thử: {paths}\n"
        f"      Hãy chạy thực nghiệm tương ứng trước (A: exp_A_datasize.py, "
        f"B: exp_B_width.py, C: exp_C_depth.py)."
    )


def load_results(logs_dir):
    a = pd.read_csv(first_existing([os.path.join(logs_dir, "exp_A_summary.csv")], "exp_A_summary.csv"))
    b = pd.read_csv(first_existing(
        [os.path.join(logs_dir, "exp_B_summary.csv"), "exp_B_summary.csv"], "exp_B_summary.csv"))
    c = pd.read_csv(first_existing([os.path.join(logs_dir, "exp_C_summary.csv")], "exp_C_summary.csv"))
    return a, b, c


def build_tables(a, b, c):
    """Bảng chung: mỗi dòng = một cấu hình (experiment, config, n_params, mse_mean, mse_std)."""
    rows = []
    for _, r in a.iterrows():
        rows.append({"experiment": "A", "config": f"A = {int(r['size'])} mẫu, 1 tầng x {A_HIDDEN}",
                     "train_size": int(r["size"]), "n_params": A_PARAMS,
                     "val_mse_mean": float("nan"), "test_mse_mean": r["mse_mean"], "test_mse_std": r["mse_std"]})
    for _, r in b.iterrows():
        rows.append({"experiment": "B", "config": f"B = {int(r['nodes_B'])} nơ-ron, 1 tầng",
                     "train_size": RANK_SIZE, "n_params": int(r["n_params"]),
                     "val_mse_mean": r.get("val_mse_mean", float("nan")),
                     "test_mse_mean": r["test_mse_mean"], "test_mse_std": r["test_mse_std"]})
    for _, r in c.iterrows():
        rows.append({"experiment": "C", "config": f"C = {int(r['layers_C'])} tầng x 16 nơ-ron",
                     "train_size": RANK_SIZE, "n_params": int(r["total_params"]),
                     "val_mse_mean": r.get("val_mse_mean", float("nan")),
                     "test_mse_mean": r["test_mse_mean"], "test_mse_std": r["test_mse_std"]})
    return pd.DataFrame(rows)


def make_figure(a, b, c, fig_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    col = {"A": "#2980b9", "B": "#0f766e", "C": "#c0392b"}

    ax = axes[0, 0]
    ax.plot(a["size"], a["mse_mean"], "o-", color=col["A"], linewidth=2)
    ax.fill_between(a["size"], a["mse_mean"] - a["mse_std"], a["mse_mean"] + a["mse_std"],
                    color=col["A"], alpha=0.18)
    ax.set_xlabel("Số mẫu huấn luyện A"); ax.set_ylabel("Test MSE")
    ax.set_title("(a) Thực nghiệm A: quy mô dữ liệu (1 tầng x 128, 80 epochs)", fontsize=10, fontweight="bold")

    ax = axes[0, 1]
    ax.plot(range(len(b)), b["test_mse_mean"], "o-", color=col["B"], linewidth=2)
    ax.fill_between(range(len(b)), b["test_mse_mean"] - b["test_mse_std"],
                    b["test_mse_mean"] + b["test_mse_std"], color=col["B"], alpha=0.18)
    ax.set_xticks(range(len(b))); ax.set_xticklabels([str(int(v)) for v in b["nodes_B"]], fontsize=8)
    ax.set_yscale("log"); ax.set_xlabel("Số nơ-ron tầng ẩn B"); ax.set_ylabel("Test MSE (thang log)")
    ax.set_title("(b) Thực nghiệm B: độ rộng (1000 mẫu, 100 epochs)", fontsize=10, fontweight="bold")

    ax = axes[1, 0]
    ax.plot(c["layers_C"], c["test_mse_mean"], "o-", color=col["C"], linewidth=2)
    ax.fill_between(c["layers_C"], c["test_mse_mean"] - c["test_mse_std"],
                    c["test_mse_mean"] + c["test_mse_std"], color=col["C"], alpha=0.18)
    ax.set_xticks(c["layers_C"]); ax.set_yscale("log")
    ax.set_xlabel("Số tầng ẩn C (16 nơ-ron/tầng)"); ax.set_ylabel("Test MSE (thang log)")
    ax.set_title("(c) Thực nghiệm C: độ sâu (1000 mẫu, 100 epochs)", fontsize=10, fontweight="bold")

    ax = axes[1, 1]
    ax.errorbar(b["n_params"], b["test_mse_mean"], yerr=b["test_mse_std"], fmt="o-",
                color=col["B"], capsize=3, label="B: 1 tầng, thay đổi độ rộng")
    ax.errorbar(c["total_params"], c["test_mse_mean"], yerr=c["test_mse_std"], fmt="s--",
                color=col["C"], capsize=3, label="C: thay đổi độ sâu (16 nơ-ron/tầng)")
    a1000 = a[a["size"] == RANK_SIZE]
    if len(a1000):
        ax.errorbar([A_PARAMS], a1000["mse_mean"], yerr=a1000["mse_std"], fmt="*", markersize=13,
                    color=col["A"], capsize=3, label="A = 1000 (1 tầng x 128, 80 epochs)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Số tham số huấn luyện được (thang log)"); ax.set_ylabel("Test MSE (thang log)")
    ax.set_title("(d) Hiệu quả theo số tham số: độ rộng vs độ sâu (1000 mẫu)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    for ax in axes.ravel():
        ax.grid(True, linestyle="--", alpha=0.5, which="both")
    plt.suptitle("So sánh đối chiếu Thực nghiệm A - B - C (Mean ± Std, 3 seeds)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
    plt.savefig(fig_path, dpi=300)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs_dir", default="logs")
    p.add_argument("--figures_dir", default="figures")
    a_ = p.parse_args()

    a, b, c = load_results(a_.logs_dir)
    table = build_tables(a, b, c)

    # Xếp hạng các cấu hình cùng 1000 mẫu train (A = 1000, mọi B, mọi C)
    # QUY ĐỊNH: không dùng tập test để chọn cấu hình -> xếp hạng theo VALIDATION MSE (B, C).
    # Dòng A (không có val trong summary) xếp cuối, chỉ để đối chiếu.
    sel = "val_mse_mean" if table.loc[table["experiment"] == "C", "val_mse_mean"].notna().all() else "test_mse_mean"
    if sel == "test_mse_mean":
        print("[CẢNH BÁO] exp_C_summary.csv thiếu cột val_mse_mean (chạy bản exp_C_depth.py mới). "
              "Tạm xếp hạng theo test MSE - CHỈ để tham khảo, không dùng để 'chọn' cấu hình trong báo cáo.")
    rank = table[table["train_size"] == RANK_SIZE].sort_values(sel, na_position="last").reset_index(drop=True)
    rank.index += 1
    rank.index.name = "rank"
    rank_path = os.path.join(a_.logs_dir, "exp_ABC_ranking_1000samples.csv")
    rank.to_csv(rank_path)

    best_rows = []
    for exp in ["A", "B", "C"]:
        t = table[table["experiment"] == exp].sort_values("test_mse_mean")
        best_rows.append(t.iloc[0])
    best = pd.DataFrame(best_rows)
    best_path = os.path.join(a_.logs_dir, "exp_ABC_best_configs.csv")
    best.to_csv(best_path, index=False)

    make_figure(a, b, c, os.path.join(a_.figures_dir, "exp_ABC_comparison.png"))

    print("=" * 78)
    print(" CẤU HÌNH TỐT NHẤT CỦA TỪNG THỰC NGHIỆM (theo Test MSE trung bình)")
    print("=" * 78)
    for _, r in best.iterrows():
        print(f"[{r['experiment']}] {r['config']:<28} | params = {int(r['n_params']):5d} | "
              f"Test MSE = {r['test_mse_mean']:.6f} ± {r['test_mse_std']:.6f}")

    top = rank.iloc[0]
    tied = rank[rank["test_mse_mean"] <= top["test_mse_mean"] + top["test_mse_std"]]  # theo test, chỉ mô tả
    print("\n" + "=" * 78)
    print(f" XẾP HẠNG CÁC CẤU HÌNH CÙNG {RANK_SIZE} MẪU TRAIN, SẮP THEO {sel} (top 8)")
    print("=" * 78)
    for i, r in rank.head(8).iterrows():
        print(f"{i:2d}. {r['config']:<28} | params = {int(r['n_params']):5d} | "
              f"val = {r['val_mse_mean']:.6f} | test = {r['test_mse_mean']:.6f} ± {r['test_mse_std']:.6f}")
    print(f"\n[*] Chọn theo {sel}: {top['config']}. Số cấu hình có test MSE trung bình nằm trong 1 std của nó "
          f"(chưa phân biệt được rõ về mặt thống kê): {len(tied)}")
    for _, r in tied.iterrows():
        print(f"      - {r['config']}")
    print(f"\n[*] Đã lưu: {rank_path}\n[*] Đã lưu: {best_path}")
    print(f"[*] Đã lưu: {os.path.join(a_.figures_dir, 'exp_ABC_comparison.png')}")


if __name__ == "__main__":
    main()
