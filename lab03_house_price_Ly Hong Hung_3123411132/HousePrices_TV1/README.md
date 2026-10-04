# House Prices - Thành viên 1

Phần việc: Data Understanding, EDA, Preprocessing, Feature Engineering và Feature Selection.

## 1. Chuẩn bị dữ liệu
Đặt hai file của Kaggle vào:

```text
data/train.csv
data/test.csv
```

## 2. Chạy code

```bash
python member1_house_prices.py --train data/train.csv --test data/test.csv --output_dir results/member1
```

Hoặc mở notebook:

```text
notebooks/01_member1_EDA_feature_engineering.ipynb
```

## 3. Output quan trọng

```text
results/member1/
├── audit.json
├── metadata.json
├── engineered_features.csv
├── feature_set_summary.csv
├── eda/
│   ├── 01_top_missing_values.png
│   ├── 02_saleprice_distribution.png
│   ├── 03_log_saleprice_distribution.png
│   ├── 04_top_numeric_correlation.png
│   ├── 05_grlivarea_vs_saleprice.png
│   └── numeric_correlation_ranking.csv
├── feature_selection/
│   ├── mutual_information_ranking.csv
│   ├── full_features.csv
│   ├── top_20_features.csv
│   ├── top_40_features.csv
│   └── top_60_features.csv
└── processed/
    ├── X_train_full.csv
    ├── X_val_full.csv
    ├── X_test_full.csv
    ├── X_train_top_20.csv
    ├── X_val_top_20.csv
    ├── X_test_top_20.csv
    ├── X_train_top_40.csv
    ├── X_val_top_40.csv
    ├── X_test_top_40.csv
    ├── X_train_top_60.csv
    ├── X_val_top_60.csv
    ├── X_test_top_60.csv
    ├── y_train.csv
    ├── y_val.csv
    └── test_ids.csv
```

## 4. Giao cho TV2/TV3

- Cùng dùng random_state=42.
- Cùng dùng train/validation split đã tạo.
- Không fit feature selection trên validation/test.
- Dùng các feature set full/top20/top40/top60 để chạy ML và MLP.
- Kaggle dùng cột `Id,SalePrice`; metric là RMSE giữa log(prediction) và log(observation).
