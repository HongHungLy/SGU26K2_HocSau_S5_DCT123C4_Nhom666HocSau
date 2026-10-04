"""
Lab 03 - House Prices: Advanced Regression Techniques
Member 1: Data Understanding, EDA, Preprocessing, Feature Engineering & Feature Selection

Expected input:
    data/train.csv
    data/test.csv

Usage:
    python member1_house_prices.py --train data/train.csv --test data/test.csv --output_dir results/member1

The script intentionally does NOT train the final ML/MLP models. It prepares a
clean, reproducible feature matrix and produces feature rankings/feature sets for
Members 2 and 3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TARGET = "SalePrice"
ID_COL = "Id"
# These columns match the preprocessing logic in the supplied class/code.
DROP_COLUMNS = ["Alley", "PoolQC", "Fence", "MiscFeature"]
TOP_K_VALUES = [20, 40, 60]


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def safe_log_target(y: pd.Series) -> pd.Series:
    """Competition-aligned target scale for ranking/EDA."""
    return np.log1p(y.clip(lower=0))


# -----------------------------------------------------------------------------
# 1. Load and audit
# -----------------------------------------------------------------------------

def load_data(train_path: str | Path, test_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    if TARGET not in train.columns:
        raise ValueError(f"'{TARGET}' not found in training data")
    if ID_COL not in train.columns or ID_COL not in test.columns:
        raise ValueError(f"'{ID_COL}' must exist in both train and test")

    return train, test


def audit_dataframe(df: pd.DataFrame, name: str) -> Dict[str, object]:
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(exclude=np.number).columns
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]

    duplicate_rows = int(df.duplicated().sum())

    return {
        "name": name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_columns": int(len(numeric_cols)),
        "categorical_columns": int(len(categorical_cols)),
        "duplicate_rows": duplicate_rows,
        "missing_columns": int(len(missing)),
        "top_missing": missing.head(20).to_dict(),
    }


# -----------------------------------------------------------------------------
# 2. Feature engineering
# -----------------------------------------------------------------------------

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable aggregate/age features used in the experiments."""
    out = df.copy()

    def add_if_possible(name: str, cols: Iterable[str], weights: Iterable[float] | None = None) -> None:
        cols = list(cols)
        if not all(c in out.columns for c in cols):
            return
        vals = [pd.to_numeric(out[c], errors="coerce").fillna(0) for c in cols]
        if weights is None:
            total = sum(vals)
        else:
            total = sum(v * w for v, w in zip(vals, weights))
        out[name] = total

    # Total finished/basement living area.
    add_if_possible("TotalSF", ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"])

    # Equivalent bathroom count.
    add_if_possible(
        "TotalBathrooms",
        ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"],
        weights=[1.0, 0.5, 1.0, 0.5],
    )

    # Total porch/deck area.
    add_if_possible(
        "TotalPorchSF",
        ["OpenPorchSF", "3SsnPorch", "EnclosedPorch", "ScreenPorch", "WoodDeckSF"],
    )

    # Age-related variables.
    if {"YrSold", "YearBuilt"}.issubset(out.columns):
        out["AgeAtSale"] = pd.to_numeric(out["YrSold"], errors="coerce") - pd.to_numeric(
            out["YearBuilt"], errors="coerce"
        )

    if {"YrSold", "YearRemodAdd"}.issubset(out.columns):
        out["RemodAge"] = pd.to_numeric(out["YrSold"], errors="coerce") - pd.to_numeric(
            out["YearRemodAdd"], errors="coerce"
        )

    # Overall quality x living area interaction.
    if {"OverallQual", "GrLivArea"}.issubset(out.columns):
        out["OverallQual_GrLivArea"] = pd.to_numeric(out["OverallQual"], errors="coerce") * pd.to_numeric(
            out["GrLivArea"], errors="coerce"
        )

    # Garage capacity and area interaction.
    if {"GarageCars", "GarageArea"}.issubset(out.columns):
        out["GarageCapacityArea"] = pd.to_numeric(out["GarageCars"], errors="coerce") * pd.to_numeric(
            out["GarageArea"], errors="coerce"
        )

    return out


# -----------------------------------------------------------------------------
# 3. EDA
# -----------------------------------------------------------------------------

def make_eda(train: pd.DataFrame, output_dir: Path) -> None:
    ensure_dir(output_dir)

    # Missing-value chart.
    missing = train.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0].head(20)
    if len(missing) > 0:
        plt.figure(figsize=(10, 6))
        missing.sort_values().plot(kind="barh")
        plt.title("Top 20 Features by Missing Values")
        plt.xlabel("Number of missing values")
        plt.tight_layout()
        plt.savefig(output_dir / "01_top_missing_values.png", dpi=180)
        plt.close()

    # Target distribution.
    plt.figure(figsize=(9, 5))
    plt.hist(train[TARGET], bins=40)
    plt.title("Distribution of SalePrice")
    plt.xlabel("SalePrice")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "02_saleprice_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.hist(safe_log_target(train[TARGET]), bins=40)
    plt.title("Distribution of log1p(SalePrice)")
    plt.xlabel("log1p(SalePrice)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "03_log_saleprice_distribution.png", dpi=180)
    plt.close()

    # Numeric correlation ranking against log target.
    numeric = train.select_dtypes(include=np.number).copy()
    if TARGET in numeric.columns:
        corr = numeric.corr(numeric_only=True)[TARGET].drop(TARGET).abs().sort_values(ascending=False)
        top_corr = corr.head(20).sort_values()
        if len(top_corr) > 0:
            plt.figure(figsize=(10, 7))
            top_corr.plot(kind="barh")
            plt.title("Top 20 Absolute Correlations with SalePrice")
            plt.xlabel("Absolute Pearson correlation")
            plt.tight_layout()
            plt.savefig(output_dir / "04_top_numeric_correlation.png", dpi=180)
            plt.close()

        corr_table = pd.DataFrame({"feature": corr.index, "abs_pearson_corr": corr.values})
        save_table(corr_table, output_dir / "numeric_correlation_ranking.csv")

    # A key relationship from the competition/paper context.
    if {"GrLivArea", TARGET}.issubset(train.columns):
        plt.figure(figsize=(9, 5))
        plt.scatter(train["GrLivArea"], train[TARGET], alpha=0.6)
        plt.title("GrLivArea vs SalePrice")
        plt.xlabel("GrLivArea")
        plt.ylabel("SalePrice")
        plt.tight_layout()
        plt.savefig(output_dir / "05_grlivarea_vs_saleprice.png", dpi=180)
        plt.close()

        outliers = train.loc[train["GrLivArea"] > 4000, ["GrLivArea", TARGET]].sort_values("GrLivArea")
        save_table(outliers, output_dir / "grlivarea_over_4000_candidates.csv")


# -----------------------------------------------------------------------------
# 4. Preprocessing
# -----------------------------------------------------------------------------

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
    ])

    # sklearn compatibility across versions.
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", encoder),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    return list(preprocessor.get_feature_names_out())


# -----------------------------------------------------------------------------
# 5. Feature selection
# -----------------------------------------------------------------------------

def rank_encoded_features(
    X_train_encoded: np.ndarray,
    y_train_log: pd.Series,
    feature_names: List[str],
) -> pd.DataFrame:
    """Rank encoded features using mutual information fitted on train only."""
    y = np.asarray(y_train_log, dtype=float)
    scores = mutual_info_regression(
        X_train_encoded,
        y,
        random_state=RANDOM_STATE,
    )
    ranking = pd.DataFrame({
        "feature": feature_names,
        "mutual_information": scores,
    }).sort_values("mutual_information", ascending=False).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking[["rank", "feature", "mutual_information"]]


def select_feature_sets(
    X_train_encoded: np.ndarray,
    X_val_encoded: np.ndarray,
    X_test_encoded: np.ndarray,
    y_train_log: pd.Series,
    feature_names: List[str],
    top_k_values: Iterable[int],
) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {
        "full": {
            "train": X_train_encoded,
            "val": X_val_encoded,
            "test": X_test_encoded,
            "feature_names": feature_names,
        }
    }

    for k in top_k_values:
        k_eff = min(int(k), X_train_encoded.shape[1])
        selector = SelectKBest(score_func=mutual_info_regression, k=k_eff)
        selector.fit(X_train_encoded, np.asarray(y_train_log, dtype=float))
        mask = selector.get_support()
        names = [n for n, keep in zip(feature_names, mask) if keep]
        out[f"top_{k_eff}"] = {
            "train": selector.transform(X_train_encoded),
            "val": selector.transform(X_val_encoded),
            "test": selector.transform(X_test_encoded),
            "feature_names": names,
        }
    return out


# -----------------------------------------------------------------------------
# 6. Main preparation pipeline
# -----------------------------------------------------------------------------

def run(train_path: str | Path, test_path: str | Path, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    eda_dir = output_dir / "eda"
    processed_dir = output_dir / "processed"
    ranking_dir = output_dir / "feature_selection"
    ensure_dir(output_dir)
    ensure_dir(eda_dir)
    ensure_dir(processed_dir)
    ensure_dir(ranking_dir)

    train, test = load_data(train_path, test_path)

    # Audit before modifications.
    audit = {
        "train": audit_dataframe(train, "train"),
        "test": audit_dataframe(test, "test"),
    }
    with open(output_dir / "audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    # Preserve IDs for the eventual Kaggle submission.
    test_ids = test[ID_COL].copy()

    # EDA on raw training data.
    make_eda(train, eda_dir)

    # Drop ID and high-missing categorical columns following the supplied code logic.
    drop_train = [c for c in DROP_COLUMNS if c in train.columns]
    drop_train.append(ID_COL)
    drop_train = list(dict.fromkeys(drop_train))
    drop_test = [c for c in DROP_COLUMNS if c in test.columns]
    drop_test.append(ID_COL)
    drop_test = list(dict.fromkeys(drop_test))

    X_raw = train.drop(columns=drop_train + [TARGET], errors="ignore").copy()
    X_test_raw = test.drop(columns=drop_test, errors="ignore").copy()
    y = train[TARGET].copy()

    # Feature engineering before train/validation split.
    X_fe = add_engineered_features(X_raw)
    X_test_fe = add_engineered_features(X_test_raw)

    added = sorted(set(X_fe.columns) - set(X_raw.columns))
    pd.DataFrame({"engineered_feature": added}).to_csv(
        output_dir / "engineered_features.csv", index=False, encoding="utf-8-sig"
    )

    # Same split protocol for Members 2 and 3.
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_fe,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    # Keep test ordering intact.
    preprocessor = build_preprocessor(X_train_raw)
    X_train_enc = preprocessor.fit_transform(X_train_raw)
    X_val_enc = preprocessor.transform(X_val_raw)
    X_test_enc = preprocessor.transform(X_test_fe)
    feature_names = get_feature_names(preprocessor)

    # Save encoded full matrices as CSV for easy reuse.
    X_train_full_df = pd.DataFrame(X_train_enc, columns=feature_names, index=X_train_raw.index)
    X_val_full_df = pd.DataFrame(X_val_enc, columns=feature_names, index=X_val_raw.index)
    X_test_full_df = pd.DataFrame(X_test_enc, columns=feature_names)

    X_train_full_df.to_csv(processed_dir / "X_train_full.csv", index=False)
    X_val_full_df.to_csv(processed_dir / "X_val_full.csv", index=False)
    X_test_full_df.to_csv(processed_dir / "X_test_full.csv", index=False)
    y_train.to_frame(TARGET).to_csv(processed_dir / "y_train.csv", index=False)
    y_val.to_frame(TARGET).to_csv(processed_dir / "y_val.csv", index=False)
    test_ids.to_frame(ID_COL).to_csv(processed_dir / "test_ids.csv", index=False)

    # Feature ranking on TRAIN ONLY.
    y_train_log = safe_log_target(y_train)
    ranking = rank_encoded_features(X_train_enc, y_train_log, feature_names)
    save_table(ranking, ranking_dir / "mutual_information_ranking.csv")

    # Top-20/40/60 and full feature sets.
    feature_sets = select_feature_sets(
        X_train_enc,
        X_val_enc,
        X_test_enc,
        y_train_log,
        feature_names,
        TOP_K_VALUES,
    )

    summary_rows = []
    for name, bundle in feature_sets.items():
        n = len(bundle["feature_names"])
        summary_rows.append({"feature_set": name, "n_features": n})

        prefix = name
        pd.DataFrame(bundle["train"], columns=bundle["feature_names"]).to_csv(
            processed_dir / f"X_train_{prefix}.csv", index=False
        )
        pd.DataFrame(bundle["val"], columns=bundle["feature_names"]).to_csv(
            processed_dir / f"X_val_{prefix}.csv", index=False
        )
        pd.DataFrame(bundle["test"], columns=bundle["feature_names"]).to_csv(
            processed_dir / f"X_test_{prefix}.csv", index=False
        )
        pd.DataFrame({"feature": bundle["feature_names"]}).to_csv(
            ranking_dir / f"{prefix}_features.csv", index=False, encoding="utf-8-sig"
        )

    pd.DataFrame(summary_rows).to_csv(output_dir / "feature_set_summary.csv", index=False)

    metadata = {
        "random_state": RANDOM_STATE,
        "target": TARGET,
        "id_column": ID_COL,
        "dropped_columns": drop_train,
        "engineered_features": added,
        "train_rows": int(X_train_raw.shape[0]),
        "validation_rows": int(X_val_raw.shape[0]),
        "full_feature_count_after_encoding": int(X_train_enc.shape[1]),
        "top_k_values": TOP_K_VALUES,
        "note": "Feature selection is fitted on training split only. Test IDs are kept separately for Kaggle submission.",
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("MEMBER 1 PIPELINE COMPLETED")
    print("=" * 72)
    print(f"Train shape: {train.shape}")
    print(f"Test shape : {test.shape}")
    print(f"Dropped    : {drop_train}")
    print(f"Engineered : {added}")
    print(f"Encoded features: {X_train_enc.shape[1]}")
    print(f"Train/Val  : {X_train_enc.shape[0]} / {X_val_enc.shape[0]}")
    print(f"Outputs    : {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--test", default="data/test.csv")
    parser.add_argument("--output_dir", default="results/member1")
    args = parser.parse_args()
    run(args.train, args.test, args.output_dir)
