"""
preprocess.py -- Data loading, cleaning, encoding, feature engineering & scaling.

Migrated from the original preprocessing.py.
All paths and parameters are loaded from YAML configs (no hardcoded values).

Pipeline steps:
  1. Load raw CSV and strip quote characters from string columns.
  2. Drop constant columns (zipcodeOri, zipMerchant).
  3. Map `age` and `gender` to numeric values.
  4. Label-encode high-cardinality categoricals (customer, merchant, category).
  5. Log-transform the `amount` column to reduce skewness.
  6. StandardScaler normalisation (critical for Autoencoder convergence).
  7. Temporal train/test split based on the `step` column.
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --------------- make project root importable ---------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import get_paths, get_model_config


# ============================================================
# 1. LOAD & CLEAN
# ============================================================
def load_and_clean(path: str = None) -> pd.DataFrame:
    """Read the BankSim CSV and remove stray quote characters."""
    if path is None:
        paths = get_paths()
        path = paths["data"]["raw_csv"]

    cfg = get_model_config()["preprocessing"]

    df = pd.read_csv(path)

    # The raw file wraps some values in single-quotes -> strip them
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip("'\" ")

    target_col = cfg["target_col"]
    print(f"[INFO] Loaded {len(df):,} rows x {df.shape[1]} columns from {os.path.basename(path)}")
    print(f"[INFO] Fraud distribution:\n{df[target_col].value_counts(normalize=True).to_string()}\n")
    return df


# ============================================================
# 2. DROP IRRELEVANT COLUMNS
# ============================================================
def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop constant / irrelevant columns specified in config."""
    cfg = get_model_config()["preprocessing"]
    cols_to_drop = cfg["cols_to_drop"]
    df = df.drop(columns=cols_to_drop, errors="ignore")
    print(f"[INFO] Dropped columns: {cols_to_drop}")
    return df


# ============================================================
# 3. ENCODE `age` AND `gender`
# ============================================================
def encode_age_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Convert age and gender to numeric using predefined mappings."""
    cfg = get_model_config()["preprocessing"]
    age_map = cfg["age_map"]
    gender_map = cfg["gender_map"]

    df["age"] = df["age"].astype(str).map(age_map).fillna(3).astype(int)
    df["gender"] = df["gender"].astype(str).map(gender_map).fillna(2).astype(int)
    print("[INFO] Encoded `age` and `gender` to numeric.")
    return df


# ============================================================
# 4. LABEL-ENCODE HIGH-CARDINALITY CATEGORICALS
# ============================================================
def label_encode_categoricals(df: pd.DataFrame):
    """
    Apply LabelEncoder to customer, merchant, category.
    Returns the modified DataFrame AND a dict of fitted encoders
    (needed if we want to inverse-transform later).
    """
    cfg = get_model_config()["preprocessing"]
    categorical_cols = cfg["categorical_cols"]

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"[INFO] LabelEncoded `{col}` -> {len(le.classes_):,} unique values")
    return df, encoders


# ============================================================
# 5. LOG-TRANSFORM AMOUNT
# ============================================================
def log_transform_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log1p (log(1+x)) to the amount column.
    log1p is numerically stable for small values and avoids log(0).
    """
    df["amount"] = np.log1p(df["amount"])
    print("[INFO] Applied log1p transform to `amount`.")
    return df


# ============================================================
# 6. STANDARD SCALING
# ============================================================
def scale_features(X_train: np.ndarray, X_test: np.ndarray):
    """
    Fit StandardScaler on training data only, then transform both sets.
    This prevents data leakage from test set statistics.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("[INFO] StandardScaler fitted on training data and applied to both sets.")
    return X_train_scaled, X_test_scaled, scaler


# ============================================================
# 7. TEMPORAL TRAIN / TEST SPLIT
# ============================================================
def temporal_split(df: pd.DataFrame, train_ratio: float = None):
    """
    Split based on the `step` column to respect the temporal order.
    Uses the first `train_ratio` fraction of unique steps for training
    and the remaining steps for testing.  NO random shuffling.
    """
    cfg = get_model_config()["preprocessing"]
    if train_ratio is None:
        train_ratio = cfg["train_ratio"]
    step_col = cfg["step_col"]
    target_col = cfg["target_col"]

    unique_steps = sorted(df[step_col].unique())
    cutoff_idx = int(len(unique_steps) * train_ratio)
    cutoff_step = unique_steps[cutoff_idx]

    train_mask = df[step_col] < cutoff_step
    test_mask = ~train_mask

    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    print(f"[INFO] Temporal split at step {cutoff_step}  "
          f"(train steps: 0-{cutoff_step - 1}, test steps: {cutoff_step}-{unique_steps[-1]})")
    print(f"[INFO] Train size: {len(df_train):,} | Test size: {len(df_test):,}")
    print(f"[INFO] Train fraud rate: {df_train[target_col].mean():.4%}")
    print(f"[INFO] Test  fraud rate: {df_test[target_col].mean():.4%}\n")
    return df_train, df_test


# ============================================================
# 8. FULL PREPROCESSING PIPELINE (convenience function)
# ============================================================
def run_preprocessing():
    """
    Execute the full preprocessing pipeline and return everything
    the downstream modules need.

    Returns
    -------
    X_train_scaled : np.ndarray   - scaled training features
    X_test_scaled  : np.ndarray   - scaled testing features
    y_train        : np.ndarray   - training labels
    y_test         : np.ndarray   - testing labels
    feature_names  : list[str]    - ordered feature column names
    scaler         : StandardScaler (fitted)
    encoders       : dict[str, LabelEncoder]
    """
    cfg = get_model_config()["preprocessing"]
    target_col = cfg["target_col"]
    step_col = cfg["step_col"]

    # --- Load & clean ---
    df = load_and_clean()

    # --- Drop constant columns ---
    df = drop_columns(df)

    # --- Encode age & gender ---
    df = encode_age_gender(df)

    # --- Label-encode categoricals ---
    df, encoders = label_encode_categoricals(df)

    # --- Log-transform amount ---
    df = log_transform_amount(df)

    # --- Temporal split (before scaling to avoid leakage) ---
    df_train, df_test = temporal_split(df)

    # --- Separate features & target ---
    feature_cols = [c for c in df_train.columns if c not in [target_col, step_col]]
    X_train = df_train[feature_cols].values
    X_test = df_test[feature_cols].values
    y_train = df_train[target_col].values
    y_test = df_test[target_col].values

    # --- Scale ---
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    print(f"[INFO] Final feature matrix shape: train {X_train_scaled.shape}, test {X_test_scaled.shape}")
    print(f"[INFO] Feature columns: {feature_cols}\n")

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler, encoders


# -- quick test --
if __name__ == "__main__":
    run_preprocessing()
