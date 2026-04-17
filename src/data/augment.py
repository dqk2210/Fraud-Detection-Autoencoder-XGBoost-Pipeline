"""
augment.py -- SMOTE oversampling and class balancing utilities.

Provides functions to handle the extreme class imbalance typical
in fraud detection datasets (~1% fraud vs ~99% normal).
"""

import sys
import os
import numpy as np
from imblearn.over_sampling import SMOTE

# --------------- make project root importable ---------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import get_train_config


def apply_smote(X_train: np.ndarray, y_train: np.ndarray):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique)
    to balance the training set.

    Parameters
    ----------
    X_train : np.ndarray - Training feature matrix
    y_train : np.ndarray - Training labels

    Returns
    -------
    X_resampled : np.ndarray - Resampled features
    y_resampled : np.ndarray - Resampled labels
    """
    train_cfg = get_train_config()
    smote_cfg = train_cfg.get("smote", {})

    sm = SMOTE(
        random_state=smote_cfg.get("random_state", 42),
        sampling_strategy=smote_cfg.get("sampling_strategy", "auto"),
    )

    print(f"[INFO] Before SMOTE: class 0 = {(y_train == 0).sum():,}, "
          f"class 1 = {(y_train == 1).sum():,}")

    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

    print(f"[INFO] After  SMOTE: class 0 = {(y_resampled == 0).sum():,}, "
          f"class 1 = {(y_resampled == 1).sum():,}")
    print(f"[INFO] Resampled shape: X={X_resampled.shape}, y={y_resampled.shape}\n")

    return X_resampled, y_resampled


def get_class_weights(y: np.ndarray) -> dict:
    """
    Compute class weights inversely proportional to class frequencies.

    Parameters
    ----------
    y : np.ndarray - Binary labels (0/1)

    Returns
    -------
    dict : {0: weight_0, 1: weight_1}
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    weight_dict = dict(zip(classes, weights))

    print(f"[INFO] Class weights: {weight_dict}")
    return weight_dict


# -- quick test --
if __name__ == "__main__":
    # Smoke test with synthetic data
    rng = np.random.RandomState(42)
    X = rng.randn(1000, 5)
    y = np.array([0] * 950 + [1] * 50)
    X_res, y_res = apply_smote(X, y)
    get_class_weights(y)
