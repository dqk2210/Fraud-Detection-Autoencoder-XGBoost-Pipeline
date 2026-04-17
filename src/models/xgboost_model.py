"""
xgboost_model.py -- Build and train an XGBoost classifier.

Migrated from the original src/models/classifier.py (training portion).
All hyperparameters loaded from YAML configs.
"""

import sys
import os
import numpy as np
import joblib

# --------------- make project root importable ---------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import get_paths, get_model_config, get_train_config
from src.utils import ensure_dir

from xgboost import XGBClassifier


def build_xgboost() -> XGBClassifier:
    """
    Create an XGBoost classifier with parameters from model_config.yaml.

    Returns
    -------
    XGBClassifier (not yet fitted)
    """
    model_cfg = get_model_config()["xgboost"]
    clf = XGBClassifier(**model_cfg)
    print(f"[INFO] XGBoost classifier created with params: {model_cfg}")
    return clf


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                   logger=None) -> XGBClassifier:
    """
    Train an XGBoost classifier with parameters tuned for imbalanced fraud data.
    `scale_pos_weight` in config compensates for the ~80:1 class ratio.

    Parameters
    ----------
    X_train : np.ndarray - Training features
    y_train : np.ndarray - Training labels
    logger  : logging.Logger (optional)

    Returns
    -------
    XGBClassifier (fitted)
    """
    paths = get_paths()
    train_cfg = get_train_config()["xgboost"]

    clf = build_xgboost()

    if logger:
        logger.info(f"Starting XGBoost training with {X_train.shape[0]:,} samples, "
                     f"{X_train.shape[1]} features")

    clf.fit(
        X_train, y_train,
        verbose=train_cfg.get("verbose", True),
    )

    msg = "XGBoost training complete."
    print(f"[INFO] {msg}")
    if logger:
        logger.info(msg)

    # Save model
    save_path = paths["models_saved"]["xgboost"]
    ensure_dir(os.path.dirname(save_path))
    joblib.dump(clf, save_path)

    msg = f"XGBoost model saved to {save_path}"
    print(f"[INFO] {msg}")
    if logger:
        logger.info(msg)

    return clf


# -- quick test --
if __name__ == "__main__":
    print("xgboost_model.py loaded successfully -- no standalone test.")
