"""
random_forest.py -- Build and train a Random Forest classifier.

New model added to the pipeline for multi-model comparison.
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

from sklearn.ensemble import RandomForestClassifier


def build_random_forest() -> RandomForestClassifier:
    """
    Create a Random Forest classifier with parameters from model_config.yaml.

    Returns
    -------
    RandomForestClassifier (not yet fitted)
    """
    model_cfg = get_model_config()["random_forest"]
    clf = RandomForestClassifier(**model_cfg)
    print(f"[INFO] Random Forest classifier created with params: {model_cfg}")
    return clf


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                         logger=None) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Parameters
    ----------
    X_train : np.ndarray - Training features
    y_train : np.ndarray - Training labels
    logger  : logging.Logger (optional)

    Returns
    -------
    RandomForestClassifier (fitted)
    """
    paths = get_paths()
    train_cfg = get_train_config()["random_forest"]

    clf = build_random_forest()

    if logger:
        logger.info(f"Starting Random Forest training with {X_train.shape[0]:,} samples, "
                     f"{X_train.shape[1]} features")

    clf.fit(X_train, y_train)

    msg = "Random Forest training complete."
    print(f"[INFO] {msg}")
    if logger:
        logger.info(msg)

    # Save model
    save_path = paths["models_saved"]["random_forest"]
    ensure_dir(os.path.dirname(save_path))
    joblib.dump(clf, save_path)

    msg = f"Random Forest model saved to {save_path}"
    print(f"[INFO] {msg}")
    if logger:
        logger.info(msg)

    return clf


# -- quick test --
if __name__ == "__main__":
    print("random_forest.py loaded successfully -- no standalone test.")
