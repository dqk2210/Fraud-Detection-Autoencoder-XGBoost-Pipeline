"""
train_all.py -- Sequential training script with logging.

Trains each model sequentially, freeing GPU memory between runs:
  1. Autoencoder  (unsupervised, on normal-only data)
  2. Random Forest (supervised, on hybrid features)
  3. XGBoost       (supervised, on hybrid features)

For each model:
  - Writes per-model log to logs/<model_name>_train.log
  - Appends a summary row to logs/run_summary.log
  - Saves best weights to models_saved/
  - Saves epoch checkpoints to models_saved/checkpoints/

Usage:
    python -m src.models.train_all
"""

import sys
import os
import gc
import time
import json
import numpy as np

# --------------- make project root importable ---------------
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, PROJECT_ROOT)

from src.config import get_paths, get_model_config, get_train_config, ensure_output_dirs
from src.utils import setup_logger, timestamp, ensure_dir
from src.data.preprocess import run_preprocessing
from src.features.build_features import extract_latent_features, build_hybrid_features
from src.evaluation.metrics import evaluate_model


def _clear_gpu():
    """Release GPU memory between model runs."""
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()


def _append_summary(log_path: str, model_name: str, metrics: dict, elapsed: float):
    """Append a summary row to the run summary log."""
    ts = timestamp()
    line = (
        f"{ts} | {model_name:<20s} | "
        f"Accuracy={metrics.get('accuracy', 0):.4f} | "
        f"Precision={metrics.get('precision', 0):.4f} | "
        f"Recall={metrics.get('recall', 0):.4f} | "
        f"F1={metrics.get('f1', 0):.4f} | "
        f"ROC-AUC={metrics.get('roc_auc', 0):.4f} | "
        f"PR-AUC={metrics.get('pr_auc', 0):.4f} | "
        f"Time={elapsed:.1f}s\n"
    )
    ensure_dir(os.path.dirname(log_path))
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)


def train_all():
    """Run the full sequential training pipeline."""
    total_start = time.time()

    # --- Setup ---
    ensure_output_dirs()
    paths = get_paths()
    summary_log = paths["logs"]["run_summary"]

    print("=" * 70)
    print("   TRAIN ALL MODELS -- Sequential Pipeline")
    print("=" * 70, "\n")

    # =========================================================
    # STEP 1: PREPROCESSING
    # =========================================================
    print("-" * 50)
    print("STEP 1: Data Preprocessing & Temporal Split")
    print("-" * 50)

    (X_train_scaled, X_test_scaled,
     y_train, y_test,
     feature_names, scaler, encoders) = run_preprocessing()

    # =========================================================
    # STEP 2: AUTOENCODER
    # =========================================================
    print("\n" + "-" * 50)
    print("STEP 2: Autoencoder Training (normal-only)")
    print("-" * 50)

    ae_logger = setup_logger("autoencoder", paths["logs"]["autoencoder"])
    ae_start = time.time()

    from src.models.autoencoder import build_autoencoder, train_autoencoder

    # Select ONLY normal transactions for AE training
    normal_mask = (y_train == 0)
    X_train_normal = X_train_scaled[normal_mask]
    ae_logger.info(f"Normal training samples for AE: {X_train_normal.shape[0]:,} "
                   f"(out of {X_train_scaled.shape[0]:,} total)")

    input_dim = X_train_scaled.shape[1]
    autoencoder, encoder = build_autoencoder(input_dim)
    history = train_autoencoder(autoencoder, X_train_normal, logger=ae_logger)

    # Extract latent features for downstream classifiers
    latent_train, latent_test = extract_latent_features(
        encoder, X_train_scaled, X_test_scaled
    )
    X_train_hybrid = build_hybrid_features(X_train_scaled, latent_train)
    X_test_hybrid = build_hybrid_features(X_test_scaled, latent_test)

    # Evaluate AE as anomaly detector (using reconstruction error)
    ae_pred_proba = np.mean((X_test_scaled - autoencoder.predict(X_test_scaled)) ** 2, axis=1)
    ae_pred_binary = (ae_pred_proba > np.percentile(ae_pred_proba, 95)).astype(int)
    ae_metrics = evaluate_model(y_test, ae_pred_binary, ae_pred_proba, "autoencoder")

    ae_elapsed = time.time() - ae_start
    ae_logger.info(f"Autoencoder training complete in {ae_elapsed:.1f}s")
    _append_summary(summary_log, "autoencoder", ae_metrics, ae_elapsed)

    # Free GPU memory
    _clear_gpu()

    # =========================================================
    # STEP 3: RANDOM FOREST
    # =========================================================
    print("\n" + "-" * 50)
    print("STEP 3: Random Forest Training (hybrid features)")
    print("-" * 50)

    rf_logger = setup_logger("random_forest", paths["logs"]["random_forest"])
    rf_start = time.time()

    from src.models.random_forest import train_random_forest

    rf_clf = train_random_forest(X_train_hybrid, y_train, logger=rf_logger)

    rf_pred = rf_clf.predict(X_test_hybrid)
    rf_proba = rf_clf.predict_proba(X_test_hybrid)[:, 1]
    rf_metrics = evaluate_model(y_test, rf_pred, rf_proba, "random_forest")

    rf_elapsed = time.time() - rf_start
    rf_logger.info(f"Random Forest training complete in {rf_elapsed:.1f}s")
    _append_summary(summary_log, "random_forest", rf_metrics, rf_elapsed)

    _clear_gpu()

    # =========================================================
    # STEP 4: XGBOOST
    # =========================================================
    print("\n" + "-" * 50)
    print("STEP 4: XGBoost Training (hybrid features)")
    print("-" * 50)

    xgb_logger = setup_logger("xgboost", paths["logs"]["xgboost"])
    xgb_start = time.time()

    from src.models.xgboost_model import train_xgboost

    xgb_clf = train_xgboost(X_train_hybrid, y_train, logger=xgb_logger)

    xgb_pred = xgb_clf.predict(X_test_hybrid)
    xgb_proba = xgb_clf.predict_proba(X_test_hybrid)[:, 1]
    xgb_metrics = evaluate_model(y_test, xgb_pred, xgb_proba, "xgboost")

    xgb_elapsed = time.time() - xgb_start
    xgb_logger.info(f"XGBoost training complete in {xgb_elapsed:.1f}s")
    _append_summary(summary_log, "xgboost", xgb_metrics, xgb_elapsed)

    _clear_gpu()

    # =========================================================
    # SUMMARY
    # =========================================================
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print("   ALL MODELS TRAINED SUCCESSFULLY")
    print("=" * 70)
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print(f"  Run summary: {summary_log}")
    print(f"  Metrics: {paths['reports']['metrics_dir']}")
    print("=" * 70)

    return {
        "autoencoder": ae_metrics,
        "random_forest": rf_metrics,
        "xgboost": xgb_metrics,
    }


if __name__ == "__main__":
    train_all()
