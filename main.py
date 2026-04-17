"""
main.py -- Fraud Detection Pipeline Orchestrator
=================================================
Runs the complete end-to-end pipeline:

  Step 1 > Data Preprocessing   (load -> clean -> encode -> log-transform -> scale)
  Step 2 > Temporal Split       (80/20 by step, NO random shuffle)
  Step 3 > Autoencoder          (train on normal-only -> extract latent features)
  Step 4 > Hybrid Features      (original + latent concatenation)
  Step 5 > XGBoost Classifier   (train on hybrid features)
  Step 6 > Evaluation           (Classification Report, PR-AUC, Confusion Matrix)

Usage:
    python main.py
"""

import sys
import os
import time
import numpy as np

# -- Ensure project root is on the path --
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.config import get_paths, ensure_output_dirs
from src.data.preprocess import run_preprocessing
from src.features.build_features import extract_latent_features, build_hybrid_features
from src.models.autoencoder import build_autoencoder, train_autoencoder
from src.models.xgboost_model import train_xgboost
from src.evaluation.metrics import evaluate_model
from src.evaluation.plot_results import plot_all_for_model, plot_training_loss


def main():
    start = time.time()
    ensure_output_dirs()
    paths = get_paths()

    print("=" * 70)
    print("   FRAUD DETECTION PIPELINE -- Autoencoder + XGBoost")
    print("=" * 70, "\n")

    # -------------------------------------------------
    # STEP 1 & 2: PREPROCESSING + TEMPORAL SPLIT
    # -------------------------------------------------
    print("-" * 50)
    print("STEP 1-2: Data Preprocessing & Temporal Split")
    print("-" * 50)
    (X_train_scaled, X_test_scaled,
     y_train, y_test,
     feature_names, scaler, encoders) = run_preprocessing()

    # -------------------------------------------------
    # STEP 3: AUTOENCODER (train on normal data only)
    # -------------------------------------------------
    print("-" * 50)
    print("STEP 3: Build & Train Autoencoder (normal-only)")
    print("-" * 50)

    # Select ONLY normal transactions for AE training
    normal_mask = (y_train == 0)
    X_train_normal = X_train_scaled[normal_mask]
    print(f"[INFO] Normal training samples for AE: {X_train_normal.shape[0]:,} "
          f"(out of {X_train_scaled.shape[0]:,} total)\n")

    input_dim = X_train_scaled.shape[1]
    autoencoder, encoder = build_autoencoder(input_dim)

    # Train on normal data only -- the AE learns to reconstruct "normal"
    history = train_autoencoder(autoencoder, X_train_normal)
    plot_training_loss(history, "autoencoder")

    # -------------------------------------------------
    # STEP 4: EXTRACT LATENT FEATURES & BUILD HYBRID SET
    # -------------------------------------------------
    print("\n" + "-" * 50)
    print("STEP 4: Extract Latent Features & Build Hybrid Set")
    print("-" * 50)

    latent_train, latent_test = extract_latent_features(
        encoder, X_train_scaled, X_test_scaled
    )

    X_train_hybrid = build_hybrid_features(X_train_scaled, latent_train)
    X_test_hybrid = build_hybrid_features(X_test_scaled, latent_test)

    # -------------------------------------------------
    # STEP 5: TRAIN XGBoost CLASSIFIER
    # -------------------------------------------------
    print("\n" + "-" * 50)
    print("STEP 5: Train XGBoost on Hybrid Features")
    print("-" * 50)

    clf = train_xgboost(X_train_hybrid, y_train)

    # -------------------------------------------------
    # STEP 6: EVALUATE
    # -------------------------------------------------
    print("\n" + "-" * 50)
    print("STEP 6: Evaluation")
    print("-" * 50)

    y_pred = clf.predict(X_test_hybrid)
    y_proba = clf.predict_proba(X_test_hybrid)[:, 1]
    metrics = evaluate_model(y_test, y_pred, y_proba, "xgboost")
    plot_all_for_model(y_test, y_pred, y_proba, "xgboost")

    # -------------------------------------------------
    # SUMMARY
    # -------------------------------------------------
    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print("   PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  PR-AUC            : {metrics['pr_auc']:.4f}")
    print(f"  Average Precision : {metrics['average_precision']:.4f}")
    print(f"  ROC-AUC           : {metrics['roc_auc']:.4f}")
    print(f"  Elapsed time      : {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  Reports saved to  : {paths['reports']['base_dir']}")
    print("=" * 70)

    return clf, encoder, autoencoder, metrics


if __name__ == "__main__":
    main()
