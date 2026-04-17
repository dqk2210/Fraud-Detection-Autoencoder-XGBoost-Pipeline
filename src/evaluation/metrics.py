"""
metrics.py -- Model evaluation metrics.

Computes and exports standard classification metrics for fraud detection.
Results are saved as JSON to reports/metrics/<model_name>_scores.json.
"""

import sys
import os
import json
import numpy as np

# --------------- make project root importable ---------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import get_paths
from src.utils import timestamp, ensure_dir

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    auc,
    classification_report,
    confusion_matrix,
)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                   y_prob: np.ndarray, model_name: str) -> dict:
    """
    Compute classification metrics and save to JSON.

    Parameters
    ----------
    y_true      : np.ndarray - Ground truth binary labels
    y_pred      : np.ndarray - Predicted binary labels
    y_prob      : np.ndarray - Predicted probabilities (for positive class)
    model_name  : str        - Name of the model (used for file naming)

    Returns
    -------
    dict : {accuracy, precision, recall, f1, roc_auc, pr_auc}
    """
    paths = get_paths()
    metrics_dir = paths["reports"]["metrics_dir"]
    ensure_dir(metrics_dir)

    # Compute metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # ROC-AUC
    try:
        roc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc = 0.0

    # PR-AUC
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = float(auc(recall_curve, precision_curve))
    except ValueError:
        pr_auc = 0.0

    # Average Precision
    try:
        ap = float(average_precision_score(y_true, y_prob))
    except ValueError:
        ap = 0.0

    metrics = {
        "model_name": model_name,
        "timestamp": timestamp(),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "average_precision": ap,
    }

    # Print classification report
    print(f"\n{'=' * 60}")
    print(f"  EVALUATION: {model_name}")
    print(f"{'=' * 60}")
    report = classification_report(
        y_true, y_pred,
        target_names=["Normal (0)", "Fraud (1)"],
        digits=4,
        zero_division=0,
    )
    print(report)

    # Print confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        print(f"  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
        print(f"  Fraud Recall (Sensitivity) = {rec:.4f}")
        print(f"  Fraud Precision             = {prec:.4f}")

    print(f"  ROC-AUC  = {roc:.4f}")
    print(f"  PR-AUC   = {pr_auc:.4f}")
    print(f"{'=' * 60}\n")

    # Save to JSON
    ts = timestamp()
    save_path = os.path.join(metrics_dir, f"{model_name}_scores.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved to {save_path}")

    return metrics


# -- quick test --
if __name__ == "__main__":
    rng = np.random.RandomState(42)
    y_true = np.array([0]*950 + [1]*50)
    y_pred = np.array([0]*940 + [1]*10 + [0]*5 + [1]*45)
    y_prob = rng.rand(1000)
    evaluate_model(y_true, y_pred, y_prob, "test_model")
