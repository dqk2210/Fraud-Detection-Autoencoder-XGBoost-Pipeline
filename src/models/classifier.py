"""
classifier.py -- Train an XGBoost classifier on the hybrid feature set
and evaluate with fraud-detection-appropriate metrics.

Evaluation focus:
  * Classification Report (per-class precision / recall / F1)
  * Precision-Recall AUC  (PR-AUC) -- better than ROC-AUC for imbalanced data
  * Confusion Matrix heatmap
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Windows compatibility
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc,
    average_precision_score,
    roc_auc_score,
)

# --------------- make project root importable ---------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.config import XGB_PARAMS, REPORTS_DIR, RANDOM_SEED


# ============================================================
# 1. TRAIN XGBoost
# ============================================================
def train_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    """
    Train an XGBoost classifier with parameters tuned for imbalanced fraud data.
    `scale_pos_weight` in config compensates for the ~80:1 class ratio.
    """
    clf = XGBClassifier(**XGB_PARAMS)
    clf.fit(
        X_train, y_train,
        verbose=True,
    )
    print("[INFO] XGBoost training complete.")
    return clf


# ============================================================
# 2. EVALUATION -- Classification Report
# ============================================================
def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray):
    """Print sklearn classification report with explicit target names."""
    report = classification_report(
        y_true, y_pred,
        target_names=["Normal (0)", "Fraud (1)"],
        digits=4,
    )
    print("\n" + "=" * 60)
    print("              CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)
    return report


# ============================================================
# 3. EVALUATION -- Confusion Matrix
# ============================================================
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    """Plot and save an annotated confusion-matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Normal (0)", "Fraud (1)"]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax,
    )
    ax.set_title("Confusion Matrix", fontsize=14)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    fig.tight_layout()

    save_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    fig.savefig(save_path, dpi=150)
    print(f"[INFO] Confusion matrix saved to {save_path}")
    plt.close(fig)

    # Also print the raw numbers
    tn, fp, fn, tp = cm.ravel()
    print(f"  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
    print(f"  Fraud Recall (Sensitivity) = {tp / (tp + fn):.4f}")
    print(f"  Fraud Precision             = {tp / (tp + fp):.4f}")
    return cm


# ============================================================
# 4. EVALUATION -- Precision-Recall Curve & PR-AUC
# ============================================================
def plot_precision_recall_curve(y_true: np.ndarray, y_scores: np.ndarray):
    """
    Plot the Precision-Recall curve and compute the PR-AUC.
    This is the most informative metric for highly imbalanced datasets
    because it focuses on the minority (fraud) class.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    ap = average_precision_score(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(recall, precision, linewidth=2,
            label=f"PR Curve (AUC = {pr_auc:.4f})")
    ax.axhline(y=y_true.mean(), color="r", linestyle="--", linewidth=1,
               label=f"Baseline (fraud rate = {y_true.mean():.4f})")
    ax.set_title("Precision-Recall Curve", fontsize=14)
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    fig.tight_layout()

    save_path = os.path.join(REPORTS_DIR, "precision_recall_curve.png")
    fig.savefig(save_path, dpi=150)
    print(f"\n[INFO] PR-Curve saved to {save_path}")
    print(f"[INFO] PR-AUC = {pr_auc:.4f}  |  Average Precision = {ap:.4f}")
    plt.close(fig)

    return pr_auc, ap


# ============================================================
# 5. FULL EVALUATION PIPELINE
# ============================================================
def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray):
    """
    Run the full evaluation suite:
      1. Predictions & probability scores
      2. Classification report
      3. Confusion matrix
      4. PR-AUC curve
      5. ROC-AUC (for reference)
    """
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]  # P(fraud)

    # 1. Classification Report
    print_classification_report(y_test, y_pred)

    # 2. Confusion Matrix
    plot_confusion_matrix(y_test, y_pred)

    # 3. PR-AUC
    pr_auc, ap = plot_precision_recall_curve(y_test, y_proba)

    # 4. ROC-AUC (for reference)
    roc = roc_auc_score(y_test, y_proba)
    print(f"[INFO] ROC-AUC = {roc:.4f}")

    return {
        "pr_auc": pr_auc,
        "average_precision": ap,
        "roc_auc": roc,
    }


# -- quick test --
if __name__ == "__main__":
    print("classifier.py loaded successfully -- no standalone test.")
