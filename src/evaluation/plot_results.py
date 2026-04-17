"""
plot_results.py -- Save evaluation plots per model with timestamps.

Every figure is saved with the filename pattern:
    <model_name>_<plot_type>_<YYYYMMDD_HHMMSS>.png

This prevents any overwrite between training runs.

Supported plots:
  - Training loss / accuracy curves
  - Confusion matrix heatmap
  - ROC curve
  - Precision-Recall curve
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Windows compatibility
import matplotlib.pyplot as plt
import seaborn as sns

# --------------- make project root importable ---------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import get_paths
from src.utils import timestamp, ensure_dir

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)


def _get_save_path(model_name: str, plot_type: str) -> str:
    """Generate a timestamped save path for a plot."""
    paths = get_paths()
    figures_dir = paths["reports"]["figures_dir"]
    ensure_dir(figures_dir)
    ts = timestamp()
    return os.path.join(figures_dir, f"{model_name}_{plot_type}_{ts}.png")


def plot_training_loss(history, model_name: str):
    """
    Plot training vs. validation loss curves.

    Parameters
    ----------
    history    : keras History object or dict with 'loss' and 'val_loss' keys
    model_name : str - model identifier for filename
    """
    h = history.history if hasattr(history, "history") else history

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(h["loss"], label="Training Loss", linewidth=2)
    if "val_loss" in h:
        ax.plot(h["val_loss"], label="Validation Loss", linewidth=2)
    ax.set_title(f"{model_name} — Training Loss", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    save_path = _get_save_path(model_name, "training_loss")
    fig.savefig(save_path, dpi=150)
    print(f"[INFO] Training loss plot saved to {save_path}")
    plt.close(fig)


def plot_training_accuracy(history, model_name: str):
    """
    Plot training vs. validation accuracy curves.

    Parameters
    ----------
    history    : keras History object or dict with 'accuracy' keys
    model_name : str - model identifier for filename
    """
    h = history.history if hasattr(history, "history") else history

    if "accuracy" not in h:
        print(f"[WARN] No accuracy metric found for {model_name}, skipping accuracy plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(h["accuracy"], label="Training Accuracy", linewidth=2)
    if "val_accuracy" in h:
        ax.plot(h["val_accuracy"], label="Validation Accuracy", linewidth=2)
    ax.set_title(f"{model_name} — Training Accuracy", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    save_path = _get_save_path(model_name, "training_accuracy")
    fig.savefig(save_path, dpi=150)
    print(f"[INFO] Training accuracy plot saved to {save_path}")
    plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          model_name: str):
    """
    Plot and save an annotated confusion-matrix heatmap.

    Parameters
    ----------
    y_true     : np.ndarray - Ground truth labels
    y_pred     : np.ndarray - Predicted labels
    model_name : str        - model identifier for filename
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Normal (0)", "Fraud (1)"]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax,
    )
    ax.set_title(f"{model_name} — Confusion Matrix", fontsize=14)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    fig.tight_layout()

    save_path = _get_save_path(model_name, "confusion_matrix")
    fig.savefig(save_path, dpi=150)
    print(f"[INFO] Confusion matrix saved to {save_path}")
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray,
                   model_name: str):
    """
    Plot and save the ROC curve.

    Parameters
    ----------
    y_true     : np.ndarray - Ground truth labels
    y_prob     : np.ndarray - Predicted probabilities
    model_name : str        - model identifier for filename
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fpr, tpr, linewidth=2,
            label=f"ROC Curve (AUC = {roc_auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Random Baseline")
    ax.set_title(f"{model_name} — ROC Curve", fontsize=14)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    fig.tight_layout()

    save_path = _get_save_path(model_name, "roc_curve")
    fig.savefig(save_path, dpi=150)
    print(f"[INFO] ROC curve saved to {save_path}")
    plt.close(fig)


def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray,
                                model_name: str):
    """
    Plot the Precision-Recall curve and compute the PR-AUC.

    Parameters
    ----------
    y_true     : np.ndarray - Ground truth labels
    y_prob     : np.ndarray - Predicted probabilities
    model_name : str        - model identifier for filename
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_val = auc(recall, precision)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(recall, precision, linewidth=2,
            label=f"PR Curve (AUC = {pr_auc_val:.4f})")
    ax.axhline(y=y_true.mean(), color="r", linestyle="--", linewidth=1,
               label=f"Baseline (fraud rate = {y_true.mean():.4f})")
    ax.set_title(f"{model_name} — Precision-Recall Curve", fontsize=14)
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    fig.tight_layout()

    save_path = _get_save_path(model_name, "precision_recall")
    fig.savefig(save_path, dpi=150)
    print(f"[INFO] PR-Curve saved to {save_path}")
    print(f"[INFO] PR-AUC = {pr_auc_val:.4f}  |  Average Precision = {ap:.4f}")
    plt.close(fig)


def plot_all_for_model(y_true: np.ndarray, y_pred: np.ndarray,
                       y_prob: np.ndarray, model_name: str,
                       history=None):
    """
    Generate all evaluation plots for a single model.

    Parameters
    ----------
    y_true     : ground truth labels
    y_pred     : predicted labels
    y_prob     : predicted probabilities
    model_name : model identifier
    history    : optional training history (keras History or dict)
    """
    if history is not None:
        plot_training_loss(history, model_name)
        plot_training_accuracy(history, model_name)
    plot_confusion_matrix(y_true, y_pred, model_name)
    plot_roc_curve(y_true, y_prob, model_name)
    plot_precision_recall_curve(y_true, y_prob, model_name)
    print(f"[INFO] All plots generated for {model_name}\n")


# -- quick test --
if __name__ == "__main__":
    rng = np.random.RandomState(42)
    y_t = np.array([0]*950 + [1]*50)
    y_p = np.array([0]*940 + [1]*10 + [0]*5 + [1]*45)
    y_pr = rng.rand(1000)
    plot_all_for_model(y_t, y_p, y_pr, "test_model")
