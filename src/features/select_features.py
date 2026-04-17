"""
select_features.py -- Feature selection methods.

Provides methods for dimensionality reduction and selecting
the most informative features for model training.
"""

import sys
import os
import numpy as np
import pandas as pd

# --------------- make project root importable ---------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def select_by_correlation(df: pd.DataFrame, threshold: float = 0.95) -> list:
    """
    Identify and return columns to drop based on high pairwise correlation.

    Parameters
    ----------
    df        : pd.DataFrame - Feature DataFrame (no target column)
    threshold : float        - Correlation threshold (default 0.95)

    Returns
    -------
    list : Column names to drop (one from each highly correlated pair)
    """
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    print(f"[INFO] Features with correlation > {threshold}: {to_drop}")
    print(f"[INFO] Dropping {len(to_drop)} features out of {len(df.columns)}")
    return to_drop


def select_by_importance(model, feature_names: list, top_k: int = None) -> list:
    """
    Select top-K features based on tree-based feature importance scores.

    Parameters
    ----------
    model         : A fitted tree-based model with `feature_importances_` attribute
    feature_names : list[str] - Names of all features
    top_k         : int       - Number of top features to keep (None = keep all)

    Returns
    -------
    list : Names of selected features, sorted by importance (descending)
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("[INFO] Feature importance ranking:")
    for rank, idx in enumerate(indices):
        print(f"  {rank + 1:3d}. {feature_names[idx]:<30s} = {importances[idx]:.4f}")

    if top_k is not None:
        selected = [feature_names[i] for i in indices[:top_k]]
        print(f"\n[INFO] Selected top {top_k} features: {selected}")
        return selected
    else:
        return [feature_names[i] for i in indices]


# -- quick test --
if __name__ == "__main__":
    # Smoke test with synthetic data
    df = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
    df["f5"] = df["f0"] + np.random.randn(100) * 0.01  # Highly correlated with f0
    to_drop = select_by_correlation(df, threshold=0.95)
    print(f"Columns to drop: {to_drop}")
