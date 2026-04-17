"""
build_features.py -- Feature construction utilities.

Provides functions to build hybrid feature sets by combining
original preprocessed features with autoencoder latent features.

Migrated from the original src/features/autoencoder.py (feature extraction logic).
"""

import sys
import os
import numpy as np

# --------------- make project root importable ---------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import get_model_config


def extract_latent_features(encoder, X_train: np.ndarray, X_test: np.ndarray):
    """
    Pass both train and test data through the encoder to extract
    the latent-space representation (bottleneck activations).

    Parameters
    ----------
    encoder  : keras.Model   - Encoder-only model
    X_train  : np.ndarray    - Scaled training features
    X_test   : np.ndarray    - Scaled testing features

    Returns
    -------
    latent_train : np.ndarray, shape (n_train, LATENT_DIM)
    latent_test  : np.ndarray, shape (n_test,  LATENT_DIM)
    """
    model_cfg = get_model_config()
    batch_size = 256  # AE batch size for prediction

    latent_train = encoder.predict(X_train, batch_size=batch_size)
    latent_test = encoder.predict(X_test, batch_size=batch_size)

    print(f"[INFO] Latent features extracted -> train {latent_train.shape}, test {latent_test.shape}")
    return latent_train, latent_test


def build_hybrid_features(X_original: np.ndarray,
                          X_latent: np.ndarray) -> np.ndarray:
    """
    Concatenate the original preprocessed features with the
    autoencoder latent features to form the hybrid feature set.

    hybrid = [original_features | latent_features]

    Parameters
    ----------
    X_original : np.ndarray - Original scaled features
    X_latent   : np.ndarray - Latent features from encoder

    Returns
    -------
    np.ndarray - Hybrid feature matrix
    """
    hybrid = np.hstack([X_original, X_latent])
    print(f"[INFO] Hybrid feature matrix: {hybrid.shape}  "
          f"(original {X_original.shape[1]} + latent {X_latent.shape[1]})")
    return hybrid


# -- quick test --
if __name__ == "__main__":
    X_orig = np.random.randn(100, 6)
    X_lat = np.random.randn(100, 8)
    hybrid = build_hybrid_features(X_orig, X_lat)
    print(f"Hybrid shape: {hybrid.shape}")
