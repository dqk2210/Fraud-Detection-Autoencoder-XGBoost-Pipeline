"""
autoencoder.py -- Build, train, and use the Autoencoder for feature extraction.

Architecture overview
---------------------
The Autoencoder is trained ONLY on normal (non-fraud) transactions so that it
learns a compressed representation of "normal" behaviour.

    Input (n_features)
      |
    Dense 32  [ReLU]          <- Encoder layer 1
      |
    Dense 16  [ReLU]          <- Encoder layer 2
      |
    Dense  8  [ReLU]          <- * Latent / Bottleneck layer
      |
    Dense 16  [ReLU]          <- Decoder layer 1 (mirror)
      |
    Dense 32  [ReLU]          <- Decoder layer 2 (mirror)
      |
    Dense n_features [linear] <- Reconstruction layer

Loss  : Mean Squared Error (MSE)
Optim : Adam
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Windows compatibility
import matplotlib.pyplot as plt

# --------------- make project root importable ---------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.config import (
    ENCODER_LAYERS, LATENT_DIM,
    AE_EPOCHS, AE_BATCH_SIZE, AE_LEARNING_RATE,
    AE_VALIDATION_SPLIT, AE_PATIENCE,
    REPORTS_DIR, MODELS_DIR, RANDOM_SEED,
)

# TensorFlow / Keras imports  (defer to avoid slow top-level import)
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


def set_seeds(seed: int = RANDOM_SEED):
    """Ensure reproducibility across NumPy and TensorFlow."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# 1. BUILD THE AUTOENCODER
# ============================================================
def build_autoencoder(input_dim: int):
    """
    Construct a symmetric Autoencoder using the Keras Functional API.

    Parameters
    ----------
    input_dim : int
        Number of input features (after preprocessing & scaling).

    Returns
    -------
    autoencoder  : keras.Model  - full encoder-decoder model
    encoder      : keras.Model  - encoder-only model (for feature extraction)
    """
    set_seeds()

    # ---- Input ----
    input_layer = Input(shape=(input_dim,), name="ae_input")

    # ---- Encoder ----
    # Progressively compress the representation:
    #   input_dim -> 32 -> 16 -> 8 (latent)
    x = input_layer
    for i, units in enumerate(ENCODER_LAYERS):
        x = Dense(units, activation="relu", name=f"encoder_{i+1}")(x)

    # Bottleneck (latent space)
    latent = Dense(LATENT_DIM, activation="relu", name="latent")(x)

    # ---- Decoder ----
    # Mirror the encoder layers in reverse:
    #   8 -> 16 -> 32 -> input_dim
    x = latent
    for i, units in enumerate(reversed(ENCODER_LAYERS)):
        x = Dense(units, activation="relu", name=f"decoder_{i+1}")(x)

    # Output layer reconstructs the original input (linear activation)
    output_layer = Dense(input_dim, activation="linear", name="ae_output")(x)

    # ---- Compile full autoencoder ----
    autoencoder = Model(inputs=input_layer, outputs=output_layer, name="autoencoder")
    autoencoder.compile(
        optimizer=Adam(learning_rate=AE_LEARNING_RATE),
        loss="mse",
    )

    # ---- Encoder-only model (for latent feature extraction) ----
    encoder = Model(inputs=input_layer, outputs=latent, name="encoder")

    autoencoder.summary()
    return autoencoder, encoder


# ============================================================
# 2. TRAIN THE AUTOENCODER (on normal transactions only)
# ============================================================
def train_autoencoder(autoencoder: Model, X_train_normal: np.ndarray):
    """
    Train the autoencoder on ONLY the normal (non-fraud) training samples.

    Using EarlyStopping & ReduceLROnPlateau to avoid over-fitting and
    help convergence respectively.

    Parameters
    ----------
    autoencoder     : compiled Keras Model
    X_train_normal  : np.ndarray, shape (n_normal, n_features)

    Returns
    -------
    history : keras History object (contains training & validation loss)
    """
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=AE_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = autoencoder.fit(
        X_train_normal, X_train_normal,    # input == target (reconstruction)
        epochs=AE_EPOCHS,
        batch_size=AE_BATCH_SIZE,
        validation_split=AE_VALIDATION_SPLIT,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    # Save the trained autoencoder weights
    save_path = os.path.join(MODELS_DIR, "autoencoder.weights.h5")
    autoencoder.save_weights(save_path)
    print(f"\n[INFO] Autoencoder weights saved to {save_path}")

    return history


# ============================================================
# 3. PLOT TRAINING LOSS
# ============================================================
def plot_training_loss(history):
    """Visualise training vs. validation reconstruction loss."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.history["loss"], label="Training Loss", linewidth=2)
    ax.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
    ax.set_title("Autoencoder Reconstruction Loss (MSE)", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    save_path = os.path.join(REPORTS_DIR, "ae_training_loss.png")
    fig.savefig(save_path, dpi=150)
    print(f"[INFO] Training loss plot saved to {save_path}")
    plt.close(fig)


# ============================================================
# 4. EXTRACT LATENT FEATURES
# ============================================================
def extract_latent_features(encoder: Model,
                            X_train: np.ndarray,
                            X_test: np.ndarray):
    """
    Pass both train and test data through the encoder to extract
    the latent-space representation (bottleneck activations).

    Returns
    -------
    latent_train : np.ndarray, shape (n_train, LATENT_DIM)
    latent_test  : np.ndarray, shape (n_test,  LATENT_DIM)
    """
    latent_train = encoder.predict(X_train, batch_size=AE_BATCH_SIZE)
    latent_test = encoder.predict(X_test, batch_size=AE_BATCH_SIZE)

    print(f"[INFO] Latent features extracted -> train {latent_train.shape}, test {latent_test.shape}")
    return latent_train, latent_test


# ============================================================
# 5. BUILD HYBRID FEATURE SET
# ============================================================
def build_hybrid_features(X_original: np.ndarray,
                          X_latent: np.ndarray) -> np.ndarray:
    """
    Concatenate the original preprocessed features with the
    autoencoder latent features to form the hybrid feature set.

    hybrid = [original_features | latent_features]
    """
    hybrid = np.hstack([X_original, X_latent])
    print(f"[INFO] Hybrid feature matrix: {hybrid.shape}  "
          f"(original {X_original.shape[1]} + latent {X_latent.shape[1]})")
    return hybrid


# -- quick test --
if __name__ == "__main__":
    # Smoke test: build and print summary for a dummy 7-feature input
    ae, enc = build_autoencoder(input_dim=7)
