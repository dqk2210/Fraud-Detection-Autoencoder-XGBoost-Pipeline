"""
autoencoder.py -- Build, train, and use the Autoencoder for feature extraction.

Migrated from the original src/features/autoencoder.py.
All hyperparameters loaded from YAML configs.

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

import sys
import os
import gc
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Windows compatibility
import matplotlib.pyplot as plt

# --------------- make project root importable ---------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import get_paths, get_model_config, get_train_config
from src.utils import setup_logger, timestamp, ensure_dir

# TensorFlow / Keras imports
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def set_seeds(seed: int = None):
    """Ensure reproducibility across NumPy and TensorFlow."""
    if seed is None:
        seed = get_train_config()["random_seed"]
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
    model_cfg = get_model_config()["autoencoder"]

    encoder_layers = model_cfg["encoder_layers"]
    latent_dim = model_cfg["latent_dim"]
    activation = model_cfg["activation"]
    output_activation = model_cfg["output_activation"]
    loss = model_cfg["loss"]

    train_cfg = get_train_config()["autoencoder"]
    learning_rate = train_cfg["learning_rate"]

    # ---- Input ----
    input_layer = Input(shape=(input_dim,), name="ae_input")

    # ---- Encoder ----
    # Progressively compress the representation:
    #   input_dim -> 32 -> 16 -> 8 (latent)
    x = input_layer
    for i, units in enumerate(encoder_layers):
        x = Dense(units, activation=activation, name=f"encoder_{i+1}")(x)

    # Bottleneck (latent space)
    latent = Dense(latent_dim, activation=activation, name="latent")(x)

    # ---- Decoder ----
    # Mirror the encoder layers in reverse:
    #   8 -> 16 -> 32 -> input_dim
    x = latent
    for i, units in enumerate(reversed(encoder_layers)):
        x = Dense(units, activation=activation, name=f"decoder_{i+1}")(x)

    # Output layer reconstructs the original input (linear activation)
    output_layer = Dense(input_dim, activation=output_activation, name="ae_output")(x)

    # ---- Compile full autoencoder ----
    autoencoder = Model(inputs=input_layer, outputs=output_layer, name="autoencoder")
    autoencoder.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss,
    )

    # ---- Encoder-only model (for latent feature extraction) ----
    encoder = Model(inputs=input_layer, outputs=latent, name="encoder")

    autoencoder.summary()
    return autoencoder, encoder


# ============================================================
# 2. TRAIN THE AUTOENCODER (on normal transactions only)
# ============================================================
def train_autoencoder(autoencoder: Model, X_train_normal: np.ndarray,
                      logger=None):
    """
    Train the autoencoder on ONLY the normal (non-fraud) training samples.

    Using EarlyStopping & ReduceLROnPlateau to avoid over-fitting and
    help convergence respectively.

    Parameters
    ----------
    autoencoder     : compiled Keras Model
    X_train_normal  : np.ndarray, shape (n_normal, n_features)
    logger          : logging.Logger (optional)

    Returns
    -------
    history : keras History object (contains training & validation loss)
    """
    paths = get_paths()
    train_cfg = get_train_config()["autoencoder"]

    epochs = train_cfg["epochs"]
    batch_size = train_cfg["batch_size"]
    validation_split = train_cfg["validation_split"]
    patience = train_cfg["early_stopping_patience"]
    ckpt_every = train_cfg.get("checkpoint_every_n_epochs", 10)

    # Checkpoint directory
    ckpt_dir = paths["models_saved"]["checkpoints_dir"]
    ensure_dir(ckpt_dir)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=train_cfg["reduce_lr_factor"],
            patience=train_cfg["reduce_lr_patience"],
            min_lr=train_cfg["reduce_lr_min"],
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=os.path.join(ckpt_dir, "ae_epoch_{epoch:03d}.weights.h5"),
            save_weights_only=True,
            save_freq=ckpt_every,
            verbose=0,
        ),
    ]

    if logger:
        logger.info(f"Starting AE training: {epochs} epochs, batch={batch_size}, "
                     f"val_split={validation_split}")

    history = autoencoder.fit(
        X_train_normal, X_train_normal,    # input == target (reconstruction)
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    # Save the best autoencoder weights
    save_path = paths["models_saved"]["autoencoder"]
    ensure_dir(os.path.dirname(save_path))
    autoencoder.save_weights(save_path)

    msg = f"Autoencoder weights saved to {save_path}"
    print(f"\n[INFO] {msg}")
    if logger:
        logger.info(msg)

    return history


# -- quick test --
if __name__ == "__main__":
    # Smoke test: build and print summary for a dummy 7-feature input
    ae, enc = build_autoencoder(input_dim=7)
