"""
config.py -- Central configuration for the Fraud Detection pipeline.
All hyperparameters, file paths, and architecture settings live here
so every module reads from a single source of truth.
"""

import os

# ----------------------------------------------
# 1. Paths
# ----------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "bs140513_032310.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODELS_DIR = os.path.join(BASE_DIR, "src", "models", "saved")

# Create directories if they don't exist
for _d in [PROCESSED_DIR, REPORTS_DIR, MODELS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ----------------------------------------------
# 2. Data Preprocessing
# ----------------------------------------------
# Columns that are constant and carry no predictive value
COLS_TO_DROP = ["zipcodeOri", "zipMerchant"]

# Categorical columns to encode with LabelEncoder
CATEGORICAL_COLS = ["customer", "merchant", "category"]

# Target column
TARGET_COL = "fraud"

# Temporal column used for train/test split
STEP_COL = "step"

# Train-test split ratio (by step percentile)
TRAIN_RATIO = 0.80

# ----------------------------------------------
# 3. Autoencoder Architecture
# ----------------------------------------------
# Encoder layer sizes (input -> bottleneck)
ENCODER_LAYERS = [32, 16]

# Latent / bottleneck dimension
LATENT_DIM = 8

# Decoder mirrors the encoder automatically
# So decoder will be: LATENT_DIM -> 16 -> 32 -> input_dim

# Training
AE_EPOCHS = 100
AE_BATCH_SIZE = 256
AE_LEARNING_RATE = 1e-3
AE_VALIDATION_SPLIT = 0.1

# Early stopping patience
AE_PATIENCE = 10

# ----------------------------------------------
# 4. XGBoost Classifier
# ----------------------------------------------
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 80,        # ~ ratio of normal/fraud to handle imbalance
    "eval_metric": "aucpr",
    "use_label_encoder": False,
    "random_state": 42,
    "n_jobs": -1,
}

# ----------------------------------------------
# 5. Random seed for reproducibility
# ----------------------------------------------
RANDOM_SEED = 42
