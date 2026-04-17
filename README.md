# Fraud Detection: Autoencoder + XGBoost Multistage Pipeline

A high-performance fraud detection system for the **BankSim** dataset, combining **Unsupervised Deep Learning** (Autoencoder) with **Supervised Gradient Boosting** (XGBoost) and **Random Forest** for multi-model comparison.

---

## Project Structure

```
Fraud_Detection/
├── data/
│   ├── raw/               # Original immutable data (never overwrite)
│   ├── interim/           # SMOTE output, encoded data, intermediate steps
│   ├── processed/         # Final cleaned data ready for modeling
│   └── external/          # Any supplementary external datasets
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Basic stats, shape, null checks
│   ├── 02_eda_visualization.ipynb     # Distribution plots, correlation heatmap
│   ├── 03_feature_engineering.ipynb   # Feature selection, SMOTE, scaling
│   ├── 04_model_experiments.ipynb     # Training runs, hyperparameter tuning
│   └── 05_results_analysis.ipynb      # Final metric comparison across models
│
├── src/
│   ├── data/
│   │   ├── preprocess.py      # Cleaning, scaling, encoding
│   │   └── augment.py         # SMOTE and class balancing
│   ├── features/
│   │   ├── build_features.py  # Feature construction (hybrid features)
│   │   └── select_features.py # Feature selection methods
│   ├── models/
│   │   ├── autoencoder.py     # Autoencoder architecture & training
│   │   ├── random_forest.py   # Random Forest classifier
│   │   ├── xgboost_model.py   # XGBoost classifier
│   │   └── train_all.py       # Sequential training with logging
│   ├── evaluation/
│   │   ├── metrics.py         # Precision, Recall, F1, AUC-ROC, AUC-PR
│   │   └── plot_results.py    # Save timestamped plots per model
│   ├── utils.py               # Shared utility functions
│   ├── config.py              # Load YAML configs
│   └── __init__.py
│
├── reports/
│   ├── figures/               # All EDA and evaluation plots
│   ├── metrics/               # scores.json per model
│   ├── comparisons/           # Multi-model comparison plots
│   └── fraud_report.html      # Final HTML summary report
│
├── logs/                      # Per-model training logs
├── models_saved/              # Saved model weights & checkpoints
│   └── checkpoints/
│
├── config/
│   ├── model_config.yaml      # Hyperparameters per model
│   ├── train_config.yaml      # Epochs, batch size, learning rate
│   └── paths.yaml             # All data/output paths
│
├── main.py                    # Quick pipeline (AE + XGBoost)
├── requirements.txt
├── README.md
└── PROJECT_SUMMARY.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Main Pipeline (Autoencoder + XGBoost)

```bash
python main.py
```

This runs the original pipeline:
1. Load & preprocess the BankSim dataset
2. Train the Autoencoder on normal transactions only
3. Extract latent features and build hybrid feature set
4. Train XGBoost on hybrid features
5. Evaluate and save results

### 3. Run All Models Sequentially

```bash
python -m src.models.train_all
```

This trains all three models in sequence:
- **Autoencoder** → unsupervised anomaly detection
- **Random Forest** → supervised classification on hybrid features
- **XGBoost** → supervised classification on hybrid features

Each model:
- Writes its log to `logs/<model_name>_train.log`
- Saves metrics to `reports/metrics/<model_name>_scores.json`
- Saves plots to `reports/figures/` with timestamps
- Appends a summary row to `logs/run_summary.log`

---

## Notebooks (Run in Order)

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `01_data_exploration.ipynb` | Load raw data, inspect shape, dtypes, null counts, class distribution |
| 02 | `02_eda_visualization.ipynb` | Class imbalance chart, correlation heatmap, amount/time distributions |
| 03 | `03_feature_engineering.ipynb` | StandardScaler, SMOTE oversampling, temporal split, save processed data |
| 04 | `04_model_experiments.ipynb` | Train Autoencoder, Random Forest, XGBoost with evaluation |
| 05 | `05_results_analysis.ipynb` | Load all metrics JSONs, compare models in tables and charts |

> **Tip:** Run notebooks from the `notebooks/` directory so relative paths resolve correctly.

---

## Configuration

All hyperparameters and paths are managed via YAML files in `config/`:

- **`paths.yaml`** — Directory and file paths (no hardcoding in source code)
- **`model_config.yaml`** — Model architectures and hyperparameters
- **`train_config.yaml`** — Training settings (epochs, batch size, etc.)

To modify any setting, edit the corresponding YAML file — no code changes needed.

---

## Key Results

| Metric | Value |
|--------|-------|
| PR-AUC | 0.9211 |
| Fraud Recall | 95.21% |
| ROC-AUC | 0.9984 |
| Overall Accuracy | 99.06% |

---

## Tech Stack

- **Python 3.10+**
- **TensorFlow / Keras** — Autoencoder
- **XGBoost** — Gradient boosting classifier
- **scikit-learn** — Random Forest, preprocessing, metrics
- **imbalanced-learn** — SMOTE oversampling
- **Pandas / NumPy** — Data manipulation
- **Matplotlib / Seaborn** — Visualization
