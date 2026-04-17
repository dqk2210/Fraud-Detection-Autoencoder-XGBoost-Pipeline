# Fraud Detection: Autoencoder + XGBoost Multistage Pipeline

## 1. Project Overview & Performance Summary
This project implements a high-performance fraud detection system for the BankSim dataset. By combining **Unsupervised Deep Learning** (Autoencoder) with **Supervised Gradient Boosting** (XGBoost), the model achieves exceptional sensitivity to fraudulent transactions while maintaining a robust overall performance.

### Key Performance Metrics
| Metric | Value |
| :--- | :--- |
| **PR-AUC (Precision-Recall AUC)** | **0.9211** |
| **Fraud Recall (Sensitivity)** | **95.21%** |
| **Fraud Precision** | **53.64%** |
| **ROC-AUC** | **0.9984** |
| **Overall Accuracy** | **99.06%** |

**Confusion Matrix (Test Set):**
- **True Positives (Detected Fraud):** 1,371
- **False Negatives (Missed Fraud):** 69
- **False Positives (False Alarms):** 1,185
- **True Negatives (Correct Legitimate):** 130,606

---

## 2. Data & Preprocessing Details
- **Dataset Size:** 594,643 transactions.
- **Class Imbalance:** Only **1.21%** of transactions are fraudulent.
- **Temporal Strategy:** Instead of random shuffling, we used a **Temporal Split** (80/20) at Step 144. This ensures the model is tested on "future" data, simulating real-world deployment.
- **Feature Engineering:**
    - **Log Transformation:** Applied `log1p` to the `amount` column to normalize its distribution.
    - **Categorical Encoding:** `age` and `gender` mapped to numeric; `customer`, `merchant`, and `category` encoded via `LabelEncoder`.
    - **Standardization:** All features scaled using `StandardScaler` (fit strictly on training data).

---

## 3. Multistage Architecture
### Stage 1: The Autoencoder (Unsupervised)
We built a symmetric Deep Autoencoder to learn the "latent signature" of normal transactions.
- **Architecture:** `6 (Input) -> 32 -> 16 -> 8 (Latent) -> 16 -> 32 -> 6 (Output)`.
- **Training:** 100 epochs on **normal transactions only**.
- **Result:** The 8-dimensional bottleneck (latent) layer captures high-level patterns of legitimate behavior.

![Autoencoder Training Loss](reports/ae_training_loss.png)
*Figure 1: Autoencoder convergence showing rapid minimization of reconstruction error.*

### Stage 2: Hybrid Feature Set
We combined the **6 original features** with the **8 latent features** extracted from the Autoencoder, creating a **14-dimensional hybrid feature matrix**. This provides the final classifier with both raw data and deep-learned behavioral patterns.

### Stage 3: XGBoost Classifier (Supervised)
- **Configuration:** 300 estimators, depth of 6, and `scale_pos_weight=80` to specifically target the minority fraud class.
- **Evaluation:** Optimized for `aucpr` (Area Under Precision-Recall Curve).

---

## 4. Visual Evaluation Results

### Precision-Recall Curve
The PR-Curve is the "gold standard" for imbalanced data. Our model maintains high precision even at very high recall levels.
![Precision-Recall Curve](reports/precision_recall_curve.png)

### Confusion Matrix
Visualizing the trade-off between sensitivity and false alarms.
![Confusion Matrix](reports/confusion_matrix.png)

---

## 5. Execution Summary
- **Total Parameters:** 1,774 (Autoencoder).
- **Training Time:** ~4.1 minutes (Full pipeline).
- **Tooling:** Python, TensorFlow/Keras, XGBoost, Scikit-Learn, Pandas, Matplotlib.

---
*Generated on: 2026-04-16*