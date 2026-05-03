# Naval Propulsion Predictive Maintenance (Machine Learning)

End-to-end machine learning pipeline for **condition-based maintenance (CBM)** of naval gas turbine propulsion systems, designed to predict component degradation before failure.

---

## 🚀 Key Result

**XGBoost achieves R² = 0.9928 (RMSE < 1e-3)** for simultaneous prediction of compressor and turbine decay coefficients.

This enables highly accurate **early-warning maintenance decisions** in naval propulsion systems.

![Predicted vs Actual](assets/predicted_vs_actual.png)
![Feature Importance](assets/feature_importance.png)

---

## 🧠 Problem Statement

Gas turbine propulsion systems degrade over time, leading to reduced efficiency and potential failure.

This project predicts:
- GT Compressor Decay
- GT Turbine Decay

using real-time sensor data to enable **condition-based maintenance (CBM)** before failures occur.

---

## ⚙️ Dataset

**UCI Naval Propulsion Plants Dataset**  
- 11,934 samples  
- 16 sensor features  
- 2 continuous targets  
- CODLAG propulsion system (diesel-electric + gas turbine simulation)

🔗 Source: https://archive.ics.uci.edu/dataset/316/condition+based+maintenance+of+naval+propulsion+plants

---

## 🏗️ Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Feature distributions (histograms, box plots)
- Correlation heatmaps
- Pairwise feature relationships
- Statistical summaries

### 2. Data Preprocessing
- Removal of zero-variance features
- IQR-based outlier inspection (retained for valid operating states)
- StandardScaler applied (train-only fit to avoid leakage)

### 3. Feature Engineering
- Compressor Pressure Ratio: Πc = P2 / P1  
- Turbine Power Proxy: WT = Torque × RPM  
- Operating mode classification (Low / Medium / High)

---

## 🤖 Models Used

| Model | Type | Description |
|------|------|-------------|
| Ridge Regression | Linear | Baseline model with L2 regularization |
| Random Forest | Ensemble | Bagging-based non-linear model |
| XGBoost ⭐ | Boosting | Best performing model |
| ANN (Keras) | Neural Network | Deep learning baseline |

---

## 📊 Performance Summary

| Model | RMSE | MAE | R² |
|------|------|-----|----|
| Ridge Regression | 3.475e-3 | 2.726e-3 | 0.9058 |
| Random Forest | 8.75e-4 | **4.18e-4** | 0.9923 |
| **XGBoost** | **8.60e-4** | 6.02e-4 | **0.9928** |
| ANN | 4.120e-3 | 3.245e-3 | 0.7685 |

---

## ⚙️ Engineering Interpretation

- Compressor decay prediction error ≈ **1.9% of operating range**
- Turbine decay prediction error ≈ **3.07% of operating range**

👉 Both are within acceptable thresholds for **real-world CBM early warning systems**

---

## 📈 Output Visualizations

Generated automatically in `/outputs`:

- Feature distributions
- Correlation heatmap
- Pair plots
- Learning curves
- Residual plots
- Q-Q plots
- Predicted vs Actual comparison
- Feature importance analysis
- Model comparison charts

---

## 🚀 Quickstart

```bash
# 1. Clone repository
git clone https://github.com/TheGuyisonhub/naval-cbm-ml.git
cd naval-cbm-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add dataset
# Download data.txt from UCI and place in data/data.txt

# 4. Run pipeline
python main.py
