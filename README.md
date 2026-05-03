<h1 align="center">Naval Propulsion Predictive Maintenance</h1>
<p align="center">
Machine Learning system for predicting gas turbine degradation using XGBoost (R² = 0.9928)
</p>
<p align="center">
<img src="https://img.shields.io/badge/Python-3.10-blue">
<img src="https://img.shields.io/badge/ML-XGBoost-orange">
<img src="https://img.shields.io/badge/Project-Completed-green">
<img src="https://img.shields.io/badge/Dataset-UCI-lightgrey">
</p>

---

## 🚀 Overview

This project builds an end-to-end **condition-based maintenance (CBM)** system for naval gas turbine propulsion systems.

It predicts:
- Compressor Decay
- Turbine Decay

using 16 real-time sensor readings from a CODLAG propulsion system.

👉 Best Model: **XGBoost (R² = 0.9928)**

---

## 📊 Key Result

<p align="center">
  <b>XGBoost achieves R² = 0.9928 with RMSE < 1e-3</b>
</p>
<p align="center">
  <img src="assets/predicted_vs_actual.png" width="45%">
  <img src="assets/feature_importance.png" width="45%">
</p>

---

## 🧠 Problem Statement

Gas turbine components degrade over time, leading to:
- Efficiency loss
- Increased failure risk
- High maintenance cost

This system enables **early fault detection before failure occurs**.

---

## ⚙️ Pipeline Overview

### 🔹 Data Processing
- Feature cleaning
- Outlier handling (IQR)
- Standardization (no leakage)

### 🔹 Feature Engineering
- Pressure Ratio (Πc)
- Power Proxy (Torque × RPM)
- Operating Mode classification

### 🔹 Models Used
- Ridge Regression
- Random Forest
- XGBoost ⭐
- ANN (Keras)

---

## 📈 Model Performance

| Model | RMSE | MAE | R² |
|------|------|-----|-----|
| Ridge | 3.475e-3 | 2.726e-3 | 0.9058 |
| Random Forest | 8.75e-4 | 4.18e-4 | 0.9923 |
| **XGBoost** | **8.60e-4** | 6.02e-4 | **0.9928** |
| ANN | 4.120e-3 | 3.245e-3 | 0.7685 |

---

## ⚙️ Engineering Insight

- Compressor error ≈ **1.9% of operating range**
- Turbine error ≈ **3.07% of operating range**

👉 Both values are within acceptable limits for **real-world CBM systems**

---

## 📊 Visual Diagnostics

<p align="center">
  <img src="assets/learning_curves.png" width="48%">
  <img src="assets/overfitting_diagnosis.png" width="48%">
</p>
<p align="center">
  <img src="assets/residual_plots.png" width="48%">
  <img src="assets/residual_qq_plots.png" width="48%">
</p>
<p align="center">
  <img src="assets/predicted_vs_actual.png" width="48%">
  <img src="assets/feature_importance.png" width="48%">
</p>

---

## 🚀 Quickstart

```bash
git clone https://github.com/TheGuyisonhub/naval-cbm-ml.git
cd naval-cbm-ml
pip install -r requirements.txt
# Add dataset (UCI data.txt → data/)
python main.py
```

---

## 📁 Structure

```
naval-cbm-ml/
├── main.py
├── requirements.txt
├── data/README.md        ← download UCI data.txt here
├── outputs/              ← plots + CSVs (auto-generated)
├── assets/               ← plots for README display
└── report/report.pdf     ← IEEE-format paper
```

---

## 📄 Report

Full IEEE-format paper in [`report/`](./report/).

---

## 🏫 Authors

Muhammad Wasea · Masooma Raza · Muhammad Azan Malik · Muhammad Abdur Rehman Shahzad  
CS-333 Applied AI & ML — NUST CEME, ME-45
