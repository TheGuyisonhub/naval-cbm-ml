<p align="center">
  <img src="assets/banner.png" width="100%" alt="Naval Propulsion Predictive Maintenance Banner">
</p>

<h1 align="center">Naval Propulsion Predictive Maintenance System</h1>

<p align="center">
  End-to-end machine learning system for gas turbine degradation prediction using XGBoost
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Model-XGBoost-orange?style=flat-square" alt="XGBoost">
  <img src="https://img.shields.io/badge/R²-0.9928-brightgreen?style=flat-square" alt="R2 Score">
  <img src="https://img.shields.io/badge/Dataset-UCI-lightgrey?style=flat-square" alt="Dataset">
  <img src="https://img.shields.io/badge/Status-Completed-green?style=flat-square" alt="Status">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="License">
</p>

---

## Table of Contents

- [Overview](#overview)
- [Key Highlights](#key-highlights)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [System Architecture](#system-architecture)
- [Methodology](#methodology)
- [Models Used](#models-used)
- [Model Performance](#model-performance)
- [Engineering Insights](#engineering-insights)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Usage](#usage)
- [Visual Results](#visual-results)
- [Results Interpretation](#results-interpretation)
- [Report](#report)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## Overview

This project presents a predictive maintenance system for naval gas turbine propulsion plants. The goal is to estimate turbine and compressor degradation using sensor readings and machine learning models, enabling early maintenance planning and improved operational reliability.

The solution is built around a full machine learning workflow:

- **Data preprocessing** — cleaning, validation, outlier removal, feature scaling
- **Feature engineering** — physics-based and statistical feature creation
- **Model training** — multiple regression models compared
- **Evaluation** — RMSE, MAE, R², residual analysis, learning curves
- **Visualization** — predicted vs actual, feature importance, residual and learning curve plots

The best-performing model in this project is **XGBoost**, achieving an **R² score of 0.9928**.

---

## Key Highlights

<table align="center">
  <tr>
    <td align="center"><h3>🎯 Objective</h3>Predict turbine and compressor decay coefficients from sensor readings</td>
    <td align="center"><h3>📊 Performance</h3>R² = 0.9928 with XGBoost on UCI Naval dataset</td>
    <td align="center"><h3>🧠 Approach</h3>Hybrid ML pipeline with physics-informed feature engineering</td>
  </tr>
</table>

---

## Problem Statement

Gas turbine propulsion systems degrade over time, which can lead to:

- reduced operational efficiency
- increased maintenance cost
- unexpected failures at sea
- mission-critical risks in naval operations

Traditional time-based maintenance doesn't account for actual system condition. This project addresses that gap by using sensor data to predict degradation **before** failure becomes severe, supporting **condition-based maintenance (CBM)** strategies.

---

## Dataset

The project uses the **UCI Naval Propulsion Plants dataset** for condition-based maintenance research.

| Property | Details |
|---|---|
| **Source** | UCI Machine Learning Repository |
| **Domain** | Naval propulsion / gas turbine condition monitoring |
| **Samples** | ~11,934 readings |
| **Features** | 16 sensor-based and operational measurements |
| **Targets** | 2 decay coefficients (compressor + turbine) |

### Input Features

The 16 features include sensor readings and operational measurements such as:

- Lever Position (lp)
- Ship Speed (v)
- Gas Turbine shaft torque (GTT)
- Gas Turbine rate of revolutions (GTn)
- Gas Generator rate of revolutions (GGn)
- Starboard / Port Propeller Torque (Ts / Tp)
- HP Turbine exit temperature (T48)
- GT Compressor inlet / outlet air temperature (T1 / T2)
- HP Turbine exit pressure (P48)
- GT Compressor inlet / outlet air pressure (P1 / P2)
- GT exhaust gas pressure (Pexh)
- Turbine Injection Control (TIC)
- Fuel flow (mf)

### Prediction Targets

- **GT Compressor Decay State Coefficient** (`kMc`)
- **GT Turbine Decay State Coefficient** (`kMt`)

---

## System Architecture

```text
┌─────────────────────────────────────┐
│       Sensor Inputs (16 features)   │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│   Data Preprocessing                │
│   • Missing value handling          │
│   • Outlier removal (IQR method)    │
│   • Feature scaling (StandardScaler)│
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│   Feature Engineering               │
│   • Pressure ratio calculations     │
│   • Power proxy features            │
│   • Interaction terms               │
│   • Operational mode grouping       │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│   Model Training & Comparison       │
│   Ridge / Random Forest / XGBoost / ANN │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│   Prediction (Decay Coefficients)   │
│   kMc (Compressor) + kMt (Turbine)  │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│   Maintenance Decision Support      │
└─────────────────────────────────────┘
```

---

## Methodology

### 1. Data Processing

- Missing value handling
- Data validation and type checks
- Outlier handling using IQR method
- Feature scaling using `StandardScaler`

### 2. Feature Engineering

- Pressure ratio calculations (P2/P1, P48/P2)
- Power proxy features from torque × RPM
- Interaction terms between correlated sensors
- Operational mode grouping via lever position
- Statistical transformations where distribution required it

### 3. Model Training

Four regression models were trained and compared:

| Model | Role |
|---|---|
| Ridge Regression | Linear baseline |
| Random Forest | Non-linear ensemble baseline |
| XGBoost | Primary model (best performer) |
| ANN (Keras) | Deep learning baseline |

### 4. Evaluation Metrics

Models were evaluated using:

- **RMSE** — Root Mean Squared Error
- **MAE** — Mean Absolute Error
- **R²** — Coefficient of determination
- Residual analysis plots
- Learning curves
- Predicted vs Actual scatter plots

---

## Models Used

### Ridge Regression
Strong regularized linear baseline. Handles multicollinearity in sensor readings. Achieved R² of 0.9058.

### Random Forest
Captures non-linear relationships between sensor readings and degradation. Performed close to XGBoost with R² of 0.9923.

### XGBoost ⭐ Best Model
Gradient boosted trees optimized for structured tabular data. Best overall performance due to its ability to model complex sensor interactions. Achieved **R² = 0.9928**.

### ANN (Keras)
Deep learning baseline with fully connected layers. Underperformed compared to tree-based models on this tabular dataset (R² = 0.7685).

---

## Model Performance

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Ridge Regression | 3.475e-3 | 2.726e-3 | 0.9058 |
| Random Forest | 8.75e-4 | 4.18e-4 | 0.9923 |
| **XGBoost** | **8.60e-4** | **6.02e-4** | **0.9928** |
| ANN | 4.120e-3 | 3.245e-3 | 0.7685 |

> XGBoost outperforms all other models in RMSE and R², confirming its suitability for this structured sensor regression task.

---

## Engineering Insights

- Compressor decay prediction error is approximately **1.9% of the operating range**
- Turbine decay prediction error is approximately **3.07% of the operating range**
- The model generalizes well across different vessel operating conditions (speed, load, fuel flow)
- Feature importance analysis shows that **GT shaft torque**, **HP turbine exit temperature**, and **compressor pressure ratio** are the strongest predictors of decay state

These results indicate the system is suitable for real-world condition-based maintenance scenarios.

---

## Project Structure

```text
naval-cbm-ml/
│
├── main.py                     # Main training and evaluation pipeline
├── requirements.txt            # Python dependencies
│
├── data/
│   └── naval_propulsion.dat    # UCI dataset file
│
├── assets/
│   ├── banner.png              # Repository banner image
│   ├── predicted_vs_actual.png # Model output visualization
│   ├── feature_importance.png  # XGBoost feature importance chart
│   ├── residual_plots.png      # Residual distribution and scatter
│   └── learning_curves.png     # Training/validation learning curves
│
├── outputs/
│   ├── xgboost_model.pkl       # Saved trained XGBoost model
│   ├── metrics_summary.csv     # All model evaluation metrics
│   └── plots/                  # All generated visualization files
│
├── report/
│   └── report.pdf              # Full IEEE-style technical report
│
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10 or newer
- pip
- Git

### Clone the Repository

```bash
git clone https://github.com/TheGuyisonhub/naval-cbm-ml.git
cd naval-cbm-ml
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
tensorflow
joblib
```

---

## Quickstart

```bash
python main.py
```

This will:
1. Load and preprocess the dataset
2. Engineer features
3. Train all four models
4. Evaluate and compare performance
5. Save trained models to `outputs/`
6. Generate and save all visualizations to `assets/`

---

## Usage

### 1. Prepare the Dataset

Download the UCI Naval Propulsion dataset and place it in the `data/` directory:

```text
data/naval_propulsion.dat
```

Dataset source: [UCI Machine Learning Repository — Condition Based Maintenance of Naval Propulsion Plants](https://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants)

### 2. Run the Pipeline

```bash
python main.py
```

### 3. View Outputs

After running, check:

```text
outputs/          → saved model files + metrics CSV
assets/           → all generated plots
```

### 4. Load Saved Model for Inference

```python
import joblib
import numpy as np

model = joblib.load("outputs/xgboost_model.pkl")

# Example: pass preprocessed sensor readings as numpy array
sensor_input = np.array([[...]])  # shape: (1, n_features)
prediction = model.predict(sensor_input)
print("Predicted decay coefficients:", prediction)
```

---

## Visual Results

<p align="center">
  <img src="assets/predicted_vs_actual.png" width="48%" alt="Predicted vs Actual">
  <img src="assets/feature_importance.png" width="48%" alt="Feature Importance">
</p>

<p align="center">
  <img src="assets/residual_plots.png" width="48%" alt="Residual Plots">
  <img src="assets/learning_curves.png" width="48%" alt="Learning Curves">
</p>

---

## Results Interpretation

| Metric | Meaning |
|---|---|
| **High R² (0.9928)** | Model predictions are very close to true degradation values |
| **Low RMSE (8.60e-4)** | Average prediction error is extremely small |
| **Low MAE (6.02e-4)** | Consistent accuracy across all samples |
| **Stable Residuals** | Errors are not skewed — model generalizes well |

The high performance across both targets (compressor and turbine decay) confirms that the engineered features successfully capture the underlying physics of turbine degradation. The model is robust across different operating conditions including speed, load, and fuel flow regimes.

---

## Report

The full IEEE-style technical report covering problem formulation, methodology, results, and analysis is available at:

```text
report/report.pdf
```

---

## Technologies Used

| Category | Tools |
|---|---|
| **Language** | Python 3.10 |
| **Data** | NumPy, Pandas |
| **ML Models** | Scikit-learn, XGBoost, TensorFlow/Keras |
| **Visualization** | Matplotlib, Seaborn |
| **Model Persistence** | Joblib |
| **Version Control** | Git, GitHub |

---

## Future Improvements

- [ ] Real-time IoT deployment using ESP32 or Raspberry Pi
- [ ] LSTM-based temporal degradation modeling
- [ ] Streamlit or FastAPI dashboard for live monitoring
- [ ] Digital twin integration for simulation-driven maintenance
- [ ] Automated CI/CD deployment pipeline
- [ ] SHAP-based explainability dashboard for maintenance teams
- [ ] Multi-output regression refinement per target

---

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add: your feature description'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request

Please ensure your changes are tested, clearly documented, and consistent with the existing code style.

---

## License

This project is released under the **MIT License**.

```text
MIT License

Copyright (c) 2025 Muhammad Wasea Mughal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## Acknowledgements

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/) for the Naval Propulsion dataset
- NUST College of E&ME for academic infrastructure and support
- The open-source Python ML ecosystem (scikit-learn, XGBoost, Keras)
- Research community in predictive maintenance and CBM

---

## Contact

**Author:** Muhammad Wasea Mughal  
**Email:** muhammadwasea04@gmail.com  
**GitHub:** [github.com/TheGuyisonhub](https://github.com/TheGuyisonhub)  
**LinkedIn:** [linkedin.com/in/muhammad-wasea-b852492a2](https://linkedin.com/in/muhammad-wasea-b852492a2)  
**Repository:** [github.com/TheGuyisonhub/naval-cbm-ml](https://github.com/TheGuyisonhub/naval-cbm-ml)

---

<p align="center">
  <b>Built for predictive maintenance research in mechanical and naval systems</b><br>
  <sub>NUST College of E&ME · 2025</sub>
</p>
