# TODO — Naval Propulsion CBM ML System

## 🚀 Immediate (Pre-Submission / GitHub Finalization)

- [ ] Add dataset access method (UCI link + optional auto-download script)
- [ ] Embed key figures in README (`predicted_vs_actual`, `feature_importance`)
- [ ] Sync `requirements.txt` with execution environment (freeze versions)
- [ ] Add lightweight CI check (GitHub Actions):
  - Run `python main.py --smoke-test`
  - Validate pipeline executes end-to-end without errors
- [ ] Ensure `.gitignore` excludes dataset, models, and large artifacts

---

## ⚙️ Model & Performance Enhancements

- [ ] Replace GridSearchCV with **Optuna** for efficient hyperparameter optimization
- [ ] Add **5-fold cross-validation reporting (mean ± std R² / RMSE)**
- [ ] Implement **uncertainty estimation**
  - Quantile regression or conformal prediction intervals
- [ ] Improve ANN architecture tuning (batch size, dropout, regularization)

---

## 🧠 Explainability & Interpretability

- [ ] Integrate **SHAP analysis for XGBoost**
  - Global feature importance
  - Local prediction explanations
- [ ] Replace permutation-based ANN interpretation with SHAP-compatible surrogate analysis
- [ ] Add feature sensitivity analysis for key sensors

---

## 📈 Advanced Modeling Extensions

- [ ] Time-series modeling (LSTM / GRU) for degradation trajectories
- [ ] Transformer-based sequence learning for long-range dependencies
- [ ] Multi-task learning architecture (shared backbone for both targets)
- [ ] Stacking ensemble (RF + XGBoost + ANN → meta-learner)

---

## 🛠️ Engineering & Deployment

- [ ] Convert pipeline into reusable Python package:
  - `predict(sensor_input)` API
- [ ] Add Docker container for reproducible inference
- [ ] Benchmark inference latency (ms/sample)
- [ ] Evaluate deployment feasibility for embedded/edge systems

---

## 🌐 Portfolio & Presentation

- [ ] Add GitHub profile feature section (pinned repository)
- [ ] Write LinkedIn technical post (problem → approach → results → impact)
- [ ] Add repository badges (Python version, dataset source, license)
- [ ] Record short demo (plots + pipeline execution walkthrough)
- [ ] Add architecture diagram (data → model → output flow)

---

## ✅ Completed (Core ML Pipeline)

- [x] Exploratory Data Analysis (distribution, correlation, pairplots)
- [x] Data preprocessing (scaling, feature filtering, leakage prevention)
- [x] Feature engineering (pressure ratio, power proxy, operating modes)
- [x] Ridge Regression baseline (RidgeCV)
- [x] Random Forest (GridSearchCV)
- [x] XGBoost optimized model
- [x] ANN (Keras with EarlyStopping + LR scheduling)
- [x] Multi-output regression (2 targets)
- [x] Statistical validation (Wilcoxon signed-rank test)
- [x] Model comparison (R², RMSE, MAE)
- [x] Overfitting analysis (train/val/test split evaluation)
- [x] Learning curves + residual diagnostics
- [x] Feature importance across models
- [x] Engineering interpretation (% operating range error)
- [x] IEEE-style technical report
- [x] Model persistence (.pkl, .keras exports)
