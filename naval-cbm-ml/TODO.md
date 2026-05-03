# TODO — Naval CBM ML Project

## Immediate (Before Submission)
- [ ] Upload `data.txt` link in README (or add download script)
- [ ] Add selected output figures to `assets/` and embed in README
- [ ] Verify `requirements.txt` versions match Colab environment
- [ ] Add GitHub Actions CI — `python main.py --smoke-test` on push

## Short-Term Improvements
- [ ] **SHAP values** — explainability for XGBoost predictions (replace permutation importance for ANN too)
- [ ] **Streamlit dashboard** — real-time inference UI with sliders for each sensor input
- [ ] **Hyperparameter tuning** — replace GridSearchCV with Optuna for faster + smarter search
- [ ] **Cross-validation metrics** — report mean ± std R² across 5 folds, not just single split
- [ ] **Confidence intervals** — quantile regression or conformal prediction for uncertainty bounds

## Modelling Extensions
- [ ] **LSTM** — treat each operating profile as a time series, predict degradation trajectory
- [ ] **Transformer** — position-sensitive attention for long-range sensor dependencies
- [ ] **Multi-task learning** — shared ANN trunk for compressor + turbine (vs current independent heads)
- [ ] **Ensemble stacking** — meta-learner on top of RF + XGBoost predictions

## Engineering / Deployment
- [ ] Package as Python module with `predict(sensor_dict)` API
- [ ] Docker container for reproducible inference
- [ ] Validate on real vessel data (when available)
- [ ] Benchmark inference speed (ms/sample) for embedded deployment

## Portfolio / GitHub
- [ ] Add repo to GitHub profile README under "ML Projects"
- [ ] Write a LinkedIn post / project write-up (tag NUST CEME, CS-333)
- [ ] Add badges: Python version, license, dataset source
- [ ] Record a short demo video of the pipeline output plots

## Done ✓
- [x] EDA (histograms, boxplots, heatmap, pairplot, target distributions)
- [x] Preprocessing (zero-variance removal, IQR outlier analysis, StandardScaler)
- [x] Feature engineering (Pressure Ratio, Power Proxy, Operating Mode)
- [x] Ridge Regression baseline with RidgeCV
- [x] Random Forest with GridSearchCV (5-fold)
- [x] XGBoost with GridSearchCV (5-fold)
- [x] ANN (Keras) with EarlyStopping + ReduceLROnPlateau
- [x] Per-target RMSE (Compressor + Turbine)
- [x] Wilcoxon signed-rank statistical significance testing
- [x] Overfitting diagnosis (Train / Val / Test R²)
- [x] Learning curves for all 4 models
- [x] Residual plots + Q-Q plots
- [x] Feature importance (Ridge coef, RF Gini, XGB Gain, ANN permutation)
- [x] Predicted vs Actual scatter plots
- [x] Engineering interpretation (RMSE as % of operating range)
- [x] IEEE-format paper (4 pages, LaTeX)
- [x] Models saved (pkl + .keras)
