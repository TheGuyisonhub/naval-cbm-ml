#!/usr/bin/env bash
# =============================================================================
# setup_repo.sh — One-shot GitHub repo initialisation for naval-cbm-ml
# Run this from inside the naval-cbm-ml/ directory
# =============================================================================
set -e

REPO_NAME="naval-cbm-ml"
GITHUB_USER="TheGuyisonhub"   # your GitHub handle

echo "=== Setting up $REPO_NAME ==="

# 1. Copy main pipeline
echo "[1] Copying main.py ..."
cp ../prooject/main.py .                   # adjust path if needed

# 2. Copy outputs (plots + CSVs only — skip large .pkl / .keras)
echo "[2] Copying output plots & CSVs ..."
mkdir -p outputs
cp ../prooject/outputs/*.png outputs/
cp ../prooject/outputs/*.csv outputs/

# 3. Copy report PDF
echo "[3] Copying report ..."
mkdir -p report
cp "../prooject/Report/Predictive Modeling for Naval Propulsion Plant Condition Based Maintenance Using Machine Learning.pdf" \
   report/report.pdf

# 4. Placeholder data dir (actual data.txt is gitignored)
mkdir -p data
echo "# Download data.txt from UCI and place it here" > data/README.md

# 5. Assets — copy a few key plots for README display
mkdir -p assets
cp outputs/model_comparison_chart.png assets/
cp outputs/feature_importance.png assets/
cp outputs/predicted_vs_actual.png assets/
cp outputs/learning_curves.png assets/

# 6. Git init and first commit
echo "[4] Initialising git ..."
git init
git add .
git commit -m "Initial commit: complete CBM ML pipeline for naval propulsion plant

- 4 models: Ridge, Random Forest, XGBoost, ANN (Keras)
- XGBoost best: R²=0.9928, RMSE=8.60e-4
- Full EDA, feature engineering, hyperparameter tuning
- Statistical significance testing (Wilcoxon)
- IEEE-format report included"

# 7. Push to GitHub (requires gh CLI or pre-configured remote)
echo "[5] Creating GitHub repo and pushing ..."
gh repo create "$REPO_NAME" \
   --public \
   --description "ML pipeline for condition-based maintenance of naval gas turbine propulsion plants. XGBoost R²=0.9928." \
   --push \
   --source=.

echo ""
echo "=== Done! ==="
echo "Repo live at: https://github.com/$GITHUB_USER/$REPO_NAME"
