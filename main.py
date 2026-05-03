# =============================================================================
# CS-333: Applied AI & Machine Learning — Lab Project
# NUST College of E&ME, Rawalpindi
# Predictive Modeling for Naval Propulsion Plant Condition-Based Maintenance
# Dataset  : UCI CBM Dataset (Naval Propulsion Plants)
# Task     : Multi-Output Regression
# Targets  : GT Compressor Decay Coefficient & GT Turbine Decay Coefficient
# Models   : Ridge Regression | Random Forest | XGBoost | ANN (Keras)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — IMPORTS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import joblib

matplotlib.use("Agg")  # Non-interactive backend — all plots saved to disk
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import wilcoxon

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

warnings.filterwarnings("ignore")

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "data.txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Robust dataset check ──────────────────────────────────────────────────────
if not os.path.exists(DATA_PATH):
    print(f"❌ Dataset not found at expected path:\n   {DATA_PATH}")

    alt_path = os.path.join(BASE_DIR, "data.txt")
    if os.path.exists(alt_path):
        print("✔ Found dataset in project root. Using fallback path.")
        DATA_PATH = alt_path
    else:
        raise FileNotFoundError(
            "\nDataset not found!\n"
            "👉 Place 'data.txt' inside a folder named 'data/'\n"
            "   Example: prooject/data/data.txt\n"
        )

# ── Global figure store (all figures collected here, saved together at end) ───
FIGURES = {}  # { filename : matplotlib Figure }

# ── Column names (from UCI CBM Features.txt) ──────────────────────────────────
FEATURE_NAMES = [
    "Lever_Position",           # lp   [ ]
    "Ship_Speed",               # v    [knots]
    "GT_Shaft_Torque",          # GTT  [kN·m]
    "GT_Rate_Revolution",       # GTn  [rpm]
    "GG_Rate_Revolution",       # GGn  [rpm]
    "Starboard_Prop_Torque",    # Ts   [kN]
    "Port_Prop_Torque",         # Tp   [kN]
    "HP_Turbine_Exit_Temp",     # T48  [°C]
    "Compressor_Inlet_Temp",    # T1   [°C]
    "Compressor_Out_Temp",      # T2   [°C]
    "HP_Turbine_Exit_Press",    # P48  [bar]
    "Comp_Inlet_Pressure",      # P1   [bar]
    "Comp_Outlet_Pressure",     # P2   [bar]
    "GT_Exhaust_Pressure",      # Pexh [bar]
    "Turbine_Inject_Control",   # TIC  [%]
    "Fuel_Flow",                # mf   [kg/s]
]
TARGET_NAMES = [
    "GT_Compressor_Decay",      # target 1
    "GT_Turbine_Decay",         # target 2
]
ALL_COLS = FEATURE_NAMES + TARGET_NAMES

print("=" * 70)
print("  Naval Propulsion Plant — Condition-Based Maintenance ML Pipeline")
print("=" * 70)

# =============================================================================
# SECTION 1 — LOAD DATA
# =============================================================================
print("\n[1] Loading dataset ...")

df = pd.read_csv(
    DATA_PATH,
    sep=r"\s+",
    header=None,
    names=ALL_COLS,
    encoding="latin-1",
)
print(f"    Loaded : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"    Features: {len(FEATURE_NAMES)}   Targets: {len(TARGET_NAMES)}")

# ── Categorical feature: Operating Mode (derived from Lever_Position) ─────────
df['Operating_Mode'] = pd.cut(df['Lever_Position'], bins=3, labels=['Low', 'Medium', 'High'])
df['Operating_Mode'] = df['Operating_Mode'].cat.codes  # encode to 0/1/2
FEATURE_NAMES.append('Operating_Mode')
print(f"    Categorical feature added: Operating_Mode (0=Low, 1=Medium, 2=High)")

# =============================================================================
# SECTION 2 — EXPLORATORY DATA ANALYSIS  (compute + stage; plot at end)
# =============================================================================
print("\n[2] Exploratory Data Analysis ...")

# ── 2.1 Descriptive statistics (mean, std, skewness, kurtosis) ───────────────
desc = df.describe().T
desc["skewness"] = df.skew(numeric_only=True)
desc["kurtosis"] = df.kurtosis(numeric_only=True)
print("\n    ── Descriptive Statistics ──")
print(desc[["mean", "std", "min", "max", "skewness", "kurtosis"]].round(4).to_string())
desc.to_csv(os.path.join(OUTPUT_DIR, "descriptive_stats.csv"))

# ── 2.2 Missing value analysis ───────────────────────────────────────────────
print("\n    ── Missing Value Analysis ──")
missing = df.isnull().sum()
print(f"    Total missing values: {missing.sum()}")
print("    Strategy: UCI CBM is complete — no imputation required")

# ── 2.3 Stage: Feature histograms ────────────────────────────────────────────
fig_hist, axes = plt.subplots(4, 4, figsize=(20, 14))
axes = axes.flatten()
for i, col in enumerate(FEATURE_NAMES[:16]):
    axes[i].hist(df[col], bins=40, edgecolor="white", alpha=0.85)
    axes[i].set_title(col, fontsize=8, fontweight="bold")
    axes[i].set_xlabel("Value", fontsize=7)
    axes[i].set_ylabel("Count", fontsize=7)
    axes[i].tick_params(labelsize=7)
fig_hist.suptitle("Feature Distributions — Naval Propulsion Dataset",
                  fontsize=13, fontweight="bold")
plt.tight_layout()
FIGURES["eda_histograms.png"] = fig_hist

# ── 2.4 Stage: Box plots (outlier overview) ───────────────────────────────────
fig_box, axes = plt.subplots(4, 4, figsize=(20, 14))
axes = axes.flatten()
for i, col in enumerate(FEATURE_NAMES[:16]):
    axes[i].boxplot(
        df[col],
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor="lightcoral", color="darkred"),
        medianprops=dict(color="black", linewidth=1.5),
    )
    axes[i].set_title(col, fontsize=8, fontweight="bold")
    axes[i].tick_params(labelsize=7)
fig_box.suptitle("Box Plots — Outlier Overview (Pre-Preprocessing)",
                 fontsize=13, fontweight="bold")
plt.tight_layout()
FIGURES["eda_boxplots.png"] = fig_box

# ── 2.5 Stage: Correlation heatmap ───────────────────────────────────────────
corr = df.corr(numeric_only=True)
fig_heat, ax = plt.subplots(figsize=(18, 14))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    square=True,
    ax=ax,
    annot_kws={"size": 6},
    linewidths=0.3,
    cbar_kws={"shrink": 0.6},
)
ax.set_title("Pearson Correlation Heatmap — All Features + Targets",
             fontsize=13, fontweight="bold")
plt.tight_layout()
FIGURES["eda_heatmap.png"] = fig_heat

# ── 2.6 Stage: Pair plot (top 5 correlated features with targets) ─────────────
top5 = (
    corr[TARGET_NAMES]
    .abs()
    .mean(axis=1)
    .drop(TARGET_NAMES)
    .nlargest(5)
    .index.tolist()
)
pair_cols = top5 + TARGET_NAMES

# Sampling keeps the pairplot usable without changing the required functionality
pair_df = df[pair_cols].sample(n=min(2500, len(df)), random_state=RANDOM_SEED)
fig_pair = sns.pairplot(
    pair_df,
    diag_kind="kde",
    plot_kws={"alpha": 0.25, "s": 7},
)
fig_pair.fig.suptitle("Pair Plot — Top 5 Correlated Features vs Targets",
                      y=1.02, fontsize=12, fontweight="bold")
FIGURES["eda_pairplot.png"] = fig_pair.fig

# ── 2.7 Stage: Target distributions ──────────────────────────────────────────
fig_tgt, axes = plt.subplots(1, 2, figsize=(12, 4))
for i, t in enumerate(TARGET_NAMES):
    axes[i].hist(df[t], bins=40, edgecolor="white", alpha=0.85)
    axes[i].axvline(df[t].mean(), linestyle="--",
                    label=f"Mean = {df[t].mean():.4f}")
    axes[i].set_title(f"Target Distribution: {t}", fontsize=10, fontweight="bold")
    axes[i].set_xlabel("Decay Coefficient")
    axes[i].set_ylabel("Count")
    axes[i].legend(fontsize=8)
fig_tgt.suptitle("Target Variable Distributions", fontsize=12, fontweight="bold")
plt.tight_layout()
FIGURES["eda_targets.png"] = fig_tgt

print("    EDA figures staged (will be saved at end).")

# =============================================================================
# SECTION 3 — PREPROCESSING
# =============================================================================
print("\n[3] Preprocessing ...")
df_clean = df.copy()

# ── 3.1 Outlier detection — IQR method ───────────────────────────────────────
print("    Outlier detection (IQR 1.5× rule) ...")
outlier_counts = {}
for col in FEATURE_NAMES:
    Q1, Q3 = df_clean[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    n_out = ((df_clean[col] < Q1 - 1.5 * IQR) | (df_clean[col] > Q3 + 1.5 * IQR)).sum()
    outlier_counts[col] = int(n_out)

total_outliers = sum(outlier_counts.values())
print(f"    IQR flagged entries (across all features): {total_outliers}")
for k, v in outlier_counts.items():
    if v > 0:
        print(f"      {k}: {v}")

print("    Decision: RETAIN all rows — outliers are valid physical operating states")

# ── 3.2 Z-score cross-check (informational) ───────────────────────────────────
z = np.abs(stats.zscore(df_clean[FEATURE_NAMES], nan_policy="omit"))
extreme_z = int((z > 3).sum())
print(f"    Z-score |z|>3 entries: {extreme_z} — retained (same justification)")

# ── 3.3 Compute pressure ratio before dropping zero-variance columns ──────────
# Comp_Inlet_Pressure is constant (0.998) — use its scalar value directly
COMP_INLET_PRESSURE_CONST = df_clean["Comp_Inlet_Pressure"].iloc[0]

# ── 3.4 Zero-variance feature removal ────────────────────────────────────────
zero_var = [c for c in FEATURE_NAMES if df_clean[c].std() == 0]
print(f"    Zero-variance features dropped: {zero_var}")
df_clean.drop(columns=zero_var, inplace=True)
FEATURE_NAMES[:] = [f for f in FEATURE_NAMES if f not in zero_var]

# ── 3.5 Separate features and targets ────────────────────────────────────────
X = df_clean[FEATURE_NAMES].copy()
y = df_clean[TARGET_NAMES].copy()
# =============================================================================
# SECTION 4 — FEATURE ENGINEERING
# =============================================================================
print("\n[4] Feature Engineering ...")

# Derived Feature 1 — Compressor Pressure Ratio (P2 / P1)
X["Compressor_Pressure_Ratio"] = X["Comp_Outlet_Pressure"] / COMP_INLET_PRESSURE_CONST

# Derived Feature 2 — Turbine Power Proxy (GTT × GTn)
X["Turbine_Power_Proxy"] = X["GT_Shaft_Torque"] * X["GT_Rate_Revolution"]

FEATURE_NAMES_ENG = list(X.columns)
print(f"    Original features : {len(FEATURE_NAMES)}")
print(f"    Derived features  : 2  (Compressor_Pressure_Ratio, Turbine_Power_Proxy)")
print(f"    Total features    : {X.shape[1]}")

# =============================================================================
# SECTION 5 — TRAIN / VALIDATION / TEST SPLIT  (70 / 15 / 15)
#   Identical split reused across ALL models (no data leakage)
# =============================================================================
print("\n[5] Train / Val / Test split  70 / 15 / 15 ...")

X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_SEED
)

X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv,
    test_size=(0.15 / 0.85),   # keeps val at exactly 15% of full set
    random_state=RANDOM_SEED
)

n = len(X)
print(f"    Train      : {len(X_train):,}  ({len(X_train) / n * 100:.1f}%)")
print(f"    Validation : {len(X_val):,}   ({len(X_val) / n * 100:.1f}%)")
print(f"    Test       : {len(X_test):,}   ({len(X_test) / n * 100:.1f}%)")

# ── StandardScaler — fit ONLY on training set ─────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc = scaler.transform(X_val)
X_test_sc = scaler.transform(X_test)

y_train_np = y_train.values
y_val_np = y_val.values
y_test_np = y_test.values

print("    StandardScaler fitted on train only → applied to val & test.")

# =============================================================================
# SECTION 6 — HELPER FUNCTIONS
# =============================================================================
def adjusted_r2(r2_val, n_samples, n_features):
    """Adjusted R²: penalises adding non-informative predictors."""
    return 1 - (1 - r2_val) * (n_samples - 1) / (n_samples - n_features - 1)


def compute_metrics(y_true, y_pred, n_features, label=""):
    """
    Compute RMSE, MAE, R², Adjusted R² for multi-output regression.
    Metrics are averaged across both targets; per-target values also stored.
    """
    rmse_l, mae_l, r2_l, ar2_l = [], [], [], []
    for i in range(y_true.shape[1]):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        ar2 = adjusted_r2(r2, len(y_true), n_features)

        rmse_l.append(rmse)
        mae_l.append(mae)
        r2_l.append(r2)
        ar2_l.append(ar2)

    m = {
        "RMSE": float(np.mean(rmse_l)),
        "MAE": float(np.mean(mae_l)),
        "R2": float(np.mean(r2_l)),
        "Adj_R2": float(np.mean(ar2_l)),
        "RMSE_comp": rmse_l[0],
        "RMSE_turb": rmse_l[1],
        "R2_comp": r2_l[0],
        "R2_turb": r2_l[1],
    }
    if label:
        print(f"    {label:<22} RMSE={m['RMSE']:.6f}  MAE={m['MAE']:.6f}"
              f"  R²={m['R2']:.6f}  AdjR²={m['Adj_R2']:.6f}")
    return m


def overfitting_diagnosis(train_r2, val_r2, test_r2, model_name):
    """Print a one-line overfitting / underfitting diagnosis."""
    gap = train_r2 - test_r2
    if test_r2 < 0.70:
        status = "UNDERFITTING  ← model too simple / insufficient features"
    elif gap > 0.10:
        status = "OVERFITTING   ← regularisation or more data recommended"
    elif gap > 0.05:
        status = "Mild overfitting — acceptable"
    else:
        status = "Well-fitted   ✓"
    print(f"    {model_name:<22} Train={train_r2:.4f}  Val={val_r2:.4f}"
          f"  Test={test_r2:.4f}  → {status}")


# Storage dicts
RESULTS = {}     # model_name → metrics (test set)
AUX_RESULTS = {}  # model_name + ' (Train)' / '(Val)' → metrics
PREDICTIONS = {}  # model_name → y_pred array (test set)
MODELS = {}      # model_name → fitted model object
MODEL_NAMES = ["Ridge Regression", "Random Forest", "XGBoost", "ANN"]

# Shorthand key mapping for train/val lookups
KEY_MAP = {
    "Ridge Regression": ("Ridge (Train)", "Ridge (Val)"),
    "Random Forest": ("RF (Train)", "RF (Val)"),
    "XGBoost": ("XGB (Train)", "XGB (Val)"),
    "ANN": ("ANN (Train)", "ANN (Val)"),
}

N_FEAT = X_train_sc.shape[1]

# =============================================================================
# SECTION 7 — MODEL 1: RIDGE REGRESSION  (Baseline — linear family)
# =============================================================================
print("\n[6.1] Model 1: Ridge Regression (Baseline) ...")

alphas_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0]
ridge = MultiOutputRegressor(RidgeCV(alphas=alphas_grid, cv=5))
ridge.fit(X_train_sc, y_train_np)

y_pred_ridge_tr = ridge.predict(X_train_sc)
y_pred_ridge_val = ridge.predict(X_val_sc)
y_pred_ridge_test = ridge.predict(X_test_sc)

RESULTS["Ridge Regression"] = compute_metrics(
    y_test_np, y_pred_ridge_test, N_FEAT, "Ridge [TEST]"
)
AUX_RESULTS["Ridge (Train)"] = compute_metrics(y_train_np, y_pred_ridge_tr, N_FEAT)
AUX_RESULTS["Ridge (Val)"] = compute_metrics(y_val_np, y_pred_ridge_val, N_FEAT)
PREDICTIONS["Ridge Regression"] = y_pred_ridge_test
MODELS["Ridge Regression"] = ridge

best_alphas = [est.alpha_ for est in ridge.estimators_]
print(f"    CV-selected alpha → Compressor: {best_alphas[0]:.3f} | "
      f"Turbine: {best_alphas[1]:.3f}")
overfitting_diagnosis(
    AUX_RESULTS["Ridge (Train)"]["R2"],
    AUX_RESULTS["Ridge (Val)"]["R2"],
    RESULTS["Ridge Regression"]["R2"],
    "Ridge Regression",
)

# =============================================================================
# SECTION 8 — MODEL 2: RANDOM FOREST  (Ensemble / Bagging family)
# =============================================================================
print("\n[6.2] Model 2: Random Forest Regressor ...")

rf_param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "max_features": ["sqrt", 0.5],
}

rf_cv = GridSearchCV(
    RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
    rf_param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=0,
)
rf_cv.fit(X_train_sc, y_train_np)
rf_best = rf_cv.best_estimator_
print(f"    Best params: {rf_cv.best_params_}")

y_pred_rf_tr = rf_best.predict(X_train_sc)
y_pred_rf_val = rf_best.predict(X_val_sc)
y_pred_rf_test = rf_best.predict(X_test_sc)

RESULTS["Random Forest"] = compute_metrics(
    y_test_np, y_pred_rf_test, N_FEAT, "RF    [TEST]"
)
AUX_RESULTS["RF (Train)"] = compute_metrics(y_train_np, y_pred_rf_tr, N_FEAT)
AUX_RESULTS["RF (Val)"] = compute_metrics(y_val_np, y_pred_rf_val, N_FEAT)
PREDICTIONS["Random Forest"] = y_pred_rf_test
MODELS["Random Forest"] = rf_best

overfitting_diagnosis(
    AUX_RESULTS["RF (Train)"]["R2"],
    AUX_RESULTS["RF (Val)"]["R2"],
    RESULTS["Random Forest"]["R2"],
    "Random Forest",
)

# =============================================================================
# SECTION 9 — MODEL 3: XGBOOST  (Ensemble / Gradient Boosting family)
# =============================================================================
print("\n[6.3] Model 3: XGBoost Regressor ...")

xgb_param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_alpha": [0, 0.1],
    "reg_lambda": [1.0, 5.0],
}

xgb_base = xgb.XGBRegressor(random_state=RANDOM_SEED, verbosity=0, n_jobs=-1)
xgb_multi = MultiOutputRegressor(xgb_base)
xgb_grid_multi = {f"estimator__{k}": v for k, v in xgb_param_grid.items()}

xgb_cv = GridSearchCV(
    xgb_multi,
    xgb_grid_multi,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=0,
)
xgb_cv.fit(X_train_sc, y_train_np)
xgb_best = xgb_cv.best_estimator_
print(f"    Best params: {xgb_cv.best_params_}")

y_pred_xgb_tr = xgb_best.predict(X_train_sc)
y_pred_xgb_val = xgb_best.predict(X_val_sc)
y_pred_xgb_test = xgb_best.predict(X_test_sc)

RESULTS["XGBoost"] = compute_metrics(
    y_test_np, y_pred_xgb_test, N_FEAT, "XGB   [TEST]"
)
AUX_RESULTS["XGB (Train)"] = compute_metrics(y_train_np, y_pred_xgb_tr, N_FEAT)
AUX_RESULTS["XGB (Val)"] = compute_metrics(y_val_np, y_pred_xgb_val, N_FEAT)
PREDICTIONS["XGBoost"] = y_pred_xgb_test
MODELS["XGBoost"] = xgb_best

overfitting_diagnosis(
    AUX_RESULTS["XGB (Train)"]["R2"],
    AUX_RESULTS["XGB (Val)"]["R2"],
    RESULTS["XGBoost"]["R2"],
    "XGBoost",
)

# =============================================================================
# SECTION 10 — MODEL 4: ARTIFICIAL NEURAL NETWORK  (Neural family)
# =============================================================================
print("\n[6.4] Model 4: Artificial Neural Network (Keras) ...")


def build_ann(input_dim, units=256, dropout=0.2, lr=1e-3):
    """
    Fully-connected regression ANN for multi-output prediction.
    Dropout layers act as regularisation against overfitting.
    BatchNormalization stabilises training and speeds convergence.
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(units, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            layers.Dense(units // 2, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            layers.Dense(units // 4, activation="relu"),
            layers.Dense(2),
        ],
        name="NavalPropulsion_ANN",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )
    return model


ann = build_ann(input_dim=N_FEAT)

early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=40,
    restore_best_weights=True,
    verbose=0,
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=12,
    min_lr=1e-6,
    verbose=0,
)

history = ann.fit(
    X_train_sc,
    y_train_np,
    validation_data=(X_val_sc, y_val_np),
    epochs=500,
    batch_size=64,
    callbacks=[early_stop, reduce_lr],
    verbose=0,
)

stopped_epoch = len(history.history["loss"])
print(f"    EarlyStopping triggered at epoch {stopped_epoch}/500")

# Cache ANN predictions once for reuse in metrics + feature importance
ann_train_pred = ann.predict(X_train_sc, verbose=0)
ann_val_pred = ann.predict(X_val_sc, verbose=0)
ann_test_pred = ann.predict(X_test_sc, verbose=0)

RESULTS["ANN"] = compute_metrics(y_test_np, ann_test_pred, N_FEAT, "ANN   [TEST]")
AUX_RESULTS["ANN (Train)"] = compute_metrics(y_train_np, ann_train_pred, N_FEAT)
AUX_RESULTS["ANN (Val)"] = compute_metrics(y_val_np, ann_val_pred, N_FEAT)
PREDICTIONS["ANN"] = ann_test_pred
MODELS["ANN"] = ann

overfitting_diagnosis(
    AUX_RESULTS["ANN (Train)"]["R2"],
    AUX_RESULTS["ANN (Val)"]["R2"],
    RESULTS["ANN"]["R2"],
    "ANN",
)

# ── Stage: ANN training history plot ─────────────────────────────────────────
fig_ann, axes = plt.subplots(1, 2, figsize=(13, 4))
epochs_range = range(1, stopped_epoch + 1)

axes[0].plot(epochs_range, history.history["loss"], label="Train Loss (MSE)")
axes[0].plot(epochs_range, history.history["val_loss"], label="Val Loss (MSE)")
axes[0].set_title("ANN — MSE Loss vs Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, history.history["mae"], label="Train MAE")
axes[1].plot(epochs_range, history.history["val_mae"], label="Val MAE")
axes[1].set_title("ANN — MAE vs Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MAE")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig_ann.suptitle("ANN Training History  (EarlyStopping + ReduceLROnPlateau)",
                 fontsize=12, fontweight="bold")
plt.tight_layout()
FIGURES["ann_training_history.png"] = fig_ann

# =============================================================================
# SECTION 11 — EVALUATION & MODEL COMPARISON
# =============================================================================
print("\n[7] Evaluation & Comparison ...")

# ── 11.1 Comparison table ─────────────────────────────────────────────────────
rows = []
for m in MODEL_NAMES:
    r = RESULTS[m]
    rows.append(
        {
            "Model": m,
            "RMSE": round(r["RMSE"], 6),
            "MAE": round(r["MAE"], 6),
            "R²": round(r["R2"], 6),
            "Adj R²": round(r["Adj_R2"], 6),
            "RMSE Comp.": round(r["RMSE_comp"], 6),
            "RMSE Turb.": round(r["RMSE_turb"], 6),
        }
    )

comp_df = pd.DataFrame(rows).set_index("Model")
print("\n    ── Model Comparison Table (Test Set) ──")
print(comp_df.to_string())
comp_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"))

best_name = comp_df["R²"].idxmax()
print(f"\n    Best model by R²: {best_name}")

# ── 11.2 Wilcoxon signed-rank significance test ───────────────────────────────
print("\n    ── Wilcoxon Signed-Rank Test (Best vs Others) ──")
best_err = np.mean(np.abs(y_test_np - PREDICTIONS[best_name]), axis=1)

for m in MODEL_NAMES:
    if m == best_name:
        continue
    other_err = np.mean(np.abs(y_test_np - PREDICTIONS[m]), axis=1)

    stat, p_val = wilcoxon(best_err, other_err)

    sig = "✓ Significant (p<0.05)" if p_val < 0.05 else "✗ Not significant"
    print(f"    {best_name} vs {m:<22} stat={stat:.1f}  p={p_val:.4e}  → {sig}")

# ── 11.3 Overfitting / underfitting summary ───────────────────────────────────
print("\n    ── Overfitting / Underfitting Diagnosis (Train / Val / Test R²) ──")
for m in MODEL_NAMES:
    tk, vk = KEY_MAP[m]
    overfitting_diagnosis(
        AUX_RESULTS[tk]["R2"],
        AUX_RESULTS[vk]["R2"],
        RESULTS[m]["R2"],
        m,
    )

# =============================================================================
# SECTION 12 — STAGE ALL REMAINING PLOTS
# =============================================================================
print("\n[8] Staging remaining plots ...")

# ── 12.1 Residual plots (fitted vs residuals) ─────────────────────────────────
fig_res, axes = plt.subplots(4, 2, figsize=(14, 20))
fig_res.suptitle("Residual Plots — All Models (Both Targets)",
                 fontsize=13, fontweight="bold")

for ri, m in enumerate(MODEL_NAMES):
    yp = PREDICTIONS[m]
    for ci, ti in enumerate([0, 1]):
        ax = axes[ri][ci]
        res = y_test_np[:, ti] - yp[:, ti]
        ax.scatter(yp[:, ti], res, alpha=0.25, s=6)
        ax.axhline(0, linestyle="--", linewidth=1.2)

        sorted_idx = np.argsort(yp[:, ti])
        window = max(1, len(sorted_idx) // 40)
        smooth_y = np.convolve(res[sorted_idx], np.ones(window) / window, mode="same")
        ax.plot(yp[:, ti][sorted_idx], smooth_y, linewidth=1.5, alpha=0.8, label="Trend")

        rmse_key = "RMSE_comp" if ti == 0 else "RMSE_turb"
        ax.text(
            0.04,
            0.93,
            f"RMSE={RESULTS[m][rmse_key]:.5f}",
            transform=ax.transAxes,
            fontsize=8,
            color="darkred",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )
        ax.set_xlabel("Fitted Values", fontsize=8)
        ax.set_ylabel("Residuals", fontsize=8)
        ax.set_title(f"{m} — {TARGET_NAMES[ti]}", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)

plt.tight_layout()
FIGURES["residual_plots.png"] = fig_res

# ── 12.2 Q-Q plots (residual normality) ──────────────────────────────────────
fig_qq, axes = plt.subplots(4, 2, figsize=(12, 18))
fig_qq.suptitle("Q-Q Plots — Residual Normality Check",
                fontsize=13, fontweight="bold")

for ri, m in enumerate(MODEL_NAMES):
    yp = PREDICTIONS[m]
    for ci, ti in enumerate([0, 1]):
        ax = axes[ri][ci]
        res = y_test_np[:, ti] - yp[:, ti]
        stats.probplot(res, dist="norm", plot=ax)
        ax.set_title(f"{m} — {TARGET_NAMES[ti]}", fontsize=9, fontweight="bold")

plt.tight_layout()
FIGURES["residual_qq_plots.png"] = fig_qq

# ── 12.3 Learning curves (train size vs score) ────────────────────────────────
print("    Computing learning curves — sklearn models (may take ~1-2 min) ...")
fig_lc, axes = plt.subplots(2, 2, figsize=(16, 12))
fig_lc.suptitle(
    "Learning Curves — Underfitting / Overfitting Diagnosis (Train size vs R²)",
    fontsize=13,
    fontweight="bold",
)

sklearn_lc_models = {
    "Ridge Regression": (ridge, axes[0][0]),
    "Random Forest": (rf_best, axes[0][1]),
    "XGBoost": (xgb_best, axes[1][0]),
}
train_sizes = np.linspace(0.10, 1.0, 8)

for m_name, (model, ax) in sklearn_lc_models.items():
    tr_sz, tr_sc, val_sc = learning_curve(
        model,
        X_train_sc,
        y_train_np,
        train_sizes=train_sizes,
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )

    tr_mean, val_mean = tr_sc.mean(axis=1), val_sc.mean(axis=1)
    tr_std, val_std = tr_sc.std(axis=1), val_sc.std(axis=1)

    ax.plot(tr_sz, tr_mean, "o-", label="Train R²")
    ax.plot(tr_sz, val_mean, "s-", label="CV Val R²")
    ax.fill_between(tr_sz, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15)
    ax.fill_between(tr_sz, val_mean - val_std, val_mean + val_std, alpha=0.15)
    ax.set_title(f"{m_name}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("R²")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])

    gap = tr_mean[-1] - val_mean[-1]
    if val_mean[-1] < 0.70:
        diag = "UNDERFITTING"
    elif gap > 0.10:
        diag = "OVERFITTING"
    else:
        diag = "Well-fitted"

    ax.text(
        0.04,
        0.08,
        diag,
        transform=ax.transAxes,
        fontsize=9,
        color="darkred",
        fontweight="bold",
        bbox=dict(facecolor="lightyellow", edgecolor="gray", alpha=0.8),
    )

ax_ann = axes[1][1]
ep_rng = range(1, stopped_epoch + 1)

ax_ann.plot(ep_rng, history.history["loss"], label="Train Loss (MSE)")
ax_ann.plot(ep_rng, history.history["val_loss"], label="Val Loss (MSE)")

ax_ann.set_title("ANN — Learning Curve (Loss vs Epoch)", fontsize=10, fontweight="bold")
ax_ann.set_xlabel("Epoch")
ax_ann.set_ylabel("MSE Loss")
ax_ann.legend()
ax_ann.grid(True, alpha=0.3)

ax_ann.text(
    0.04, 0.08,
    f"Stopped at epoch {stopped_epoch}",
    transform=ax_ann.transAxes,
    fontsize=8,
    color="darkred",
    bbox=dict(facecolor="lightyellow", edgecolor="gray", alpha=0.8),
)

plt.tight_layout()
FIGURES["learning_curves.png"] = fig_lc

# ── 12.4 Feature importance ───────────────────────────────────────────────────
print("    Computing feature importance ...")
fig_fi, axes = plt.subplots(2, 2, figsize=(18, 14))
fig_fi.suptitle("Feature Importance — All Models", fontsize=13, fontweight="bold")
feat_arr = np.array(FEATURE_NAMES_ENG)

# Ridge — mean absolute coefficient across both targets
ridge_coefs = np.mean([np.abs(est.coef_) for est in ridge.estimators_], axis=0)
idx = np.argsort(ridge_coefs)
axes[0][0].barh(feat_arr[idx], ridge_coefs[idx])
axes[0][0].set_title("Ridge Regression — |Coefficient|",
                     fontsize=10, fontweight="bold")
axes[0][0].set_xlabel("Mean |Coefficient|")
axes[0][0].tick_params(labelsize=7)

# Random Forest — built-in impurity-based importance
rf_imp = rf_best.feature_importances_
idx = np.argsort(rf_imp)
axes[0][1].barh(feat_arr[idx], rf_imp[idx])
axes[0][1].set_title("Random Forest — Feature Importance",
                     fontsize=10, fontweight="bold")
axes[0][1].set_xlabel("Importance (Gini impurity)")
axes[0][1].tick_params(labelsize=7)

# XGBoost — mean gain across both target estimators
xgb_imp = np.mean([est.feature_importances_ for est in xgb_best.estimators_], axis=0)
idx = np.argsort(xgb_imp)
axes[1][0].barh(feat_arr[idx], xgb_imp[idx])
axes[1][0].set_title("XGBoost — Feature Importance (Gain)",
                     fontsize=10, fontweight="bold")
axes[1][0].set_xlabel("Importance (Mean Gain)")
axes[1][0].tick_params(labelsize=7)

# ANN — permutation importance (manual implementation)
rng = np.random.default_rng(RANDOM_SEED)
baseline_r2 = np.mean(
    [
        r2_score(y_test_np[:, i], ann_test_pred[:, i])
        for i in range(2)
    ]
)

ann_imp = np.zeros(X_test_sc.shape[1], dtype=float)
n_repeats = 10

for feat_idx in range(X_test_sc.shape[1]):
    scores = []
    for _ in range(n_repeats):
        X_perm = X_test_sc.copy()
        rng.shuffle(X_perm[:, feat_idx])
        perm_preds = ann.predict(X_perm, verbose=0)
        perm_r2 = np.mean([r2_score(y_test_np[:, i], perm_preds[:, i]) for i in range(2)])
        scores.append(baseline_r2 - perm_r2)
    ann_imp[feat_idx] = float(np.mean(scores))

idx = np.argsort(ann_imp)
axes[1][1].barh(feat_arr[idx], ann_imp[idx])
axes[1][1].set_title("ANN — Permutation Importance (R² decrease)",
                     fontsize=10, fontweight="bold")
axes[1][1].set_xlabel("Mean R² Decrease")
axes[1][1].tick_params(labelsize=7)

plt.tight_layout()
FIGURES["feature_importance.png"] = fig_fi

# ── 12.5 Predicted vs Actual scatter ─────────────────────────────────────────
fig_pva, axes = plt.subplots(4, 2, figsize=(14, 20))
fig_pva.suptitle("Predicted vs Actual — All Models",
                 fontsize=13, fontweight="bold")

for ri, m in enumerate(MODEL_NAMES):
    yp = PREDICTIONS[m]
    for ci, ti in enumerate([0, 1]):
        ax = axes[ri][ci]
        actual = y_test_np[:, ti]
        pred = yp[:, ti]

        ax.scatter(actual, pred, alpha=0.25, s=6)
        lim = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
        ax.plot(lim, lim, linestyle="--", linewidth=1.5, label="Ideal (y=x)")
        r2k = "R2_comp" if ti == 0 else "R2_turb"
        ax.text(
            0.04,
            0.92,
            f"R²={RESULTS[m][r2k]:.5f}",
            transform=ax.transAxes,
            fontsize=8,
            color="darkred",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
        ax.set_xlabel("Actual", fontsize=8)
        ax.set_ylabel("Predicted", fontsize=8)
        ax.set_title(f"{m} — {TARGET_NAMES[ti]}", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)

plt.tight_layout()
FIGURES["predicted_vs_actual.png"] = fig_pva

# ── 12.6 Model performance bar chart ─────────────────────────────────────────
fig_bar, axes = plt.subplots(1, 3, figsize=(15, 5))
fig_bar.suptitle("Model Performance Summary (Test Set)",
                 fontsize=13, fontweight="bold")
colors = ["steelblue", "darkorange", "forestgreen", "mediumpurple"]

axes[0].bar(MODEL_NAMES, [RESULTS[m]["RMSE"] for m in MODEL_NAMES], color=colors)
axes[0].set_title("RMSE  (lower = better)")
axes[0].set_ylabel("RMSE")
axes[0].tick_params(axis="x", rotation=15)

axes[1].bar(MODEL_NAMES, [RESULTS[m]["MAE"] for m in MODEL_NAMES], color=colors)
axes[1].set_title("MAE  (lower = better)")
axes[1].set_ylabel("MAE")
axes[1].tick_params(axis="x", rotation=15)

axes[2].bar(MODEL_NAMES, [RESULTS[m]["R2"] for m in MODEL_NAMES], color=colors)
axes[2].set_title("R²  (higher = better)")
axes[2].set_ylabel("R²")
axes[2].set_ylim([0, 1.05])
axes[2].tick_params(axis="x", rotation=15)

plt.tight_layout()
FIGURES["model_comparison_chart.png"] = fig_bar

# ── 12.7 Train / Val / Test R² — overfitting summary bar chart ───────────────
fig_ovf, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(MODEL_NAMES))
w = 0.25
tr_r2s = [AUX_RESULTS[KEY_MAP[m][0]]["R2"] for m in MODEL_NAMES]
val_r2s = [AUX_RESULTS[KEY_MAP[m][1]]["R2"] for m in MODEL_NAMES]
te_r2s = [RESULTS[m]["R2"] for m in MODEL_NAMES]

ax.bar(x - w, tr_r2s, w, label="Train R²")
ax.bar(x, val_r2s, w, label="Validation R²")
ax.bar(x + w, te_r2s, w, label="Test R²")
ax.set_xticks(x)
ax.set_xticklabels(MODEL_NAMES, rotation=12)
ax.set_ylabel("R² Score")
ax.set_ylim([0, 1.12])
ax.set_title("Train / Validation / Test R² — Overfitting Diagnosis",
             fontsize=12, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
FIGURES["overfitting_diagnosis.png"] = fig_ovf

# =============================================================================
# SECTION 13 — SAVE ALL FIGURES  (single batch at the end)
# =============================================================================
print("\n[9] Saving all plots ...")
for fname, fig in FIGURES.items():
    fpath = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {fpath}")

# =============================================================================
# SECTION 14 — SAVE TRAINED MODELS
# =============================================================================
print("\n[10] Saving trained models ...")

# Save sklearn models
joblib.dump(MODELS["Ridge Regression"], os.path.join(OUTPUT_DIR, "ridge_model.pkl"))
joblib.dump(MODELS["Random Forest"], os.path.join(OUTPUT_DIR, "rf_model.pkl"))
joblib.dump(MODELS["XGBoost"], os.path.join(OUTPUT_DIR, "xgb_model.pkl"))

# Save ANN separately (Keras format)
MODELS["ANN"].save(os.path.join(OUTPUT_DIR, "ann_model.keras"))

print(f"    Models saved to: {os.path.abspath(OUTPUT_DIR)}")

# =============================================================================
# SECTION 15 — SAMPLE INFERENCE (REAL-WORLD USAGE)
# =============================================================================
print("\n[11] Running sample inference ...")

# Take one real sample from test set
sample = X_test.iloc[0:1]
sample_scaled = scaler.transform(sample)

print("\n    Sample Input Features:")
print(sample.to_string(index=False))

print("\n    Predictions from all models:")

for m in MODEL_NAMES:
    model = MODELS[m]

    if m == "ANN":
        pred = model.predict(sample_scaled, verbose=0)
    else:
        pred = model.predict(sample_scaled)

    print(f"\n    {m}:")
    print(f"      GT Compressor Decay  = {pred[0][0]:.6f}")
    print(f"      GT Turbine Decay     = {pred[0][1]:.6f}")


# =============================================================================
# SECTION 16 — FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("  FINAL RESULTS SUMMARY")
print("=" * 70)
print(comp_df.to_string())

best = RESULTS[best_name]
print(f"\n  ✓ Best Model : {best_name}")
print(f"    RMSE   = {best['RMSE']:.6f}")
print(f"    MAE    = {best['MAE']:.6f}")
print(f"    R²     = {best['R2']:.6f}")
print(f"    Adj R² = {best['Adj_R2']:.6f}")

print("\n  Engineering Interpretation:")
comp_range = 1.00 - 0.95    # compressor decay: range ~0.05
turb_range = 1.00 - 0.975   # turbine decay:   range ~0.025
comp_pct = best["RMSE_comp"] / comp_range * 100
turb_pct = best["RMSE_turb"] / turb_range * 100

print(f"  GT Compressor Decay RMSE = {best['RMSE_comp']:.5f}")
print(f"    → {comp_pct:.2f}% of operating range (0.95–1.0)")
print(
    "    Operationally acceptable — suitable for early fault warning"
    if comp_pct < 20
    else "    Marginal — model needs more data or features"
)

print(f"  GT Turbine Decay RMSE    = {best['RMSE_turb']:.5f}")
print(f"    → {turb_pct:.2f}% of operating range (0.975–1.0)")
print("    Operationally acceptable" if turb_pct < 20 else "    Marginal")

print(f"\n  Outputs saved to : {os.path.abspath(OUTPUT_DIR)}/")
print("=" * 70)