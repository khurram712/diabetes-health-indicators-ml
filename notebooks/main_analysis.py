#!/usr/bin/env python
# coding: utf-8
# ===========================================================================
#  main_analysis.py  ──  Diabetes Health Indicators ML Project
#  Run this file directly or convert to Jupyter Notebook:
#      pip install jupytext
#      jupytext --to notebook main_analysis.py
# ===========================================================================

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  CELL 0 — Imports & Configuration                                       │
# └─────────────────────────────────────────────────────────────────────────┘

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV, cross_val_score,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Add project root to path so src/ modules can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import (
    load_data, inspect_data,
    handle_missing_values, remove_duplicates,
    label_encode_column, standardize_features,
    prepare_targets, get_feature_matrix, save_processed,
)
from src.evaluation import (
    evaluate_binary, evaluate_multiclass, evaluate_regression,
    plot_roc_curve, plot_confusion_matrix_binary, plot_confusion_matrix_multi,
    plot_actual_vs_predicted, plot_residuals,
    compare_models, results_to_dataframe,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Paths
DATA_RAW       = os.path.join(PROJECT_ROOT, "data", "raw",       "diabetes_dataset.csv")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed", "diabetes_clean.csv")
MODELS_DIR     = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("✅ Setup complete.\n")


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  CELL 1 — Load & Inspect                                                │
# └─────────────────────────────────────────────────────────────────────────┘

df_raw = load_data(DATA_RAW)
inspect_data(df_raw)


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  CELL 2 — Data Preprocessing                                            │
# └─────────────────────────────────────────────────────────────────────────┘

df = df_raw.copy()

# 2a. Remove duplicate rows
df = remove_duplicates(df)

# 2b. Handle missing values (median imputation for numerics)
df = handle_missing_values(df, strategy="median")

# 2c. Encode the multiclass target (diabetes_stage) with LabelEncoder
#     so it becomes integers 0, 1, 2, …
if "diabetes_stage" in df.columns:
    df, le_stage = label_encode_column(df, "diabetes_stage")

# 2d. Encode the binary target if it is not already numeric
if "diagnosed_diabetes" in df.columns and df["diagnosed_diabetes"].dtype == object:
    df, le_diab = label_encode_column(df, "diagnosed_diabetes")

# 2e. Save processed CSV
save_processed(df, DATA_PROCESSED)

df.head()


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  CELL 3 — Exploratory Data Analysis (EDA)                               │
# └─────────────────────────────────────────────────────────────────────────┘

TARGET_COLS = ["diagnosed_diabetes", "diabetes_stage", "diabetes_risk_score"]
feature_cols = [c for c in df.columns if c not in TARGET_COLS]

# ── 3a. Distribution of target variables ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Binary
if "diagnosed_diabetes" in df.columns:
    df["diagnosed_diabetes"].value_counts().plot.bar(
        ax=axes[0], color=["#4CAF50", "#F44336"], edgecolor="white"
    )
    axes[0].set_title("Binary: Diagnosed Diabetes\n(0 = No, 1 = Yes)")
    axes[0].set_xlabel("")

# Multiclass
if "diabetes_stage" in df.columns:
    df["diabetes_stage"].value_counts().sort_index().plot.bar(
        ax=axes[1], color=sns.color_palette("Set2"), edgecolor="white"
    )
    axes[1].set_title("Multiclass: Diabetes Stage")
    axes[1].set_xlabel("")

# Regression target
if "diabetes_risk_score" in df.columns:
    axes[2].hist(df["diabetes_risk_score"], bins=30,
                 color="#2196F3", edgecolor="white")
    axes[2].set_title("Regression: Diabetes Risk Score")
    axes[2].set_xlabel("Risk Score")

plt.suptitle("Target Variable Distributions", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()


# ── 3b. Correlation Heat-map ───────────────────────────────────────────────
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()

plt.figure(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm",
            center=0, linewidths=0.4)
plt.title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()


# ── 3c. Feature distributions — Histograms ────────────────────────────────
num_features = [c for c in feature_cols if df[c].dtype != object][:12]
df[num_features].hist(bins=25, figsize=(16, 9), color="#5C6BC0", edgecolor="white")
plt.suptitle("Feature Distributions (Histograms)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()


# ── 3d. Boxplots: feature vs binary target ────────────────────────────────
if "diagnosed_diabetes" in df.columns:
    plot_features = num_features[:8]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()
    for i, feat in enumerate(plot_features):
        sns.boxplot(x="diagnosed_diabetes", y=feat, data=df,
                    palette=["#4CAF50", "#F44336"], ax=axes[i])
        axes[i].set_title(feat)
    plt.suptitle("Feature vs Diagnosed Diabetes (Box Plots)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ── 3e. Top correlations with binary target ───────────────────────────────
if "diagnosed_diabetes" in df.columns:
    top_corr = (
        corr["diagnosed_diabetes"]
        .drop("diagnosed_diabetes")
        .abs()
        .sort_values(ascending=False)
        .head(10)
    )
    print("\n🔝 Top 10 Features Correlated with 'diagnosed_diabetes':")
    print(top_corr.to_string())


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  CELL 4 — Data Splitting                                                │
# └─────────────────────────────────────────────────────────────────────────┘

targets = prepare_targets(df)
X = get_feature_matrix(df, TARGET_COLS)

splits = {}

# ── Binary ────────────────────────────────────────────────────────────────
if "binary" in targets:
    y_bin = targets["binary"]
    X_tr_b, X_te_b, y_tr_b, y_te_b = train_test_split(
        X, y_bin, test_size=0.2, random_state=RANDOM_STATE, stratify=y_bin
    )
    X_tr_b_sc, X_te_b_sc, scaler_bin = standardize_features(X_tr_b, X_te_b)
    splits["binary"] = (X_tr_b, X_te_b, X_tr_b_sc, X_te_b_sc, y_tr_b, y_te_b)
    print(f"Binary  — Train: {len(y_tr_b)}, Test: {len(y_te_b)}")

# ── Multiclass ────────────────────────────────────────────────────────────
if "multiclass" in targets:
    y_mc = targets["multiclass"]
    X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(
        X, y_mc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_mc
    )
    X_tr_m_sc, X_te_m_sc, scaler_mc = standardize_features(X_tr_m, X_te_m)
    splits["multiclass"] = (X_tr_m, X_te_m, X_tr_m_sc, X_te_m_sc, y_tr_m, y_te_m)
    print(f"Multiclass — Train: {len(y_tr_m)}, Test: {len(y_te_m)}")

# ── Regression ────────────────────────────────────────────────────────────
if "regression" in targets:
    y_reg = targets["regression"]
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        X, y_reg, test_size=0.2, random_state=RANDOM_STATE
    )
    X_tr_r_sc, X_te_r_sc, scaler_reg = standardize_features(X_tr_r, X_te_r)
    splits["regression"] = (X_tr_r, X_te_r, X_tr_r_sc, X_te_r_sc, y_tr_r, y_te_r)
    print(f"Regression — Train: {len(y_tr_r)}, Test: {len(y_te_r)}")


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  CELL 5 — Baseline Model Training & Evaluation                          │
# └─────────────────────────────────────────────────────────────────────────┘

baseline_results = {"binary": {}, "multiclass": {}, "regression": {}}
roc_probs = {}

# ════════════════════════════════════════════════════════════
# 5A — BINARY CLASSIFICATION
# ════════════════════════════════════════════════════════════
if "binary" in splits:
    X_tr_b, X_te_b, X_tr_b_sc, X_te_b_sc, y_tr_b, y_te_b = splits["binary"]

    binary_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree":       DecisionTreeClassifier(
            max_depth=5, random_state=RANDOM_STATE),
        "KNN":                 KNeighborsClassifier(n_neighbors=5),
    }

    for name, model in binary_models.items():
        # LR and KNN benefit from scaling; DT does not require it
        if name in ("Logistic Regression", "KNN"):
            model.fit(X_tr_b_sc, y_tr_b)
            y_pred = model.predict(X_te_b_sc)
            y_prob = model.predict_proba(X_te_b_sc)[:, 1]
            roc_probs[name] = y_prob
        else:
            model.fit(X_tr_b, y_tr_b)
            y_pred = model.predict(X_te_b)
            y_prob = model.predict_proba(X_te_b)[:, 1]
            roc_probs[name] = y_prob

        metrics = evaluate_binary(y_te_b, y_pred, y_prob, model_name=name)
        baseline_results["binary"][name] = metrics
        plot_confusion_matrix_binary(y_te_b, y_pred, model_name=name,
                                      labels=["No Diabetes", "Diabetes"])

    plot_roc_curve(y_te_b, roc_probs,
                   title="ROC Curves — Binary Classification (Baseline)")
    compare_models(baseline_results["binary"], metric="f1",
                   title="Baseline Binary — F1 Score Comparison")
    print("\nBaseline Binary Results:")
    print(results_to_dataframe(baseline_results["binary"]))


# ════════════════════════════════════════════════════════════
# 5B — MULTICLASS CLASSIFICATION
# ════════════════════════════════════════════════════════════
if "multiclass" in splits:
    X_tr_m, X_te_m, X_tr_m_sc, X_te_m_sc, y_tr_m, y_te_m = splits["multiclass"]

    multiclass_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, multi_class="auto", random_state=RANDOM_STATE),
        "Decision Tree":       DecisionTreeClassifier(
            max_depth=6, random_state=RANDOM_STATE),
        "KNN":                 KNeighborsClassifier(n_neighbors=5),
    }

    for name, model in multiclass_models.items():
        if name in ("Logistic Regression", "KNN"):
            model.fit(X_tr_m_sc, y_tr_m)
            y_pred = model.predict(X_te_m_sc)
        else:
            model.fit(X_tr_m, y_tr_m)
            y_pred = model.predict(X_te_m)

        metrics = evaluate_multiclass(y_te_m, y_pred, model_name=name)
        baseline_results["multiclass"][name] = metrics
        plot_confusion_matrix_multi(y_te_m, y_pred, model_name=name)

    compare_models(baseline_results["multiclass"], metric="macro_f1",
                   title="Baseline Multiclass — Macro F1 Comparison")
    print("\nBaseline Multiclass Results:")
    print(results_to_dataframe(baseline_results["multiclass"]))


# ════════════════════════════════════════════════════════════
# 5C — REGRESSION
# ════════════════════════════════════════════════════════════
if "regression" in splits:
    X_tr_r, X_te_r, X_tr_r_sc, X_te_r_sc, y_tr_r, y_te_r = splits["regression"]

    regression_models = {
        "Linear Regression":       LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(
            max_depth=5, random_state=RANDOM_STATE),
    }

    for name, model in regression_models.items():
        if name == "Linear Regression":
            model.fit(X_tr_r_sc, y_tr_r)
            y_pred = model.predict(X_te_r_sc)
        else:
            model.fit(X_tr_r, y_tr_r)
            y_pred = model.predict(X_te_r)

        metrics = evaluate_regression(y_te_r, y_pred, model_name=name)
        baseline_results["regression"][name] = metrics
        plot_actual_vs_predicted(y_te_r, y_pred, model_name=name)
        plot_residuals(y_te_r, y_pred, model_name=name)

    print("\nBaseline Regression Results:")
    print(results_to_dataframe(baseline_results["regression"]))


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  CELL 6 — Hyperparameter Tuning                                         │
# └─────────────────────────────────────────────────────────────────────────┘

tuned_results  = {"binary": {}, "multiclass": {}, "regression": {}}
best_models    = {}
cv_strategy    = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ════════════════════════════════════════════════════════════
# 6A — Binary: Tune Logistic Regression & Decision Tree
# ════════════════════════════════════════════════════════════
if "binary" in splits:
    X_tr_b, X_te_b, X_tr_b_sc, X_te_b_sc, y_tr_b, y_te_b = splits["binary"]

    # --- Logistic Regression ---
    lr_param_grid = {
        "C"      : [0.01, 0.1, 1, 10, 100],
        "solver" : ["lbfgs", "liblinear"],
        "penalty": ["l2"],
    }
    lr_gs = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        lr_param_grid, cv=cv_strategy, scoring="f1", n_jobs=-1, verbose=0
    )
    lr_gs.fit(X_tr_b_sc, y_tr_b)
    best_lr_bin = lr_gs.best_estimator_
    print(f"\n🔍 Best LR params (binary): {lr_gs.best_params_}")

    y_pred_lr = best_lr_bin.predict(X_te_b_sc)
    y_prob_lr = best_lr_bin.predict_proba(X_te_b_sc)[:, 1]
    m = evaluate_binary(y_te_b, y_pred_lr, y_prob_lr,
                         model_name="Tuned LR (Binary)")
    tuned_results["binary"]["Tuned Logistic Regression"] = m
    best_models["binary_lr"] = best_lr_bin

    # --- Decision Tree ---
    dt_param_grid = {
        "max_depth"        : [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf" : [1, 2, 4],
    }
    dt_gs = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        dt_param_grid, cv=cv_strategy, scoring="f1", n_jobs=-1, verbose=0
    )
    dt_gs.fit(X_tr_b, y_tr_b)
    best_dt_bin = dt_gs.best_estimator_
    print(f"\n🔍 Best DT params (binary): {dt_gs.best_params_}")

    y_pred_dt = best_dt_bin.predict(X_te_b)
    y_prob_dt = best_dt_bin.predict_proba(X_te_b)[:, 1]
    m = evaluate_binary(y_te_b, y_pred_dt, y_prob_dt,
                         model_name="Tuned DT (Binary)")
    tuned_results["binary"]["Tuned Decision Tree"] = m
    best_models["binary_dt"] = best_dt_bin

    # Final ROC comparison (baseline vs tuned)
    roc_all = dict(roc_probs)
    roc_all["Tuned LR"]  = y_prob_lr
    roc_all["Tuned DT"]  = y_prob_dt
    plot_roc_curve(y_te_b, roc_all,
                   title="ROC Curves — Baseline vs Tuned (Binary)")
    compare_models(tuned_results["binary"], metric="f1",
                   title="Tuned Binary — F1 Score Comparison")


# ════════════════════════════════════════════════════════════
# 6B — Multiclass: Tune Decision Tree (Random Search)
# ════════════════════════════════════════════════════════════
if "multiclass" in splits:
    X_tr_m, X_te_m, X_tr_m_sc, X_te_m_sc, y_tr_m, y_te_m = splits["multiclass"]

    dt_param_dist = {
        "max_depth"        : [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf" : [1, 2, 4, 8],
        "criterion"        : ["gini", "entropy"],
    }
    dt_rs = RandomizedSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        dt_param_dist, n_iter=30, cv=5, scoring="f1_macro",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=0
    )
    dt_rs.fit(X_tr_m, y_tr_m)
    best_dt_mc = dt_rs.best_estimator_
    print(f"\n🔍 Best DT params (multiclass): {dt_rs.best_params_}")

    y_pred_dt_mc = best_dt_mc.predict(X_te_m)
    m = evaluate_multiclass(y_te_m, y_pred_dt_mc,
                             model_name="Tuned DT (Multiclass)")
    tuned_results["multiclass"]["Tuned Decision Tree"] = m
    best_models["multiclass_dt"] = best_dt_mc

    # Tune LR for multiclass too
    lr_mc_gs = GridSearchCV(
        LogisticRegression(max_iter=1000, multi_class="auto",
                           random_state=RANDOM_STATE),
        {"C": [0.01, 0.1, 1, 10]},
        cv=5, scoring="f1_macro", n_jobs=-1
    )
    lr_mc_gs.fit(X_tr_m_sc, y_tr_m)
    best_lr_mc = lr_mc_gs.best_estimator_
    print(f"\n🔍 Best LR params (multiclass): {lr_mc_gs.best_params_}")

    y_pred_lr_mc = best_lr_mc.predict(X_te_m_sc)
    m = evaluate_multiclass(y_te_m, y_pred_lr_mc,
                             model_name="Tuned LR (Multiclass)")
    tuned_results["multiclass"]["Tuned Logistic Regression"] = m
    best_models["multiclass_lr"] = best_lr_mc

    compare_models(tuned_results["multiclass"], metric="macro_f1",
                   title="Tuned Multiclass — Macro F1 Comparison")


# ════════════════════════════════════════════════════════════
# 6C — Regression: Tune Decision Tree Regressor
# ════════════════════════════════════════════════════════════
if "regression" in splits:
    X_tr_r, X_te_r, X_tr_r_sc, X_te_r_sc, y_tr_r, y_te_r = splits["regression"]

    dt_reg_grid = {
        "max_depth"        : [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf" : [1, 2, 4],
    }
    dt_reg_gs = GridSearchCV(
        DecisionTreeRegressor(random_state=RANDOM_STATE),
        dt_reg_grid, cv=5, scoring="neg_mean_squared_error",
        n_jobs=-1, verbose=0
    )
    dt_reg_gs.fit(X_tr_r, y_tr_r)
    best_dt_reg = dt_reg_gs.best_estimator_
    print(f"\n🔍 Best DT params (regression): {dt_reg_gs.best_params_}")

    y_pred_dt_reg = best_dt_reg.predict(X_te_r)
    m = evaluate_regression(y_te_r, y_pred_dt_reg,
                             model_name="Tuned DT (Regression)")
    tuned_results["regression"]["Tuned Decision Tree"] = m
    best_models["regression_dt"] = best_dt_reg

    plot_actual_vs_predicted(y_te_r, y_pred_dt_reg,
                              model_name="Tuned Decision Tree Regressor")
    plot_residuals(y_te_r, y_pred_dt_reg,
                   model_name="Tuned Decision Tree Regressor")

    print("\nTuned Regression Results:")
    print(results_to_dataframe(tuned_results["regression"]))


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  CELL 7 — Export Best Models to models/                                 │
# └─────────────────────────────────────────────────────────────────────────┘

# Pick the best binary model by F1
if "binary" in tuned_results and tuned_results["binary"]:
    best_bin_name = max(tuned_results["binary"],
                        key=lambda n: tuned_results["binary"][n]["f1"])
    key_map = {
        "Tuned Logistic Regression": "binary_lr",
        "Tuned Decision Tree":       "binary_dt",
    }
    final_binary = best_models[key_map[best_bin_name]]
    joblib.dump(final_binary,
                os.path.join(MODELS_DIR, "binary_model.joblib"))
    joblib.dump(scaler_bin,
                os.path.join(MODELS_DIR, "binary_scaler.joblib"))
    print(f"✅ Saved binary model  ({best_bin_name})")

# Pick best multiclass model by macro_f1
if "multiclass" in tuned_results and tuned_results["multiclass"]:
    best_mc_name = max(tuned_results["multiclass"],
                       key=lambda n: tuned_results["multiclass"][n]["macro_f1"])
    key_map_mc = {
        "Tuned Logistic Regression": "multiclass_lr",
        "Tuned Decision Tree":       "multiclass_dt",
    }
    final_mc = best_models[key_map_mc[best_mc_name]]
    joblib.dump(final_mc,
                os.path.join(MODELS_DIR, "multiclass_model.joblib"))
    joblib.dump(scaler_mc,
                os.path.join(MODELS_DIR, "multiclass_scaler.joblib"))
    print(f"✅ Saved multiclass model ({best_mc_name})")

# Best regression model by R²
if "regression" in tuned_results and tuned_results["regression"]:
    best_reg_name = max(tuned_results["regression"],
                        key=lambda n: tuned_results["regression"][n]["R²"])
    final_reg = best_models["regression_dt"]
    joblib.dump(final_reg,
                os.path.join(MODELS_DIR, "regression_model.joblib"))
    joblib.dump(scaler_reg,
                os.path.join(MODELS_DIR, "regression_scaler.joblib"))
    print(f"✅ Saved regression model ({best_reg_name})")

# Save feature column list (needed by Streamlit app)
joblib.dump(list(X.columns),
            os.path.join(MODELS_DIR, "feature_columns.joblib"))
print("✅ Feature column list saved.")

print("\n🎉 Full pipeline complete! Models exported to models/")
