"""
evaluation.py
-------------
Reusable functions for evaluating and visualising model performance
across binary classification, multiclass classification, and regression.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    mean_absolute_error, mean_squared_error, r2_score,
)

# ── Consistent colour palette ──────────────────────────────────────────────
PALETTE = sns.color_palette("Set2")
sns.set_theme(style="whitegrid", palette=PALETTE)


# ══════════════════════════════════════════════════════════════════════════
# A. BINARY CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════

def evaluate_binary(y_true, y_pred, y_prob=None,
                    model_name: str = "Model") -> dict:
    """
    Compute and print binary classification metrics.

    Parameters
    ----------
    y_true     : True labels
    y_pred     : Predicted labels
    y_prob     : Predicted probabilities for the positive class (optional)
    model_name : Label used in printed output

    Returns
    -------
    Dictionary with metric values.
    """
    metrics = {
        "accuracy" : accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall"   : recall_score(y_true, y_pred, zero_division=0),
        "f1"       : f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)

    print(f"\n{'='*55}")
    print(f"  Binary Evaluation — {model_name}")
    print(f"{'='*55}")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    print(f"\n{classification_report(y_true, y_pred, zero_division=0)}")
    return metrics


def plot_roc_curve(y_true, y_prob_dict: dict,
                   title: str = "ROC Curves – Binary Classification") -> None:
    """
    Overlay ROC curves for multiple models.

    Parameters
    ----------
    y_true        : True binary labels
    y_prob_dict   : {'Model Name': predicted_probabilities, ...}
    title         : Plot title
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (name, prob) in enumerate(y_prob_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc = roc_auc_score(y_true, prob)
        ax.plot(fpr, tpr, lw=2, color=PALETTE[i],
                label=f"{name} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_binary(y_true, y_pred,
                                  model_name: str = "Model",
                                  labels=None) -> None:
    """Plot a styled confusion matrix for binary classification."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════
# B. MULTICLASS CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════

def evaluate_multiclass(y_true, y_pred,
                         model_name: str = "Model") -> dict:
    """
    Compute and print multiclass classification metrics.

    Returns
    -------
    Dictionary with accuracy and macro-averaged F1.
    """
    metrics = {
        "accuracy"  : accuracy_score(y_true, y_pred),
        "macro_f1"  : f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    print(f"\n{'='*55}")
    print(f"  Multiclass Evaluation — {model_name}")
    print(f"{'='*55}")
    for k, v in metrics.items():
        print(f"  {k:<14}: {v:.4f}")
    print(f"\n{classification_report(y_true, y_pred, zero_division=0)}")
    return metrics


def plot_confusion_matrix_multi(y_true, y_pred,
                                 model_name: str = "Model",
                                 labels=None) -> None:
    """Plot a heatmap confusion matrix for multiclass problems."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════
# C. REGRESSION
# ══════════════════════════════════════════════════════════════════════════

def evaluate_regression(y_true, y_pred,
                         model_name: str = "Model") -> dict:
    """
    Compute and print regression metrics.

    Returns
    -------
    Dictionary with MAE, MSE, RMSE, and R².
    """
    mse = mean_squared_error(y_true, y_pred)
    metrics = {
        "MAE"  : mean_absolute_error(y_true, y_pred),
        "MSE"  : mse,
        "RMSE" : np.sqrt(mse),
        "R²"   : r2_score(y_true, y_pred),
    }
    print(f"\n{'='*55}")
    print(f"  Regression Evaluation — {model_name}")
    print(f"{'='*55}")
    for k, v in metrics.items():
        print(f"  {k:<6}: {v:.4f}")
    return metrics


def plot_actual_vs_predicted(y_true, y_pred,
                              model_name: str = "Model") -> None:
    """Scatter plot of actual vs predicted values."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.4, color=PALETTE[0], edgecolors="none")
    lims = [min(y_true.min(), y_pred.min()),
            max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1.5, label="Perfect Prediction")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Actual vs Predicted — {model_name}")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true, y_pred, model_name: str = "Model") -> None:
    """Plot residuals (errors) to check for bias or heteroscedasticity."""
    residuals = np.array(y_true) - np.array(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Residual scatter
    axes[0].scatter(y_pred, residuals, alpha=0.4, color=PALETTE[1])
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Predicted")

    # Residual histogram
    axes[1].hist(residuals, bins=30, color=PALETTE[2], edgecolor="white")
    axes[1].set_xlabel("Residual")
    axes[1].set_title("Residual Distribution")

    fig.suptitle(f"Residual Analysis — {model_name}", fontweight="bold")
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════
# D. COMPARISON UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def compare_models(results: dict, metric: str = "accuracy",
                   title: str = "Model Comparison") -> None:
    """
    Bar chart comparing a single metric across multiple models.

    Parameters
    ----------
    results : {'Model Name': metrics_dict, ...}
    metric  : Key in the metrics dict to plot
    title   : Chart title
    """
    names  = list(results.keys())
    values = [results[n].get(metric, 0) for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names, values, color=PALETTE[:len(names)])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(metric.upper())
    ax.set_title(title)
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.show()


def results_to_dataframe(results: dict) -> pd.DataFrame:
    """Convert a results dictionary to a tidy DataFrame for display."""
    return pd.DataFrame(results).T.round(4)
