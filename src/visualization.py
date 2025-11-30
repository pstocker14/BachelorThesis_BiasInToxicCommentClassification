from __future__ import annotations
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve,
    average_precision_score, confusion_matrix
)

#region Performance visualizations

def plot_performance_bar(metrics_dict: Dict[str, float], title="Model Performance"):
    """
    Visualize accuracy, precision, recall, F1, ROC-AUC, PR-AUC as a bar chart.

    Args:
        metrics_dict (dict): Dictionary containing metric names and their values.
        title (str): Title of the plot.
    """

    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    plt.figure(figsize=(8, 5))
    plt.bar(names, values)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, title="ROC Curve"):
    """
    Plot ROC curve using true labels and predicted probabilities.

    Args:
        y_true (np.ndarray): True binary labels.
        y_proba (np.ndarray): Predicted probabilities for the positive class.
        title (str): Title of the plot.
    """

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # diagonal
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_pr_curve(y_true: np.ndarray, y_proba: np.ndarray, title="Precision–Recall Curve"):
    """
    Plot Precision–Recall curve and display Average Precision.

    Args:
        y_true (np.ndarray): True binary labels.
        y_proba (np.ndarray): Predicted probabilities for the positive class.
        title (str): Title of the plot.
    """

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title="Confusion Matrix"):
    """
    Plot a confusion matrix.

    Args:
        y_true (np.ndarray): True binary labels.
        y_pred (np.ndarray): Predicted binary labels.
        title (str): Title of the plot.
    """

    cm = confusion_matrix(y_true, y_pred)
    classes = ["Non-Toxic (0)", "Toxic (1)"]

    plt.figure(figsize=(5, 4))

    # Draw white blocks (no color) using imshow with a constant array
    plt.imshow(np.ones_like(cm), interpolation="nearest", cmap="gray_r")

    # Add borders manually (outer grid)
    for i in range(cm.shape[0] + 1):
        plt.axhline(i - 0.5, color="black", linewidth=1)
    for j in range(cm.shape[1] + 1):
        plt.axvline(j - 0.5, color="black", linewidth=1)

    # Title (same as before)
    plt.title(title)

    # X and Y axis ticks
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Add numeric values (always black text)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     ha="center", va="center", color="black", fontsize=13)

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()

#endregion

#region Fairness visualizations

def plot_auc_bars(df_metrics: pd.DataFrame, figsize=(10,5)):
    """
    Bar chart of Subgroup AUC, BPSN AUC, BNSP AUC per subgroup.

    Args:
        df_metrics (pd.DataFrame): DataFrame containing subgroup metrics.
        figsize (tuple): Figure size.
    """
    cols = ["subgroup_auc", "bpsn_auc", "bnsp_auc"]
    data = df_metrics.set_index("subgroup")[cols]
    ax = data.plot(kind="bar", figsize=figsize)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUC")
    ax.set_title("Subgroup, BPSN, BNSP AUCs — Baseline")
    ax.legend(title="Metric")
    plt.tight_layout()
    plt.show()

def plot_rate_gaps_bars(df_metrics: pd.DataFrame, figsize=(10,5)):
    """
    Bar chart of FPR/FNR gaps (sg - bg). Zero line for reference.

    Args:
        df_metrics (pd.DataFrame): DataFrame containing subgroup metrics.
        figsize (tuple): Figure size.
    """
    cols = ["gap_fpr", "gap_fnr"]
    data = df_metrics.set_index("subgroup")[cols]
    ax = data.plot(kind="bar", figsize=figsize)
    ax.axhline(0.0, linestyle="--")
    ax.set_ylabel("Gap (Subgroup - Background)")
    ax.set_title("FPR/FNR Gaps")
    ax.legend(title="Gap")
    plt.tight_layout()
    plt.show()

def print_small_sample_warnings(df_metrics: pd.DataFrame, min_rows: int = 50, min_pos: int = 5, min_neg: int = 5):
    """
    Print warnings for subgroups with small samples (to contextualize NaNs/instability).

    Args:
        df_metrics (pd.DataFrame): DataFrame containing subgroup metrics.
        min_rows (int): Minimum total rows for subgroup.
        min_pos (int): Minimum positive samples for subgroup.
        min_neg (int): Minimum negative samples for subgroup.
    """
    for _, r in df_metrics.iterrows():
        msgs = []
        if r["n_subgroup"] < min_rows:
            msgs.append(f"n_subgroup={r['n_subgroup']}")
        if r["n_sg_pos"] < min_pos:
            msgs.append(f"n_sg_pos={r['n_sg_pos']}")
        if r["n_sg_neg"] < min_neg:
            msgs.append(f"n_sg_neg={r['n_sg_neg']}")
        if msgs:
            print(f"[WARN] {r['subgroup']}: small sample -> " + ", ".join(msgs))

#endregion