from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, Iterable
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

#region Overall metrics computation

def compute_overall_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """
    Compute overall classification metrics.

    Args:
        y_true (array-like of shape (n_samples)): True labels.
        y_pred (array-like of shape (n_samples)): Predicted labels.
        y_proba (array-like of shape (n_samples)): Predicted probabilities.

    Returns:
        metrics (dict): Dictionary containing accuracy, precision, recall, F1-score,
                      ROC AUC, and Average Precision.
    """
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1-score
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0) # avoid division by zero
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0) # = Sensitivity
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    
    # ROC AUC and Average Precision
    metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    metrics['pr_auc'] = average_precision_score(y_true, y_proba)
    
    return metrics

#endregion

#region Fairness metrics computation

def subgroup_auc(
        df: pd.DataFrame,
        subgroup_col: str,
        label_col: str = "labelled_as_toxic",
        y_proba_col: str = "predicted_proba",
        subgroup_mention_threshold: float = 0.5
) -> float:
    """
    Compute the AUC for a specific subgroup.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        subgroup_col (str): Column name for the subgroup indicator (boolean).
        label_col (str): Column name for the true labels.
        y_proba_col (str): Column name for the predicted probabilities.
        subgroup_mention_threshold (float): Threshold for considering subgroup mentions.

    Returns:
        float: AUC score for the specified subgroup.
    """
    subgroup_df, *_ = _split_by_subgroup(df, subgroup_col, label_col, subgroup_mention_threshold)
    
    # Extract true labels and predicted probabilities (as numpy arrays for consistency and ability to use numpy operations)
    y_true  = np.asarray(subgroup_df[label_col].values)
    y_proba = np.asarray(subgroup_df[y_proba_col].values)

    return _calculate_safe_auc(y_true, y_proba)
    
def bpsn_auc(
        df: pd.DataFrame,
        subgroup_col: str,
        label_col: str = "labelled_as_toxic",
        y_proba_col: str = "predicted_proba",
        subgroup_mention_threshold: float = 0.5
) -> float:
    """
    Compute the Background Positive, Subgroup Negative (BPSN) AUC.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        subgroup_col (str): Column name for the subgroup indicator (boolean).
        label_col (str): Column name for the true labels.
        y_proba_col (str): Column name for the predicted probabilities.
        subgroup_mention_threshold (float): Threshold for considering subgroup mentions.

    Returns:
        float: BPSN AUC score.
    """
    _, _, _, sg_neg, bg_pos, _ = _split_by_subgroup(df, subgroup_col, label_col, subgroup_mention_threshold)
    
    # Combine subgroup negatives with background positives
    bpsn_df = pd.concat([sg_neg, bg_pos], ignore_index=True)

    return _calculate_safe_auc(
        y_true = np.asarray(bpsn_df[label_col].values),
        y_proba = np.asarray(bpsn_df[y_proba_col].values)
    )

def bnsp_auc(
        df: pd.DataFrame,
        subgroup_col: str,
        label_col: str = "labelled_as_toxic",
        y_proba_col: str = "predicted_proba",
        subgroup_mention_threshold: float = 0.5
) -> float:
    """
    Compute the Background Negative, Subgroup Positive (BNSP) AUC.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        subgroup_col (str): Column name for the subgroup indicator (boolean).
        label_col (str): Column name for the true labels.
        y_proba_col (str): Column name for the predicted probabilities.
        subgroup_mention_threshold (float): Threshold for considering subgroup mentions.

    Returns:
        float: BNSP AUC score.
    """
    _, _, sg_pos, _, _, bg_neg = _split_by_subgroup(df, subgroup_col, label_col, subgroup_mention_threshold)
    
    # Combine subgroup positives with background negatives
    bnsp_df = pd.concat([sg_pos, bg_neg], ignore_index=True)

    return _calculate_safe_auc(
        y_true = np.asarray(bnsp_df[label_col].values),
        y_proba = np.asarray(bnsp_df[y_proba_col].values)
    )

def rate_gaps_equalized_odds(
        df: pd.DataFrame,
        subgroup_col: str,
        label_col: str = "labelled_as_toxic",
        y_proba_col: str = "predicted_proba",
        y_pred_col: Optional[str] = None,
        threshold: float = 0.5,
        subgroup_mention_threshold: float = 0.5,
        min_pos: int = 5,
        min_neg: int = 5
) -> Dict[str, float]:
    """
    Compute the Equalized Odds Rate Gaps for a specific subgroup.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        subgroup_col (str): Column name for the subgroup indicator (boolean).
        label_col (str): Column name for the true labels.
        y_proba_col (str): Column name for the predicted probabilities.
        y_pred_col (Optional[str]): Column name for the predicted labels. If None, labels are derived from probabilities.
        threshold (float): Decision threshold for classification. (used if y_pred_col is None)
        subgroup_mention_threshold (float): Threshold for considering subgroup mentions.
        min_pos (int): Minimum number of positive examples required to compute rates.
        min_neg (int): Minimum number of negative examples required to compute rates.
    Returns:
        Dict[str, float]: Dictionary containing FPR/FNR for subgroup/background and their gaps (sg - bg).
    """
    sg_df, bg_df, *_ = _split_by_subgroup(df, subgroup_col, label_col, subgroup_mention_threshold)
    
    # Helper function to compute FPR and FNR
    def _compute_rates(data: pd.DataFrame) -> Tuple[float, float]:
        if data.empty:
            return (np.nan, np.nan)
        
        y_true = np.asarray(data[label_col].values)
        # use predicted labels if provided, else derive from predicted probabilities
        y_pred = np.asarray(data[y_pred_col].values) if y_pred_col else (np.asarray(data[y_proba_col].values) >= threshold).astype(int)
        
        n_pos = int((y_true == 1).sum())
        n_neg = int((y_true == 0).sum())

        # if too few positive/negative examples, the rates are unreliable
        if n_pos < min_pos or n_neg < min_neg:
            return (np.nan, np.nan)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel() # ravel to flatten the array into 1D
        
        # Calculate False Positive Rate (FPR) and False Negative Rate (FNR)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
        
        return fpr, fnr
    
    # Helper function to calculate the gap between two rates
    def _calculate_gap(a: float, b: float) -> float:
        if (a != a) or (b != b):  # NaN check
            return np.nan
        return float(a - b)
    
    sg_fpr, sg_fnr = _compute_rates(sg_df)
    bg_fpr, bg_fnr = _compute_rates(bg_df)

    return {
        "sg_fpr": sg_fpr,
        "sg_fnr": sg_fnr,
        "bg_fpr": bg_fpr,
        "bg_fnr": bg_fnr,
        "gap_fpr": _calculate_gap(sg_fpr, bg_fpr),
        "gap_fnr": _calculate_gap(sg_fnr, bg_fnr)
    }

def evaluate_subgroups(
    df: pd.DataFrame,
    subgroup_cols: Iterable[str],
    label_col: str = "labelled_as_toxic",
    y_proba_col: str = "predicted_proba",
    y_pred_col: Optional[str] = None,
    threshold: float = 0.5,
    subgroup_mention_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Returns a tidy DataFrame with all fairness metrics per subgroup.
    Also includes counts to help interpret small-sample NaNs.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        subgroup_cols (Iterable[str]): List of subgroup column names.
        label_col (str): Column name for the true labels.
        y_proba_col (str): Column name for the predicted probabilities.
        y_pred_col (Optional[str]): Column name for the predicted labels (used for rate gaps). If None, labels are derived from probabilities.
        threshold (float): Decision threshold for classification.
        subgroup_mention_threshold (float): Threshold for considering subgroup mentions.
    """
    rows = []
    for col in subgroup_cols:
        sg, bg, sg_pos, sg_neg, bg_pos, bg_neg = _split_by_subgroup(df, col, label_col, subgroup_mention_threshold)

        row = {
            "subgroup": col,
            "n_subgroup": int(len(sg)),
            "n_bg": int(len(bg)),
            "n_sg_pos": int((sg[label_col]==1).sum()),
            "n_sg_neg": int((sg[label_col]==0).sum()),
            "subgroup_auc": subgroup_auc(df, col, label_col, y_proba_col, subgroup_mention_threshold),
            "bpsn_auc": bpsn_auc(df, col, label_col, y_proba_col, subgroup_mention_threshold),
            "bnsp_auc": bnsp_auc(df, col, label_col, y_proba_col, subgroup_mention_threshold)
        }
        row.update(rate_gaps_equalized_odds(df, col, label_col, y_proba_col, y_pred_col, threshold, subgroup_mention_threshold))
        rows.append(row)

    out = pd.DataFrame(rows)
    # Sort: worst first
    return out.sort_values(by=["subgroup_auc", "bpsn_auc", "bnsp_auc"], ascending=[True, True, True]).reset_index(drop=True)

#endregion

#region Private helper functions for calculations

def _split_by_subgroup(
    df: pd.DataFrame,
    subgroup_col: str,
    label_col: str,
    subgroup_mention_threshold: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (sg, bg, sg_pos, sg_neg, bg_pos, bg_neg)
    sg/bg: subgroup/background partitions
    *_pos/*_neg: further split by ground-truth label

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        subgroup_col (str): Column name for the subgroup indicator (boolean).
        label_col (str): Column name for the true labels.
        subgroup_mention_threshold (float): Threshold for considering subgroup mentions.
    """
    mask_sg = df[subgroup_col] >= subgroup_mention_threshold
    sg = df[mask_sg] # subgroup
    bg = df[~mask_sg] # background is everything not in subgroup

    sg_pos = sg[sg[label_col] == 1]
    sg_neg = sg[sg[label_col] == 0]
    bg_pos = bg[bg[label_col] == 1]
    bg_neg = bg[bg[label_col] == 0]
    return sg, bg, sg_pos, sg_neg, bg_pos, bg_neg

def _calculate_safe_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray
) -> float:
    """
    Safely calculate AUC, returning NaN if calculation fails.

    Args:
        y_true (np.ndarray): True binary labels.
        y_proba (np.ndarray): Predicted probabilities for the positive class.

    Returns:
        float: AUC score or NaN if calculation fails.
    """

    if np.unique(y_true).size < 2: # AUC is undefined if there is only one class present
        return np.nan
    
    try:
        return float(roc_auc_score(y_true, y_proba))
    except Exception: # catch any exception during AUC calculation 
        return np.nan

#endregion