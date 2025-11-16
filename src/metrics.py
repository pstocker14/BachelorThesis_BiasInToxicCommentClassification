from typing import Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

def compute_overall_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """
    Compute overall classification metrics.

    Parameters:
    - y_true: array-like of shape (n_samples) - True labels.
    - y_pred: array-like of shape (n_samples) - Predicted labels.
    - y_proba: array-like of shape (n_samples) - Predicted probabilities.

    Returns:
    - metrics: dict - Dictionary containing accuracy, precision, recall, F1-score,
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