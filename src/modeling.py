from typing import Dict, Tuple, Any, Optional, Literal
import pandas as pd

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.calibration import CalibratedClassifierCV as CCCV #TODO: do we need this? make some research

def train_lr_tfidf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weight: str = "balanced",
    C: float = 1.0,
    penalty: Literal["l1", "l2", "elasticnet", None] = "l2",
    solver: Literal["newton-cg", "lbfgs", "liblinear", "sag", "saga"] = "saga", #saga is the best solver for TF-IDF text classification and large sparse datasets
    random_state: int = 42,
    max_iter: int = 1000,
    calibrate: bool = True
) -> Dict[str, Any]:
    """Train a Logistic Regression model with TF-IDF features.

    Args:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation feature matrix.
        y_val (np.ndarray): Validation labels.
        class_weight (str): Class weight strategy for handling class imbalance.
        C (float): Inverse of regularization strength.
        penalty (Literal["l1", "l2", "elasticnet", None]): Type of regularization to use.
        solver (Literal["newton-cg", "lbfgs", "liblinear", "sag", "saga"]): Algorithm to use in the optimization problem.
        random_state (int): Random state for reproducibility.
        max_iter (int): Maximum number of iterations for the solver.
        calibrate (bool): Whether to calibrate the classifier.

    Returns:
        Dict[str, Any]: A dictionary containing the trained model and, if calibrated, the uncalibrated model (for evaluation purposes).
    """
    lr_model = LR(class_weight=class_weight, C=C, penalty=penalty, solver=solver, random_state=random_state, max_iter=max_iter)
    
    if calibrate:
        lr_model.fit(X_train, y_train) # Fit the base model first

        model = CCCV(lr_model, method="sigmoid", cv="prefit") # Use prefit since we will fit lr_model separately
        model.fit(X_val, y_val)
    else:
        model = lr_model
        model.fit(X_train, y_train)

    return {
        "model": model,
        "uncalibrated_model": lr_model if calibrate else None
    }

def extract_feature_and_label_arrays(
    data: Dict[str, pd.Series[Any]],
    feature_col: str = "x",
    label_col: str = "y"
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract feature and label arrays from the dictionary."""
    
    x = data[feature_col].to_numpy()
    y = data[label_col].to_numpy()
    
    return x, y