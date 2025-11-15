from typing import Dict, Tuple, Any, Optional, Literal
import pandas as pd

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.calibration import CalibratedClassifierCV as CCCV #TODO: do we need this? make some research

def train_lr_tfidf(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
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
        lr_model.fit(x_train, y_train) # Fit the base model first

        model = CCCV(lr_model, method="sigmoid", cv="prefit") # Use prefit since we will fit lr_model separately
        model.fit(x_val, y_val)
    else:
        model = lr_model
        model.fit(x_train, y_train)

    return {
        "model": model,
        "uncalibrated_model": lr_model if calibrate else None
    }

def predict_with_model(
    model: Any,
    x: np.ndarray,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict labels using the trained model and a specified threshold.

    Args:
        model (Any): Trained model.
        x (np.ndarray): Feature matrix for prediction.
        threshold (float): Decision threshold for classifying positive classification.

    Returns:
        np.ndarray: Predicted labels.
    """
    y_proba = model.predict_proba(x)[:, 1]  # Probability of the positive class
    y_pred = (y_proba >= threshold).astype(int) # Convert probabilities to binary labels based on the threshold
    
    return (y_proba, y_pred)

def extract_feature_and_label_arrays(
    data: Dict[str, Any],
    feature_col: str = "x",
    label_col: str = "y"
) -> Tuple[Any, Any]:
    """Extract feature and label arrays from the dictionary."""
    
    x = data[feature_col]
    y = data[label_col]
    
    return x, y