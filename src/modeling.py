from typing import Dict, Tuple, Any, Optional, Literal
import pandas as pd

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.calibration import CalibratedClassifierCV as CCCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
from fairlearn.reductions import ExponentiatedGradient

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

PARAM_GRID_LR = [
    # 1) liblinear: supports L1 and L2
    {
        "solver": ["liblinear"],
        "penalty": ["l1", "l2"],
        "C": [0.1, 0.3, 1, 3, 10],
        "class_weight": [None, "balanced"]
    },

    # 2) lbfgs: supports ONLY L2
    {
        "solver": ["lbfgs"],
        "penalty": ["l2"],
        "C": [0.1, 0.3, 1, 3, 10],
        "class_weight": [None, "balanced"]
    },
]
# other solvers are possible but not included due to either limited regularization support or longer training times (e.g., around 2h for one try for 'saga'...)
# more info on solvers: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression-solvers

def train_lr_tfidf(
    x_train: np.ndarray,
    y_train: np.ndarray,
    parameter_tuning: bool = True,
    class_weight: str = "balanced",
    C: float = 1.0,
    penalty: Literal["l1", "l2", "elasticnet", None] = "l2",
    solver: Literal["newton-cg", "lbfgs", "liblinear", "sag", "saga"] = "lbfgs",
    random_state: int = 42,
    max_iter: int = 1000,
    calibrate: bool = True
) -> Dict[str, Any]:
    """Train a Logistic Regression model with TF-IDF features.

    Args:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training labels.
        parameter_tuning (bool): Whether to perform hyperparameter tuning. If True, the following hyperparameter arguments are ignored: class_weight, C, penalty, solver, max_iter.
        class_weight (str): Class weight strategy for handling class imbalance.
        C (float): Inverse of regularization strength.
        penalty (Literal["l1", "l2", "elasticnet", None]): Type of regularization to use.
        solver (Literal["newton-cg", "lbfgs", "liblinear", "sag", "saga"]): Algorithm to use in the optimization problem.
        random_state (int): Random state for reproducibility.
        max_iter (int): Maximum number of iterations for the solver.
        calibrate (bool): Whether to calibrate the classifier.

    Returns:
        Dict[str, Any]: A dictionary containing the trained model and, if calibrated, the uncalibrated model (for evaluation purposes).
            Additionally, the best hyperparameters found during tuning (if parameter_tuning is True) are included under the key "best_params".
    """

    best_params = None
    if parameter_tuning:
        best_params = find_best_hyperparameters(x_train, y_train)

        class_weight = best_params["class_weight"]
        C = best_params["C"]
        penalty = best_params["penalty"]
        solver = best_params["solver"]

    lr_model = LR(class_weight=class_weight, C=C, penalty=penalty, solver=solver, random_state=random_state, max_iter=max_iter)
    
    if calibrate:
        model = CCCV(lr_model, method="sigmoid", cv=CV)
        lr_model.fit(x_train, y_train)  # fit uncalibrated model on training set for later use
    else:
        model = lr_model
    
    model.fit(x_train, y_train)

    return {
        "model": model,
        "uncalibrated_model": lr_model if calibrate else None,
        "best_params": best_params
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
        Tuple[np.ndarray, np.ndarray]: Predicted probabilities and binary labels.
    """
    if isinstance(model, ExponentiatedGradient):
        # For ExponentiatedGradient, compute weighted average of predict_proba from predictors
        if hasattr(model, 'predictors_') and hasattr(model, 'weights_'):
            probas = np.array([pred.predict_proba(x)[:, 1] for pred in model.predictors_])
            y_proba = np.average(probas, weights=model.weights_, axis=0)
        else:
            raise AttributeError("ExponentiatedGradient model does not have 'predictors_' or 'weights_' attributes. Ensure the model is fitted and fairlearn version supports these attributes.")
    else:
        y_proba = model.predict_proba(x)[:, 1]  # Probability of the positive class

    y_pred = (y_proba >= threshold).astype(int) # Convert probabilities to binary labels based on the threshold
    
    return (y_proba, y_pred)

def extract_feature_and_label_arrays(
    data: Dict[str, Any],
    feature_col: str = "x",
    label_col: str = "y"
) -> Tuple[Any, Any]:
    """Extract feature and label arrays from the dictionary.
    
    Args:
        data (Dict[str, Any]): Dictionary containing feature and label arrays.
        feature_col (str): Key for the feature array in the dictionary.
        label_col (str): Key for the label array in the dictionary.

    Returns:
        Tuple[Any, Any]: Feature and label arrays.
    """
    
    x = data[feature_col]
    y = data[label_col]
    
    return x, y

def find_best_hyperparameters(
    x_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: list[Dict[str, Any]] = PARAM_GRID_LR,
    scoring: str = "roc_auc",
    n_jobs: int = 1 # number of parallel jobs (1 means no parallelism; default due to OS crashes on Windows with more)
) -> Dict[str, Any]:
    """Perform hyperparameter tuning using GridSearchCV.

    Args:
        x_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training labels.
        param_grid (list[Dict[str, Any]]): list of dictionaries with parameter names as keys and lists of parameter settings to try as values.
        scoring (str): Scoring metric to optimize.
        n_jobs (int): Number of jobs to run in parallel.

    Returns:
        Dict[str, Any]: Best hyperparameters found during the search.
    """
    
    lr_model = LR(max_iter=1000)

    grid_search = GridSearchCV(
        estimator=lr_model,
        param_grid=param_grid,
        scoring=scoring,
        cv=CV,
        n_jobs=n_jobs,
        verbose=2 # for more detailed output
    )
    
    grid_search.fit(x_train, y_train)
    
    print("Best params:", grid_search.best_params_)
    print("Best CV AUC: {:.4f}".format(grid_search.best_score_))

    return grid_search.best_params_

def find_best_threshold(
    y_val: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Optional[np.ndarray] = None
) -> float:
    """Find the best decision threshold based on F1-score.

    Args:
        y_val (np.ndarray): True labels.
        y_proba (np.ndarray): Predicted probabilities for the positive class.
        thresholds (Optional[np.ndarray]): Array of thresholds to evaluate. If None, defaults to values from 0.05 to 0.95 with a step of 0.01.

    Returns:
        float: Best threshold value.
    """

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        current_f1 = f1_score(y_val, y_pred)

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    print(f"Best threshold: {best_threshold:.2f} with F1-score: {best_f1:.4f}")
    return best_threshold