from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal, Tuple

import numpy as np
import pandas as pd
import re
import os
import sys

from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, BoundedGroupLoss, ZeroOneLoss
from fairlearn.postprocessing import ThresholdOptimizer

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

#region Counterfactual Data Augmentation (CDA)

def apply_cda_augmentation(
        df: pd.DataFrame,
        text_col: str = "comment_text_processed",
        label_col: str = "labelled_as_toxic",
        identity_config: Optional[Dict[str, Dict[str, Any]]] = None,
        max_augments_per_row: int = 3,
        random_state: Optional[int] = 42
) -> pd.DataFrame:
    """
    Apply Counterfactual Data Augmentation (CDA) to a DataFrame of comments.

    The core logic:
        - Detect identity terms related to sexual orientation (/ gender identity)
          (heterosexual, gay/lesbian, bisexual, transgender, etc.).
        - For each row and each detected identity group, generate a number of
          counterfactual samples (n_augment) where the group's terms are swapped
          with terms from other groups, while the label is kept unchanged.
        - The original DataFrame rows are preserved; new rows are appended.

    CDA is intended to:
        - reduce false correlations between identity terms and toxicity,
        - balance representation across identity subgroups,
        - enforce (approximate) prediction invariance with respect to identity.

    Notes:
        - This function is applied to the *training* set only.
        - The default identity configuration and augmentation strengths are based
          on the subgroup metrics computed earlier on the base model, prioritizing highly
          biased or underrepresented subgroups.
        - After first iteration, a differentiation between toxic and non-toxic
          samples is made to fight higher false positive rates for identity mentions in non-toxic comments.
        - By default, random replacements are used, ensuring that no fixed
          mapping (e.g., always gay -> heterosexual) is encoded.

    Args:
        df (pd.DataFrame): Input DataFrame containing at least the text and label
            columns. The label column is not changed by CDA.
        text_col (str): Name of the column containing the comment text.
        label_col (str): Name of the column containing the toxicity label.
        identity_config (Optional[Dict[str, Dict[str, Any]]]): Optional custom
            identity configuration. If None, a default configuration based on
            the subgroup metrics analysis is used.
        max_augments_per_row (int): Maximum number of augmented samples to
            create per original row (to limit dataset growth).
        random_state (Optional[int]): Optional random seed for reproducibility.

    Returns:
        pd.DataFrame: New DataFrame containing the original rows plus the
            CDA-augmented rows.
    """
    if identity_config is None:
        identity_config = _build_default_identity_config()

    # Prepare RNG and compiled patterns
    rng = np.random.default_rng(random_state)
    identity_patterns = _compile_identity_patterns(identity_config)

    # Initialize logging counters for each configured group.
    augmentation_counters = {group: 0 for group in identity_config.keys()}

    augmented_rows = []

    # Process each row in the DataFrame
    for _, row in df.iterrows():
        # Skip rows with missing text.
        if pd.isna(row[text_col]):
            continue

        # Generate CDA-augmented samples for the current row.
        new_rows = _generate_cda_for_row(
            row=row,
            text_col=text_col,
            label_col=label_col,
            identity_config=identity_config,
            identity_patterns=identity_patterns,
            augmentation_counters=augmentation_counters,
            max_augments_per_row=max_augments_per_row,
            rng=rng
        )
        augmented_rows.extend(new_rows)

    # Simple logging of augmentation statistics.
    print("CDA augmentation summary (number of new samples per group):")
    for group, count in augmentation_counters.items():
        print(f"  {group}: {count}")

    # If no augmentations were created, return the original DataFrame.
    if not augmented_rows:
        return df.copy()

    # Create DataFrame for augmented rows and concatenate
    augmented_df = pd.DataFrame(augmented_rows, columns=df.columns)
    combined_df = pd.concat([df, augmented_df], ignore_index=True) # reset index to have a clean continuous index

    return combined_df

#endregion

#region Fairness-constrained learning (FCL)

def fit_fairness_constrained_pipeline(
    train_df: pd.DataFrame,
    x_train: np.ndarray,
    y_train: np.ndarray,
    identity_cols: List[str],
    best_params: Optional[Dict[str, Any]] = None,
    max_iter: int = 5000,
    constraint_type: Literal["demographic_parity", "bounded_group_loss"] = "demographic_parity",
    upper_bound: float = 0.3,
    eg_max_iter: int = 300,
    random_state: int = 42
) -> ExponentiatedGradient:
    """
    Fit a fairness-constrained logistic regression model on pre-vectorized training data.

    The core logic:
        - Take the *training* split in two forms:
              (a) the original DataFrame (for identity information), and
              (b) the already vectorized feature matrix x_train_vec together
                  with its binary labels y_train.
        - Derive a single identity group label per row from the given identity
          columns (e.g. heterosexual, bisexual, homosexual_gay_or_lesbian,
          transgender, background).
        - Train a logistic regression base model under the specified fairness
          constraint using fairlearn's ExponentiatedGradient reduction, with
          the identity group labels as sensitive_features.
        - Return the fitted fairness-constrained classifier. The TF-IDF
          vectorizer is assumed to be managed outside this function and is not
          modified here.

    This function is intended to:
        - Reuse precomputed TF-IDF features (x_train_vec) and labels (y_train)
          that were saved during the data preparation pipeline, avoiding any
          repeated calls to fit_transform on the vectorizer.
        - Implement an in-processing fairness mitigation technique that can be
          directly compared to the base model trained on the same features.
        - Keep the global pipeline structure consistent: mitigation is applied
          only at the model training stage, while validation and test sets
          remain untouched and are transformed using the previously fitted
          TF-IDF vectorizer.

    Notes:
        - The identity_cols argument should contain the same identity columns
          that are used for subgroup fairness evaluation (e.g. sexual
          orientation and gender identity groups).
        - Validation and test data are not passed into this function; they are
          later evaluated with the returned classifier.

    Args:
        train_df (pd.DataFrame): DataFrame containing the *training* split,
            including the identity columns used to derive sensitive_features.
            The number of rows in train_df must match the number of rows in
            x_train_vec and y_train.
        x_train (np.ndarray): Precomputed TF-IDF feature matrix for the
            training data, as saved in the train_tfidf_bundle joblib file.
        y_train (np.ndarray): Binary target labels (0/1) for the training data,
            aligned with x_train_vec.
        identity_cols (Optional[List[str]]): List of column names corresponding
            to the identity indicators or scores used to derive the
            sensitive_features. If None, a ValueError is raised.
        best_params (Optional[Dict[str, Any]]): Optional dictionary of best
            hyperparameters for the LogisticRegression base estimator, as
            obtained from prior hyperparameter tuning. If provided, the
            parameters 'class_weight', 'C', 'penalty', and 'solver' are taken.
        max_iter (int): Maximum number of iterations for the LogisticRegression
            base estimator. Default is 5000. Increased from typical 1000 to help
            convergence with fairness constraints on imbalanced groups.
        constraint_type (Literal["demographic_parity", "bounded_group_loss"]):
            Type of fairness constraint to enforce. Default is "demographic_parity".
            Use "bounded_group_loss" for datasets with severely imbalanced sensitive
            feature groups (e.g. when smallest group has <100 samples).
        upper_bound (float): For BoundedGroupLoss, the maximum acceptable loss
            difference between demographic groups. Default is 0.3 (30% tolerance).
            Higher values are more lenient and easier to satisfy with imbalanced groups.
        eg_max_iter (int): Maximum iterations for ExponentiatedGradient reduction.
            Default is 300. Increase this if convergence issues persist.
        random_state (int): Random seed for reproducibility of the
            fairness-constrained classifier.

    Returns:
        ExponentiatedGradient: A fitted fairness-constrained classifier that
            can be used to obtain probabilities and predictions on TF-IDF
            feature matrices produced by the existing vectorizer.
    """

    # Derive sensitive group labels from the training DataFrame
    sensitive_features = _build_identity_group_labels(
        df=train_df,
        identity_cols=identity_cols
    )

    # Train the fairness-constrained logistic regression model
    model = _train_fairness_constrained_logistic_regression(
        x_train=x_train,
        y_train=y_train,
        sensitive_features=sensitive_features,
        # use best hyperparameters if provided, else defaults
        class_weight=best_params["class_weight"] if best_params else "balanced",
        C=best_params["C"] if best_params else 1.0,
        penalty=best_params["penalty"] if best_params else "l2",
        solver=best_params["solver"] if best_params else "lbfgs",
        max_iter=max_iter,
        constraint_type=constraint_type,
        upper_bound=upper_bound,
        eg_max_iter=eg_max_iter,
        random_state=random_state
    )

    return model

#endregion

#region Equalized Odds post-processing (EO)

def fit_equalized_odds_postprocessor(
    y_val: np.ndarray,
    y_val_proba: np.ndarray,
    val_df: pd.DataFrame,
    sensitive_attribute: Literal["gender", "sexual_orientation"] = "sexual_orientation",
    identity_threshold: float = 0.5,
    base_threshold: float = 0.5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Fit a Fairlearn Equalized Odds post-processor on validation predictions and labels.

    The core logic:
        - Derive sensitive group labels from val_df for the requested sensitive_attribute
        - Filter validation rows to the subset that is eligible for the EO definition
        - Fit ThresholdOptimizer(constraints="equalized_odds") using precomputed probabilities
        - Return a reusable postprocessor bundle that can be applied to validation/test later

    Equalized Odds is intended to:
        reduce disparities in error rates (true positive rate and false positive rate)
        across the specified sensitive groups, without retraining the base classifier.

    Notes:
        - This method integrates into the global pipeline as a post-processing stage:
          it must be fitted on validation only and then applied to validation/test predictions.
        - The base classifier is not retrained and TF-IDF features are not modified.
        - Validation/test raw texts must not be augmented or altered.
        - This implementation produces hard labels via ThresholdOptimizer.predict(...);
          additional thresholding is therefore unnecessary for included rows.

    Args:
        y_val (np.ndarray): Binary validation labels (0/1), shape (n_samples,).
        y_val_proba (np.ndarray): Base model positive-class probabilities for validation,
            shape (n_samples,), values in [0, 1].
        val_df (pd.DataFrame): Validation DataFrame containing identity columns used to derive
            sensitive group labels.
        sensitive_attribute (Literal): Either "gender" or "sexual_orientation". Defaults to "sexual_orientation".
        identity_threshold (float): Threshold above which an identity is treated as present. Defaults to 0.5.
        base_threshold (float): Fallback threshold used for rows excluded from EO application. Defaults to 0.5.
        random_state (int): Random seed used for reproducible randomized predictions.

    Returns:
        Dict[str, Any]: A postprocessor bundle containing:
            - "postprocessor": fitted ThresholdOptimizer
            - "sensitive_attribute": configuration string
            - "identity_threshold": used threshold for identity presence
            - "base_threshold": fallback threshold for excluded rows
            - "random_state": seed for reproducibility
            - "fit_included_count": number of included rows used to fit EO
            - "fit_excluded_count": number of excluded rows not used to fit EO
    """
    y_val = np.asarray(y_val).reshape(-1)
    y_val_proba = np.asarray(y_val_proba).reshape(-1)

    # Validate input dimensions (sanity check)
    if len(y_val) != len(y_val_proba) or len(y_val) != len(val_df):
        raise ValueError("y_val, y_val_proba, and val_df must have the same number of rows.")

    # Derive sensitive features and inclusion mask based on chosen attribute
    if sensitive_attribute == "gender":
        sensitive, include_mask = _derive_gender_sensitive_features(df=val_df, threshold=identity_threshold)
    elif sensitive_attribute == "sexual_orientation":
        sensitive, include_mask = _derive_orientation_sensitive_features(df=val_df, threshold=identity_threshold)

    # Filter to included rows for EO fitting
    included_idx = include_mask.values
    excluded_count = int((~included_idx).sum())
    included_count = int(included_idx.sum())

    # Ensure there are included rows to fit EO on
    if included_count == 0:
        raise ValueError("No rows are eligible for EO fitting under the chosen sensitive attribute definition.")

    sensitive_included = sensitive.loc[include_mask]

    # Logging
    print("Fitting Equalized Odds post-processor")
    print(f"  sensitive_attribute: {sensitive_attribute}")
    print(f"  identity_threshold: {identity_threshold}")
    print(f"  included rows: {included_count}")
    print(f"  excluded rows: {excluded_count}")

    # Fit ThresholdOptimizer with precomputed probabilities
    estimator = _PrecomputedProbaEstimator()
    postprocessor = ThresholdOptimizer(
        estimator=estimator,
        constraints="equalized_odds",
        objective="accuracy_score",
        prefit=True, # We provide precomputed probabilities
        predict_method="predict_proba"
    )

    # Create input for fitting in the shape (n_samples, 1)
    x_fit = y_val_proba[included_idx].reshape(-1, 1)

    # X is the probability vector; estimator.predict_proba(X) simply returns those probabilities
    postprocessor.fit(
        x_fit,
        y_val[included_idx],
        sensitive_features=sensitive_included
    )

    # Create postprocessor bundle for later application and return it
    bundle = {
        "postprocessor": postprocessor,
        "sensitive_attribute": sensitive_attribute,
        "identity_threshold": identity_threshold,
        "base_threshold": base_threshold,
        "random_state": random_state,
        "fit_included_count": included_count,
        "fit_excluded_count": excluded_count
    }
    return bundle

def apply_equalized_odds_postprocessor(
    postprocessor_bundle: Dict[str, Any],
    y_proba: np.ndarray,
    df: pd.DataFrame
) -> np.ndarray:
    """
    Apply a fitted Equalized Odds post-processor to probability predictions.

    The core logic:
        - Derive sensitive group labels from df for the configured sensitive_attribute
        - Apply ThresholdOptimizer.predict(...) to rows eligible for EO
        - For excluded rows, fall back to base thresholding (y_proba >= base_threshold)
        - Return a full-length prediction vector aligned with the input rows

    Equalized Odds is intended to:
        adjust final decisions to reduce group disparities in true/false positive rates,
        while leaving the base classifier and feature representation unchanged.

    Notes:
        - This method integrates into the global pipeline after the base model has produced
          probabilities for validation/test.
        - It must not retrain the base classifier and must not modify TF-IDF or raw texts.
        - ThresholdOptimizer.predict(...) returns hard labels for included rows, so additional
          thresholding is not applied there.
        - Excluded rows are handled explicitly to ensure downstream evaluation can operate
          on the full dataset.

    Args:
        postprocessor_bundle (Dict[str, Any]): Output of fit_equalized_odds_postprocessor(...).
        y_proba (np.ndarray): Base model positive-class probabilities for the given split,
            shape (n_samples,), values in [0, 1].
        df (pd.DataFrame): DataFrame for the given split containing identity columns used to
            derive sensitive group labels.

    Returns:
        np.ndarray: Hard-label predictions (0/1) for all rows, shape (n_samples,).
    """
    # Unpack postprocessor bundle
    postprocessor = postprocessor_bundle["postprocessor"]
    sensitive_attribute = str(postprocessor_bundle.get("sensitive_attribute", "sexual_orientation"))
    identity_threshold = float(postprocessor_bundle.get("identity_threshold", 0.5))
    base_threshold = float(postprocessor_bundle.get("base_threshold", 0.5))
    rs = postprocessor_bundle.get("random_state", 42)

    # Prepare input probabilities
    y_proba = np.asarray(y_proba).reshape(-1)

    # Validate input dimensions (sanity check)
    if len(y_proba) != len(df):
        raise ValueError("y_proba and df must have the same number of rows.")

    # Derive sensitive features and inclusion mask based on chosen attribute
    if sensitive_attribute == "gender":
        sensitive, include_mask = _derive_gender_sensitive_features(df=df, threshold=identity_threshold)
    elif sensitive_attribute == "sexual_orientation":
        sensitive, include_mask = _derive_orientation_sensitive_features(df=df, threshold=identity_threshold)

    # Filter to included rows for EO application
    included_idx = include_mask.values
    included_count = int(included_idx.sum())
    excluded_count = int((~included_idx).sum())

    # Logging
    print("Applying Equalized Odds post-processor")
    print(f"  sensitive_attribute: {sensitive_attribute}")
    print(f"  identity_threshold: {identity_threshold}")
    print(f"  included rows: {included_count}")
    print(f"  excluded rows (fallback thresholding): {excluded_count}")
    print(f"  fallback base_threshold: {base_threshold}")

    # Initialize prediction array with fallback thresholding
    y_pred = (y_proba >= base_threshold).astype(int)

    # Apply EO post-processor to included rows
    if included_count > 0:
        # Get EO-adjusted predictions for included rows
        sensitive_included = sensitive.loc[include_mask]

        # create input for fitting in the shape (n_samples, 1)
        x_fit = y_proba[included_idx].reshape(-1, 1)
        y_pred_included = postprocessor.predict(
            x_fit,
            sensitive_features=sensitive_included,
            random_state=rs
        )

        # Overwrite included rows with EO predictions
        y_pred[included_idx] = np.asarray(y_pred_included).astype(int)

    return y_pred

#endregion

#region Private helper functions for CDA

def _build_default_identity_config() -> Dict[str, Dict[str, Any]]:
    """
    Build the default identity configuration used for CDA.

    The configuration defines:
    - which identity groups are considered,
    - which text tokens belong to each group,
    - which replacement terms are used for counterfactual augmentation,
    - and how many counterfactual samples to generate per sample and group.

    The augmentation strengths below reflect the subgroup statistics observed from the subgroup metrics analysis:
    - bisexual: very small n and strong bias -> 3 augmentations
    - heterosexual: small n and strong bias -> 3 augmentations
    - transgender: moderate n and notable bias -> 2 augmentations
    - homosexual_gay_or_lesbian: large n and high bias -> 1 augmentation

    Returns:
        Dict[str, Dict[str, Any]]: Mapping from group name to its configuration.
    """
    return {
        "heterosexual": {
            "terms": ["heterosexual", "straight"],
            "replacement_terms": ["homosexual", "gay", "lesbian", "bisexual", "transgender"],
            "n_augment": [3, 0]      # [non-toxic, toxic]
        },
        "homosexual_gay_or_lesbian": {
            "terms": ["gay", "lesbian", "homosexual"],
            "replacement_terms": ["heterosexual", "straight", "bisexual", "transgender"],
            "n_augment": [2, 1]       
        },
        "bisexual": {
            "terms": ["bisexual"],
            "replacement_terms": ["heterosexual", "straight", "homosexual", "gay", "lesbian", "transgender"],
            "n_augment": [3, 1],
        },
        "transgender": {
            "terms": ["transgender", "trans"],
            "replacement_terms": ["heterosexual", "straight", "homosexual", "gay", "lesbian", "bisexual"],
            "n_augment": [2, 1],
        },
    }

def _compile_identity_patterns(
    identity_config: Dict[str, Dict[str, Any]]
) -> Dict[str, List[re.Pattern]]:
    """
    Compile regular expression patterns for each identity term in the configuration.
    The patterns are word-boundary anchored to detect identity mentions in text.

    Args:
        identity_config (Dict[str, Dict[str, Any]]): Identity configuration containing "terms" per group.

    Returns:
        Dict[str, List[re.Pattern]]: Mapping from group name to a list of compiled
            regex patterns for its terms.
    """
    compiled = {}

    for group, cfg in identity_config.items():
        patterns = []
        for term in cfg.get("terms", []):
            # Use word boundaries and to find standalone tokens.
            pattern = re.compile(r"\b" + re.escape(term) + r"\b")
            patterns.append(pattern)
        compiled[group] = patterns
    return compiled

def _find_identity_groups_in_text(
        text: str,
        identity_patterns: Dict[str, List[re.Pattern]]
) -> List[str]:
    """
    Identify which identity groups are present in a given text based on regex patterns.

    Args:
        text (str): Input comment text.
        identity_patterns (Dict[str, List[re.Pattern]]): Mapping from group name
            to compiled regex patterns for each group's terms.

    Returns:
        List[str]: List of group names that are detected in the text
            (at least one pattern matches).
    """
    present_groups = []
    for group, patterns in identity_patterns.items():
        if any(p.search(text) for p in patterns):
            present_groups.append(group)
    return present_groups

def _sample_replacement_term(
        source_group: str,
        identity_config: Dict[str, Dict[str, Any]],
        rng: np.random.Generator = np.random.default_rng()
) -> str:
    """
    Randomly sample a replacement identity term from the defined options in the identity configuration.

    Args:
        source_group (str): Name of the identity group to be replaced.
        identity_config (Dict[str, Dict[str, Any]]): Identity configuration mapping.
        rng (np.random.Generator): Random number generator instance.

    Returns:
        str: Name of the sampled replacement term.
    """

    # Get candidate replacement terms from the identity configuration for the source group
    candidates = identity_config.get(source_group, {}).get("replacement_terms", [])

    if not candidates:
        # Fallback: if there are no replacement terms, return the source group (no-op).
        return source_group
    
    return str(rng.choice(candidates))

def _swap_identity_terms_in_text(
        text: str,
        source_group: str,
        identity_config: Dict[str, Dict[str, Any]],
        identity_patterns: Dict[str, List[re.Pattern]],
        rng: np.random.Generator = np.random.default_rng()
) -> str:
    """
    Create a counterfactual version of the text by swapping identity terms of the
    given source group with randomly selected terms from another group.

    The label is assumed to remain unchanged (CDA assumption), so this function
    only modifies the text.

    Notes:
        - All terms of the source group that match via regex are replaced.
        - The target group term is sampled randomly from the group's configured "replacement_terms".

    Args:
        text (str): Original comment text.
        source_group (str): Name of the identity group to be swapped.
        identity_config (Dict[str, Dict[str, Any]]): Identity configuration.
        identity_patterns (Dict[str, List[re.Pattern]]): Compiled regex patterns for each group.
        rng (np.random.Generator): Random number generator instance.

    Returns:
        str: Counterfactual text with swapped identity terms.
    """
    if source_group not in identity_config:
        # If the group is unknown, return the original text
        return text

    # Sample a replacement term for the source group
    replacement_term = _sample_replacement_term(source_group, identity_config, rng)

    # Copy the text and apply replacements for all patterns of the source group
    new_text = text
    for pattern in identity_patterns.get(source_group, []):
        new_text = pattern.sub(replacement_term, new_text)

    return new_text

def _generate_cda_for_row(
        row: pd.Series,
        text_col: str,
        label_col: str,
        identity_config: Dict[str, Dict[str, Any]],
        identity_patterns: Dict[str, List[re.Pattern]],
        augmentation_counters: Dict[str, int],
        max_augments_per_row: int = 3,
        rng: np.random.Generator = np.random.default_rng()
) -> List[Dict[str, Any]]:
    """
    Generate CDA-augmented samples for a single row.

    For each identity group detected in the row's text, this function creates a
    number of counterfactual samples equal to the group's "n_augment" setting
    in `identity_config`. The original row is not modified here; this function
    only returns new synthetic rows.

    Additionally, the function updates `augmentation_counters` to reflect how
    many augmented samples are created per identity group.

    Args:
        row (pd.Series): Single DataFrame row representing one comment.
        text_col (str): Column name containing the comment text.
        label_col (str): Name of the column containing the toxicity label.
        identity_config (Dict[str, Dict[str, Any]]): Identity configuration.
        identity_patterns (Dict[str, List[re.Pattern]]): Compiled regex patterns for each group.
        augmentation_counters (Dict[str, int]): Dictionary to count the number of
            augmentations created per identity group.
        max_augments_per_row (int): Maximum number of augmented samples to
            create per original row (to limit dataset growth).
        rng (np.random.Generator): Random number generator instance.

    Returns:
        List[Dict[str, Any]]: List of dictionaries representing new augmented rows
            (same columns as the input row, but with modified text_col).
    """
    text = str(row[text_col])

    # Find all identity groups present in the text
    present_groups = _find_identity_groups_in_text(text, identity_patterns)

    augmented_rows = []
    if not present_groups:
        # If no identity groups are present in the text, no augmentations are created
        return augmented_rows

    # Iterate over each detected group and create group-specific augmentations
    for group in present_groups:
        # Determine the toxicity label index (0 or 1) for augmentation count lookup
        toxicity_label_index = int(row[label_col])  # assuming binary labels 0/1

        # Determine how many augmentations to create for this group
        n_augment = int(identity_config.get(group, {}).get("n_augment", [0, 0])[toxicity_label_index])

        if n_augment <= 0:
            continue

        for _ in range(n_augment):
            cf_text = _swap_identity_terms_in_text(
                text=text,
                source_group=group,
                identity_config=identity_config,
                identity_patterns=identity_patterns,
                rng=rng,
            )
            # If the swapping did not change the text, we can skip it to avoid duplicates
            if cf_text == text:
                continue

            # Create a new row dictionary copying all original columns
            new_row = row.to_dict()
            new_row[text_col] = cf_text
            augmented_rows.append(new_row)

            # Update logging counters
            augmentation_counters[group] = augmentation_counters.get(group, 0) + 1

    # Limit the number of augmentations per row to avoid excessive growth
    if len(augmented_rows) > max_augments_per_row:
        # Randomly select a subset of augmentations to keep
        selected_indices = rng.choice(
            len(augmented_rows),
            size=max_augments_per_row,
            replace=False
        )
        augmented_rows = [augmented_rows[i] for i in selected_indices]

    return augmented_rows

#endregion

#region Private helper functions for FCL

def _build_identity_group_labels(
        df: pd.DataFrame,
        identity_cols: List[str],
        threshold: float = 0.5
) -> pd.Series:
    """
    Build a single categorical identity group label per row from multiple identity columns.
    The function checks each identity column in the specified order, and assigns
    the first identity that meets or exceeds the threshold. If no identity meets
    the threshold, the label 'background' is assigned.

    Notes:
        - The order of identity_cols matters. The first identity that meets
          the threshold is used.
        - The threshold can be adapted to align with how identity scores or
          probabilities are interpreted in the dataset.

    Args:
        df (pd.DataFrame): Input DataFrame containing at least the identity
            columns used to derive the group labels.
        identity_cols (List[str]): List of column names corresponding to identity
            indicators or scores (e.g. ["heterosexual", "bisexual", ...]).
        threshold (float): Threshold above which an identity value is considered
            present for a row. Defaults to 0.5.

    Returns:
        pd.Series: A Series of string labels with the same index as df, where
            each entry is either one of the identity column names or
            'background'.
    """
    labels = []

    for _, row in df[identity_cols].iterrows():
        label = "background"
        for col in identity_cols:
            value = row[col]
            if value >= threshold:
                label = col
                break
        labels.append(label)

    return pd.Series(labels, index=df.index, name="identity_group")

def _train_fairness_constrained_logistic_regression(
        x_train: np.ndarray,
        y_train: np.ndarray,
        sensitive_features: pd.Series,
        class_weight: str = "balanced",
        C: float = 1.0,
        penalty: Literal["l1", "l2", "elasticnet", None] = "l2",
        solver: Literal["newton-cg", "lbfgs", "liblinear", "sag", "saga"] = "lbfgs",
        max_iter: int = 5000,
        constraint_type: Literal["demographic_parity", "bounded_group_loss"] = "demographic_parity",
        upper_bound: float = 0.3,
        eg_max_iter: int = 300,
        random_state: int = 42
) -> ExponentiatedGradient:
    """
    Train a fairness-constrained logistic regression model using ExponentiatedGradient.
    
    The ExponentiatedGradient reduction from Fairlearn is used to enforce
    the specified fairness constraint during training.

    Constraint Types:
        - "demographic_parity": Enforces equal positive prediction rates across
          all sensitive groups. Strict and brittle with severely imbalanced groups.
        - "bounded_group_loss": Penalizes the worst-off group rather than demanding
          equality. More forgiving for imbalanced groups.

    Notes:
        - The base estimator remains logistic regression to keep the model
          family consistent with the base pipeline.
        - The ExponentiatedGradient reduction optimizes a mixture of models
          to satisfy the fairness constraints while maintaining performance.

    Args:
        x_train (np.ndarray): TF-IDF feature matrix for the training data.
        y_train (np.ndarray): Binary target labels (0/1) for the training data.
        sensitive_features (pd.Series): One-dimensional Series of identity
            group labels (e.g. 'background', 'heterosexual', 'bisexual', ...),
            aligned with x_train and y_train.
        class_weight (str): Class weight strategy for handling class imbalance.
        C (float): Inverse regularization strength for the LogisticRegression
            base estimator. Smaller values specify stronger regularization.
        penalty (Literal["l1", "l2", "elasticnet", None]): Type of regularization to use.
        solver (Literal["newton-cg", "lbfgs", "liblinear", "sag", "saga"]): Algorithm to use in the optimization problem.
        max_iter (int): Maximum number of iterations for the LogisticRegression base estimator.
            Set to 5000 as default to help convergence with fairness constraints on imbalanced groups.
        constraint_type (Literal["demographic_parity", "bounded_group_loss"]):
            Type of fairness constraint. Default is "demographic_parity".
        upper_bound (float): For BoundedGroupLoss, the maximum acceptable loss
            difference between demographic groups. Default is 0.3 (30% tolerance).
        eg_max_iter (int): Maximum iterations for ExponentiatedGradient reduction
            algorithm. Default is 300. Increase if convergence issues persist.
        random_state (int): Random seed for reproducibility of both the base
            estimator and the ExponentiatedGradient reduction.

    Returns:
        ExponentiatedGradient: A fitted fairness-constrained classifier that
            can be used for predict_proba / predict on TF-IDF feature matrices.
    """

    base_estimator = LogisticRegression(
        class_weight=class_weight,
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state
    )

    # Instantiate the appropriate constraint based on constraint_type
    if constraint_type == "demographic_parity":
        constraint = DemographicParity()
    elif constraint_type == "bounded_group_loss":
        # ZeroOneLoss() is appropriate for binary classification
        # upper_bound controls tolerance for loss difference between groups
        constraint = BoundedGroupLoss(loss=ZeroOneLoss(), upper_bound=upper_bound)
    else:
        raise ValueError(
            f"Unknown constraint_type: {constraint_type}. "
            "Choose from 'demographic_parity' or 'bounded_group_loss'."
        )

    expo_gradient = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=constraint,
        max_iter=eg_max_iter
    )

    expo_gradient.fit(x_train, y_train, sensitive_features=sensitive_features)

    return expo_gradient

#endregion

# region Private helper functions/classes for EO (post-processing)

class _PrecomputedProbaEstimator:
    """
    Minimal estimator wrapper that treats X as a precomputed probability vector.

    ThresholdOptimizer expects an estimator and will call predict_proba(X) when
    predict_method='predict_proba'. This wrapper enables fitting and predicting
    purely from already computed probabilities (without retraining a base model).
    """

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "_PrecomputedProbaEstimator":
        return self # no-op fit

    # Format X as 2D array with shape (n_samples, 2) for predict_proba
    # where column 0 is prob of class 0 and column 1 is prob of class 1
    # This is done because the input X is assumed to be a 1D array of probabilities for class 1
    def predict_proba(self, X: Any) -> np.ndarray:
        x = np.asarray(X).reshape(-1)
        x = np.clip(x, 0.0, 1.0)
        return np.column_stack([1.0 - x, x])
    
def _derive_gender_sensitive_features(
    df: pd.DataFrame,
    threshold: float = 0.5
) -> Tuple[pd.Series, pd.Series]:
    """
    Derive sensitive features for gender Equalized Odds (female vs male only).

    Args:
        df (pd.DataFrame): DataFrame containing at least the columns "female" and "male"
        threshold (float): threshold above which an identity value is considered present for a row

    Returns:
        sensitive_features: values in {"female", "male"} for included rows; "excluded" otherwise
        include_mask: True for rows included in EO fitting/application
    """
    # Create boolean flags for gender identities based on threshold
    female_flag = (df["female"] > threshold)
    male_flag = (df["male"] > threshold)

    both_flag = female_flag & male_flag
    any_flag = female_flag | male_flag

    # Rows to include (only one of gender flags is True)
    include_mask = any_flag & (~both_flag)

    # Create sensitive feature Series with "excluded" default
    sensitive = pd.Series("excluded", index=df.index, name="sensitive_gender")
    sensitive.loc[female_flag & include_mask] = "female"
    sensitive.loc[male_flag & include_mask] = "male"

    print("Equalized Odds sensitive feature derivation (gender)")
    print(f"  included female: {int((sensitive == 'female').sum())}")
    print(f"  included male: {int((sensitive == 'male').sum())}")
    print(f"  excluded none: {int((~any_flag).sum())}")
    print(f"  excluded both: {int(both_flag.sum())}")

    return sensitive, include_mask

def _derive_orientation_sensitive_features(
    df: pd.DataFrame,
    threshold: float = 0.5
) -> Tuple[pd.Series, pd.Series]:
    """
    Derive sensitive features for sexual orientation Equalized Odds:
    homosexual_gay_or_lesbian, transgender vs background (no identity mention at all).

    Args:
        df (pd.DataFrame): DataFrame containing at least the column "homosexual_gay_or_lesbian" and "has_identity"
        threshold (float): threshold above which an identity value is considered present for a row

    Returns:
        sensitive_features: values in {"homosexual_gay_or_lesbian", "transgender", "background"} for included rows;
            "excluded" otherwise
        include_mask: True for rows included in EO fitting/application
    """
    # Create boolean flags for sexual orientation identities based on threshold
    has_any_identity = (df["has_identity"]).astype(bool)
    hgl_flag = df["homosexual_gay_or_lesbian"] > threshold
    trans_flag = df["transgender"] > threshold
    bisexual_flag = df["bisexual"] > threshold

    # background: no lgbtq-identity mention at all
    background_flag = ~has_any_identity

    # Rows to include (either hgl, trans, bisexual, or background)
    include_mask = hgl_flag | trans_flag | bisexual_flag | background_flag

    # Create sensitive feature Series with "excluded" default
    sensitive = pd.Series("excluded", index=df.index, name="sensitive_orientation")
    sensitive.loc[hgl_flag] = "homosexual_gay_or_lesbian"
    sensitive.loc[trans_flag] = "transgender"
    sensitive.loc[bisexual_flag] = "bisexual"
    sensitive.loc[background_flag] = "background"

    print("Equalized Odds sensitive feature derivation (sexual orientation)")
    print(f"  included homosexual_gay_or_lesbian: {int((sensitive == 'homosexual_gay_or_lesbian').sum())}")
    print(f"  included transgender: {int((sensitive == 'transgender').sum())}")
    print(f"  included bisexual: {int((sensitive == 'bisexual').sum())}")
    print(f"  included background (no identity mention): {int((sensitive == 'background').sum())}")
    print(f"  excluded other identity mentions: {int((sensitive == 'excluded').sum())}")

    return sensitive, include_mask

# endregion