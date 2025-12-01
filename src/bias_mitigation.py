from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import re

#region CDA

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
        - This function is typically applied to the *training* set only.
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

    for _, row in df.iterrows():
        # Skip rows with missing text.
        if pd.isna(row[text_col]):
            continue

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