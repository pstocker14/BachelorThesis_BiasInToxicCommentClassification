from __future__ import annotations
from typing import Iterable, List, Optional, Sequence, Tuple

import re
import regex
import emoji
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# incomplete list of words to surely keep during text preprocessing (even if they are stopwords or low-frequency)
KEEP_WORDS = {
    # Pronouns and gendered references
    "she", "her", "hers", "he", "him", "his",
    "they", "them", "their", "theirs",
    "ze", "zir", "zie", "hir", "xe", "xem", "xyr",
    "we", "us", "our", "ours", "i", "me", "my", "mine",
    "you", "your", "yours",

    # Gender-related terms
    "woman", "women", "female", "girl", "girls", "lady", "ladies",
    "man", "men", "male", "boy", "boys", "gentleman", "gentlemen",
    "feminine", "masculine", "gender", "sex", "sexes",
    "sexism", "sexist", "misogyny", "misogynist",

    # LGBTQ+ and orientation terms
    "gay", "gays", "lesbian", "lesbians", "bisexual", "bi", "queer",
    "lgbt", "lgbtq", "lgbtqia", "lgbtqiap", "lgbtqplus",
    "trans", "transgender", "transsexual", "transition", "transitioning",
    "nonbinary", "non-binary", "genderfluid", "genderqueer", "agender",
    "intersex", "asexual", "pansexual", "homosexual", "heterosexual",
    "straight", "ally", "allies", "pride", "comingout", "coming-out",

    # Identity context and descriptors
    "feminist", "feminism", "queerphobic", "homophobic", "transphobic",
    "biphobic", "heteronormative", "equality", "diversity", "inclusion",
    "discrimination", "identity", "orientation", "pronoun", "pronouns",
    "respect", "rights", "representation",

    # Titles and address forms
    "miss", "mrs", "mr", "ms", "mx", "sir", "madam", "lady", "gentleman",

    # Family / relationship / social identity terms
    "wife", "husband", "spouse", "partner", "couple",
    "boyfriend", "girlfriend", "fiancé", "fiancee",
    "mother", "mom", "mum", "father", "dad", "parent",
    "sister", "brother", "daughter", "son", "child", "children",
    "family", "marriage", "married", "wedding",

    # Group / identity references
    "community", "minority", "majority", "group", "people",
    "womanhood", "manhood", "sisterhood", "brotherhood",

    # Equality and social justice related
    "rights", "fairness", "justice", "inclusion", "diversity",
    "equality", "equity", "representation", "bias", "discrimination",
}

# Set of English stopwords excluding the KEEP_WORDS
try:
    STOPWORDS = set(stopwords.words("english")) - KEEP_WORDS
except LookupError:
    from nltk import download
    download("stopwords")
    STOPWORDS = set(stopwords.words("english")) - KEEP_WORDS

# Precompiled regex patterns for text cleaning
_URL = re.compile(r"http\S+") # matches http:// or https:// URLs
_MENTION = re.compile(r"@\S+") # matches @mentions
_HASHTAG = re.compile(r"#\w+") # matches hashtags
_NON_ALPHA = regex.compile(r"[^\p{L}'\s]") # matches non-letter characters, apostrophes, and whitespaces (using regex module for Unicode support)
_WS = re.compile(r"\s+") # matches sequences of whitespace

# region Data splitting

def split_dataset(df: pd.DataFrame, identity_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits the dataset into training, testing, and validation sets, 
    stratified by toxicity label and identity mentions."""

    df["label"] = (df["target"] >= 0.5).astype(int) # Create binary label based on target threshold of 0.5

    df["has_identity_for_stratification"] = (df[identity_cols] > 0.5).any(axis=1).astype(int) # Create binary column indicating presence of any identity mention

    df["strat_key"] = df["label"].astype(str) + "_" + df["has_identity_for_stratification"].astype(str) # Create stratification key

    # This gives four possible groups:
    #   0_0 → non-toxic, no identity mention
    #   1_0 → toxic, no identity mention
    #   0_1 → non-toxic, mentions identity
    #   1_1 → toxic, mentions identity
    
    # Split the training data into a training and test/validation set, stratified by the strat_key
    train_data, test_data = train_test_split(df, test_size=0.3, random_state=42, stratify=df["strat_key"])

    # Split the test data part even further into a test and a validation set, stratified by the strat_key
    test_data, validation_data = train_test_split(test_data, test_size=0.5, random_state=42, stratify=test_data["strat_key"])

    # Drop helper column
    for d in (df, train_data, validation_data, test_data):
        d.drop(columns=["strat_key", "has_identity_for_stratification", "label"], inplace=True)

    # Reset the index of the dataframes
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    validation_data = validation_data.reset_index(drop=True)

    return train_data, test_data, validation_data

def reduce_dataset_size(
    df: pd.DataFrame,
    sample_size: int,
    identity_cols: list[str],
    drop_index: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reduces the dataset size to a specified sample size, stratified by identity mentions."""

    df["label"] = (df["target"] >= 0.5).astype(int) # Create binary label based on target threshold of 0.5

    df["has_identity_for_stratification"] = (df[identity_cols] > 0.5).any(axis=1).astype(int) # Create binary column indicating presence of any identity mention

    df["strat_key"] = df["has_identity_for_stratification"].astype(str) # Create stratification key

    test_size = 1 - (sample_size / len(df)) # Calculate the test size proportion to discard

    # Split the data into sampled and discarded sets, stratified by the strat_key
    sampled_df, discarded_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df["strat_key"])

    # Drop helper column
    for d in (df, sampled_df, discarded_df):
        d.drop(columns=["strat_key", "has_identity_for_stratification", "label"], inplace=True)
    # Reset the index of the dataframes
    sampled_df = sampled_df.reset_index(drop=drop_index) # let caller decide to keep original index as a column called 'index'
    discarded_df = discarded_df.reset_index(drop=drop_index)

    return sampled_df, discarded_df

#endregion

# region Text preprocessing

def clean_comment(comment_text: object) -> str:
    """Light-to-moderate cleaning tailored for online comments.
    - lowercases
    - removes URLs, @mentions, hashtags, emojis
    - keeps letters and spaces only
    - collapses whitespace
    """

    if not isinstance(comment_text, str):
        return ""
    
    text_processed = comment_text.lower()
    text_processed = _URL.sub(" ", text_processed)
    text_processed = _MENTION.sub(" ", text_processed)
    text_processed = _HASHTAG.sub(" ", text_processed)
    text_processed = emoji.replace_emoji(text_processed, replace=" ")
    text_processed = _NON_ALPHA.sub(" ", text_processed)
    text_processed = _WS.sub(" ", text_processed)

    return text_processed.strip()

#endregion

#region Column preparations

def binarize_labels(
    df: pd.DataFrame,
    threshold: float = 0.5,
    target_col: str = "target",
    new_col_name: str = "labelled_as_toxic"
) -> pd.DataFrame:
    """Add a binary label column from a continuous target score in [0,1]."""
    
    df[new_col_name] = (df[target_col] >= threshold).astype(int)
    return df

#endregion

#region Vectorizer preparation

def build_tfidf_vectorizer(
    max_features: Optional[int] = None,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 5,
    max_df: float = 0.9,
    stop_words: Optional[list[str]] = list(STOPWORDS),
    dtype: str = "float32"
) -> TfidfVectorizer:
    """Configure a TF-IDF vectorizer for toxic comment classification."""
    
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        lowercase=False,  # text is already lowercased in preprocessing
        strip_accents=None,
        min_df=min_df,
        max_df=max_df,
        stop_words=stop_words,
        dtype=dtype,
    )

#endregion