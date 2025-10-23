import os
import re
import json
import emoji
import string
import pickle
import warnings
import contractions

import pandas as pd

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from unicodedata import normalize
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------
# TEXT CLEANING HELPERS
# ---------------------------------------------------------------------

def has_html(text: str) -> bool:
    """Check if text contains HTML tags."""
    return bool(re.search(r"<.*?>", str(text)))

def get_words(s: str) -> list[str]:
    "Get all the words in a sentense using re"
    return re.findall(r"\w+", s)

def get_charecters(s: str) -> list[str]:
    "Get all the charecters in a sentense using re"
    return re.findall(r"\w", s)

def count_words(s: str) -> int:
    "Count all the words in a sentense using re and len"
    return len(re.findall(r"\w+", s))

def count_charecters(s: str) -> int:
    "Count all the charecters in a sentense using re and len"
    return len(re.findall(r"\w", s))
    
def strip_html(text: str) -> str:
    """Remove HTML tags using BeautifulSoup."""
    return BeautifulSoup(str(text), "html.parser").get_text()

def emoji_text(text: str) -> str:
    return emoji.demojize(text)

def normalize_unicode_text(text: str) -> str:
    """Normalize Unicode characters (NFKC)."""
    return normalize("NFKC", str(text))

def replace_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into one space."""
    return re.sub(r"\s+", " ", str(text)).strip()

def expand_contractions(text: str) -> str:
    """Expand common English contractions (don't -> do not)."""
    return contractions.fix(str(text))

def remove_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text)

def remove_control_chars(text: str) -> str:
    """Removing control charecter like \\n"""
    return re.sub(r"[\r\n\t]+", " ", text)
    
def remove_urls(text: str, replacement_text="") -> str:
    """Remove URLs."""
    return re.sub(r'https?://\S+|www\.\S+', replacement_text, str(text))

def remove_emails(text: str, replacement_text="") -> str:
    """Remove Emails."""
    return re.sub(r'\S+@\S+', replacement_text, text)

def normalize_elongations(token: str) -> str:
    """Reduce character elongations (soooo -> soo)."""
    return re.sub(r'(.)\1{2,}', r'\1\1', token)

def tokenize(text, tokenizer=None):
    """
    Tokenizes text into words using either a custom tokenizer or regex fallback.

    Parameters
    ----------
    text : str or list
        The input text to tokenize. If already tokenized (list), returns as-is.
    tokenizer : callable, optional
        A custom tokenizer function (e.g., nlp.tokenizer, word_tokenize).
        Must return a list of tokens.

    Returns
    -------
    list
        List of tokens.
    """
    if isinstance(text, list):
        return text
    
    if tokenizer is not None:
        return tokenizer(text)
    
    return re.findall(r"\b\w+\b", text.lower())

def remove_stopwords(tokens, stopword_set=None):
    """Remove stopwords from token list."""
    if stopword_set is None:
        stopword_set = set(stopwords.words('english'))
    return [tok for tok in tokens if tok.lower() not in stopword_set]

def lemmatize(tokens, lemmatizer=None):
    """Lemmatize token list using WordNetLemmatizer."""
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(tok) for tok in tokens]

def filter_tokens(tokens, min_len=2, alpha_only=True):
    """Filter tokens based on length and alphabetic content."""
    clean_tokens = []
    for tok in tokens:
        if alpha_only and not tok.isalpha():
            continue
        if len(tok) < min_len:
            continue
        clean_tokens.append(tok)
    return clean_tokens

def reconstruct_text(tokens):
    """Rejoin tokens into a cleaned string."""
    return " ".join(tokens)

# ---------------------------------------------------------------------
# QUALITY CHECK HELPERS
# ---------------------------------------------------------------------

def sample_original_vs_cleaned(df, n=20):
    """Return small sample for manual inspection."""
    sample = df.sample(min(n, len(df)), random_state=42)
    return sample[['review', 'cleaned_review']] if 'cleaned_review' in df.columns else sample

def compute_post_clean_stats(df):
    """Compute quick text-level summary stats."""
    stats = {
        "total_reviews": len(df),
        "avg_word_count": df['cleaned_review'].apply(lambda x: len(str(x).split())).mean(),
        "empty_reviews": (df['cleaned_review'].str.strip() == "").sum(),
        "vocab_size_est": len(set(" ".join(df['cleaned_review']).split()))
    }
    return stats
