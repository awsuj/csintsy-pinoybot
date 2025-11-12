"""
feature_extractor.py

This module contains all feature extraction logic for PinoyBot.
It provides a single function, extract_features, which takes a list
of raw tokens and converts it into a feature matrix (a pandas DataFrame)
that can be fed into a trained scikit-learn model.
"""

import pandas as pd
import re
from typing import List

def f_is_pure_symbol(token):
    """
    If token is just punctuation or symbols
    The '\W' means all symbols except '_', '_' is added for comparison
    """
    token_str = str(token)
    if re.fullmatch(r'[\W_]+', token_str):
        return 1
    return 0


def f_is_numeric(token):
    """
    If the token is a number
    """
    if str(token).isdigit():
        return 1
    return 0


def f_has_dash_duplication(token):
    """
    The token is first split on the '-' if applicable

    If parts is split into 2, and both parts the same
        return 1
        EX: araw-araw, sino-sino, etc.
    """
    token_str = str(token).lower()
    parts = token_str.split('-')

    if len(parts) == 2 and parts[0] == parts[1]:
        return 1
    return 0

def f_has_pair_duplication(token):
    """
    If there is a group of 2 letters that repeat in succession
        return 1
        EX: tatakbo, nagtatanim, etc.
    """
    token_str = str(token).lower()
    if re.search(r'([a-z]{2})\1', token_str):
        return 1
    return 0


def f_has_vowel_duplication(token):
    """
    If there is a vowel that repeats in succession
        return 1
        EX: umiiyak, nag-aaral, etc.
    """
    token_str = str(token).lower()
    if re.search(r'([aeiou])\1', token_str):
        return 1
    return 0

def f_startswith_um(token):
    """
    If length of token is less than 4 (it probably won't contain affixes)
        return 0
    If token starts with 'um' and the third letter is a vowel
        return 1
        EX: umalis, umiyak, etc.
    """
    token_str = str(token).lower()
    if len(token_str) < 4:
        return 0
    if token_str.startswith('um') and token_str[2] in 'aeiou':
        return 1
    return 0

def f_prefix_in(token):
    """
    If length of token is less than 4 (it probably won't contain affixes)
        return 0
    If token starts with 'in' and the third letter is a vowel
        return 1
        EX: inilagay, inabot, etc.
    """
    token_str = str(token).lower()
    if len(token_str) < 4:
        return 0
    if token_str.startswith('in') and token_str[2] in 'aeiou':
        return 1
    return 0

def f_prefix_ni(token):
    """
    If length of token is less than 4 (it probably won't contain affixes)
        return 0
    If token starts with 'ni' and the third letter is 'l'
        return 1
        EX: niluto, nilinis, nilakad, etc.
    """
    token_str = str(token).lower()
    if len(token_str) < 4:
        return 0
    if token_str.startswith('ni') and token_str[2] == 'l':
        return 1
    return 0

def f_prefix_ma(token):
    """
    If length of token is less than 4 (it probably won't contain affixes)
        return 0
    If token starts with 'ma'
        return 1
        EX: malakas, maganda, maingay, etc.
    """
    token_str = str(token).lower()
    if token_str.startswith('ma'):
        return 1
    return 0

def f_prefix_pa(token):
    """
    If length of token is less than 4 (it probably won't contain affixes)
        return 0
    If token starts with 'pa'
        return 1
        EX: paalis, pakain, papunta, etc.
    """
    token_str = str(token).lower()
    if token_str.startswith('pa'):
        return 1
    return 0

def f_prefix_na(token):
    """
    If length of token is less than 4 (it probably won't contain affixes)
        return 0
    If token starts with 'na'
        return 1
        EX: natapon, nabasa, nabasag, etc.
    """
    token_str = str(token).lower()
    if token_str.startswith('na'):
        return 1
    return 0

def f_prefix_mag(token):
    """
    If length of token is less than 4 (it probably won't contain affixes)
        return 0
    If token starts with 'pa'
        return 1
        EX: paalis, pakain, papunta, etc.
    """
    token_str = str(token).lower()
    if token_str.startswith('pa'):
        return 1
    return 0

def f_prefix_nag(token):

"""
suffix an and in:
when the root word ends with "o", change o to "u" and add the suffix
for suffix "in" and "an" the 3rd tot he last letter is a consonant or "u"

when the start of the root word is a vowel, the infix becomes a prefix
that means when the start of the token is a consonant, and the next 2
characters are "in" (and maybe "um"?) its pob filipino

"""


def f_has_filipino_affix_signal(token):
    """
    This function checks for:
    1. Prefix 'um-' (e.g., umalis)
    2. Prefix 'in-' (e.g., inilagay)
    3. Prefix 'ni-' (the '-in-' rule for 'L' roots, e.g., niluto)
    4. Infix '-um-' after first letter (e.g., kumain)
    5. Infix '-in-' after first letter (e.g., sinulat)
    """
    token_str = str(token).lower()

    if len(token_str) < 4:
        return 0

    # Rule 1 & 2: Prefix for vowel-starting roots
    # (e.g., umalis, umiyak, inilagay, inabot)
    if token_str.startswith('um') or token_str.startswith('in'):
        return 1

    # Rule 3: The 'ni-' prefix rule you found
    # (e.g., niluto, nilinis, nilakad)
    if token_str.startswith('ni'):
        return 1

    # Rule 4 & 5: Infix for consonant-starting roots
    # (e.g., kumain, bumili, sinulat, ginawa)
    potential_infix = token_str[1:3]
    if potential_infix == 'um' or potential_infix == 'in':
        return 1

    # [YES/NO] Starts with 'ng' (strong FIL signal).
    if str(token).lower().startswith('ng'):
        return 1

    # If none of these patterns matched, it's a 0
    return 0

# contains 'ng', ex. maingay, etc.

def f_is_standalone_ng(token):
    """[YES/NO] Is the *entire token* 'ng' (strong FIL signal)."""
    return 1 if str(token).lower() == 'ng' else 0


def f_has_loanword_letter(token):
    """[YES/NO] Contains letters C, F, J, Q, V, X, Z."""
    token_str = str(token).lower()
    return 1 if any(c in token_str for c in 'cfjqvxz') else 0


def f_has_digraph_th(token):
    """[YES/NO] Contains 'th' (strong ENG signal)."""
    return 1 if 'th' in str(token).lower() else 0


def f_is_circumfix_ka_an(token):
    """[YES/NO] Starts with 'ka' AND ends with 'an'."""
    token_str = str(token).lower()
    return 1 if token_str.startswith('ka') and token_str.endswith('an') else 0


def f_vowel_a_ratio(token):
    """[RATIO] Ratio of 'a's to all letters."""
    token_str = str(token).lower()
    letters = [c for c in token_str if c.isalpha()]
    if not letters:
        return 0
    a_count = letters.count('a')
    return a_count / len(letters)


# ----------------------------------------------------
# THE "MAIN" WRAPPER FUNCTION YOU ASKED FOR
# ----------------------------------------------------

def extract_features(tokens: List[str]) -> pd.DataFrame:
    """
    Takes a list of raw tokens and converts it into a
    feature matrix (DataFrame).

    Args:
        tokens: A list of string tokens (e.g., ['Love', 'kita', '.'])

    Returns:
        A pandas DataFrame where each row is a token and each
        column is a feature.
    """

    # List of all the feature functions to run
    feature_functions = [
        f_is_pure_symbol,
        f_is_numeric,
        f_has_reduplication,
        f_has_infix_um,
        f_starts_with_ng,
        f_is_standalone_ng,
        f_has_loanword_letter,
        f_has_digraph_th,
        f_is_circumfix_ka_an,
        f_vowel_a_ratio
        # ---
        # ADD ALL YOUR NEW FEATURES HERE!
        # ---
    ]

    # Convert the list of tokens into a pandas Series
    # This makes it easy to .apply() our functions
    words = pd.Series(tokens)

    # Create an empty DataFrame to hold our features
    X = pd.DataFrame()

    # Apply each function to the 'words' Series and store the
    # result as a new column in our feature matrix 'X'
    for func in feature_functions:
        col_name = func.__name__  # e.g., "f_is_pure_symbol"
        X[col_name] = words.apply(func)

    return X