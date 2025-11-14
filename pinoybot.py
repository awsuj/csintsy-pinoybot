"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier
This script includes the feature extraction logic to ensure compatibility
with the 'pinoybot_model_f1_validated_depth_20_2.pkl' model.
"""

import os
import pickle
import re
import pandas as pd
from typing import List
import feature_extractor as fe

# --- CONFIGURATION ---
MODEL_FILENAME = 'pinoybot_model_f1_validated_depth_20.pkl'

# Exact feature order required by the model
FEATURE_COLS = [
    "f_is_english_word", "f_is_filipino_word", "f_is_laughter_expression",
    "f_is_fully_capitalized", "f_is_pure_symbol", "f_is_numeric",
    "f_has_dash_duplication", "f_has_pair_duplication", "f_has_vowel_duplication",
    "f_prefix_um", "f_prefix_in", "f_prefix_ni", "f_prefix_ma", "f_prefix_pa",
    "f_prefix_na", "f_prefix_mag", "f_prefix_nag", "f_prefix_pala",
    "f_prefix_mala", "f_prefix_pang", "f_infix_in", "f_infix_um",
    "f_suffix_in", "f_suffix_an", "f_startswith_ng", "f_has_pair_ng",
    "f_has_pair_th", "f_contains_letters_cfjqvxz", "f_a_ratio",
    "f_k_ratio", "f_e_ratio", "f_vowel_consonant_ratio",
    "f_has_consonant_cluster", "f_is_capitalized_mid_sentence",
    "f_first_letter_ascii", "f_last_letter_ascii", "f_common_eng_bigrams",
]

# Global variable for the model
_MODEL = None
_EXTRACTOR = fe

def tag_language(tokens: List[str]) -> List[str]:
    """
    Tags each token in the input list with its predicted language.
    """
    global _MODEL

    # 1. Load model if not already loaded
    if _MODEL is None:
        if not os.path.exists(MODEL_FILENAME):
            raise FileNotFoundError(f"Model file '{MODEL_FILENAME}' not found in directory.")
        with open(MODEL_FILENAME, 'rb') as f:
            _MODEL = pickle.load(f)

    # 2. Extract features using the embedded extractor
    # This returns a DataFrame with correct column names
    df_features = fe.extract_features(tokens)

    # 3. Reorder columns to strictly match the model's training order
    # This step prevents the KeyError and ensures valid predictions
    try:
        df_features = df_features[FEATURE_COLS]
    except KeyError as e:
        print(f"Critical Error: Extractor is missing features required by the model: {e}")
        return ['OTH'] * len(tokens)

    # 4. Predict
    try:
        predictions = _MODEL.predict(df_features)
        return list(predictions)
    except Exception as e:
        print(f"Prediction Error: {e}")
        return ['OTH'] * len(tokens)

if __name__ == "__main__":
    example_tokens = ["Cup", "Baso", "Ballpen", "Ginagamit"]
    print(f"Tokens: {example_tokens}")

    try:
        tags = tag_language(example_tokens)
        print(f"Tags:   {tags}")
    except Exception as e:
        print(e)