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
from sklearn.preprocessing import OrdinalEncoder

# --- CONFIGURATION ---
MODEL_FILENAME = 'pinoybot_model_f1_validated_depth_11.pkl'
ENCODER_FILENAME = 'pinoybot_encoder_depth_11.pkl'

# Exact feature order required by the model
FEATURE_COLS = [
    'f_get_language',
    'f_oth_filter',
    'f_has_pair_vowel_word_duplication',
    'f_prefix_fil',
    'f_infix_fil',
    'f_suffix_fil',
    'f_eng_bigrams',
    'f_get_suffix_eng',
    'f_contains_letters_cfjqvxz',
    'f_a_ratio',
    'f_k_ratio',
    'f_e_ratio',
    'f_vowel_consonant_ratio',
    'f_has_consonant_cluster',
    'f_first_letter_ascii',
    'f_last_letter_ascii',
    'f_is_capitalized_mid_sentence'
]

CATEGORICAL_COLS = [
    'f_get_language',
    'f_oth_filter',
    'f_has_pair_vowel_word_duplication',
    'f_prefix_fil',
    'f_infix_fil',
    'f_suffix_fil',
    'f_eng_bigrams',
    'f_get_suffix_eng'
]

# Global variable for the model
_MODEL = None
_ENCODER = None
_EXTRACTOR = fe

def tag_language(tokens: List[str]) -> List[str]:
    """
    Tags each token in the input list with its predicted language.
    """
    global _MODEL, _ENCODER

    # 1. Load model if not already loaded
    if _MODEL is None:
        if not os.path.exists(MODEL_FILENAME):
            raise FileNotFoundError(f"Model file '{MODEL_FILENAME}' not found. Did you run train_model.py?")
        with open(MODEL_FILENAME, 'rb') as f:
            _MODEL = pickle.load(f)

    # 2. Load ENCODER if not already loaded
    if _ENCODER is None:
        if not os.path.exists(ENCODER_FILENAME):
            raise FileNotFoundError(f"Encoder file '{ENCODER_FILENAME}' not found. Please update train_model.py to save it.")
        with open(ENCODER_FILENAME, 'rb') as f:
            _ENCODER = pickle.load(f)

    # 3. Extract features using the embedded extractor
    # This returns a DataFrame with correct column names
    df_features = fe.extract_features(tokens)

    # 4. NEW STEP: Encode categorical features
    # We must use .transform() here, not .fit_transform()
    try:
        df_features_encoded = df_features.copy()
        df_features_encoded[CATEGORICAL_COLS] = _ENCODER.transform(df_features[CATEGORICAL_COLS])
    except Exception as e:
        print(f"Critical Error: Failed to encode features. {e}")
        return ['OTH'] * len(tokens)

    # 5. Reorder columns to strictly match the model's training order
    try:
        df_features_final = df_features_encoded[FEATURE_COLS]
    except KeyError as e:
        print(f"Critical Error: Extractor is missing features required by the model: {e}")
        return ['OTH'] * len(tokens)

    # 6. Predict using the encoded and reordered DataFrame
    try:
        predictions = _MODEL.predict(df_features_final)
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