"""
This file contains the main function that
handles other modules and classifies each word with a
'FIL', 'ENG', or 'OTH' tag
"""

import os
import pickle
from typing import List
import feature_extractor as fe

MODEL_FILENAME = 'pinoybot_model_f1_validated_depth_11.pkl'
ENCODER_FILENAME = 'pinoybot_encoder_depth_11.pkl'

# List of features in the same order as the model
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

# The list of features that return a string value
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

# Global variables
_MODEL = None
_ENCODER = None
_EXTRACTOR = fe

def tag_language(tokens: List[str]) -> List[str]:
    """
    Tags each token in the input list with its predicted language
    """
    global _MODEL, _ENCODER

    # Load the trained decision tree model, if cannot be found, display error
    if _MODEL is None:
        if not os.path.exists(MODEL_FILENAME):
            raise FileNotFoundError(f"[Model file '{MODEL_FILENAME}' not found]")
        with open(MODEL_FILENAME, 'rb') as f:
            _MODEL = pickle.load(f)

    # Load the feature encoder, if cannot be found, display error
    if _ENCODER is None:
        if not os.path.exists(ENCODER_FILENAME):
            raise FileNotFoundError(f"[Encoder file '{ENCODER_FILENAME}' not found]")
        with open(ENCODER_FILENAME, 'rb') as f:
            _ENCODER = pickle.load(f)

    # Converts the raw data into a feature matrix (DataFrame with rows as each token and column as each feature)
    df_features = fe.extract_features(tokens)

    # Encode categorical features into numbers that the model can use
    try:
        df_features_encoded = df_features.copy()
        df_features_encoded[CATEGORICAL_COLS] = _ENCODER.transform(df_features[CATEGORICAL_COLS])
    except Exception as e:
        print(f"[Error: Failed to encode features. {e}]")
        return ['OTH'] * len(tokens)

    # Reorder DataFrame columns to match FEATURE_COLS's order
    try:
        df_features_final = df_features_encoded[FEATURE_COLS]
    except KeyError as e:
        print(f"[Error: Extractor is missing features required by the model: {e}]")
        return ['OTH'] * len(tokens)

    # Predict the language of each string using the trained model
    try:
        predictions = _MODEL.predict(df_features_final)
        return list(predictions)
    except Exception as e:
        print(f"[Error: {e}]")
        return ['OTH'] * len(tokens)

# Tester
if __name__ == "__main__":
    example_tokens = ["Cup", "Baso", "Ballpen", "Ginagamit"]
    print(f"Tokens: {example_tokens}")

    try:
        tags = tag_language(example_tokens)
        print(f"Tags:   {tags}")
    except Exception as e:
        print(e)