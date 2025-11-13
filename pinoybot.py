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

# --- CONFIGURATION ---
MODEL_FILENAME = 'pinoybot_model_f1_validated_depth_20_2.pkl'

# Placeholder dictionaries. For best results, replace these sets
# with actual loads from 'english_words.txt' and 'filipino_words.txt'.
ENGLISH_WORDS = {"love", "apple", "computer", "school", "teacher", "friend", "today"}
FILIPINO_WORDS = {"kita", "kamusta", "ako", "ikaw", "bakit", "hindi", "talaga", "sira", "na", "ang", "ay", "yung", "grabe"}

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
    "f_has_consonant_cluster", "f_is_capitalized_mid_sentence"
]

class InternalFeatureExtractor:
    def get_vowel_consonant_ratio(self, word):
        vowels = "aeiou"
        v_count = sum(1 for c in word.lower() if c in vowels)
        c_count = sum(1 for c in word.lower() if c.isalpha() and c not in vowels)
        if c_count == 0: return v_count
        return v_count / c_count

    def extract_features(self, tokens: List[str]) -> pd.DataFrame:
        feature_list = []
        total_len = len(tokens)

        for i, token in enumerate(tokens):
            t_lower = token.lower()
            def is_true(condition): return 1 if condition else 0

            features = {
                "f_is_english_word": is_true(t_lower in ENGLISH_WORDS),
                "f_is_filipino_word": is_true(t_lower in FILIPINO_WORDS),
                "f_is_laughter_expression": is_true(bool(re.match(r'^(h[aeiou]|j[e])[hjaeiou]*$', t_lower))),
                "f_is_fully_capitalized": is_true(token.isupper()),
                "f_is_pure_symbol": is_true(not any(c.isalnum() for c in token)),
                "f_is_numeric": is_true(token.isnumeric()),
                "f_has_dash_duplication": is_true('-' in t_lower and len(set(t_lower.split('-'))) == 1 if '-' in t_lower else False),
                "f_has_pair_duplication": is_true(bool(re.search(r'(.)\1', t_lower))),
                "f_has_vowel_duplication": is_true(bool(re.search(r'([aeiou])\1', t_lower))),
                "f_prefix_um": is_true(t_lower.startswith('um')),
                "f_prefix_in": is_true(t_lower.startswith('in')),
                "f_prefix_ni": is_true(t_lower.startswith('ni')),
                "f_prefix_ma": is_true(t_lower.startswith('ma')),
                "f_prefix_pa": is_true(t_lower.startswith('pa')),
                "f_prefix_na": is_true(t_lower.startswith('na')),
                "f_prefix_mag": is_true(t_lower.startswith('mag')),
                "f_prefix_nag": is_true(t_lower.startswith('nag')),
                "f_prefix_pala": is_true(t_lower.startswith('pala')),
                "f_prefix_mala": is_true(t_lower.startswith('mala')),
                "f_prefix_pang": is_true(t_lower.startswith('pang')),
                "f_infix_in": is_true('in' in t_lower[1:-1] if len(t_lower) > 2 else False),
                "f_infix_um": is_true('um' in t_lower[1:-1] if len(t_lower) > 2 else False),
                "f_suffix_in": is_true(t_lower.endswith('in')),
                "f_suffix_an": is_true(t_lower.endswith('an')),
                "f_startswith_ng": is_true(t_lower.startswith('ng')),
                "f_has_pair_ng": is_true('ng' in t_lower),
                "f_has_pair_th": is_true('th' in t_lower),
                "f_contains_letters_cfjqvxz": is_true(any(c in 'cfjqvxz' for c in t_lower)),
                "f_a_ratio": t_lower.count('a') / len(t_lower) if len(t_lower) > 0 else 0,
                "f_k_ratio": t_lower.count('k') / len(t_lower) if len(t_lower) > 0 else 0,
                "f_e_ratio": t_lower.count('e') / len(t_lower) if len(t_lower) > 0 else 0,
                "f_vowel_consonant_ratio": self.get_vowel_consonant_ratio(token),
                "f_has_consonant_cluster": is_true(bool(re.search(r'[bcdfghjklmnpqrstvwxyz]{3,}', t_lower))),
                "f_is_capitalized_mid_sentence": is_true(token[0].isupper() and i > 0)
            }
            feature_list.append(features)

        # Convert list of dicts to DataFrame (automatically uses dict keys as column names)
        return pd.DataFrame(feature_list)

# Global variable for the model
_MODEL = None
_EXTRACTOR = InternalFeatureExtractor()

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
    df_features = _EXTRACTOR.extract_features(tokens)

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
    example_tokens = [""]
    print(f"Tokens: {example_tokens}")

    try:
        tags = tag_language(example_tokens)
        print(f"Tags:   {tags}")
    except Exception as e:
        print(e)