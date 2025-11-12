"""
feature_extractor.py

This module contains all feature functions for pinoybot
It has a function called extract_features
Which takes a list of tokens and converts it into a feature matrix (a pandas DataFrame)
That can be fed into the decision tree
"""

import pandas as pd
import re
from typing import List

# Lists of 100% correct words, that are regularly used
ENG_WORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'he', 'she', 'it', 'they',
    'you', 'we', 'i', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
    'in', 'on', 'at', 'of', 'to', 'for', 'with', 'by', 'from', 'and', 'but', 'or'
}

FIL_WORDS = {
    'ang', 'mga', 'sa', 'ng', 'na', 'pa', 'ba', 'ay', 'si', 'ni', 'kay', 'kina',
    'ako', 'ikaw', 'siya', 'tayo', 'kami', 'sila', 'ito', 'iyan', 'iyon',
    'ko', 'mo', 'niya', 'namin', 'nila', 'atin', 'inyo', 'at', 'o', 'pero', 'hindi',
    'nang', 'nina'
}

def f_is_english_word(token):
    """
    Check if it's an english word
    """
    token_str = str(token)
    if str(token).lower() in ENG_WORDS:
        return 1
    return 0

def f_is_filipino_word(token):
    """
    Checks if it's a filipino word
    """
    token_str = str(token)
    if str(token).lower() in FIL_WORDS:
        return 1
    return 0

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
    Remove all the commas in the token
    Try to convert the cleaned token into a float
    If it works
        return 1
    If it does not work
        return 0
    EX: 100,000.00 -> 100000.00 (success), 10 -> 10 (success),
        Hello -> Hello (error), . -> . (error)
    """
    token_str = str(token)
    cleaned_str = token_str.replace(",", "")
    try:
        float(cleaned_str)
        return 1
    except ValueError:
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

def f_prefix_um(token):
    """
    If length of token is less than 5 (it probably won't contain affixes)
        return 0
    If token starts with 'um' and the third letter is a vowel
        return 1
        EX: umalis, umiyak, etc.
    """
    token_str = str(token).lower()
    if len(token_str) < 5:
        return 0
    if token_str.startswith('um') and token_str[2] in 'aeiou':
        return 1
    return 0

def f_prefix_in(token):
    """
    If length of token is less than 5 (it probably won't contain affixes)
        return 0
    If token starts with 'in' and the third letter is a vowel
        return 1
        EX: inilagay, inabot, etc.
    """
    token_str = str(token).lower()
    if len(token_str) < 5:
        return 0
    if token_str.startswith('in') and token_str[2] in 'aeiou':
        return 1
    return 0

def f_prefix_ni(token):
    """
    If length of token is less than 5 (it probably won't contain affixes)
        return 0
    If token starts with 'ni' and the third letter is 'l'
        return 1
        EX: niluto, nilinis, nilakad, etc.
    """
    token_str = str(token).lower()
    if len(token_str) < 5:
        return 0
    if token_str.startswith('ni') and token_str[2] == 'l':
        return 1
    return 0

def f_prefix_ma(token):
    """
    If length of token is less than 5 (it probably won't contain affixes)
        return 0
    If token starts with 'ma'
        return 1
        EX: malakas, maganda, maingay, etc.
    """
    token_str = str(token).lower()
    if len(token_str) < 5:
        return 0
    if token_str.startswith('ma'):
        return 1
    return 0

def f_prefix_pa(token):
    """
    If length of token is less than 5 (it probably won't contain affixes)
        return 0
    If token starts with 'pa'
        return 1
        EX: paalis, pakain, papunta, pakibasa, pakisabi etc.
    """
    token_str = str(token).lower()
    if len(token_str) < 5:
        return 0
    if token_str.startswith('pa') or token_str.startswith('paki'):
        return 1
    return 0

def f_prefix_na(token):
    """
    If length of token is less than 5 (it probably won't contain affixes)
        return 0
    If token starts with 'na'
        return 1
        EX: natapon, nabasa, nabasag, etc.
    """
    token_str = str(token).lower()
    if len(token_str) < 5:
        return 0
    if token_str.startswith('na'):
        return 1
    return 0

def f_prefix_mag(token):
    """
    If length of token is less than 6 (it probably won't contain affixes)
        return 0
    If token starts with 'mag'
        return 1
        EX: magluto, magtatanim, maglilinis, etc.
    """
    token_str = str(token).lower()
    if len(token_str) < 6:
        return 0
    if token_str.startswith('mag'):
        return 1
    return 0

def f_prefix_nag(token):
    """
    If length of token is less than 6 (it probably won't contain affixes)
        return 0
    If token starts with 'nag'
        return 1
        EX: nagbayad, naglalaba, nagsasayaw, etc.
    """
    token_str = str(token).lower()
    if len(token_str) < 6:
        return 0
    if token_str.startswith('nag'):
        return 1
    return 0

def f_prefix_pala(token):
    """
    If length of token is less than 7 (it probably won't contain affixes)
        return 0
    If token starts with 'pala'
        return 1
        EX: palangiti, palabiro, palatawa, etc.
    """
    token_str = str(token).lower()
    if len(token_str) < 7:
        return 0
    if token_str.startswith('pala'):
        return 1
    return 0

def f_prefix_mala(token):
    """
    If length of token is less than 7 (it probably won't contain affixes)
        return 0
    If token starts with 'mala'
        return 1
        EX: malahayop, malaibon, malaanghel, etc.
    """
    token_str = str(token).lower()
    if len(token_str) < 7:
        return 0
    if token_str.startswith('mala'):
        return 1
    return 0

def f_prefix_pang(token):
    """
    If length of token is less than 7 (it probably won't contain affixes)
        return 0
    If token starts with 'pang'
        return 1
        EX: pangkamay, pangligo, pang-abay, etc.
    """
    token_str = str(token).lower()
    if len(token_str) < 7:
        return 0
    if token_str.startswith('pang'):
        return 1
    return 0

def f_infix_in(token):
    """
    If length of token is less than 4 (it probably won't contain affixes)
        return 0
    If token starts with a vowel
        return 0
    If the token starts with a consonant (all starting with vowels are now gone) and
    contains 'in' in the second to third letter
        return 1
        EX: kinain, tinanim, pinalo etc.
    """
    token_str = str(token).lower()
    if len(token_str) < 4:
        return 0
    if token_str[0] in 'aeiou':
        return 0
    if token_str[1:3] == 'in':
        return 1
    return 0

def f_infix_um(token):
    """
    If length of token is less than 4 (it probably won't contain affixes)
        return 0
    If token starts with a vowel
        return 0
    If the token starts with a consonant (all starting with vowels are now gone) and
    contains 'um' in the second to third letter
        return 1
        EX: pumunta, kumuha, tumawa etc.
    """
    token_str = str(token).lower()
    if len(token_str) < 4:
        return 0
    if token_str[0] in 'aeiou':
        return 0
    if token_str[1:3] == 'um':
        return 1
    return 0

def f_suffix_in(token):
    """
    If length of token is less than 4 (it probably won't contain affixes)
        return 0
    If the token ends in 'in' and the third to the last letter is a 'u' or a consonant
        return 1
        EX: kainin, lutuin, kapitin, etc.
    """
    token_str = str(token).lower()
    if len(token_str) < 4:
        return 0
    if token_str.endswith('in'):
        third_last_char = token_str[-3]
        if third_last_char == 'u' or third_last_char not in 'aeio':
            return 1
    return 0

def f_suffix_an(token):
    """
    If length of token is less than 4 (it probably won't contain affixes)
        return 0
    If the token ends in 'an' and the third to the last letter is a 'u' or a consonant
        return 1
        EX: palayan, puntahan, damitan, etc.
    """
    token_str = str(token).lower()
    if len(token_str) < 4:
        return 0
    if token_str.endswith('an'):
        third_last_char = token_str[-3]
        if third_last_char == 'u' or third_last_char not in 'aeio':
            return 1
    return 0

def f_startswith_ng(token):
    """
    If the token starts with 'ng'
        return 1
        EX: ngunit, ngiti, ngayon, etc.
    """
    token_str = str(token).lower()
    if token_str.startswith('ng'):
        return 1
    return 0

def f_has_pair_ng(token):
    """
    If length of token is less than 4, 2 letters cant fit in the middle
        return 0
    If 'ng' fits in the middle of the token, starting from index 1 to the 2nd to the last letter
        return 1
        EX: pangalan, malungkot, mangga, etc.
    """
    token_str = str(token).lower()
    if len(token_str) < 4:
        return 0
    if 'ng' in token_str[1:-1]:
        return 1
    return 0

def f_has_pair_th(token):
    """
    If the token contains 'th' in it
        return 1
        EX: the, mother, threatened, etc.
    """
    token_str = str(token).lower()
    if 'th' in token_str:
        return 1
    return 0

def f_contains_letters_cfjqvxz(token):
    """
    If the token contains any of the letters: (c, f, j, q, v, x, z)
        return 1
        EX: cabbage, jacket, fairy, etc.
    """
    token_str = str(token).lower()
    if any(c in token_str for c in 'cfjqvxz'):
        return 1
    return 0

def f_a_ratio(token):
    """
    Take all the letters from a token and count the number of times 'a' repeats
    If the token does not contain letters
        return 0
    Otherwise return the ratio of letter a's to all the letters
    Note: Filipino words use the letter 'a' the most
    """
    token_str = str(token).lower()
    letters = [c for c in token_str if c.isalpha()]
    if not letters:
        return 0
    a_count = letters.count('a')
    return a_count / len(letters)

def f_k_ratio(token):
    """
    Take all the letters from a token and count the number of times 'k' repeats
    If the token does not contain letters
        return 0
    Otherwise return the ratio of letter k's to all the letters
    Note: Filipino words use the letter 'k' more compared to english
    """
    token_str = str(token).lower()
    letters = [c for c in token_str if c.isalpha()]
    if not letters:
        return 0
    a_count = letters.count('k')
    return a_count / len(letters)

def f_e_ratio(token):
    """
    Take all the letters from a token and count the number of times 'e' repeats
    If the token does not contain letters
        return 0
    Otherwise return the ratio of letter e's to all the letters
    Note: English words use the letter 'e' the most
    """
    token_str = str(token).lower()
    letters = [c for c in token_str if c.isalpha()]
    if not letters:
        return 0
    a_count = letters.count('e')
    return a_count / len(letters)

def f_vowel_consonant_ratio(token):
    """
    Get all the letters of the token
    If there are no letters
        return 0
    Loop through the letters list
    If it's a vowel, increment v_count, else, increment v_count
    If there are no consonants
        return 0
    Otherwise, return ratio of vowels to consonants
    Note: Filipino words have a ratio closer to 1.0
    EX: pupunta = 0.75, isipin = 1.0, string = 0.2, university = 0.6
    """
    token_str = str(token).lower()
    v_count = 0
    c_count = 0
    letters = [c for c in token_str if c.isalpha()]

    if not letters:
        return 0
    for c in letters:
        if c in 'aeiou':
            v_count += 1
        else:
            c_count += 1
    if c_count == 0:
        return 100
    return v_count / c_count

def f_has_consonant_cluster(token):
    """
    consonant_pattern = match everything that is not:
        ('aeiou', \d = '0-9', \w = all symbols, '_')
        3 times in a row, or 3 consonants in a row
    If the patter exists in the token
        return 1
    """
    token_str = str(token).lower()
    consonant_pattern = r'[^aeiou\d\W_]{3,}'

    if re.search(consonant_pattern, token_str):
        return 1
    return 0

def extract_features(tokens: List[str]) -> pd.DataFrame:
    """
    Takes a list of tokens and converts it into a feature matrix.

    It returns a 2d array where each row is a token and each column is a feature.
    """
    feature_functions = [
        f_is_english_word,
        f_is_filipino_word,
        f_is_pure_symbol,
        f_is_numeric,
        f_has_dash_duplication,
        f_has_pair_duplication,
        f_has_vowel_duplication,
        f_prefix_um,
        f_prefix_in,
        f_prefix_ni,
        f_prefix_ma,
        f_prefix_pa,
        f_prefix_na,
        f_prefix_mag,
        f_prefix_nag,
        f_prefix_pala,
        f_prefix_mala,
        f_prefix_pang,
        f_infix_in,
        f_infix_um,
        f_suffix_in,
        f_suffix_an,
        f_startswith_ng,
        f_has_pair_ng,
        f_has_pair_th,
        f_contains_letters_cfjqvxz,
        f_a_ratio,
        f_k_ratio,
        f_e_ratio,
        f_vowel_consonant_ratio,
        f_has_consonant_cluster,
    ]
    """
    Convert list of tokens to a pandas series, makes it easier to apply the functions
    Create a dataframe to hold the features
    Apply each function to the 'words' Series and
    Store the result as a new column in our feature matrix 'X'
    Columns are named after the function names
    """
    words = pd.Series(tokens)
    X = pd.DataFrame()

    for func in feature_functions:
        col_name = func.__name__
        X[col_name] = words.apply(func)

    return X