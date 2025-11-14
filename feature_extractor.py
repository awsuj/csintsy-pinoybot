"""
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
    'in', 'on', 'at', 'of', 'to', 'for', 'with', 'by', 'from', 'and', 'but', 'or',
    'am', 'me', 'him', 'us', 'them', 'this', 'that', 'these', 'those',
    'be', 'been', 'do', 'does', 'did', 'have', 'has', 'had',
    'will', 'would', 'can', 'could', 'should', 'may', 'might', 'must',
    'so', 'if', 'when', 'then', 'because', 'as', 'while', 'than',
    'about', 'after', 'before', 'down', 'out', 'over', 'up', 'under', 'through',
    'not', 'no', 'yes', 'all', 'any', 'some', 'very', 'just', 'now',
    'here', 'there', 'why', 'how', 'what', 'which', 'who', 'whom'
}

FIL_WORDS = {
    'ang', 'mga', 'sa', 'ng', 'na', 'pa', 'ba', 'ay', 'si', 'ni', 'kay', 'kina',
    'ako', 'ikaw', 'siya', 'tayo', 'kami', 'sila', 'ito', 'iyan', 'iyon',
    'ko', 'mo', 'niya', 'namin', 'nila', 'atin', 'inyo', 'at', 'o', 'pero', 'hindi',
    'nang', 'nina', 'sina', 'natin', 'po', 'amin',
    'din', 'rin', 'daw', 'raw', 'lang', 'nga', 'naman', 'man', 'kasi', 'yata', 'muna',
    'kayo', 'kanila', 'kaniya', 'dito', 'diyan', 'doon', 'nito', 'niyan', 'niyon',
    'kaya', 'para', 'dahil', 'habang', 'kapag', 'kung', 'saka',
    'may', 'meron', 'wala', 'dapat', 'talaga', 'mismo'
}

def f_get_language(token):
    """
    Checks if a token is an English word, a Filipino word, or NONE of the above.
    """
    # Convert to string and lowercase it once
    token_lower = str(token).lower()

    if token_lower in ENG_WORDS:
        return "ENG"
    if token_lower in FIL_WORDS:
        return "FIL"

    return "NONE"

def f_oth_filter(token):
    """
    Filters tokens with OTH characteristics:
        Numeric
        Symbol
        Laughter
        Abbreviations
    """
    token_str = str(token)

    """
        Remove all the commas in the token
        Try to convert the cleaned token into a float
        If it works
            return "NUMERIC"
        If it does not work
            pass
        EX: 100,000.00 -> 100000.00 (success), 10 -> 10 (success),
            Hello -> Hello (error), . -> . (error)
    """
    cleaned_str = token_str.replace(",", "")
    try:
        float(cleaned_str)
        return "NUMERIC"
    except ValueError:
        pass

    """
        If token is just punctuation or symbols
            return "SYMBOL"
        The 'backslash W' means all symbols except '_', '_' is added for comparison
    """
    if re.fullmatch(r'[\W_]+', token_str):
        return "SYMBOL"

    """
        If 'haha' or 'hehe' exists in the token
            return "LAUGHTER"
    """
    token_lower = token_str.lower()
    if 'haha' in token_lower or 'hehe' in token_lower:
        return "LAUGHTER"

    """
        If the token is mane of letters and is fully capitalized
            return "ABB"
    """
    if any(c.isalpha() for c in token_str) and token_str.isupper():
        return "ABB"

    return "REGULAR"

def f_has_pair_vowel_word_duplication (token):
    """
    Filters tokens with FIL characteristics:
        Word duplication
        Pair duplication
        Vowel duplication
    """
    token_str = str(token).lower()

    """
        The token is first split on the '-' if applicable

        If parts is split into 2, and both parts the same
            return "WORD_dupe"
            EX: araw-araw, sino-sino, etc.
    """
    parts = token_str.split('-')
    if len(parts) == 2 and parts[0] == parts[1]:
        return "WORD_dupe"

    """
        If there is a group of 2 letters that repeat in succession
            return "PAIR_dupe"
            EX: tatakbo, nagtatanim, etc.
    """
    if re.search(r'([a-z]{2})\1', token_str):
        return "PAIR_dupe"

    """
        If there is a vowel that repeats in succession
            return "VOWEL_dupe"
            EX: umiiyak, nag-aaral, etc.
    """
    token_str = str(token).lower()
    if re.search(r'([aeiou])\1', token_str):
        return "VOWEL_dupe"

    return "NO_dupe"

def f_prefix_fil(token):
    """
    Filters tokens with FIL prefix characteristics:
        maki-
        paki-
        naki-
        pala-
        mala-
        pang-
        mag-
        nag-
        pag-
        um-
        in-
        ni-
        ma-
        pa-
        na-
        ng-
    """
    token_str = str(token).lower()
    t_len = len(token_str)

    # Filter for 4 width prefix
    if t_len > 4:
        """
            If token starts with 'maki'
                return "MAKI"
                EX: makikain, makisali, makipaglaro, etc.
        """
        if token_str.startswith('maki'):
            return "MAKI"

        """
            If token starts with 'paki'
                return "PAKI"
                EX: pakibasa, pakisabi, pakisama, etc.
        """
        if token_str.startswith('paki'):
            return "PAKI"

        """
            If token starts with 'naki'
                return "NAKI"
                EX: nakisakay, nakiinom, nakisama, etc.
        """
        if token_str.startswith('naki'):
            return "NAKI"

        """
            If token starts with 'pala'
                return "PALA"
                EX: palangiti, palabiro, palatawa, etc.
        """
        if token_str.startswith('pala'):
            return "PALA"

        """
            If token starts with 'mala'
                return "MALA"
                EX: malahayop, malaibon, malaanghel, etc.
        """
        if token_str.startswith('mala'):
            return "MALA"

        """
            If token starts with 'pang'
                return "PANG"
                EX: pangkamay, pangligo, pang-abay, etc.
        """
        if token_str.startswith('pang'):
            return "PANG"

    # Filter for 3 width prefix
    if t_len > 3:
        """
            If token starts with 'mag'
                return "MAG"
                EX: magluto, magtatanim, maglilinis, etc.
        """
        if token_str.startswith('mag'):
            return "MAG"

        """
            If token starts with 'nag'
                return "NAG"
                EX: nagbayad, naglalaba, nagsasayaw, etc.
        """
        if token_str.startswith('nag'):
            return "NAG"

        """
            If token starts with 'pag'
                return "PAG"
                EX: pagkain, pagpunta, pag-aaral, etc.
        """
        if token_str.startswith('pag'):
            return "PAG"

    # Filter for 2 width prefix
    if t_len > 2:
        """
            If token starts with 'um' and the third letter is a vowel
                return "UM"
                EX: umalis, umiyak, etc.
        """
        if token_str.startswith('um') and token_str[2] in 'aeiou':
            return "UM"

        """
            If token starts with 'in' and the third letter is a vowel
                return "IN"
                EX: inilagay, inabot, etc.
        """
        if token_str.startswith('in') and token_str[2] in 'aeiou':
            return "IN"

        """
            If token starts with 'ni' and the third letter is 'l'
                return "NI"
                EX: niluto, nilinis, nilakad, etc.
        """
        if token_str.startswith('ni') and token_str[2] == 'l':
            return "NI"

        """
            If token starts with 'ma'
                return "MA"
                EX: malakas, maganda, maingay, etc.
        """
        if token_str.startswith('ma'):
            return "MA"

        """
            If token starts with 'pa'
                return "PA"
                EX: paalis, pakain, papunta, etc.
        """
        if token_str.startswith('pa'):
            return "PA"

        """
            If token starts with 'na'
                return "NA"
                EX: natapon, nabasa, nabasag, etc.
        """
        if token_str.startswith('na'):
            return "NA"

        """
            If the token starts with 'ng'
                return "NG"
                EX: ngunit, ngiti, ngayon, etc.
        """
        if token_str.startswith('ng'):
            return "NG"

    return "NONE"

def f_infix_fil(token):
    """
    Filters tokens with FIL infix characteristics:
        -in-
        -um-
        -ng-
    """
    token_str = str(token).lower()
    t_len = len(token_str)

    # If token starts with a vowel and its length is more than 3
    if token_str[0] not in 'aeiou' and t_len > 3:
        """
            If the token starts with a consonant and contains 'in' in the 2nd-3rd letter
                return "IN"
                EX: kinain, tinanim, pinalo etc.
        """
        if token_str[1:3] == 'in':
            return "IN"

        """
            If the token starts with a consonant and contains 'um' in the 2nd-3rd letter
                return "UM"
                EX: pumunta, kumuha, tumawa etc.
        """
        if token_str[1:3] == 'um':
            return "UM"

    # If token is longer than 3
    if t_len > 3:
        """
            If 'ng' fits in the middle of the token, starting from index 1 to index 2
                return "NG"
                EX: pangalan, malungkot, mangga, etc.
        """
        if 'ng' in token_str[1:-1]:
            return "NG"

    return "NONE"

def f_suffix_fil(tokens):
    """
    Filters tokens with FIL suffix characteristics:
        -in
        -an
    """
    token_str = str(tokens).lower()

    # If token is bigger than 3 and 3rd to the last letter is 'u' or a consonant
    if len(token_str) > 3 and (token_str[-3] == 'u' or token_str[-3] not in 'aeio'):
        """
            If the token ends in 'in'
                return 1
                EX: kainin, lutuin, kapitin, etc.
        """
        if token_str.endswith('in'):
            return "IN"

        """
            If the token ends in 'an'
                return "AN"
                EX: palayan, puntahan, damitan, etc.
        """
        if token_str.endswith('an'):
            return "AN"

    return "NONE"

def f_eng_bigrams(token):
    """
    Filters tokens with ENG bigrams characteristics:
        th
        sh
        ch
        wh
        ck
        qu
        ion
    """
    token_str = str(token).lower()

    """
        If the token contains 'th' in it
            return "TH"
            EX: the, mother, threatened, etc.
    """
    if 'th' in token_str:
        return "TH"

    """
        If the token contains 'sh' in it
            return "SH"
            EX: shape, shrapnel, sheep, etc.
    """
    if 'sh' in token_str:
        return "SH"

    """
        If the token contains 'ch' in it
            return "CH"
            EX: church, chicken, child, etc.
    """
    if 'ch' in token_str:
        return "CH"

    """
        If the token contains 'wh' in it
            return "WH"
            EX: when, weather, which, etc.
    """
    if 'wh' in token_str:
        return "WH"

    """
        If the token contains 'ck' in it
            return "CK"
            EX: chicken, peck, duck, etc.
    """
    if 'ck' in token_str:
        return "CK"

    """
        If the token contains 'qu' in it
            return "QU"
            EX: quack, queen, quiz, etc.
    """
    if 'qu' in token_str:
        return "QU"

    """
        If the token contains 'ion' in it
            return "ION"
            EX: action, motion, emotion, etc.
    """
    if 'ion' in token_str:
        return "ION"

    return "NONE"

def f_get_suffix_eng(token):
    """
    Filters tokens with ENG suffixes characteristics:
        -tion
        -sion
        -ment
        -ness
        -able
        -ible
        -less
        -ing
        -ful
        -ity
        -es
        -ed
        -er
        -est
        -ly
        -s
        -y
    """
    token_str = str(token).lower()
    t_len = len(token_str)

    # Filter for 4 width prefix
    if t_len > 4:
        """
        If the token ends with 'tion'
            return "TION"
            EX: action, motion, emotion
        """
        if token_str.endswith('tion'):
            return 'TION'

        """
        If the token ends with 'sion'
            return "SION"
            EX: precision, confusion, vision
        """
        if token_str.endswith('sion'):
            return 'SION'

        """
        If the token ends with 'ment'
            return "MENT"
            EX: moment, payment, contentment
        """
        if token_str.endswith('ment'):
            return 'MENT'

        """
        If the token ends with 'ness'
            return "NESS"
            EX: happiness, kindness, darkness
        """
        if token_str.endswith('ness'):
            return 'NESS'

        """
        If the token ends with 'able'
            return "ABLE"
            EX: reachable, comfortable, doable
        """
        if token_str.endswith('able'):
            return 'ABLE'

        """
        If the token ends with 'ible'
            return "IBLE"
            EX: visible, terrible, flexible
        """
        if token_str.endswith('ible'):
            return 'IBLE'

        """
        If the token ends with 'less'
            return "LESS"
            EX: hopeless, useless, careless
        """
        if token_str.endswith('less'):
            return 'LESS'

    # Filter for 3 width prefix
    if t_len > 3:
        """
        If the token ends with 'ing'
            return "ING"
            EX: walking, talking, coding
        """
        if token_str.endswith('ing'):
            return 'ING'

        """
        If the token ends with 'ful'
            return "FUL"
            EX: beautiful, wonderful, painful
        """
        if token_str.endswith('ful'):
            return 'FUL'

        """
        If the token ends with 'ity'
            return "ITY"
            EX: ability, flexibility, city
        """
        if token_str.endswith('ity'):
            return 'ITY'

        """
        If the token ends with 'est'
            return "EST"
            EX: biggest, fastest, strongest
        """
        if token_str.endswith('est'):
            return 'EST'

    # Filter for 2 width prefix
    if t_len > 2:
        """
        If the token ends with 'es'
            return "ES"
            EX: boxes, wishes, goes
        """
        # 'es' must be checked before 's'
        if token_str.endswith('es'):
            return 'ES'

        """
        If the token ends with 'ed'
            return "ED"
            EX: walked, talked, coded
        """
        if token_str.endswith('ed'):
            return 'ED'

        """
        If the token ends with 'er'
            return "ER"
            EX: teacher, worker, faster
        """
        if token_str.endswith('er'):
            return 'ER'

        """
        If the token ends with 'ly'
            return "LY"
            EX: quickly, slowly, happily
        """
        if token_str.endswith('ly'):
            return 'LY'

    # Filter for 1 width prefix
    if t_len > 1:
        """
        If the token ends with 's' (and not 'es')
            return "S"
            EX: cats, dogs, runs
        """
        if token_str.endswith('s'):
            return 'S'

        """
        If the token ends with 'y'
            return "Y"
            EX: happy, sleepy, party
        """
        if token_str.endswith('y'):
            return 'Y'

    return 'NONE'

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
        return 100

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
        ('aeiou', backslash d = '0-9', backslash W = all symbols, '_')
        3 times in a row, or 3 consonants in a row
    If the patter exists in the token
        return 1
    """
    token_str = str(token).lower()
    consonant_pattern = r'[^aeiou\d\W_]{3,}'

    if re.search(consonant_pattern, token_str):
        return 1
    return 0

def f_is_capitalized_mid_sentence(token, index, tokens):
    """
    Test if the token is a capitalized mid-sentence word (Named-Entity)
    If The first letter is not capitalized
        return 0
    If it is the first token of the list
        return 0
    If the previous token is a sentence ender symbol
        return 0
    otherwise, return 1
    """
    token_str = str(token)
    if not token_str[0].isupper():
        return 0
    if index == 0:
        return 0
    prev_token = str(tokens[index - 1])
    if prev_token in ('.', '!', '?'):
        return 0
    return 1

def f_first_letter_ascii(token):
    """
    Returns the ASCII value of the first letter of the token.
    """
    return ord(token[0])

def f_last_letter_ascii(token):
    """
    Returns the ASCII value of the last letter of the token.
    """
    return ord(token[-1])

def extract_features(tokens: List[str]) -> pd.DataFrame:
    """
    Takes a list of tokens and converts it into a feature matrix.

    It returns a dataframe (2d array) where each row is a token and each column is a feature.
    """
    token_only_features = [
        f_get_language,
        f_oth_filter,
        f_has_pair_vowel_word_duplication,
        f_prefix_fil,
        f_infix_fil,
        f_suffix_fil,
        f_eng_bigrams,
        f_get_suffix_eng,
        f_contains_letters_cfjqvxz,
        f_a_ratio,
        f_k_ratio,
        f_e_ratio,
        f_vowel_consonant_ratio,
        f_has_consonant_cluster,
        f_first_letter_ascii,
        f_last_letter_ascii
    ]

    contextual_features = [
        f_is_capitalized_mid_sentence
    ]

    """
    Create a features list to store features temporarily
    For every token, a temporary features list will contain the returns of:
        token only features, and contextual_features (need index i) 
        It will then append the features list for the single token to the all_features_list
    Convert the all_features_list into a dataframe (feature matrix), then return it
    """
    all_features_list = []

    for i, token in enumerate(tokens):
        features = {}

        for func in token_only_features:
            features[func.__name__] = func(token)
        for func in contextual_features:
            features[func.__name__] = func(token, i, tokens)

        all_features_list.append(features)

    X = pd.DataFrame(all_features_list)
    return X