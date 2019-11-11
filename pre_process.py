# File: implement pre-processing functions (i.e., lookup table, tokenizing punctuations)

import numpy as np
from collections import Counter

def create_lookup_tables(text):
    # lookup tables for vocabulary
    # param text: the text of tv scripts split into words
    # return: tuple of dicts (vocab_to_int, int_to_vocab)
    word_counts = Counter(text)
    # sorting the words from most to least frequent in text occurrence
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    # create int_to_vocab dictionaries
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab

def token_lookup():
    # generates a dictionary to turn punctuations into tokens
    # return: tokenized dictionary where the key is the punctuation and the value is the token

    # Replace punctuation with tokens so we can use them in our model
    token_dict = {}
    token_dict['.'] = '<PERIOD>'
    token_dict[','] = '<COMMA>'
    token_dict['"'] = '<QUOTATION_MARK>'
    token_dict[';'] = '<SEMICOLON>'
    token_dict['!'] = '<EXCLAMATION_MARK>'
    token_dict['?'] = '<QUESTION_MARK>'
    token_dict['('] = '<LEFT_PAREN>'
    token_dict[')'] = '<RIGHT_PAREN>'
    token_dict['-'] = '<DASH>'
    token_dict['\n'] = '<NEW_LINE>'

    return token_dict

