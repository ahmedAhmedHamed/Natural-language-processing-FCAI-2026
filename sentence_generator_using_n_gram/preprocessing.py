"""
all functions can be used independently, but only run_preprocessing should be used in this assignment.
"""

import string
from typing import List
import nltk
from nltk.corpus import stopwords

nltk.download('punkt')  # for tokenization
nltk.download('stopwords')  # for lemmatization
nltk.download('punkt_tab')


def tokenize_sentences_into_words(sentences: str) -> List[str]:
    """step one"""
    tokens = nltk.word_tokenize(sentences)
    return tokens


def remove_punctuation_from_tokens(tokens: List[str]) -> List[str]:
    """step two"""
    tokens_without_punctuation = [token for token in tokens if token not in string.punctuation]
    return tokens_without_punctuation


def remove_stop_words(tokens: List[str]) -> List[str]:
    """step three"""
    english_stopwords = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in english_stopwords]
    return tokens

def convert_all_tokens_to_lower_case(tokens: List[str]) -> List[str]:
    """step four"""
    tokens = [token.lower() for token in tokens]
    return tokens

def build_a_set_of_vocabulary_from_pre_processed_corpus(tokens: List[str]) -> List[str]:
    """step five"""
    vocabulary_set = set(tokens)
    tokens = list(vocabulary_set)
    return tokens

def run_preprocessing(sentences: str):
    tokens = tokenize_sentences_into_words(sentences)
    tokens = remove_punctuation_from_tokens(tokens)
    tokens = remove_stop_words(tokens)
    tokens = convert_all_tokens_to_lower_case(tokens)
    # doing the set step as the first step would be faster, but I am doing this to be faithful to the assignment description
    # - Ahmed
    tokens = build_a_set_of_vocabulary_from_pre_processed_corpus(tokens)
    return tokens

if __name__ == '__main__':

    example = "The quick brown fox jumps over the lazy dog, and the dog barked loudly!"
    print('before: --- :', example)
    tokenized_example = run_preprocessing(example)
    print('after: --- :', tokenized_example)
