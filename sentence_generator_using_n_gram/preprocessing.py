import string
from typing import List
import nltk
from nltk.corpus import stopwords

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
    tokens = [token for token in tokens if token not in stopwords]
    return tokens

def convert_all_tokens_to_lower_case(tokens: List[str]) -> List[str]:
    """step four"""
    tokens = [token.lower() for token in tokens]
    return tokens

def build_a_set_of_vocabulary_from_pre_processed_corpus(tokens: List[str]) -> set(str):
    """step five"""
    vocabulary_set = set(tokens)
    return vocabulary_set

def run_preprocessing(sentences: str):
    tokens = tokenize_sentences_into_words(sentences)
    tokens = remove_punctuation_from_tokens(tokens)
    tokens = remove_stop_words(tokens)
    tokens = convert_all_tokens_to_lower_case(tokens)
    # doing the set step as the first step would be faster, but I am doing this to be faithful to the assignment description
    # - Ahmed
    vocabulary_set = build_a_set_of_vocabulary_from_pre_processed_corpus(tokens)
    return vocabulary_set

if __name__ == '__main__':
    print(string.punctuation)
