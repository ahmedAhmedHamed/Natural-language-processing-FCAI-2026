import string
from typing import List
import nltk

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
    pass

def convert_all_tokens_to_lower_case(tokens: List[str]) -> List[str]:
    """step four"""
    pass

def build_a_set_of_vocabulary_from_pre_processed_corpus(tokens: List[str]):
    """step five"""
    pass

def run_preprocessing(sentences: str):
    tokens = tokenize_sentences_into_words(sentences)
    tokens = remove_punctuation_from_tokens(tokens)
    tokens = remove_stop_words(tokens)
    tokens = convert_all_tokens_to_lower_case(tokens)
    vocab_set = build_a_set_of_vocabulary_from_pre_processed_corpus(tokens)
    return vocab_set

if __name__ == '__main__':
    print(string.punctuation)
