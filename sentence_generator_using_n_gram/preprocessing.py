from typing import List


def tokenize_sentences_into_words(sentences: str) -> List[str]:
    """step one"""
    pass

def remove_punctuation_from_tokens(tokens: List[str]) -> List[str]:
    """step two"""

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
