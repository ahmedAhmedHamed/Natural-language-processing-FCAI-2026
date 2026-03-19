import string
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.corpus import brown

nltk.download('punkt')  # for tokenization
nltk.download('stopwords')  # for lemmatization
nltk.download('punkt_tab')
nltk.download('brown')

##################### Preprocessing #####################

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


##################### Dictionary Building #####################

def get_tokens_from_brown(limit=200_000):
    tokens = []
    
    for sentence in brown.sents():
        tokens.extend(sentence)
        
        if len(tokens) >= limit:
            break
    
    return tokens[:limit]

def build_ngram_dict(corpus, n):
    tokens = remove_punctuation_from_tokens(corpus)
    tokens = convert_all_tokens_to_lower_case(tokens)
    # brown corpus is already tokenized
    # the set step is affecting the n-gram generation order-wise, so I discarded it, besides there is no need for uniqueness in the n-gram
    # and for the stop words, it was canceled in the last remarks ¯\_(ツ)_/¯
    # - Noran
    ngrams = {}

    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i+n])
        
        if gram in ngrams:
            ngrams[gram] += 1
        else:
            ngrams[gram] = 1

    return ngrams

if __name__ == '__main__':

    example = "The quick brown fox jumps over the lazy dog, and the dog barked loudly!"
    print('before: --- :', example)
    tokenized_example = run_preprocessing(example)
    print('after: --- :', tokenized_example)

    brown_tokens = get_tokens_from_brown()
    print(f"\n-> Total tokens from Brown corpus: {len(brown_tokens)}")
    trigrams = build_ngram_dict(brown_tokens, 3)
    print(f"-> Dictionary: {len(trigrams)} n-grams found")
    for k, v in trigrams.items():
        if(v > 20):     # to avoid large number of n-grams
            print(k, ":", v)
