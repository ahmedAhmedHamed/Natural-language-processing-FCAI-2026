import string
import random
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


##################### Sentence Generation #####################

def generate_sentences(ngram_dict, n, m, max_len, vocabulary):
    """
    Step Three: Sentence Generator
    - Starts with a random n-1 gram
    - Selects the word with the highest probability
    - Stops at max_len
    """
    # - Farouk
    generated_sentences = []

    starting_grams = [list(gram[:-1]) for gram in ngram_dict.keys()]

    for _ in range(m):
        current_context = list(random.choice(starting_grams))
        sentence = list(current_context)

        for _ in range(max_len - len(current_context)):
            best_word = None
            max_prob = -1

            for word in vocabulary:
                test_gram = tuple(current_context + [word])
                count_ngram = ngram_dict.get(test_gram, 0)
                
                if count_ngram > max_prob:
                    max_prob = count_ngram
                    best_word = word
            
            if best_word:
                sentence.append(best_word)
                current_context = sentence[-(n-1):]
            else:
                break
                
        generated_sentences.append(" ".join(sentence))
    
    return generated_sentences
