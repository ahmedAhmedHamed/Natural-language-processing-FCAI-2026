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
    # tokens = tokenize_sentences_into_words(sentences)
    tokens = remove_punctuation_from_tokens(sentences)
    # tokens = remove_stop_words(tokens)
    tokens = convert_all_tokens_to_lower_case(tokens)
    tokens = build_a_set_of_vocabulary_from_pre_processed_corpus(tokens)
    return tokens


##################### Dictionary Building #####################

def get_sentences_from_brown(limit=200_000):
    sentences = []
    count = 0
    
    for sentence in brown.sents():
        sentences.append(sentence)
        count += len(sentence)
        
        if count >= limit:
            break
    
    return sentences    # list of sentences, each sentence is a list of tokens

def build_ngram_dict(corpus, n):
    sentences = [remove_punctuation_from_tokens(sen) for sen in corpus]
    sentences = [convert_all_tokens_to_lower_case(sen) for sen in sentences]
    ngrams = {}

    for sentence in sentences:
        tokens = sentence.copy()
        tokens = ["<s>"] * (n - 1) + tokens + ["</s>"]  # add start and end tokens

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

    starting_grams = [list(gram[:-1]) for gram in ngram_dict.keys() 
                      if "<s>" not in gram[:-1] and "</s>" not in gram[:-1]]

    for _ in range(m):
        current_context = list(random.choice(starting_grams))
        sentence = list(current_context)

        for _ in range(max_len - len(current_context)):
            best_word = None
            max_prob = -1
            used_ngrams = set() # to avoid repetition
            candidates = []

            for word in vocabulary:
                test_gram = tuple(current_context + [word])
                count_ngram = ngram_dict.get(test_gram, 0)
                
                if count_ngram and test_gram not in used_ngrams:
                    candidates.append((word, count_ngram))
            
            used_ngrams.add(tuple(current_context + [best_word]))
            
            if not candidates:
                break

            for word, count_ngram in candidates:
                if count_ngram > max_prob:
                    max_prob = count_ngram
                    best_word = word

            if best_word == "</s>":
                break

            sentence.append(best_word)
            current_context = sentence[-(n-1):]
                
        generated_sentences.append(" ".join(sentence))
    
    return generated_sentences


##################### Application #####################

if __name__ == '__main__':

    brown_corpus = get_sentences_from_brown()
    corpus_flattened = [token for sentence in brown_corpus for token in sentence]
    vocabulary = run_preprocessing(corpus_flattened)

    n = 5
    m = 10
    max_len = 10

    ngram_dict = build_ngram_dict(brown_corpus, n)
    generated_sentences = generate_sentences(ngram_dict, n, m, max_len, vocabulary)

    for i, sentence in enumerate(generated_sentences, 1):
        print(f"Sentence {i}: {sentence}\n")
