import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Set
from scipy.sparse import csr_matrix

def _download_nltk_data() -> None:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')

def tokenize_text(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return word_tokenize(text)

def remove_stopwords_from_tokens(tokens: List[str], stop_words: Set[str]) -> List[str]:
    return [word for word in tokens if word.lower() not in stop_words]

def join_tokens(tokens: List[str]) -> str:
    return " ".join(tokens)

def process_single_review(text: str, stop_words: Set[str]) -> str:
    tokens = tokenize_text(text)
    filtered_tokens = remove_stopwords_from_tokens(tokens, stop_words)
    processed_text = join_tokens(filtered_tokens)
    return processed_text

def step1_preprocess_reviews(reviews_dataframe: pd.DataFrame, text_column_name: str) -> pd.DataFrame:
    _download_nltk_data()
    english_stopwords: Set[str] = set(stopwords.words('english'))
    
    processed_dataframe = reviews_dataframe.copy()
    processed_dataframe[text_column_name] = processed_dataframe[text_column_name].apply(
        lambda text: process_single_review(text, english_stopwords)
    )
    return processed_dataframe

def step2_encode_labels(reviews_dataframe: pd.DataFrame, label_column_name: str) -> Tuple[pd.DataFrame, LabelEncoder]:
    encoded_dataframe = reviews_dataframe.copy()
    label_encoder = LabelEncoder()
    encoded_dataframe[label_column_name] = label_encoder.fit_transform(encoded_dataframe[label_column_name])
    return encoded_dataframe, label_encoder

def step3_apply_vectorization(reviews_dataframe: pd.DataFrame, text_column_name: str) -> Tuple[csr_matrix, TfidfVectorizer]:
    tfidf_vectorizer = TfidfVectorizer()
    feature_vectors = tfidf_vectorizer.fit_transform(reviews_dataframe[text_column_name])
    return feature_vectors, tfidf_vectorizer

def step4_split_data(feature_vectors: csr_matrix, target_labels: pd.Series) -> Tuple[csr_matrix, csr_matrix, pd.Series, pd.Series]:
    features_training_set, features_testing_set, labels_training_set, labels_testing_set = train_test_split(
        feature_vectors, target_labels, test_size=0.20, random_state=42
    )
    return features_training_set, features_testing_set, labels_training_set, labels_testing_set

def run_pipeline(csv_file_path: str = 'amazon_reviews.csv', 
                 text_column_name: str = 'cleaned_review', 
                 label_column_name: str = 'sentiments') -> Tuple[csr_matrix, csr_matrix, pd.Series, pd.Series, LabelEncoder, TfidfVectorizer]:
    reviews_dataframe = pd.read_csv(csv_file_path)
    
    reviews_dataframe = step1_preprocess_reviews(reviews_dataframe, text_column_name)
    
    reviews_dataframe, dataset_label_encoder = step2_encode_labels(reviews_dataframe, label_column_name)
    
    feature_vectors, dataset_tfidf_vectorizer = step3_apply_vectorization(reviews_dataframe, text_column_name)
    
    target_labels = reviews_dataframe[label_column_name]
    features_training_set, features_testing_set, labels_training_set, labels_testing_set = step4_split_data(feature_vectors, target_labels)

    print(f"Original dataset shape: {reviews_dataframe.shape}")
    print(f"Training set: {features_training_set.shape[0]} samples")
    print(f"Testing set: {features_testing_set.shape[0]} samples")
    
    return features_training_set, features_testing_set, labels_training_set, labels_testing_set, dataset_label_encoder, dataset_tfidf_vectorizer

if __name__ == "__main__":
    run_pipeline()
