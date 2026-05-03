import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from typing import List, Tuple, Set, Dict
from scipy.sparse import csr_matrix

def _download_nltk_data() -> None:
    nltk_resources = {
        'corpora/stopwords': 'stopwords',
        'tokenizers/punkt': 'punkt',
        'tokenizers/punkt_tab': 'punkt_tab'
    }

    for resource_path, resource_name in nltk_resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)

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

def step5_train_classifiers(features_training_set: csr_matrix, labels_training_set: pd.Series) -> Dict[str, object]:
    trained_models: Dict[str, object] = {
        "SVM": LinearSVC(),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }

    for model in trained_models.values():
        model.fit(features_training_set, labels_training_set)

    return trained_models

def step6_print_classification_reports(
    trained_models: Dict[str, object],
    features_testing_set: csr_matrix,
    labels_testing_set: pd.Series,
    dataset_label_encoder: LabelEncoder
) -> None:
    target_names = list(dataset_label_encoder.classes_)

    for model_name, model in trained_models.items():
        predictions = model.predict(features_testing_set)
        print(f"\nClassification Report - {model_name}")
        print(
            classification_report(
                labels_testing_set,
                predictions,
                target_names=target_names,
                zero_division=0
            )
        )

def step7_predict_user_review(
    trained_model: object,
    dataset_tfidf_vectorizer: TfidfVectorizer,
    dataset_label_encoder: LabelEncoder,
    stop_words: Set[str]
) -> None:
    user_review = input("\nEnter a new review to classify (or press Enter to skip): ").strip()
    if not user_review:
        print("No input entered. Skipping user review prediction.")
        return

    processed_user_review = process_single_review(user_review, stop_words)
    user_vector = dataset_tfidf_vectorizer.transform([processed_user_review])
    predicted_label_id = trained_model.predict(user_vector)
    predicted_label_name = dataset_label_encoder.inverse_transform(predicted_label_id)[0]
    print(f"Predicted sentiment: {predicted_label_name}")

def run_pipeline(csv_file_path: str = 'amazon_reviews.csv', 
                 text_column_name: str = 'cleaned_review', 
                 label_column_name: str = 'sentiments') -> Tuple[Dict[str, object], LabelEncoder, TfidfVectorizer]:
    reviews_dataframe = pd.read_csv(csv_file_path)
    
    reviews_dataframe = step1_preprocess_reviews(reviews_dataframe, text_column_name)
    
    reviews_dataframe, dataset_label_encoder = step2_encode_labels(reviews_dataframe, label_column_name)
    
    feature_vectors, dataset_tfidf_vectorizer = step3_apply_vectorization(reviews_dataframe, text_column_name)
    
    target_labels = reviews_dataframe[label_column_name]
    features_training_set, features_testing_set, labels_training_set, labels_testing_set = step4_split_data(feature_vectors, target_labels)
    trained_models = step5_train_classifiers(features_training_set, labels_training_set)
    step6_print_classification_reports(
        trained_models,
        features_testing_set,
        labels_testing_set,
        dataset_label_encoder
    )

    english_stopwords: Set[str] = set(stopwords.words('english'))
    step7_predict_user_review(
        trained_model=trained_models["Logistic Regression"],
        dataset_tfidf_vectorizer=dataset_tfidf_vectorizer,
        dataset_label_encoder=dataset_label_encoder,
        stop_words=english_stopwords
    )

    print(f"Original dataset shape: {reviews_dataframe.shape}")
    print(f"Training set: {features_training_set.shape[0]} samples")
    print(f"Testing set: {features_testing_set.shape[0]} samples")
    
    return trained_models, dataset_label_encoder, dataset_tfidf_vectorizer

if __name__ == "__main__":
    run_pipeline()

