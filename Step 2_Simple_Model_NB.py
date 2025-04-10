import multiprocessing
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

total_data_loaded = 0  # Variable to accumulate the total size of loaded data

# Load dataset in chunks using TensorFlow
def load_data_with_tensorflow(filepath, chunksize=50000):
    global total_data_loaded
    dataset = tf.data.experimental.make_csv_dataset(
        filepath,
        batch_size=chunksize,
        label_name='type',
        select_columns=['content', 'type'],
        num_epochs=1,
        ignore_errors=True,
        header=True
    )
    for batch in tqdm(dataset.take(20), desc="Loading Data"):
        features_dict, labels = batch
        features = features_dict['content']

        features_np = features.numpy()
        labels_np = labels.numpy()

        valid_indices = []

        for i, (feature, label) in enumerate(zip(features_np, labels_np)):
            if feature.strip() and label.strip():
                valid_indices.append(i)

        valid_features = features_np[valid_indices]
        valid_labels = labels_np[valid_indices]

        total_data_loaded += sum(len(feature) for feature in valid_features)
        yield valid_features, valid_labels

# Tokenization and Preprocessing
def preprocess_data(features, labels, stop_words, stemmer):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    processed_results = pool.starmap(process_text, [(text, stop_words, stemmer) for text in features])

    pool.close()
    pool.join()

    processed_features = []
    processed_labels = []
    for result, label in zip(processed_results, labels):
        if result:
            processed_text, _ = result  # Unpack the result tuple, only need the text
            processed_features.append(processed_text)
            processed_labels.append(classify_news_type(label.decode('utf-8')))

    return processed_features, processed_labels

def process_text(text, stop_words, stemmer):
    if not text.strip():  # Check if the text is empty or contains only whitespace
        return '', 0
    
    text = text.decode('utf-8')
    tokens = word_tokenize(text.lower())
    filtered_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words and token.isalpha()]
    processed_text = ' '.join(filtered_tokens)  # Processed text with stopwords removed and stemming applied
    return processed_text, len(filtered_tokens)

def classify_news_type(news_type):
    fake_types = ['fake', 'conspiracy', 'unreliable', 'satire', 'bias']
    reliable_types = ['political', 'reliable']
    if news_type in fake_types:
        return 'Fake'
    elif news_type in reliable_types:
        return 'Reliable'
    else:
        return 'Neutral'

def main():
    filepath = 'news.csv'  # Adjust the file path as needed
    chunksize = 50000
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    dataset = load_data_with_tensorflow(filepath, chunksize=chunksize)

    X_chunks = []
    y_chunks = []

    for chunk_num, (features, labels) in enumerate(dataset, start=1):
        processed_data = preprocess_data(features, labels, stop_words, stemmer)
        processed_features, processed_labels = processed_data

        if not processed_features:
            print(f"No features loaded in chunk {chunk_num}.")
        if not processed_labels:
            print(f"No labels loaded in chunk {chunk_num}.")

        X_chunks.extend(processed_features)
        y_chunks.extend(processed_labels)

    if not X_chunks or not y_chunks:
        print("No data loaded. Please check the dataset or adjust parameters.")
        return

    X = X_chunks
    y = y_chunks

    # Split data into training (80%), validation (10%), and test (10%) sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)
    val_texts, test_texts, val_labels, test_labels = train_test_split(test_texts, test_labels, test_size=0.5, random_state=42)

    # Convert text into numerical features using TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)

    # Train Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train, train_labels)

    # Predictions
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    # Evaluation function
    def evaluate_model(y_true, y_pred, dataset_name):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print(f"{dataset_name} Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}\n")
        return accuracy, precision, recall, f1

    # Print classification reports for all sets
    print("\nClassification Report - Training Set:\n")
    print(classification_report(train_labels, train_preds, target_names= model.classes_, zero_division=1))

    print("\nClassification Report - Validation Set:\n")
    print(classification_report(val_labels, val_preds, target_names= model.classes_, zero_division=1))

    print("\nClassification Report - Test Set:\n")
    print(classification_report(test_labels, test_preds, target_names= model.classes_, zero_division=1))

    # Confusion Matrix
    def plot_confusion_matrix(y_true, y_pred, title):
        cm = confusion_matrix(y_true, y_pred)
        print(f"{title} Confusion Matrix:")
        print(cm)

    plot_confusion_matrix(train_labels, train_preds, "Training Set")
    plot_confusion_matrix(val_labels, val_preds, "Validation Set")
    plot_confusion_matrix(test_labels, test_preds, "Test Set")

if __name__ == "__main__":
    main()
