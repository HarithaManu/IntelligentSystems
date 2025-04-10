import multiprocessing
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

# Environment config to prevent warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Download necessary resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Globals
total_data_loaded = 0

def load_data_with_tensorflow(filepath: str, chunksize: int = 50000):
    global total_data_loaded

    dataset = tf.data.experimental.make_csv_dataset(
        filepath,
        batch_size=chunksize,
        label_name='type',
        select_columns=['content', 'type', 'source', 'authors'],
        num_epochs=1,
        ignore_errors=True
    )

    for batch in tqdm(dataset.take(20), desc="Loading Data", total=20):
        features_dict, labels = batch
        content = features_dict['content'].numpy()
        source = features_dict['source'].numpy()
        authors = features_dict['authors'].numpy()
        labels_np = labels.numpy()

        valid_data = [
            (c, s, a, l) for c, s, a, l in zip(content, source, authors, labels_np)
            if c.strip() and l.strip()
        ]

        if valid_data:
            valid_features, source_np, author_np, valid_labels = zip(*valid_data)
            total_data_loaded += sum(len(feat) for feat in valid_features)
            yield valid_features, source_np, author_np, valid_labels


def process_text(text: bytes, stop_words: set, stemmer: PorterStemmer) -> str:
    try:
        text = text.decode('utf-8')
    except UnicodeDecodeError:
        return ""

    tokens = word_tokenize(text.lower())
    filtered = [w for w in tokens if w.isalpha() and w not in stop_words]
    stemmed = [stemmer.stem(w) for w in filtered]
    return " ".join(stemmed)


def classify_news_type(news_type: str) -> str:
    fake = {'fake', 'conspiracy', 'unreliable', 'satire', 'bias'}
    reliable = {'reliable', 'political'}
    label = news_type.lower()
    if label in fake:
        return 'Fake'
    elif label in reliable:
        return 'Reliable'
    return 'Neutral'


def preprocess_data(features, sources, authors, labels, stop_words, stemmer):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        processed = pool.starmap(process_text, [(f, stop_words, stemmer) for f in features])

    X, S, A, Y = [], [], [], []
    for content, source, author, label in zip(processed, sources, authors, labels):
        if content.strip():
            X.append(content)
            S.append(source.decode('utf-8') if isinstance(source, bytes) else str(source))
            A.append(author.decode('utf-8') if isinstance(author, bytes) else str(author))
            Y.append(classify_news_type(label.decode('utf-8')))

    return X, S, A, Y


def main():
    filepath = 'news.csv'
    chunksize = 50000

    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    dataset = load_data_with_tensorflow(filepath, chunksize=chunksize)

    X_chunks, source_chunks, author_chunks, y_chunks = [], [], [], []

    for features, sources, authors, labels in dataset:
        X, S, A, Y = preprocess_data(features, sources, authors, labels, stop_words, stemmer)
        X_chunks.extend(X)
        source_chunks.extend(S)
        author_chunks.extend(A)
        y_chunks.extend(Y)

    if not X_chunks or not y_chunks:
        print("No valid data was loaded.")
        return

    X_combined = [f"{c} {s} {a}" for c, s, a in zip(X_chunks, source_chunks, author_chunks)]

    X_train, X_temp, y_train, y_temp = train_test_split(X_combined, y_chunks, test_size=0.2, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    X_val_tfidf = vectorizer.transform(X_val)

    param_grid = {'alpha': [0.1, 0.5, 1.0]}
    clf = GridSearchCV(MultinomialNB(), param_grid, cv=5, n_jobs=-1, verbose=1)
    clf.fit(X_train_tfidf, y_train)
    best_model = clf.best_estimator_

    # Test Set Evaluation
    y_pred_test = best_model.predict(X_test_tfidf)
    print("\n=== Test Set Metrics ===")
    print(classification_report(y_test, y_pred_test))
    print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_test, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_test, average='weighted'):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred_test, average='weighted'):.4f}")

    # Validation Set Evaluation
    y_pred_val = best_model.predict(X_val_tfidf)
    print("\n=== Validation Set Metrics ===")
    print(classification_report(y_val, y_pred_val))
    print(f"Accuracy: {accuracy_score(y_val, y_pred_val):.4f}")
    print(f"Precision: {precision_score(y_val, y_pred_val, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_val, y_pred_val, average='weighted'):.4f}")
    print(f"F1 Score: {f1_score(y_val, y_pred_val, average='weighted'):.4f}")

    print(f"\nTotal data loaded: {total_data_loaded / (1024 ** 2):.2f} MB")


if __name__ == "__main__":
    main()
