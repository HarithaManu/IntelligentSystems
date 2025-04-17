import multiprocessing
import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import csv
import os
import re
import nltk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def load_fake_news_corpus(filepath, chunksize):
    """Load FakeNewsCorpus dataset"""
    dataset = tf.data.experimental.make_csv_dataset(
        filepath,
        batch_size=chunksize,
        label_name='type',
        select_columns=['content', 'type'],
        num_epochs=1,
        ignore_errors=True,
        header=True
    )

    X = []
    y = []
    
    for batch in tqdm(dataset.take(20), desc = "Loading") :  # Adjust number of chunks as needed
        features_dict, labels = batch
        features = features_dict['content']
        
        features_np = features.numpy()
        labels_np = labels.numpy()
        
        for feature, label in zip(features_np, labels_np):
            if feature.strip() and label.strip():
                X.append(feature.decode('utf-8'))
                y.append(label.decode('utf-8'))
    
    return X, y

def load_liar_dataset(train_path, test_path):
    """Load LIAR dataset"""
    def load_split(filepath):
        data = pd.read_csv(filepath, sep='\t', header=None, encoding='latin1')
        # LIAR dataset columns: 
        # 0: ID, 1: label, 2: statement, 3: subject, 4: speaker, 5: job, 6: state, 7: party, 8-12: context
        X = data[2].tolist()  # statements
        y = data[1].tolist()  # labels
        return X, y
    
    X_train, y_train = load_split(train_path)
    X_test, y_test = load_split(test_path)
    
    return X_train, y_train, X_test, y_test

def preprocess_text(text, stop_words, stemmer):
    """Preprocess a single text"""
    if not text.strip():
        return '', 0, 0, 0, 0, 0
    
    # Count URLs, dates, numerics
    num_urls = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
    dates = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)
    num_dates = len(dates)
    numerics = re.findall(r'\b\d+\b', text)
    num_numerics = len(numerics)

    tokens = word_tokenize(text.lower())
    vocab_size = len(set(tokens))
    filtered_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words and token.isalpha()]
    vocab_size_after_filtering = len(set(filtered_tokens))
    processed_text = ' '.join(filtered_tokens)
    
    return processed_text, vocab_size, vocab_size_after_filtering, num_urls, num_dates, num_numerics

def preprocess_data(X, y, stop_words, stemmer):
    """Preprocess data in parallel"""
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = pool.starmap(preprocess_text, [(text, stop_words, stemmer) for text in X])
    pool.close()
    pool.join()
    
    processed_X = []
    processed_y = []
    stats = {
        'vocab_sizes': [],
        'vocab_sizes_after': [],
        'urls_counts': [],
        'dates_counts': [],
        'numerics_counts': []
    }
    
    for (text, vs, vs_af, urls, dates, nums), label in zip(results, y):
        if text:
            processed_X.append(text)
            processed_y.append(classify_news_type(label))
            stats['vocab_sizes'].append(vs)
            stats['vocab_sizes_after'].append(vs_af)
            stats['urls_counts'].append(urls)
            stats['dates_counts'].append(dates)
            stats['numerics_counts'].append(nums)
    
    return processed_X, processed_y, stats

def classify_news_type(news_type):
    """Classify news type into Fake/Reliable/Neutral"""
    fake_types = ['fake', 'conspiracy', 'unreliable', 'satire', 'bias', 'false', 'pants-fire']
    reliable_types = ['political', 'reliable', 'true', 'mostly-true', 'half-true']
    
    if news_type.lower() in fake_types:
        return 'Fake'
    elif news_type.lower() in reliable_types:
        return 'Reliable'
    else:
        return 'Neutral'

def build_lstm_model(max_features, embedding_dim, lstm_units, maxlen):
    model = Sequential()
    model.add(Embedding(max_features, embedding_dim, input_length=maxlen))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main():
    # Initialize NLTK
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    # Load and preprocess training data (FakeNewsCorpus)
    print("Loading FakeNewsCorpus...")
    fake_news_X, fake_news_y = load_fake_news_corpus('news_cleaned_2018_02_13.csv', 50000)
    X_train, y_train, train_stats = preprocess_data(fake_news_X, fake_news_y, stop_words, stemmer)
    
    # Load and preprocess test data (LIAR)
    print("Loading LIAR dataset...")
    liar_X_train, liar_y_train, liar_X_test, liar_y_test = load_liar_dataset(
        'train.tsv', 
        'test.tsv'
    )
    
    # Combine LIAR train and test for evaluation (since we're using FakeNewsCorpus for training)
    X_test = liar_X_train + liar_X_test
    y_test = liar_y_train + liar_y_test
    X_test, y_test, test_stats = preprocess_data(X_test, y_test, stop_words, stemmer)
    
    # Print dataset statistics
    print(f"\nTraining Data Stats (FakeNewsCorpus):")
    print(f"  Samples: {len(X_train)}")
    print(f"  Avg vocab size: {np.mean(train_stats['vocab_sizes']):.1f}")
    print(f"  Avg vocab after filtering: {np.mean(train_stats['vocab_sizes_after']):.1f}")
    
    print(f"\nTest Data Stats (LIAR):")
    print(f"  Samples: {len(X_test)}")
    print(f"  Avg vocab size: {np.mean(test_stats['vocab_sizes']):.1f}")
    print(f"  Avg vocab after filtering: {np.mean(test_stats['vocab_sizes_after']):.1f}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    y_train_cat = to_categorical(y_train_encoded)
    y_test_cat = to_categorical(y_test_encoded)
    
    # Tokenization for LSTM
    max_features = 5000
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(X_train)
    
    # Convert texts to sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Pad sequences
    maxlen = 200
    X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
    X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)
    
    # Build and train LSTM model
    print("\nTraining LSTM model...")
    embedding_dim = 128
    lstm_units = 64
    model = build_lstm_model(max_features, embedding_dim, lstm_units, maxlen)
    
    history = model.fit(
        X_train_pad,
        y_train_cat,
        epochs=5,
        batch_size=256,
        validation_data=(X_test_pad, y_test_cat),
        verbose=1
    )
    
    # Evaluate the model
    y_pred = model.predict(X_test_pad)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test_cat, axis=1)
    
    # Print evaluation metrics
    print("\nClassification Report on LIAR Dataset:")
    print(classification_report(y_test_classes, y_pred_classes, target_names=label_encoder.classes_))
    
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
    f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()