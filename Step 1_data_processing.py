import multiprocessing
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import csv
import os
import re
import matplotlib.pyplot as plt

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

total_data_loaded = 0  # Variable to accumulate the total size of loaded data

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

    num_chunks = 0
    for batch in tqdm(dataset.take(20), desc="Loading Data"):
        features_dict, labels = batch
        features = features_dict['content']

        # Convert to numpy and filter out empty rows
        features_np = features.numpy()
        labels_np = labels.numpy()

        valid_indices = []

        for i, (feature, label) in enumerate(zip(features_np, labels_np)):
            # Check if both content and type columns are non-empty
            if feature.strip() and label.strip():
                valid_indices.append(i)

        valid_features = features_np[valid_indices]
        valid_labels = labels_np[valid_indices]

        total_data_loaded += sum(len(feature) for feature in valid_features)
        yield valid_features, valid_labels

def preprocess_data(features, labels, stop_words, stemmer):
    # Define a multiprocessing pool
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    # Preprocess each feature in parallel
    processed_results = pool.starmap(process_text, [(text, stop_words, stemmer) for text in features])

    # Close the pool
    pool.close()
    pool.join()

    processed_features = []
    processed_labels = []
    vocab_sizes = []  # List to store vocabulary sizes
    vocab_sizes_after_filtering = []
    urls_counts = []
    dates_counts = []
    numerics_counts = []
    for result, label in zip(processed_results, labels):
        if result:
            processed_text, vocab_size, vocab_size_after_filtering,num_urls,num_dates, num_numerics= result  # Unpack the result tuple
            processed_features.append(processed_text)
            processed_labels.append(classify_news_type(label.decode('utf-8')))  # Decode label to string
            vocab_sizes.append(vocab_size)
            vocab_sizes_after_filtering.append(vocab_size_after_filtering)
            urls_counts.append(num_urls)
            dates_counts.append(num_dates)
            numerics_counts.append(num_numerics)

    return processed_features, processed_labels, vocab_sizes, vocab_sizes_after_filtering,urls_counts,dates_counts,numerics_counts

def process_text(text, stop_words, stemmer):
    if not text.strip():  # Check if the text is empty or contains only whitespace
        return '', 0, 0, 0  # Return empty strings and counts if the text is empty
    
    # Decode the bytes-like object to a string
    text = text.decode('utf-8')

    # Count URLs in the content
    num_urls = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))

    # Count dates in the content

    dates = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text) # Example date format: 01/01/2022
    num_dates = len(dates)

    # Count numeric values in the content

    numerics = re.findall(r'\b\d+\b', text)  # Extracts integers
    num_numerics = len(numerics)

    tokens = word_tokenize(text.lower())
    vocab_size = len(set(tokens))
    filtered_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words and token.isalpha()]

    vocab_size_after_filtering = len(set(filtered_tokens))  # Unique tokens after filtering and stemming
    processed_text = ' '.join(filtered_tokens)  # Processed text with stopwords removed and stemming applied
    #processed_text = ' '.join(tokens)
    #vocab_size_after_filtering = len(set(tokens))
    return processed_text, vocab_size, vocab_size_after_filtering, num_urls, num_dates, num_numerics

def classify_news_type(news_type):
    fake_types = ['fake', 'conspiracy','unreliable','satire','bias']
    reliable_types = ['political', 'reliable']
    if news_type in fake_types:
        return 'Fake'
    elif news_type in reliable_types:
        return 'Reliable'
    else:
        return 'Neutral'

def get_tokens_size_on_disk(tokens, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Stemmed Tokens'])  # Write header
        for token in tokens:
            writer.writerow([token])  # Write each stemmed token to a separate row
    return os.path.getsize(filename)

def bytes_to_gb(size_in_bytes):
    return size_in_bytes / (1024 ** 3)

def remove_stopwords_and_stem(text, stop_words, stemmer):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and perform stemming
    filtered_stemmed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and word.isalpha()]

    return filtered_stemmed_tokens

def main():
    filepath = 'news.csv'  # Adjust the file path as needed
    chunksize = 50000

    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    dataset = load_data_with_tensorflow(filepath, chunksize=chunksize)

    X_chunks = []
    y_chunks = []
    total_vocab_size = 0  # Initialize total_vocab_size
    total_vocab_size_after_filtering = 0
    total_tokens_size_on_disk = 0  # Initialize total size of tokens on disk


    removed_rows_per_chunk = {}  # Dictionary to store the number of removed rows per chunk


    for chunk_num, (features, labels) in enumerate(dataset, start=1):
        removed_rows = 0

        processed_data = preprocess_data(features, labels, stop_words, stemmer)
        processed_features, processed_labels, vocab_sizes, vocab_sizes_after_filtering,urls_count, dates_count, numerics_count  = processed_data

        if not processed_features:
            print(f"No features loaded in chunk {chunk_num}.")
        if not processed_labels:
            print(f"No labels loaded in chunk {chunk_num}.")

        # Calculate and print the size of tokens on disk
        for text in processed_features:
            tokens = text.split()
            tokens_filename = "tokens.csv"
            tokens_size_on_disk = get_tokens_size_on_disk(tokens, tokens_filename)
            total_tokens_size_on_disk += tokens_size_on_disk


        fake_counts = processed_labels.count('Fake')
        reliable_counts = processed_labels.count('Reliable')
        neutral_counts = processed_labels.count('Neutral')

        # Count the number of removed rows
        removed_rows = chunksize - len(processed_features)
        removed_rows_per_chunk[chunk_num] = {
        'removed_rows': removed_rows,
        'urls_count': sum(urls_count),
        'dates_count': sum(dates_count),
        'numerics_count': sum(numerics_count),
        'fake': fake_counts,
        'reliable': reliable_counts,
        'neutral': neutral_counts

         }

        X_chunks.extend(processed_features)
        y_chunks.extend(processed_labels)
    

        total_vocab_size += sum(vocab_sizes)  # Accumulate the total vocabulary size
        total_vocab_size_after_filtering += sum(vocab_sizes_after_filtering)  # Accumulate the total vocabulary size

    print(f"Total Vocab Size: {total_vocab_size} ")
    print(f"Total Vocab Size after stemming: {total_vocab_size_after_filtering}")

    if not X_chunks or not y_chunks:
        print("No data loaded. Please check the dataset or adjust parameters.")
        return

    X = X_chunks  # No need to convert to NumPy array
    y = y_chunks


   # Perform stemming
    all_text = ' '.join(X)

    # Tokenize the text excluding numeric values and non-word tokens
    #word_tokens = word_tokenize(all_text.lower())
    #filtered_words = [word for word in word_tokens if word.isalpha() and not word.isdigit()]

    
    # Determine the 100 most frequent words excluding numeric values
    #word_freq = Counter(filtered_words)
    #top_10000_words = word_freq.most_common(10000)
    #top_1000_words = word_freq.most_common(1000)
    #top_100_words = word_freq.most_common(100)
    #word_tokens_raw = word_tokenize(all_text.lower())
    #raw_words = [word for word in word_tokens_raw if word.isalpha()]
    # Word frequency BEFORE stopword removal and stemming
    #word_freq_before = Counter(raw_words)
    #top_10000_before = word_freq_before.most_common(10000)
    #filtered_words = [stemmer.stem(word) for word in raw_words if word not in stop_words]
    # Word frequency AFTER stopword removal and stemming
    #word_freq_after = Counter(filtered_words)
    #top_10000_after = word_freq_after.most_common(10000)
    
    # Extract words and frequencies
    #words_100 = [word[0] for word in top_100_words]
    #frequencies_100 = [word[1] for word in top_100_words]

    # Extract words and frequencies
    #words_1000 = [word[0] for word in top_1000_words]
    #frequencies_1000 = [word[1] for word in top_1000_words]

    # Extract words and frequencies
    #words_10000 = [word[0] for word in top_10000_words]
    #frequencies_10000 = [word[1] for word in top_10000_words]

    # Calculate ranks

    # Word frequency BEFORE stopword removal and stemming
    word_tokens_raw = word_tokenize(all_text.lower())
    raw_words = [word for word in word_tokens_raw if word.isalpha()]
    word_freq_before = Counter(raw_words)
    top_10000_before = word_freq_before.most_common(10000)

# Word frequency AFTER stopword removal and stemming
    filtered_words = [stemmer.stem(word) for word in raw_words if word not in stop_words]
    word_freq_after = Counter(filtered_words)
    top_10000_after = word_freq_after.most_common(10000)

# Verify top words before and after
    print("Top 10000 words before stemming:", top_10000_before[:10])  # Check the first few items
    print("Top 10000 words after stemming:", top_10000_after[:10])  # Check the first few items


    # Plot: Top 10000 words BEFORE removing stopwords and stemming
    words_before = [word for word, freq in top_10000_before]
    frequencies_before = [freq for word, freq in top_10000_before]
    plt.figure(figsize=(15, 6))
    plt.bar(words_before[:1000], frequencies_before[:1000], color='purple')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top 10000 Most Frequent Words - BEFORE Removing Stopwords & Stemming')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # Plotting 100
    """
    plt.figure(figsize=(15, 6))
    plt.bar(words_100[:100], frequencies_100[:100], color='green')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top 100 Words and Their Frequencies')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    print("Top 100 most frequent words (excluding numeric values) in the whole dataset:")
    for word, freq in top_100_words:
        print(f"{word}: {freq}")

        # Plotting 1000
    plt.figure(figsize=(15, 6))
    plt.bar(words_1000[:1000], frequencies_1000[:1000], color='green')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top 1000 Words and Their Frequencies')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
  
    """
    # Plot: Top 10000 words AFTER removing stopwords and stemming
    words_after = [word for word, freq in top_10000_after]
    frequencies_after = [freq for word, freq in top_10000_after]

    plt.figure(figsize=(15, 6))
    plt.bar(words_after[:1000], frequencies_after[:1000], color='purple')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top 10000 Most Frequent Words - AFTER Removing Stopwords & Stemming')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

        # Plotting 10000
    """
    plt.figure(figsize=(15, 6))
    plt.bar(words_10000[:10000], frequencies_10000[:10000], color='green')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top 10000 Words and Their Frequencies')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    """
  
    # Print total data loaded
    print(f"Total data loaded: {total_data_loaded / (1024 ** 3):.6f} GB")

    # Print results
    print(f"Total size of all tokens on disk: {bytes_to_gb(total_tokens_size_on_disk):.6f} GB")

    # Classify news types
    for label in set(y):
        news_type = classify_news_type(label)  # No need to decode label
        print(f"Label: {label}, News Type: {news_type}")

    # Print removed rows per chunk
    print("\nRemoved Rows per Chunk:")
    for chunk_num, data in removed_rows_per_chunk.items():
        print(f"Chunk {chunk_num}:")
        print(f"  Removed rows: {data['removed_rows']}")
        print(f"  URLs count: {data['urls_count']}")
        print(f"  Dates count: {data['dates_count']}")
        print(f"  Numeric values count: {data['numerics_count']}")
        print(f"  fake: {data['fake']}")
        print(f"  reliable: {data['reliable']}")
        print(f"  neutral: {data['neutral']}")

    # Extract data for plotting
    chunks = list(removed_rows_per_chunk.keys())
    removed_rows = [data['removed_rows'] for data in removed_rows_per_chunk.values()]
    urls_counts = [data['urls_count'] for data in removed_rows_per_chunk.values()]
    dates_counts = [data['dates_count'] for data in removed_rows_per_chunk.values()]
    numeric_values_counts = [data['numerics_count'] for data in removed_rows_per_chunk.values()]
    fake_counts = [data['fake'] for data in removed_rows_per_chunk.values()]
    reliable_counts = [data['reliable'] for data in removed_rows_per_chunk.values()]
    neutral_counts = [data['neutral'] for data in removed_rows_per_chunk.values()]


    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(chunks, removed_rows, label='Removed Rows', color='red', alpha=0.9)
    plt.bar(chunks, urls_counts, bottom=removed_rows, label='URLs Count', color='green')
    plt.bar(chunks, dates_counts, bottom=[i+j for i,j in zip(removed_rows, urls_counts)], label='Dates Count', color='blue')
    plt.bar(chunks, numeric_values_counts, bottom=[i+j+k for i,j,k in zip(removed_rows, urls_counts, dates_counts)], label='Numeric Values Count', color='yellow')
    plt.xlabel('Chunk')
    plt.ylabel('Count')
    plt.title('Counts of Removed Rows, URLs, Dates, Numeric Values, per Chunk')
    plt.legend()
    plt.xticks(chunks)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()

