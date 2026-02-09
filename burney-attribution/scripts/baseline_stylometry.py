#!/usr/bin/env python3
"""
Traditional stylometry baseline using Burrows' Delta.

Implements feature extraction and Delta distance calculation for
authorship attribution on 18th-century novels.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import re


class BurrowsDelta:
    """
    Burrows' Delta stylometric distance measure.

    Based on: Burrows, John. "'Delta': a Measure of Stylistic Difference
    and a Guide to Likely Authorship." Literary and Linguistic Computing 17.3 (2002).
    """

    def __init__(self, n_features=100):
        """
        Initialize Delta calculator.

        Args:
            n_features: Number of most frequent words to use (default 100)
        """
        self.n_features = n_features
        self.vocab = None
        self.scaler = StandardScaler()

    def extract_features(self, texts, authors):
        """
        Extract stylometric features from texts.

        Args:
            texts: List of text strings
            authors: List of author labels

        Returns:
            feature_matrix: numpy array of shape (n_texts, n_features)
            feature_names: list of feature names
        """
        # Build vocabulary from all texts
        word_counter = Counter()
        for text in texts:
            words = self._tokenize(text)
            word_counter.update(words)

        # Get most frequent words
        self.vocab = [word for word, count in word_counter.most_common(self.n_features)]

        # Extract features for each text
        features = []
        for text in texts:
            features.append(self._text_to_features(text))

        return np.array(features), self.vocab

    def _tokenize(self, text):
        """Simple word tokenization (lowercase, alphabetic only)."""
        text = text.lower()
        words = re.findall(r'\b[a-z]+\b', text)
        return words

    def _text_to_features(self, text):
        """Convert text to feature vector (word frequencies)."""
        words = self._tokenize(text)
        total_words = len(words)

        if total_words == 0:
            return np.zeros(self.n_features)

        word_counts = Counter(words)

        # Calculate relative frequencies for MFW
        features = []
        for word in self.vocab:
            freq = (word_counts[word] / total_words) * 1000  # per thousand words
            features.append(freq)

        return np.array(features)

    def fit(self, X, y):
        """Fit the scaler on training data."""
        self.scaler.fit(X)
        return self

    def transform(self, X):
        """Transform features to z-scores."""
        return self.scaler.transform(X)

    def calculate_delta(self, X_train, X_test):
        """
        Calculate Burrows' Delta distance.

        Args:
            X_train: Training features (already z-scored)
            X_test: Test features (already z-scored)

        Returns:
            distances: Array of shape (n_test, n_train) with Delta distances
        """
        # Delta = mean absolute difference of z-scores
        distances = np.zeros((len(X_test), len(X_train)))

        for i, test_vec in enumerate(X_test):
            for j, train_vec in enumerate(X_train):
                delta = np.mean(np.abs(test_vec - train_vec))
                distances[i, j] = delta

        return distances


def load_corpus(processed_dir, metadata_file):
    """
    Load processed corpus and metadata.

    Returns:
        texts: List of text strings
        authors: List of author labels
        titles: List of titles
        file_paths: List of file paths
    """
    metadata = pd.read_csv(metadata_file)

    texts = []
    authors = []
    titles = []
    file_paths = []

    for _, row in metadata.iterrows():
        file_path = processed_dir / row['file_path']

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        texts.append(text)
        authors.append(row['author'])
        titles.append(row['title'])
        file_paths.append(row['file_path'])

    return texts, authors, titles, file_paths


def split_data(texts, authors, titles, file_paths, test_size=0.2, random_state=42):
    """
    Split data by work (to avoid leakage from multi-volume works).
    Try to maintain author representation in both sets.

    Args:
        texts, authors, titles, file_paths: Corpus data
        test_size: Proportion for test set
        random_state: Random seed

    Returns:
        train and test splits
    """
    np.random.seed(random_state)

    # Group by (author, title) to keep volumes together
    work_groups = {}
    for i, (author, title) in enumerate(zip(authors, titles)):
        key = (author, title)
        if key not in work_groups:
            work_groups[key] = []
        work_groups[key].append(i)

    # Group works by author for stratification
    author_works = {}
    for work_key in work_groups.keys():
        author = work_key[0]
        if author not in author_works:
            author_works[author] = []
        author_works[author].append(work_key)

    # Stratified split: try to get each author in both train and test
    train_indices = []
    test_indices = []

    for author, works in author_works.items():
        np.random.shuffle(works)

        # For authors with multiple works, split them
        if len(works) > 1:
            n_test = max(1, int(len(works) * test_size))
            test_author_works = works[:n_test]
            train_author_works = works[n_test:]
        else:
            # Single work: put in train (we need training examples for all authors)
            train_author_works = works
            test_author_works = []

        # Collect indices
        for work in train_author_works:
            train_indices.extend(work_groups[work])
        for work in test_author_works:
            test_indices.extend(work_groups[work])

    # Extract splits
    train_texts = [texts[i] for i in train_indices]
    train_authors = [authors[i] for i in train_indices]
    train_titles = [titles[i] for i in train_indices]
    train_paths = [file_paths[i] for i in train_indices]

    test_texts = [texts[i] for i in test_indices]
    test_authors = [authors[i] for i in test_indices]
    test_titles = [titles[i] for i in test_indices]
    test_paths = [file_paths[i] for i in test_indices]

    return (train_texts, train_authors, train_titles, train_paths,
            test_texts, test_authors, test_titles, test_paths)


def main():
    """Run baseline stylometry evaluation."""
    print("="*60)
    print("BURROWS' DELTA BASELINE")
    print("="*60)

    # Set up paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    processed_dir = project_dir / 'data' / 'processed'
    metadata_file = project_dir / 'data' / 'metadata.csv'
    output_dir = project_dir / 'outputs'
    output_dir.mkdir(exist_ok=True)

    # Load corpus
    print("\nLoading corpus...")
    texts, authors, titles, file_paths = load_corpus(processed_dir, metadata_file)
    print(f"Loaded {len(texts)} texts")

    # Split data
    print("\nSplitting data by work...")
    (train_texts, train_authors, train_titles, train_paths,
     test_texts, test_authors, test_titles, test_paths) = split_data(
        texts, authors, titles, file_paths, test_size=0.2, random_state=42
    )

    print(f"Train: {len(train_texts)} texts")
    print(f"Test: {len(test_texts)} texts")

    print("\nTrain works:")
    for author, title in sorted(set(zip(train_authors, train_titles))):
        print(f"  - {author}: {title}")

    print("\nTest works:")
    for author, title in sorted(set(zip(test_authors, test_titles))):
        print(f"  - {author}: {title}")

    # Extract features
    print(f"\n Extracting stylometric features (top {100} words)...")
    delta = BurrowsDelta(n_features=100)
    X_train, vocab = delta.extract_features(train_texts, train_authors)
    print(f"Train features: {X_train.shape}")

    # Transform test set using same vocabulary
    X_test = np.array([delta._text_to_features(text) for text in test_texts])
    print(f"Test features: {X_test.shape}")

    # Fit scaler and transform
    print("\nNormalizing features (z-scores)...")
    delta.fit(X_train, train_authors)
    X_train_scaled = delta.transform(X_train)
    X_test_scaled = delta.transform(X_test)

    # Calculate Delta distances
    print("\nCalculating Delta distances...")
    distances = delta.calculate_delta(X_train_scaled, X_test_scaled)

    # Classify by nearest neighbor
    predictions = []
    for i in range(len(X_test)):
        nearest_idx = np.argmin(distances[i])
        predictions.append(train_authors[nearest_idx])

    # Evaluate
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    accuracy = accuracy_score(test_authors, predictions)
    print(f"\nAccuracy: {accuracy:.2%}")

    print("\nClassification Report:")
    print(classification_report(test_authors, predictions, zero_division=0))

    print("\nConfusion Matrix:")
    unique_authors = sorted(set(train_authors + test_authors))
    cm = confusion_matrix(test_authors, predictions, labels=unique_authors)
    cm_df = pd.DataFrame(cm, index=unique_authors, columns=unique_authors)
    print(cm_df)

    # Detailed results
    print("\n" + "="*60)
    print("DETAILED PREDICTIONS")
    print("="*60)
    for i, (true, pred, title) in enumerate(zip(test_authors, predictions, test_titles)):
        status = "✓" if true == pred else "✗"
        print(f"{status} {test_paths[i]}")
        print(f"  True: {true:12} Predicted: {pred:12} [{title}]")

    # Save results
    results = {
        'accuracy': accuracy,
        'predictions': predictions,
        'test_authors': test_authors,
        'test_titles': test_titles,
        'test_paths': test_paths,
        'confusion_matrix': cm,
        'author_labels': unique_authors
    }

    with open(output_dir / 'baseline_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"\n✓ Results saved to {output_dir / 'baseline_results.pkl'}")


if __name__ == "__main__":
    main()
