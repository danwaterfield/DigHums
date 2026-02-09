#!/usr/bin/env python3
"""
Prepare data for ECCO-BERT fine-tuning.

Creates train/val/test splits, tokenizes text, and saves as HuggingFace datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm
import json


def load_corpus(processed_dir, metadata_file):
    """Load processed corpus and metadata."""
    metadata = pd.read_csv(metadata_file)

    corpus = []
    for _, row in metadata.iterrows():
        file_path = processed_dir / row['file_path']
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        corpus.append({
            'text': text,
            'author': row['author'],
            'title': row['title'],
            'year': row['year'],
            'file_path': row['file_path']
        })

    return corpus


def split_by_work(corpus, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42):
    """
    Split corpus by work (stratified by author where possible).

    Args:
        corpus: List of dicts with text and metadata
        train_size, val_size, test_size: Split proportions
        random_state: Random seed

    Returns:
        train, val, test: Lists of corpus items
    """
    np.random.seed(random_state)

    # Group by (author, title) to keep volumes together
    works = {}
    for item in corpus:
        key = (item['author'], item['title'])
        if key not in works:
            works[key] = []
        works[key].append(item)

    # Group works by author
    author_works = {}
    for work_key, items in works.items():
        author = work_key[0]
        if author not in author_works:
            author_works[author] = []
        author_works[author].append((work_key, items))

    # Stratified split
    train_items = []
    val_items = []
    test_items = []

    for author, author_work_list in author_works.items():
        np.random.shuffle(author_work_list)

        n_works = len(author_work_list)
        if n_works == 1:
            # Single work: all to train
            for work_key, items in author_work_list:
                train_items.extend(items)
        elif n_works == 2:
            # Two works: train and val
            train_items.extend(author_work_list[0][1])
            val_items.extend(author_work_list[1][1])
        elif n_works == 3:
            # Three works: train, val, test (one each)
            train_items.extend(author_work_list[0][1])
            val_items.extend(author_work_list[1][1])
            test_items.extend(author_work_list[2][1])
        else:
            # Four or more: ensure at least one in train, distribute rest
            train_items.extend(author_work_list[0][1])  # Always include first in train

            remaining = author_work_list[1:]
            n_remaining = len(remaining)
            n_val = max(1, int(n_remaining * val_size / (val_size + test_size)))
            n_test = n_remaining - n_val

            for work_key, items in remaining[:n_val]:
                val_items.extend(items)
            for work_key, items in remaining[n_val:]:
                test_items.extend(items)

    return train_items, val_items, test_items


def chunk_text(text, tokenizer, chunk_size=512, stride=256):
    """
    Split text into overlapping chunks of tokens.

    Args:
        text: Input text string
        tokenizer: HuggingFace tokenizer
        chunk_size: Number of tokens per chunk
        stride: Step size between chunks (for overlap)

    Returns:
        List of text chunks
    """
    # Tokenize entire text
    tokens = tokenizer.encode(text, add_special_tokens=False)

    # Create chunks with sliding window
    chunks = []
    for i in range(0, len(tokens), stride):
        chunk_tokens = tokens[i:i+chunk_size]

        if len(chunk_tokens) < 64:  # Skip very short chunks
            continue

        # Decode back to text
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)

        # Stop if we've reached the end
        if i + chunk_size >= len(tokens):
            break

    return chunks


def prepare_dataset(corpus_items, tokenizer, author_to_id, chunk_size=512, stride=256):
    """
    Convert corpus items to dataset with chunked texts.

    Args:
        corpus_items: List of corpus dicts
        tokenizer: HuggingFace tokenizer
        author_to_id: Dict mapping author names to integer IDs
        chunk_size: Tokens per chunk
        stride: Stride for sliding window

    Returns:
        HuggingFace Dataset
    """
    data = []

    for item in tqdm(corpus_items, desc="Processing texts"):
        chunks = chunk_text(item['text'], tokenizer, chunk_size, stride)

        for chunk in chunks:
            data.append({
                'text': chunk,
                'label': author_to_id[item['author']],
                'author': item['author'],
                'title': item['title'],
                'year': item['year']
            })

    return Dataset.from_dict({
        'text': [d['text'] for d in data],
        'label': [d['label'] for d in data],
        'author': [d['author'] for d in data],
        'title': [d['title'] for d in data],
        'year': [d['year'] for d in data]
    })


def main():
    """Prepare BERT training data."""
    print("="*60)
    print("PREPARING DATA FOR ECCO-BERT")
    print("="*60)

    # Paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    processed_dir = project_dir / 'data' / 'processed'
    metadata_file = project_dir / 'data' / 'metadata.csv'
    output_dir = project_dir / 'data' / 'bert_data'
    output_dir.mkdir(exist_ok=True)

    # Load tokenizer
    print("\nLoading ECCO-BERT tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("Brendan/bert-base-uncased-ecco")
        print("✓ Loaded ECCO-BERT tokenizer")
    except Exception as e:
        print(f"⚠ Could not load ECCO-BERT tokenizer: {e}")
        print("Falling back to bert-base-uncased...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Load corpus
    print("\nLoading corpus...")
    corpus = load_corpus(processed_dir, metadata_file)
    print(f"Loaded {len(corpus)} texts")

    # Split data
    print("\nSplitting data (70/15/15 by work)...")
    train_corpus, val_corpus, test_corpus = split_by_work(corpus)

    print(f"Train: {len(train_corpus)} texts")
    print(f"Val: {len(val_corpus)} texts")
    print(f"Test: {len(test_corpus)} texts")

    # Show split details
    for split_name, split_corpus in [("Train", train_corpus), ("Val", val_corpus), ("Test", test_corpus)]:
        print(f"\n{split_name} works:")
        works = set((item['author'], item['title']) for item in split_corpus)
        for author, title in sorted(works):
            print(f"  - {author}: {title}")

    # Create author to ID mapping
    all_authors = sorted(set(item['author'] for item in corpus))
    author_to_id = {author: i for i, author in enumerate(all_authors)}
    id_to_author = {i: author for author, i in author_to_id.items()}

    print(f"\n{len(author_to_id)} authors: {all_authors}")

    # Save label mapping
    with open(output_dir / 'label_mapping.json', 'w') as f:
        json.dump({'author_to_id': author_to_id, 'id_to_author': id_to_author}, f, indent=2)

    # Prepare datasets with chunking
    print("\nCreating chunked datasets (512 tokens, 256 stride)...")
    train_dataset = prepare_dataset(train_corpus, tokenizer, author_to_id, chunk_size=512, stride=256)
    val_dataset = prepare_dataset(val_corpus, tokenizer, author_to_id, chunk_size=512, stride=256)
    test_dataset = prepare_dataset(test_corpus, tokenizer, author_to_id, chunk_size=512, stride=256)

    print(f"\nTrain chunks: {len(train_dataset)}")
    print(f"Val chunks: {len(val_dataset)}")
    print(f"Test chunks: {len(test_dataset)}")

    # Show class distribution
    print("\nClass distribution (chunks):")
    for split_name, dataset in [("Train", train_dataset), ("Val", val_dataset), ("Test", test_dataset)]:
        print(f"\n{split_name}:")
        label_counts = {}
        for label in dataset['label']:
            author = id_to_author[label]
            label_counts[author] = label_counts.get(author, 0) + 1

        for author in sorted(label_counts.keys()):
            count = label_counts[author]
            pct = count / len(dataset) * 100
            print(f"  {author:12} {count:5} chunks ({pct:.1f}%)")

    # Combine into DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })

    # Save
    print(f"\nSaving datasets to {output_dir}...")
    dataset_dict.save_to_disk(str(output_dir / 'chunked_datasets'))

    print("\n✓ Data preparation complete!")
    print(f"\nDatasets saved to: {output_dir / 'chunked_datasets'}")
    print(f"Label mapping saved to: {output_dir / 'label_mapping.json'}")


if __name__ == "__main__":
    main()
