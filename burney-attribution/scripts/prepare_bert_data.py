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


def chunk_corpus(corpus, tokenizer, chunk_size=512, stride=256):
    """
    Chunk all texts in corpus and return list of chunk dicts.

    Args:
        corpus: List of corpus dicts with text and metadata
        tokenizer: HuggingFace tokenizer
        chunk_size: Tokens per chunk
        stride: Stride for sliding window

    Returns:
        List of chunk dicts with text and metadata
    """
    all_chunks = []

    for item in tqdm(corpus, desc="Chunking texts"):
        chunks = chunk_text(item['text'], tokenizer, chunk_size, stride)

        for chunk in chunks:
            all_chunks.append({
                'text': chunk,
                'author': item['author'],
                'title': item['title'],
                'year': item['year']
            })

    return all_chunks


def split_chunks_stratified(chunks, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42):
    """
    Split chunks into train/val/test stratified by author.

    Ensures all authors appear in all splits by splitting each author's chunks
    according to the specified proportions.

    Args:
        chunks: List of chunk dicts with 'author' field
        train_size, val_size, test_size: Split proportions
        random_state: Random seed

    Returns:
        train_chunks, val_chunks, test_chunks: Lists of chunk dicts
    """
    np.random.seed(random_state)

    # Group chunks by author
    author_chunks = {}
    for chunk in chunks:
        author = chunk['author']
        if author not in author_chunks:
            author_chunks[author] = []
        author_chunks[author].append(chunk)

    # Split each author's chunks
    train_chunks = []
    val_chunks = []
    test_chunks = []

    for author, chunks_list in author_chunks.items():
        # Shuffle chunks for this author
        np.random.shuffle(chunks_list)

        n_chunks = len(chunks_list)
        n_train = int(n_chunks * train_size)
        n_val = int(n_chunks * val_size)
        # Test gets remainder to ensure we use all chunks

        train_chunks.extend(chunks_list[:n_train])
        val_chunks.extend(chunks_list[n_train:n_train+n_val])
        test_chunks.extend(chunks_list[n_train+n_val:])

    return train_chunks, val_chunks, test_chunks


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


def chunks_to_dataset(chunks, author_to_id):
    """
    Convert list of chunk dicts to HuggingFace Dataset.

    Args:
        chunks: List of chunk dicts with text and metadata
        author_to_id: Dict mapping author names to integer IDs

    Returns:
        HuggingFace Dataset
    """
    return Dataset.from_dict({
        'text': [chunk['text'] for chunk in chunks],
        'label': [author_to_id[chunk['author']] for chunk in chunks],
        'author': [chunk['author'] for chunk in chunks],
        'title': [chunk['title'] for chunk in chunks],
        'year': [chunk['year'] for chunk in chunks]
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

    # Create author to ID mapping
    all_authors = sorted(set(item['author'] for item in corpus))
    author_to_id = {author: i for i, author in enumerate(all_authors)}
    id_to_author = {i: author for author, i in author_to_id.items()}

    print(f"\n{len(author_to_id)} authors: {all_authors}")

    # Save label mapping
    with open(output_dir / 'label_mapping.json', 'w') as f:
        json.dump({'author_to_id': author_to_id, 'id_to_author': id_to_author}, f, indent=2)

    # Chunk all texts
    print("\nChunking all texts (512 tokens, 256 stride)...")
    all_chunks = chunk_corpus(corpus, tokenizer, chunk_size=512, stride=256)
    print(f"Created {len(all_chunks)} chunks total")

    # Split chunks stratified by author
    print("\nSplitting chunks (70/15/15 stratified by author)...")
    train_chunks, val_chunks, test_chunks = split_chunks_stratified(all_chunks)

    print(f"Train: {len(train_chunks)} chunks")
    print(f"Val: {len(val_chunks)} chunks")
    print(f"Test: {len(test_chunks)} chunks")

    # Convert to datasets
    print("\nConverting to HuggingFace datasets...")
    train_dataset = chunks_to_dataset(train_chunks, author_to_id)
    val_dataset = chunks_to_dataset(val_chunks, author_to_id)
    test_dataset = chunks_to_dataset(test_chunks, author_to_id)

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
