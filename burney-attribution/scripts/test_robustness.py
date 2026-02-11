#!/usr/bin/env python3
"""
Robustness tests for BERT authorship attribution.

Tests to ensure model is learning authorial style, not memorizing content.
"""

import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import re


def load_text(file_path):
    """Load and clean text from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Remove Project Gutenberg boilerplate
    start_markers = ['*** START OF', 'START OF THE PROJECT GUTENBERG']
    for marker in start_markers:
        if marker in text:
            text = text[text.find(marker):].split('\n', 1)[1]
            break

    end_markers = ['*** END OF', 'END OF THE PROJECT GUTENBERG']
    for marker in end_markers:
        if marker in text:
            text = text[:text.find(marker)]
            break

    text = re.sub(r'\s+', ' ', text).strip()
    return text


def chunk_text(text, tokenizer, chunk_size=512, stride=256):
    """Split text into overlapping chunks."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    for i in range(0, len(tokens), stride):
        chunk_tokens = tokens[i:i + chunk_size]
        if len(chunk_tokens) < 100:
            continue
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)

        if i + chunk_size >= len(tokens):
            break

    return chunks


def predict_chunks(chunks, model, tokenizer, device, id_to_author):
    """Run predictions on chunks and return detailed results."""
    predictions = []
    confidences = []
    all_probs = []

    model.eval()
    with torch.no_grad():
        for chunk in tqdm(chunks, desc="Predicting", leave=False):
            inputs = tokenizer(
                chunk,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

            pred_id = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_id].item()

            predictions.append(id_to_author[pred_id])
            confidences.append(confidence)
            all_probs.append({id_to_author[i]: probs[0][i].item() for i in range(len(id_to_author))})

    return predictions, confidences, all_probs


def main():
    """Run robustness tests."""
    print("="*70)
    print("ROBUSTNESS TESTING")
    print("="*70)
    print("\nTesting if model is truly learning authorial style...")
    print()

    # Paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    corpus_dir = project_dir.parent
    model_dir = project_dir / 'models' / 'bert_authorship' / 'final'
    data_dir = project_dir / 'data' / 'bert_data'

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

    with open(data_dir / 'label_mapping.json', 'r') as f:
        label_info = json.load(f)
    id_to_author = {int(k): v for k, v in label_info['id_to_author'].items()}

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)

    print(f"\nModel trained on: {', '.join(label_info['author_to_id'].keys())}")
    print()

    # TEST 1: Out-of-sample authors (NOT in training set)
    print("="*70)
    print("TEST 1: OUT-OF-SAMPLE AUTHORS")
    print("="*70)
    print("Testing authors NOT in training set - model should be uncertain\n")

    out_of_sample_tests = [
        {
            'author': 'walpole',
            'work': 'Castle of Otranto',
            'file': corpus_dir / 'HoraceWalpole' / 'CastleOfOtranto.txt',
            'description': 'Gothic novel by author NOT in training set'
        },
        {
            'author': 'beckford',
            'work': 'Vathek',
            'file': corpus_dir / 'WilliamBeckford' / 'Vathek.txt',
            'description': 'Oriental Gothic by author NOT in training set'
        },
        {
            'author': 'smith',
            'work': 'Emmeline',
            'file': corpus_dir / 'CharlotteSmith' / 'Emmeline.txt',
            'description': 'Domestic novel by author NOT in training set'
        }
    ]

    oos_results = {}

    for test in out_of_sample_tests:
        if not test['file'].exists():
            print(f"⚠️  {test['work']} not found, skipping...")
            continue

        print(f"\n📖 Testing: {test['work']} (by {test['author'].title()})")
        print(f"   {test['description']}")

        text = load_text(test['file'])
        chunks = chunk_text(text, tokenizer)
        print(f"   {len(chunks)} chunks")

        predictions, confidences, all_probs = predict_chunks(
            chunks[:50],  # Sample first 50 chunks for speed
            model, tokenizer, device, id_to_author
        )

        # Analyze
        from collections import Counter
        pred_counts = Counter(predictions)
        avg_confidence = np.mean(confidences)
        max_author = max(pred_counts, key=pred_counts.get)
        max_percentage = pred_counts[max_author] / len(predictions) * 100

        print(f"   📊 Top prediction: {max_author} ({max_percentage:.1f}% of chunks)")
        print(f"   🎯 Average confidence: {avg_confidence:.3f}")

        # Distribution of predictions
        print("   Prediction distribution:")
        for author, count in sorted(pred_counts.items(), key=lambda x: -x[1])[:3]:
            pct = count / len(predictions) * 100
            print(f"      {author:12} {count:3} chunks ({pct:5.1f}%)")

        # Flag if too confident on wrong author
        if avg_confidence > 0.8:
            print("   ⚠️  WARNING: High confidence on unknown author - possible overfitting!")
        elif max_percentage > 80:
            print("   ⚠️  WARNING: Strongly assigns to one author - should be more uncertain!")
        else:
            print("   ✅ Good: Model shows appropriate uncertainty")

        oos_results[test['work']] = {
            'predictions': dict(pred_counts),
            'avg_confidence': avg_confidence,
            'top_author': max_author,
            'top_percentage': max_percentage
        }

    # TEST 2: Radcliffe's first novel (NOT in training)
    print("\n" + "="*70)
    print("TEST 2: TEMPORAL ROBUSTNESS")
    print("="*70)
    print("Testing Radcliffe's FIRST novel (not in training set)\n")

    first_novel_file = corpus_dir / 'AnnRadcliffe' / 'CastlesOfAthlinAndDunbayne.txt'
    if first_novel_file.exists():
        print("📖 Testing: Castles of Athlin and Dunbayne (1789)")
        print("   Radcliffe's debut - NOT in training set")

        text = load_text(first_novel_file)
        chunks = chunk_text(text, tokenizer)
        print(f"   {len(chunks)} chunks")

        predictions, confidences, _ = predict_chunks(
            chunks, model, tokenizer, device, id_to_author
        )

        from collections import Counter
        pred_counts = Counter(predictions)
        accuracy = pred_counts['radcliffe'] / len(predictions) if 'radcliffe' in pred_counts else 0
        avg_confidence = np.mean(confidences)

        print(f"   ✅ Identified as Radcliffe: {accuracy:.1%}")
        print(f"   🎯 Average confidence: {avg_confidence:.3f}")

        if accuracy > 0.8:
            print("   ✅ EXCELLENT: Model recognizes Radcliffe's style in unseen work!")
        else:
            print("   ⚠️  Model struggles with early work - style may have evolved")
    else:
        print("⚠️  Castles of Athlin not found")

    # TEST 3: Very short chunks (degradation test)
    print("\n" + "="*70)
    print("TEST 3: CHUNK SIZE ROBUSTNESS")
    print("="*70)
    print("Testing if accuracy degrades with shorter chunks\n")

    test_file = corpus_dir / 'FrancesBurney' / 'Evelina.txt'
    text = load_text(test_file)

    for chunk_size in [512, 256, 128]:
        chunks = chunk_text(text, tokenizer, chunk_size=chunk_size, stride=chunk_size//2)
        predictions, confidences, _ = predict_chunks(
            chunks[:100],  # Sample
            model, tokenizer, device, id_to_author
        )

        accuracy = sum(1 for p in predictions if p == 'burney') / len(predictions)
        avg_conf = np.mean(confidences)

        print(f"   Chunk size {chunk_size:3}: {accuracy:.1%} accuracy, {avg_conf:.3f} confidence")

    # Summary
    print("\n" + "="*70)
    print("ROBUSTNESS SUMMARY")
    print("="*70)

    print("\n📊 Out-of-sample authors (should show uncertainty):")
    for work, result in oos_results.items():
        status = "✅" if result['avg_confidence'] < 0.8 else "⚠️"
        print(f"   {status} {work}: {result['avg_confidence']:.3f} confidence")

    print("\n✅ Robustness tests complete!")
    print("   See output above for detailed analysis")


if __name__ == "__main__":
    main()
