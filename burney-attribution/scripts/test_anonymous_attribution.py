#!/usr/bin/env python3
"""
Test anonymous attribution: Can BERT identify authors from their anonymous works?

This script tests whether the trained BERT model can correctly identify authors
when given text from works they published anonymously.
"""

import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import re


def load_text(file_path):
    """Load and clean text from file, removing Project Gutenberg boilerplate."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Remove Project Gutenberg header
    start_markers = [
        '*** START OF',
        'START OF THE PROJECT GUTENBERG',
        '*** START OF THE PROJECT'
    ]
    for marker in start_markers:
        if marker in text:
            text = text[text.find(marker):].split('\n', 1)[1]
            break

    # Remove Project Gutenberg footer
    end_markers = [
        '*** END OF',
        'END OF THE PROJECT GUTENBERG',
        '*** END OF THE PROJECT'
    ]
    for marker in end_markers:
        if marker in text:
            text = text[:text.find(marker)]
            break

    # Clean up
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def chunk_text(text, tokenizer, chunk_size=512, stride=256):
    """Split text into overlapping chunks."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    for i in range(0, len(tokens), stride):
        chunk_tokens = tokens[i:i + chunk_size]
        if len(chunk_tokens) < 100:  # Skip very short chunks
            continue
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)

        if i + chunk_size >= len(tokens):
            break

    return chunks


def predict_chunks(chunks, model, tokenizer, device, id_to_author):
    """Run model predictions on text chunks."""
    predictions = []
    confidences = []

    model.eval()
    with torch.no_grad():
        for chunk in tqdm(chunks, desc="Predicting"):
            # Tokenize
            inputs = tokenizer(
                chunk,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(device)

            # Predict
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

            pred_id = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_id].item()

            predictions.append(id_to_author[pred_id])
            confidences.append(confidence)

    return predictions, confidences


def analyze_predictions(predictions, confidences, true_author):
    """Analyze prediction results."""
    from collections import Counter

    # Count predictions
    pred_counts = Counter(predictions)
    total = len(predictions)

    # Calculate accuracy
    correct = sum(1 for p in predictions if p == true_author)
    accuracy = correct / total if total > 0 else 0

    # Average confidence
    avg_confidence = np.mean(confidences)

    # Confidence for true author
    true_author_confidences = [
        conf for pred, conf in zip(predictions, confidences)
        if pred == true_author
    ]
    avg_true_confidence = np.mean(true_author_confidences) if true_author_confidences else 0

    return {
        'total_chunks': total,
        'accuracy': accuracy,
        'correct_chunks': correct,
        'predictions': dict(pred_counts),
        'avg_confidence': avg_confidence,
        'avg_true_author_confidence': avg_true_confidence
    }


def main():
    """Run anonymous attribution test."""
    print("="*70)
    print("TESTING ANONYMOUS ATTRIBUTION")
    print("="*70)
    print("\nQuestion: Can BERT identify authors from their anonymous works?")
    print()

    # Paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    corpus_dir = project_dir.parent
    model_dir = project_dir / 'models' / 'bert_authorship' / 'final'
    data_dir = project_dir / 'data' / 'bert_data'

    # Check if model exists
    if not model_dir.exists():
        print("❌ Model not found!")
        print(f"Expected location: {model_dir}")
        print("\nPlease train the model first using: python scripts/train_bert.py")
        return

    # Test cases: works that were published anonymously
    test_cases = [
        {
            'author': 'burney',
            'work': 'Evelina',
            'file': corpus_dir / 'FrancesBurney' / 'Evelina.txt',
            'publication': '1778',
            'attribution': 'Anonymous - "By a Lady"',
            'description': "Burney's debut novel, published anonymously to protect her reputation"
        },
        {
            'author': 'radcliffe',
            'work': 'A Sicilian Romance',
            'file': corpus_dir / 'AnnRadcliffe' / 'ASicilianRomance.txt',
            'publication': '1790',
            'attribution': 'Anonymous - "By the Authoress of The Castles of Athlin and Dunbayne"',
            'description': "Radcliffe's second novel, still published without her name"
        },
        {
            'author': 'edgeworth',
            'work': 'Castle Rackrent',
            'file': corpus_dir / 'MariaEdgeworth' / 'CastleRackrent.txt',
            'publication': '1800',
            'attribution': 'Anonymous',
            'description': "First historical novel in English, published without attribution"
        }
    ]

    # Load model and tokenizer
    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

    # Load label mapping
    with open(data_dir / 'label_mapping.json', 'r') as f:
        label_info = json.load(f)
    id_to_author = {int(k): v for k, v in label_info['id_to_author'].items()}

    # Device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)

    # Run tests
    results = {}

    for i, test in enumerate(test_cases, 1):
        print("\n" + "="*70)
        print(f"TEST {i}/{len(test_cases)}: {test['work']} by {test['author'].title()}")
        print("="*70)
        print(f"📖 Work: {test['work']} ({test['publication']})")
        print(f"✍️  Original attribution: {test['attribution']}")
        print(f"ℹ️  {test['description']}")
        print()

        # Check if file exists
        if not test['file'].exists():
            print(f"❌ File not found: {test['file']}")
            continue

        # Load and chunk text
        print("Loading text...")
        text = load_text(test['file'])
        print(f"Text length: {len(text):,} characters")

        print("Chunking text...")
        chunks = chunk_text(text, tokenizer)
        print(f"Created {len(chunks)} chunks")

        # Predict
        print("Running predictions...")
        predictions, confidences = predict_chunks(
            chunks, model, tokenizer, device, id_to_author
        )

        # Analyze
        analysis = analyze_predictions(predictions, confidences, test['author'])
        results[test['work']] = {
            'test_info': test,
            'analysis': analysis
        }

        # Print results
        print("\n" + "-"*70)
        print("RESULTS")
        print("-"*70)
        print(f"✅ Accuracy: {analysis['accuracy']:.1%} ({analysis['correct_chunks']}/{analysis['total_chunks']} chunks)")
        print(f"📊 Average confidence: {analysis['avg_confidence']:.3f}")
        print(f"🎯 Confidence when correct: {analysis['avg_true_author_confidence']:.3f}")
        print("\nPredictions breakdown:")
        for author, count in sorted(analysis['predictions'].items(), key=lambda x: -x[1]):
            pct = count / analysis['total_chunks'] * 100
            marker = "✅" if author == test['author'] else "  "
            print(f"  {marker} {author:12} {count:4} chunks ({pct:5.1f}%)")

    # Overall summary
    print("\n" + "="*70)
    print("SUMMARY: Can BERT Identify Anonymous Authors?")
    print("="*70)

    for work, result in results.items():
        info = result['test_info']
        analysis = result['analysis']
        status = "✅ YES" if analysis['accuracy'] > 0.8 else "⚠️ PARTIAL" if analysis['accuracy'] > 0.5 else "❌ NO"
        print(f"\n{info['work']} ({info['author'].title()}):")
        print(f"  {status} - {analysis['accuracy']:.1%} accuracy")
        print(f"  Published: {info['attribution']}")

    # Save results
    output_file = project_dir / 'results' / 'anonymous_attribution_test.json'
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump({
            'test_date': '2025-02-10',
            'model': str(model_dir),
            'device': device,
            'results': {
                work: {
                    'author': result['test_info']['author'],
                    'work': result['test_info']['work'],
                    'publication': result['test_info']['publication'],
                    'attribution': result['test_info']['attribution'],
                    'accuracy': result['analysis']['accuracy'],
                    'total_chunks': result['analysis']['total_chunks'],
                    'correct_chunks': result['analysis']['correct_chunks'],
                    'predictions': result['analysis']['predictions'],
                    'avg_confidence': result['analysis']['avg_confidence']
                }
                for work, result in results.items()
            }
        }, f, indent=2)

    print(f"\n\n✅ Results saved to: {output_file}")
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
