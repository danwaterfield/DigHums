#!/usr/bin/env python3
"""
Fine-tune BERT for authorship attribution with proper work-level splits.

Key differences from v1:
- Train/test split by WORK, not by chunk -- no data leakage
- Non-overlapping chunks (stride == chunk_size)
- Validation set drawn from training works only (random chunk sample)
- Reports both holdout and LOWO metrics
"""

import json
import time
import numpy as np
from pathlib import Path
from collections import Counter
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch

from corpus import load_metadata, load_work_text, get_project_paths, holdout_split, lowo_splits


def chunk_text_tokens(text: str, tokenizer, chunk_size: int = 512) -> list[str]:
    """
    Chunk text into non-overlapping token windows.
    No overlap = no leakage between chunks.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i : i + chunk_size]
        if len(chunk_tokens) < 64:
            continue
        chunks.append(tokenizer.decode(chunk_tokens))
    return chunks


def works_to_chunks(works, processed_dir, tokenizer, chunk_size=512):
    """Convert works to labelled chunks. Returns list of dicts."""
    all_chunks = []
    for work in works:
        text = load_work_text(work, processed_dir)
        chunks = chunk_text_tokens(text, tokenizer, chunk_size)
        for c in chunks:
            all_chunks.append({
                "text": c,
                "author": work.author,
                "title": work.title,
            })
    return all_chunks


def make_dataset(chunks, author_to_id):
    """Convert chunk dicts to HuggingFace Dataset."""
    return Dataset.from_dict({
        "text": [c["text"] for c in chunks],
        "label": [author_to_id[c["author"]] for c in chunks],
        "author": [c["author"] for c in chunks],
        "title": [c["title"] for c in chunks],
    })


def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def train_and_evaluate(
    train_works, test_works, all_authors, paths, tokenizer, run_name="holdout"
):
    """Train BERT on train_works, evaluate on test_works."""
    processed_dir = paths["processed"]

    author_to_id = {a: i for i, a in enumerate(sorted(all_authors))}
    id_to_author = {i: a for a, i in author_to_id.items()}
    num_labels = len(author_to_id)

    # Chunk texts (non-overlapping)
    print(f"  Chunking {len(train_works)} train works...")
    train_chunks = works_to_chunks(train_works, processed_dir, tokenizer)

    print(f"  Chunking {len(test_works)} test works...")
    test_chunks = works_to_chunks(test_works, processed_dir, tokenizer)

    print(f"  Train: {len(train_chunks)} chunks, Test: {len(test_chunks)} chunks")

    # Split 10% of train chunks for validation (random, from train works only)
    np.random.seed(42)
    indices = np.random.permutation(len(train_chunks))
    val_size = max(1, int(len(train_chunks) * 0.1))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    val_chunk_list = [train_chunks[i] for i in val_indices]
    train_chunk_list = [train_chunks[i] for i in train_indices]

    train_ds = make_dataset(train_chunk_list, author_to_id)
    val_ds = make_dataset(val_chunk_list, author_to_id)
    test_ds = make_dataset(test_chunks, author_to_id)

    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text", "author", "title"])
    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["text", "author", "title"])
    test_ds = test_ds.map(tokenize_fn, batched=True, remove_columns=["text", "author", "title"])

    train_ds.set_format("torch")
    val_ds.set_format("torch")
    test_ds.set_format("torch")

    # Load fresh model
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,
        id2label=id_to_author,
        label2id=author_to_id,
    )

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Device: {device}")

    output_dir = paths["models"] / f"bert_v2_{run_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        save_total_limit=2,
        report_to="none",
        fp16=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # Test evaluation
    test_pred = trainer.predict(test_ds)
    test_preds = np.argmax(test_pred.predictions, axis=1)
    test_labels = test_pred.label_ids
    test_acc = accuracy_score(test_labels, test_preds)

    # Per-class report
    present_labels = sorted(set(test_labels))
    target_names = [id_to_author[i] for i in present_labels]
    report = classification_report(
        test_labels, test_preds,
        labels=present_labels,
        target_names=target_names,
        zero_division=0,
    )

    # Work-level majority vote
    work_votes = []
    chunk_idx = 0
    for work in test_works:
        n = sum(1 for c in test_chunks if c["title"] == work.title and c["author"] == work.author)
        if n == 0:
            continue
        work_preds = test_preds[chunk_idx : chunk_idx + n]
        work_labels_arr = test_labels[chunk_idx : chunk_idx + n]
        chunk_idx += n

        pred_authors = [id_to_author[p] for p in work_preds]
        vote = Counter(pred_authors).most_common(1)[0][0]
        correct = vote == work.author
        work_votes.append({
            "author": work.author,
            "title": work.title,
            "vote": vote,
            "correct": correct,
            "n_chunks": n,
            "chunk_acc": float(accuracy_score(work_labels_arr, work_preds)),
        })

    # Save model
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    return test_acc, report, test_preds.tolist(), test_labels.tolist(), work_votes


def main():
    t0 = time.time()
    paths = get_project_paths()
    works = load_metadata(paths["metadata"])
    all_authors = sorted(set(w.author for w in works))

    print("=" * 65)
    print("BERT AUTHORSHIP ATTRIBUTION v2 (work-level splits)")
    print("=" * 65)
    print(f"\nCorpus: {len(works)} works from {len(all_authors)} authors")
    print(f"Authors: {', '.join(all_authors)}")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # --- Holdout evaluation ---
    print("\n" + "=" * 65)
    print("HOLDOUT EVALUATION")
    print("=" * 65)

    train_works, test_works = holdout_split(works)
    test_authors = set(w.author for w in test_works)

    print(f"\nTrain: {len(train_works)} works")
    print(f"Test:  {len(test_works)} works")
    print("\nHeld-out test works:")
    for w in test_works:
        print(f"  {w.author:12} - {w.title} ({w.year})")

    acc, report, preds, labels, work_votes = train_and_evaluate(
        train_works, test_works, all_authors, paths, tokenizer, run_name="holdout"
    )

    print(f"\nHoldout Chunk Accuracy: {acc:.1%}")
    print(report)

    print("Work-level majority vote:")
    for wv in work_votes:
        mark = "OK" if wv["correct"] else "MISS"
        print(f"  [{mark:4}] {wv['author']:12} - {wv['title']:35} "
              f"-> {wv['vote']} (chunks: {wv['chunk_acc']:.1%})")

    works_correct = sum(1 for wv in work_votes if wv["correct"])
    works_total = len(work_votes)
    print(f"\nHoldout Work Accuracy: {works_correct}/{works_total} = "
          f"{works_correct/works_total:.1%}" if works_total else "N/A")

    # Save results
    paths["results"].mkdir(exist_ok=True)
    result = {
        "model": "bert-base-uncased",
        "split": "work-level holdout",
        "holdout_chunk_accuracy": acc,
        "holdout_work_accuracy": f"{works_correct}/{works_total}",
        "holdout_work_accuracy_pct": works_correct / works_total if works_total else 0,
        "work_votes": work_votes,
        "report": report,
    }

    outfile = paths["results"] / "bert_v2_holdout.json"
    with open(outfile, "w") as f:
        json.dump(result, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed / 60:.1f} min. Results: {outfile}")


if __name__ == "__main__":
    main()
