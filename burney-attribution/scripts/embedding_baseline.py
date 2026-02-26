#!/usr/bin/env python3
"""
Sentence-transformer + SVM baseline with leave-one-work-out evaluation.

Extracts embeddings from a pre-trained sentence-transformer, then trains
a linear SVM classifier. This gives honest accuracy numbers because
train and test never share text from the same novel.

Optimised: embeds the full corpus once, then re-indexes for each fold.
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter, defaultdict

from corpus import load_metadata, load_work_text, get_project_paths, lowo_splits, holdout_split

EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_WORDS = 500
MIN_CHUNK_WORDS = 50


def chunk_text(text: str, chunk_size: int = CHUNK_WORDS) -> list[str]:
    """Split text into non-overlapping word-level chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i : i + chunk_size]
        if len(chunk_words) < MIN_CHUNK_WORDS:
            continue
        chunks.append(" ".join(chunk_words))
    return chunks


def embed_corpus(works, processed_dir, model):
    """
    Embed every work once. Returns a dict keyed by work.key with
    embeddings array and metadata per work.
    """
    corpus_data = {}
    total_chunks = 0

    for i, work in enumerate(works):
        text = load_work_text(work, processed_dir)
        chunks = chunk_text(text)
        if not chunks:
            print(f"  WARN: {work.author}/{work.title} produced 0 chunks", flush=True)
            continue

        t0 = time.time()
        embeddings = model.encode(chunks, show_progress_bar=False, batch_size=128)
        dt = time.time() - t0

        corpus_data[work.key] = {
            "work": work,
            "embeddings": embeddings,
            "labels": [work.author] * len(chunks),
            "n_chunks": len(chunks),
        }
        total_chunks += len(chunks)
        print(f"  [{i+1:2}/{len(works)}] {work.author:12} {work.title:35} "
              f"{len(chunks):4} chunks  ({dt:.1f}s)", flush=True)

    print(f"\n  Total: {total_chunks} chunks embedded", flush=True)
    return corpus_data


def gather(works, corpus_data):
    """Stack embeddings and labels for a list of works."""
    embs, labels = [], []
    for w in works:
        d = corpus_data.get(w.key)
        if d is None:
            continue
        embs.append(d["embeddings"])
        labels.extend(d["labels"])
    return np.vstack(embs), labels


def run_holdout(works, corpus_data):
    """Run a single holdout split evaluation."""
    train_works, test_works = holdout_split(works)

    print("\n--- Holdout Split ---")
    print(f"Train: {len(train_works)} works, Test: {len(test_works)} works")
    print("\nHeld-out test works:")
    for w in test_works:
        print(f"  {w.author:12} - {w.title} ({w.year})")

    X_train, y_train = gather(train_works, corpus_data)
    X_test, y_test = gather(test_works, corpus_data)
    print(f"\nTrain: {X_train.shape[0]} chunks, Test: {X_test.shape[0]} chunks")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LinearSVC(max_iter=10000, C=1.0, dual="auto")
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nHoldout Chunk Accuracy: {acc:.1%}")

    print("\nWork-level majority vote:")
    work_votes_correct = 0
    work_votes_total = 0
    for w in test_works:
        d = corpus_data.get(w.key)
        if d is None:
            continue
        w_emb = d["embeddings"]
        w_pred = clf.predict(scaler.transform(w_emb))
        vote = Counter(w_pred).most_common(1)[0][0]
        correct = vote == w.author
        work_votes_correct += int(correct)
        work_votes_total += 1
        mark = "OK" if correct else "MISS"
        print(f"  [{mark:4}] {w.author:12} - {w.title:35} -> {vote} "
              f"({'correct' if correct else 'WRONG'})", flush=True)

    print(f"\nHoldout Work Accuracy: {work_votes_correct}/{work_votes_total}")

    print("\nClassification Report (chunk-level):")
    print(classification_report(y_test, y_pred, zero_division=0))

    return acc, y_test, y_pred


def run_lowo(works, corpus_data):
    """Run full leave-one-work-out cross-validation (fast: no re-embedding)."""
    print("\n--- Leave-One-Work-Out Cross-Validation ---")
    print("(Only authors with 2+ works are tested)\n")

    fold_results = []

    for fold_name, held_out, train_works in lowo_splits(works):
        d = corpus_data.get(held_out.key)
        if d is None:
            continue

        X_train, y_train = gather(train_works, corpus_data)
        X_test = d["embeddings"]
        y_test = d["labels"]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LinearSVC(max_iter=10000, C=1.0, dual="auto")
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)

        acc = accuracy_score(y_test, y_pred)
        correct = sum(1 for t, p in zip(y_test, y_pred) if t == p)

        vote_counts = Counter(y_pred)
        work_prediction = vote_counts.most_common(1)[0][0]
        work_correct = work_prediction == held_out.author

        fold_results.append({
            "fold": fold_name,
            "author": held_out.author,
            "title": held_out.title,
            "year": held_out.year,
            "chunk_accuracy": float(acc),
            "n_chunks": len(y_test),
            "correct_chunks": correct,
            "work_prediction": work_prediction,
            "work_correct": work_correct,
        })

        vote_mark = "OK" if work_correct else "MISS"
        print(f"  [{vote_mark:4}] {held_out.author:12} - {held_out.title:35} "
              f"chunks:{acc:5.1%}  vote:{work_prediction:12} "
              f"({'correct' if work_correct else 'WRONG'})", flush=True)

    chunk_accs = [r["chunk_accuracy"] for r in fold_results]
    work_correct = sum(1 for r in fold_results if r["work_correct"])
    work_total = len(fold_results)

    print(f"\n  Chunk-level LOWO mean:  {np.mean(chunk_accs):.1%}")
    print(f"  Work-level LOWO vote:   {work_correct}/{work_total} = {work_correct/work_total:.1%}")
    print(f"  Chunk median:           {np.median(chunk_accs):.1%}")
    print(f"  Chunk min:              {np.min(chunk_accs):.1%}  ({fold_results[int(np.argmin(chunk_accs))]['title']})")
    print(f"  Chunk max:              {np.max(chunk_accs):.1%}  ({fold_results[int(np.argmax(chunk_accs))]['title']})")

    print("\n  Per-author summary:")
    author_accs = defaultdict(list)
    author_votes = defaultdict(list)
    for r in fold_results:
        author_accs[r["author"]].append(r["chunk_accuracy"])
        author_votes[r["author"]].append(r["work_correct"])
    for author in sorted(author_accs):
        cmean = np.mean(author_accs[author])
        vcorrect = sum(author_votes[author])
        vtotal = len(author_votes[author])
        print(f"    {author:12} chunks:{cmean:5.1%}  works:{vcorrect}/{vtotal}")

    return fold_results


def main():
    t0 = time.time()
    paths = get_project_paths()
    works = load_metadata(paths["metadata"])

    print("=" * 65, flush=True)
    print("EMBEDDING BASELINE: Sentence-Transformer + SVM", flush=True)
    print("=" * 65, flush=True)
    print(f"\nCorpus: {len(works)} works from {len(set(w.author for w in works))} authors")
    print(f"Embedding model: {EMBED_MODEL}")
    print(f"Chunk size: {CHUNK_WORDS} words, non-overlapping\n")

    print(f"Loading model...", flush=True)
    model = SentenceTransformer(EMBED_MODEL)
    print(f"Model loaded.\n", flush=True)

    print("Embedding full corpus (one pass)...", flush=True)
    corpus_data = embed_corpus(works, paths["processed"], model)

    embed_time = time.time() - t0
    print(f"\nEmbedding complete in {embed_time:.0f}s\n", flush=True)

    holdout_acc, _, _ = run_holdout(works, corpus_data)
    lowo_results = run_lowo(works, corpus_data)

    paths["results"].mkdir(exist_ok=True)
    lowo_chunk_accs = [r["chunk_accuracy"] for r in lowo_results]
    lowo_work_correct = sum(1 for r in lowo_results if r["work_correct"])
    output = {
        "model": EMBED_MODEL,
        "classifier": "LinearSVC",
        "chunk_size_words": CHUNK_WORDS,
        "overlap": False,
        "holdout_accuracy": float(holdout_acc),
        "lowo_chunk_mean_accuracy": float(np.mean(lowo_chunk_accs)),
        "lowo_work_accuracy": f"{lowo_work_correct}/{len(lowo_results)}",
        "lowo_work_accuracy_pct": float(lowo_work_correct / len(lowo_results)) if lowo_results else 0,
        "lowo_folds": lowo_results,
    }

    outfile = paths["results"] / "embedding_baseline.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. Results saved to {outfile}", flush=True)


if __name__ == "__main__":
    main()
