#!/usr/bin/env python3
"""
Narrative pace analysis engine.

Usage:
    python3 gazetteer/analyse_narrative_pace.py
    python3 gazetteer/analyse_narrative_pace.py --limit 3
"""
import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import spacy
from scipy.signal import savgol_filter

import sys
sys.path.insert(0, str(Path(__file__).parent))
from narrative_pace_corpus import load_corpus
from narrative_pace_classify import classify_sentence

GAZETTEER_DIR = Path(__file__).parent
DB_PATH = GAZETTEER_DIR / "narrative_pace.db"
JSON_PATH = GAZETTEER_DIR / "narrative_pace_data.json"
SENSORY_DB = GAZETTEER_DIR / "sensory.db"
METADATA_PATH = Path(__file__).parent.parent / "burney-attribution" / "data" / "metadata_v2.csv"
ROOT_PATH = Path(__file__).parent.parent / "burney-attribution" / "data" / "processed"
LEXICON_VERSION = "1.0"
N_RESAMPLE = 200
CATEGORIES = ["dialogue", "singulative", "iterative", "description", "fid", "commentary"]


def _init_db(db_path: Path) -> sqlite3.Connection:
    """Drop and recreate tables for a fresh run."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executescript("""
        DROP TABLE IF EXISTS environmental;
        DROP TABLE IF EXISTS sentences;
        DROP TABLE IF EXISTS novels;

        CREATE TABLE novels (
            id TEXT PRIMARY KEY,
            author TEXT NOT NULL,
            title TEXT NOT NULL,
            year INTEGER NOT NULL,
            genre TEXT NOT NULL,
            word_count INTEGER NOT NULL,
            sentence_count INTEGER NOT NULL,
            volume_boundaries TEXT NOT NULL
        );

        CREATE TABLE sentences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            novel_id TEXT NOT NULL REFERENCES novels(id),
            sentence_index INTEGER NOT NULL,
            position REAL NOT NULL,
            text TEXT NOT NULL,
            is_dialogue INTEGER NOT NULL,
            singulative REAL NOT NULL,
            iterative REAL NOT NULL,
            description REAL NOT NULL,
            fid REAL NOT NULL,
            commentary REAL NOT NULL,
            dominant_category TEXT NOT NULL
        );

        CREATE TABLE environmental (
            decade INTEGER PRIMARY KEY,
            smell INTEGER NOT NULL,
            smoke INTEGER NOT NULL,
            pollution INTEGER NOT NULL,
            total INTEGER NOT NULL
        );
    """)
    conn.commit()
    return conn


def _split_paragraphs(text: str, max_chars: int = 10_000) -> list:
    """Split text into paragraph-sized chunks no larger than max_chars.

    Paragraphs are delimited by blank lines.  Chunks that exceed max_chars
    are further split on sentence-ending punctuation so spaCy's parser never
    receives an enormous string at once.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            # Split on sentence boundaries (greedy)
            current = []
            current_len = 0
            # Simple split on ". " / "! " / "? "
            import re as _re
            parts = _re.split(r'(?<=[.!?])\s+', para)
            for part in parts:
                if current_len + len(part) > max_chars and current:
                    chunks.append(" ".join(current))
                    current = [part]
                    current_len = len(part)
                else:
                    current.append(part)
                    current_len += len(part) + 1
            if current:
                chunks.append(" ".join(current))
    return chunks


def _process_novel(nlp, novel: dict) -> list:
    """Run spaCy + classify_sentence on every sentence in the novel.

    Text is split into paragraph-sized chunks before being passed to spaCy so
    that the parser never receives an enormous document.  Sentence indices and
    positions are computed across the full novel.

    Returns a list of score dicts with keys:
        sentence_index, position, text, is_dialogue,
        singulative, iterative, description, fid, commentary, dominant_category
    """
    nlp.max_length = 2_000_000
    text = novel["text"]
    epistolary = novel.get("genre", "").lower() == "epistolary"

    chunks = _split_paragraphs(text)

    # Single pass: collect results, then fix positions once we know total n
    results = []
    for doc in nlp.pipe(chunks, batch_size=32):
        for sent in doc.sents:
            if not sent.text.strip():
                continue
            sent_doc = sent.as_doc()
            classification = classify_sentence(sent_doc, epistolary=epistolary)
            row = {
                "sentence_index": len(results),
                "position": 0.0,  # placeholder; fixed below
                "text": sent.text,
                "is_dialogue": classification["is_dialogue"],
                "singulative": classification["singulative"],
                "iterative": classification["iterative"],
                "description": classification["description"],
                "fid": classification["fid"],
                "commentary": classification["commentary"],
                "dominant_category": classification["dominant_category"],
            }
            results.append(row)

    n = len(results)
    for i, row in enumerate(results):
        row["position"] = i / max(n - 1, 1)

    return results


def _smooth_and_resample(sentence_scores: list, n_out: int = N_RESAMPLE) -> dict:
    """Apply Savitzky-Golay smoothing and resample each category to n_out points.

    Returns a dict with key 'positions' (list of n_out floats) and one key per
    category in CATEGORIES (each a list of n_out floats).
    """
    if not sentence_scores:
        positions = list(np.linspace(0.0, 1.0, n_out))
        return {"positions": positions, **{cat: [0.0] * n_out for cat in CATEGORIES}}

    n = len(sentence_scores)
    raw_positions = np.array([s["position"] for s in sentence_scores])

    # Build raw arrays per category
    raw = {}
    for cat in CATEGORIES:
        if cat == "dialogue":
            raw[cat] = np.array([1.0 if s["is_dialogue"] else 0.0 for s in sentence_scores])
        else:
            raw[cat] = np.array([s[cat] for s in sentence_scores])

    # Savitzky-Golay: window must be odd and >= polyorder+1
    window = max(11, int(n * 0.05))
    if window % 2 == 0:
        window += 1
    # window must not exceed n
    if window > n:
        window = n if n % 2 == 1 else max(1, n - 1)
        if window < 3:
            window = 1  # degenerate: skip smoothing

    out_positions = np.linspace(0.0, 1.0, n_out)
    result = {"positions": out_positions.tolist()}

    for cat in CATEGORIES:
        if window >= 3:
            try:
                smoothed = savgol_filter(raw[cat], window_length=window, polyorder=2)
            except Exception:
                smoothed = raw[cat]
        else:
            smoothed = raw[cat]
        # Clamp to [0, 1]
        smoothed = np.clip(smoothed, 0.0, 1.0)
        # Resample to n_out points via linear interpolation
        resampled = np.interp(out_positions, raw_positions, smoothed)
        result[cat] = resampled.tolist()

    return result


def _compute_summary(sentence_scores: list) -> dict:
    """Compute whole-novel category proportions, normalised to sum to 1.0."""
    if not sentence_scores:
        return {cat: 1.0 / len(CATEGORIES) for cat in CATEGORIES}

    totals = {cat: 0.0 for cat in CATEGORIES}
    for s in sentence_scores:
        totals["dialogue"] += 1.0 if s["is_dialogue"] else 0.0
        for cat in CATEGORIES:
            if cat != "dialogue":
                totals[cat] += s[cat]

    grand_total = sum(totals.values())
    if grand_total > 0:
        return {cat: totals[cat] / grand_total for cat in CATEGORIES}
    else:
        return {cat: 1.0 / len(CATEGORIES) for cat in CATEGORIES}


def _load_environmental(sensory_db: Path) -> list:
    """Query sensory.db for decade-level environmental evidence counts."""
    if not sensory_db.exists():
        return []
    conn = sqlite3.connect(sensory_db)
    cur = conn.cursor()
    cur.execute("""
        SELECT (date_min/10)*10 AS decade,
            SUM(CASE WHEN modality='olfactory' THEN 1 ELSE 0 END) AS smell,
            SUM(CASE WHEN modality='thermal' THEN 1 ELSE 0 END) AS smoke,
            SUM(CASE WHEN modality IN ('olfactory','thermal') THEN 1 ELSE 0 END) AS pollution,
            COUNT(*) AS total
        FROM sensory_evidence
        WHERE source_type!='fiction' AND date_min IS NOT NULL
        GROUP BY decade ORDER BY decade
    """)
    rows = [
        {"decade": row[0], "smell": row[1], "smoke": row[2],
         "pollution": row[3], "total": row[4]}
        for row in cur.fetchall()
    ]
    conn.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description="Narrative pace analysis engine")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N novels (for testing)")
    parser.add_argument("--db", type=Path, default=DB_PATH,
                        help="SQLite output path")
    parser.add_argument("--json", type=Path, default=JSON_PATH,
                        help="JSON output path")
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH,
                        help="Metadata CSV path")
    parser.add_argument("--root", type=Path, default=ROOT_PATH,
                        help="Corpus root directory")
    args = parser.parse_args()

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm", exclude=["ner"])

    print("Loading corpus...")
    corpus = load_corpus(args.metadata, args.root)
    if args.limit is not None:
        corpus = corpus[: args.limit]
    print(f"  {len(corpus)} novel(s) to process")

    print("Initialising database...")
    conn = _init_db(args.db)
    cur = conn.cursor()

    novels_json = []

    for idx, novel in enumerate(corpus, 1):
        print(f"  [{idx}/{len(corpus)}] {novel['author']} — {novel['title']} ({novel['year']})")

        sentence_scores = _process_novel(nlp, novel)
        sentence_count = len(sentence_scores)
        print(f"    {sentence_count} sentences classified")

        # Insert novel row
        cur.execute(
            """INSERT INTO novels
               (id, author, title, year, genre, word_count, sentence_count, volume_boundaries)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                novel["id"],
                novel["author"],
                novel["title"],
                int(novel["year"]),
                novel["genre"],
                novel["word_count"],
                sentence_count,
                json.dumps(novel["volume_boundaries"]),
            ),
        )

        # Insert sentence rows in batches
        batch = []
        for s in sentence_scores:
            batch.append((
                novel["id"],
                s["sentence_index"],
                s["position"],
                s["text"],
                1 if s["is_dialogue"] else 0,
                s["singulative"],
                s["iterative"],
                s["description"],
                s["fid"],
                s["commentary"],
                s["dominant_category"],
            ))
        cur.executemany(
            """INSERT INTO sentences
               (novel_id, sentence_index, position, text, is_dialogue,
                singulative, iterative, description, fid, commentary, dominant_category)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            batch,
        )
        conn.commit()

        # Smooth and build arc
        arc = _smooth_and_resample(sentence_scores, n_out=N_RESAMPLE)
        summary = _compute_summary(sentence_scores)

        novels_json.append({
            "id": novel["id"],
            "author": novel["author"],
            "title": novel["title"],
            "year": int(novel["year"]),
            "genre": novel["genre"],
            "word_count": novel["word_count"],
            "sentence_count": sentence_count,
            "volume_boundaries": novel["volume_boundaries"],
            "arc": arc,
            "summary": summary,
        })

    # Environmental data
    print("Loading environmental data from sensory.db...")
    env_rows = _load_environmental(SENSORY_DB)
    for row in env_rows:
        cur.execute(
            "INSERT OR REPLACE INTO environmental (decade, smell, smoke, pollution, total) VALUES (?,?,?,?,?)",
            (row["decade"], row["smell"], row["smoke"], row["pollution"], row["total"]),
        )
    conn.commit()
    conn.close()
    print(f"  {len(env_rows)} decade(s) of environmental data stored")

    # Write JSON
    output = {
        "novels": novels_json,
        "environmental": env_rows,
        "lexicon_version": LEXICON_VERSION,
        "generated": datetime.now(timezone.utc).isoformat(),
    }
    args.json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"JSON written to {args.json}")
    print("Done.")


if __name__ == "__main__":
    main()
