"""
Pass 1 sensory extraction pipeline.

For each source, scans the text for sensory terms (lexicon-based),
extracts a context window around each match, geocodes to a venue,
and writes to the sensory_evidence table.

Usage:
    python3 gazetteer/extract_sensory.py            # dry run, print stats
    python3 gazetteer/extract_sensory.py --write    # write to sensory.db
"""

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sensory_db import init_db, DB_PATH_DEFAULT
from sensory_lexicon import tag_modalities
from venue_geocoder import geocode_passage

SOURCES_DIR  = Path(__file__).parent / "sources"
CATALOG_PATH = Path(__file__).parent / "sources_catalog.csv"
VENUES_PATH  = Path(__file__).parent / "venues.csv"

WINDOW_CHARS = 400
STRIDE_CHARS = 200


def load_venues(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def strip_gutenberg(text: str) -> str:
    """Remove Project Gutenberg header/footer boilerplate."""
    start_markers = ["*** START OF", "***START OF"]
    end_markers   = ["*** END OF",   "***END OF"]
    for m in start_markers:
        idx = text.find(m)
        if idx != -1:
            nl = text.find("\n", idx)
            text = text[nl+1:]
            break
    for m in end_markers:
        idx = text.find(m)
        if idx != -1:
            text = text[:idx]
            break
    return text.strip()


def extract_from_text(
    text: str,
    source_id: str,
    source_type: str,
    author: str,
    title: str,
    pub_year: int,
    date_min: int,
    date_max: int,
    venues: list[dict],
    conn,
    write: bool = False,
) -> list[dict]:
    """
    Scan text for sensory passages. Return list of evidence dicts.
    If write=True, also INSERT into sensory_evidence.
    """
    results = []
    last_offset: dict[str, int] = {}

    text_len = len(text)

    for m in re.finditer(r"\S+", text):
        word_start = m.start()
        fragment = text[max(0, word_start - WINDOW_CHARS):
                        word_start + WINDOW_CHARS]
        matches = tag_modalities(fragment)
        if not matches:
            continue

        for term, modality in matches:
            prev = last_offset.get(modality, -STRIDE_CHARS * 2)
            if word_start - prev < STRIDE_CHARS:
                continue
            last_offset[modality] = word_start

            ctx_start = max(0, word_start - WINDOW_CHARS)
            ctx_end   = min(text_len, word_start + WINDOW_CHARS)
            passage   = text[ctx_start:ctx_end].strip()

            geo = geocode_passage(text, passage[:60], venues)

            pos = round(word_start / text_len, 4) if text_len > 0 else 0.0

            row = {
                "source_id":   source_id,
                "venue_id":    geo["venue_id"]   if geo else None,
                "venue_name":  geo["venue_name"] if geo else None,
                "lat":         geo["lat"]        if geo else None,
                "lon":         geo["lon"]        if geo else None,
                "source_type": source_type,
                "author":      author,
                "title":       title,
                "pub_year":    pub_year,
                "date_min":    date_min,
                "date_max":    date_max,
                "modality":    modality,
                "text":        passage[:500],
                "context":     term,
                "char_offset": word_start,
                "pos":         pos,
                "confidence":  1.0,
            }
            results.append(row)

            if write:
                conn.execute("""
                    INSERT OR IGNORE INTO sources
                    (source_id, source_type, author, title,
                     pub_year, date_min, date_max)
                    VALUES
                    (:source_id, :source_type, :author, :title,
                     :pub_year, :date_min, :date_max)
                """, row)
                conn.execute("""
                    INSERT INTO sensory_evidence
                    (source_id, venue_id, venue_name, lat, lon,
                     source_type, author, title, pub_year, date_min,
                     date_max, modality, text, context, char_offset,
                     pos, confidence)
                    VALUES
                    (:source_id, :venue_id, :venue_name, :lat, :lon,
                     :source_type, :author, :title, :pub_year, :date_min,
                     :date_max, :modality, :text, :context, :char_offset,
                     :pos, :confidence)
                """, row)

    if write:
        conn.commit()
    return results


def run(write: bool = False):
    venues = load_venues(VENUES_PATH)
    conn   = init_db(DB_PATH_DEFAULT)

    with open(CATALOG_PATH, newline="", encoding="utf-8") as f:
        catalog = list(csv.DictReader(f))

    total = 0
    for entry in catalog:
        sid = entry["source_id"]
        if sid == "fiction_corpus":
            print(f"  [skip] fiction_corpus — use extract_fiction.py")
            continue

        fp = SOURCES_DIR / entry["file_path"]
        if not fp.exists():
            print(f"  [missing] {fp}")
            continue

        text = strip_gutenberg(fp.read_text(encoding="utf-8", errors="replace"))
        rows = extract_from_text(
            text=text,
            source_id=sid,
            source_type=entry["source_type"],
            author=entry["author"],
            title=entry["title"],
            pub_year=int(entry["pub_year"]) if entry["pub_year"] else 0,
            date_min=int(entry["date_min"]) if entry["date_min"] else 0,
            date_max=int(entry["date_max"]) if entry["date_max"] else 0,
            venues=venues,
            conn=conn,
            write=write,
        )
        geocoded = sum(1 for r in rows if r["venue_id"])
        print(f"  {sid:30s}  {len(rows):4d} passages  "
              f"{geocoded:3d} geocoded  "
              f"({entry['source_type']})")
        total += len(rows)

    print(f"\nTotal: {total} passages extracted")
    if not write:
        print("(dry run — pass --write to persist)")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    run(write=args.write)
