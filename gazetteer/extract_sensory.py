"""
Pass 1 sensory extraction pipeline.

For each source, scans the text for sensory terms (lexicon-based),
extracts a context window around each match, geocodes to a venue,
and writes to the sensory_evidence table.

Sources listed in sources_catalog.csv may optionally carry
primary_cities metadata so city-filtered aliases can be applied during
geocoding.

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
from sensory_lexicon import tag_modalities, tag_valence
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


def normalize_source_text(text: str) -> str:
    """Normalize common OCR glyphs found in scanned early printed texts."""
    return (text
            .replace("\u00ad", "")
            .replace("\ufb00", "ff")
            .replace("\ufb01", "fi")
            .replace("\ufb02", "fl")
            .replace("\ufb03", "ffi")
            .replace("\ufb04", "ffl")
            .replace("ſ", "s")
            .replace("∫", "s")
            .replace("Æ", "AE")
            .replace("æ", "ae")
            .replace("Œ", "OE")
            .replace("œ", "oe")
            .replace("\f", "\n"))


def _parse_year(value: str) -> int:
    return int(value) if value else 0


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
    primary_cities: str = "",
) -> list[dict]:
    """
    Scan text for sensory passages. Return list of evidence dicts.
    If write=True, also INSERT into sensory_evidence.
    """
    results = []
    last_offset: dict[str, int] = {}

    text_len = len(text)

    for pos_start in range(0, text_len, STRIDE_CHARS):
        fragment = text[pos_start: pos_start + WINDOW_CHARS * 2]
        matches = tag_modalities(fragment)
        if not matches:
            continue

        for term, modality in matches:
            prev = last_offset.get(modality, -STRIDE_CHARS * 2)
            if pos_start - prev < STRIDE_CHARS:
                continue
            last_offset[modality] = pos_start

            # Find absolute position of term in full text, centred on it
            term_pos_in_frag = fragment.lower().find(term.lower())
            abs_term_pos = pos_start + (term_pos_in_frag if term_pos_in_frag >= 0 else 0)

            ctx_start = max(0, abs_term_pos - WINDOW_CHARS // 2)
            ctx_end   = min(text_len, abs_term_pos + WINDOW_CHARS // 2)
            passage   = text[ctx_start:ctx_end].strip()

            geo = geocode_passage(
                text,
                passage,
                venues,
                primary_cities=primary_cities,
                anchor_pos=abs_term_pos,
            )

            pos = round(abs_term_pos / text_len, 4) if text_len > 0 else 0.0

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
                "char_offset": abs_term_pos,
                "pos":         pos,
                "confidence":  1.0,
                "valence":     tag_valence(passage),
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
                    INSERT OR IGNORE INTO sensory_evidence
                    (source_id, venue_id, venue_name, lat, lon,
                     source_type, author, title, pub_year, date_min,
                     date_max, modality, text, context, char_offset,
                     pos, confidence, valence)
                    VALUES
                    (:source_id, :venue_id, :venue_name, :lat, :lon,
                     :source_type, :author, :title, :pub_year, :date_min,
                     :date_max, :modality, :text, :context, :char_offset,
                     :pos, :confidence, :valence)
                """, row)

    if write:
        conn.commit()
    return results


def run(write: bool = False, source_ids: set[str] | None = None):
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
        if source_ids and sid not in source_ids:
            continue

        fp = SOURCES_DIR / entry["file_path"]
        if not fp.exists():
            print(f"  [missing] {fp}")
            continue

        text = normalize_source_text(
            strip_gutenberg(fp.read_text(encoding="utf-8", errors="replace"))
        )
        rows = extract_from_text(
            text=text,
            source_id=sid,
            source_type=entry["source_type"],
            author=entry["author"],
            title=entry["title"],
            pub_year=_parse_year(entry["pub_year"]),
            date_min=_parse_year(entry["date_min"]),
            date_max=_parse_year(entry["date_max"]),
            venues=venues,
            conn=conn,
            write=write,
            primary_cities=entry.get("primary_cities", ""),
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
    parser.add_argument(
        "--source-id",
        action="append",
        default=[],
        help="Restrict extraction to one or more source_id values from sources_catalog.csv",
    )
    args = parser.parse_args()
    run(write=args.write, source_ids=set(args.source_id) or None)
