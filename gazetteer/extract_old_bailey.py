#!/usr/bin/env python3
"""
Phase 2: Old Bailey Proceedings extraction.

Queries the Old Bailey API by venue alias, caches raw JSON responses,
extracts sensory passages using the existing lexicon tagger, and writes
to sensory.db with venue_id set directly (no geocoding needed).

Usage:
    python3 gazetteer/extract_old_bailey.py            # dry run
    python3 gazetteer/extract_old_bailey.py --write    # write to sensory.db
"""

import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sensory_db import init_db, DB_PATH_DEFAULT
from sensory_lexicon import tag_modalities, tag_valence
from validate_venues import VENUE_ALIASES

VENUES_PATH = Path(__file__).parent / "venues.csv"
CACHE_DIR   = Path(__file__).parent / "sources" / "legal"
API_BASE    = "https://www.dhi.ac.uk/api/data/oldbailey_record"

DATE_MIN   = 1660
DATE_MAX   = 1820
PAGE_SIZE  = 10
SLEEP_SECS = 0.3

# IDs of the venues added in Phase 2 to query against Old Bailey
TARGET_IDS = {
    "LON074", "LON075", "LON076", "LON077", "LON078", "LON079",
    "LON080", "LON081",
    "LON082", "LON083", "LON084", "LON085",
    "LON086", "LON087", "LON088", "LON089", "LON090", "LON091",
    "LON092", "LON093",
    "LON094", "LON095",
}


def parse_date(date_int: int) -> int:
    """Extract year from YYYYMMDD integer."""
    return date_int // 10000


def alias_slug(alias: str) -> str:
    """Convert alias to a safe filename fragment."""
    return re.sub(r"[^a-z0-9]+", "_", alias.lower()).strip("_")


def fetch_page(alias: str, offset: int, cache_path: Path) -> dict:
    """Fetch one API page, returning cached JSON if available."""
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    encoded = urllib.parse.quote(alias)
    url = f"{API_BASE}?text={encoded}&_limit={PAGE_SIZE}&_offset={offset}"
    with urllib.request.urlopen(url, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    time.sleep(SLEEP_SECS)
    return data


def extract_trial(
    text: str,
    venue_id: str,
    venue_name: str,
    lat: float,
    lon: float,
    reference: str,
    year: int,
) -> list[dict]:
    """Extract sensory passages from a single trial text.

    Venue is known from the search alias — no geocoding required.
    All legal passages default to valence='unpleasant'.
    """
    results: list[dict] = []
    seen_offsets: set[int] = set()

    for term, modality in tag_modalities(text):
        term_pos = text.lower().find(term.lower())
        if term_pos < 0 or term_pos in seen_offsets:
            continue
        seen_offsets.add(term_pos)

        ctx_start = max(0, term_pos - 200)
        ctx_end   = min(len(text), term_pos + 200)
        passage   = text[ctx_start:ctx_end].strip()

        results.append({
            "source_id":   f"old_bailey_{reference}",
            "venue_id":    venue_id,
            "venue_name":  venue_name,
            "lat":         float(lat),
            "lon":         float(lon),
            "source_type": "legal",
            "author":      "Old Bailey Proceedings",
            "title":       f"Trial {reference}",
            "pub_year":    year,
            "date_min":    year,
            "date_max":    year,
            "modality":    modality,
            "text":        passage[:500],
            "context":     term,
            "char_offset": term_pos,
            "pos":         round(term_pos / max(len(text), 1), 4),
            "confidence":  1.0,
            "valence":     "unpleasant",
        })

    return results


def ingest_venue(
    venue: dict,
    aliases: list[tuple],
    conn,
    write: bool = False,
) -> list[dict]:
    """Query Old Bailey API for all aliases of one venue."""
    vid  = venue["id"]
    lat  = venue["lat"]
    lon  = venue["lon"]
    name = venue["name"]
    all_rows: list[dict] = []

    for alias, _city_filter in aliases:
        offset = 0
        while True:
            slug       = alias_slug(alias)
            cache_path = CACHE_DIR / f"{vid}_{slug}_p{offset // PAGE_SIZE}.json"
            data       = fetch_page(alias, offset, cache_path)

            records = data.get("records", [])
            if not records:
                break

            for record in records:
                date_int = record.get("date", 0)
                year     = parse_date(date_int)
                if not (DATE_MIN <= year <= DATE_MAX):
                    continue

                ref  = record.get("reference", f"unknown_{date_int}")
                text = record.get("text", "")
                if not text:
                    continue

                rows = extract_trial(text, vid, name, lat, lon, ref, year)
                all_rows.extend(rows)

                if write:
                    for row in rows:
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

            if len(records) < PAGE_SIZE:
                break
            offset += PAGE_SIZE

    if write:
        conn.commit()

    return all_rows


def run(write: bool = False) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with open(VENUES_PATH, newline="", encoding="utf-8") as f:
        venues = {row["id"]: row for row in csv.DictReader(f)}

    conn  = init_db(DB_PATH_DEFAULT)
    total = 0

    for vid in sorted(TARGET_IDS):
        venue   = venues.get(vid)
        aliases = VENUE_ALIASES.get(vid, [])
        if not venue or not aliases:
            print(f"  [skip] {vid} — no venue or no aliases")
            continue

        rows = ingest_venue(venue, aliases, conn, write=write)
        print(f"  {vid:8s} {venue['name']:35s} {len(rows):4d} passages")
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
