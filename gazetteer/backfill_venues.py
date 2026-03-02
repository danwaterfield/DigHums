"""
Backfill venue_id for fiction passages where it is NULL.

Uses updated VENUE_ALIASES from validate_venues.py to geocode each
passage via a ±500-word context window in the source text.

Usage:
    python3 gazetteer/backfill_venues.py            # dry run, print stats
    python3 gazetteer/backfill_venues.py --write    # write to sensory.db
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sensory_db import init_db, DB_PATH_DEFAULT
from venue_geocoder import geocode_passage, _alias_map_cache

CORPUS_ROOT  = Path(__file__).parent.parent
CORPUS_CSV   = Path(__file__).parent / "corpus_dates.csv"
VENUES_CSV   = Path(__file__).parent / "venues.csv"


def load_venues(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_corpus(path: Path) -> dict[str, dict]:
    """Return {title -> corpus_entry} for fast lookup."""
    with open(path, newline="", encoding="utf-8") as f:
        return {row["title"]: row for row in csv.DictReader(f)}


def backfill(write: bool = False, db_path: Path = DB_PATH_DEFAULT) -> dict:
    """Geocode fiction passages without a venue_id.

    Returns stats dict with keys:
        total_unlinked, geocoded, unchanged
    """
    conn = init_db(db_path)

    rows = conn.execute(
        "SELECT id, source_id, title, text, char_offset "
        "FROM sensory_evidence "
        "WHERE source_type = 'fiction' AND (venue_id IS NULL OR venue_id = '')"
    ).fetchall()

    stats = {"total_unlinked": len(rows), "geocoded": 0, "unchanged": 0}

    if not rows:
        conn.close()
        return stats

    venues   = load_venues(VENUES_CSV)
    corpus   = load_corpus(CORPUS_CSV)

    # Cache full text per file path to avoid repeated reads
    text_cache: dict[str, str] = {}

    # Invalidate any stale alias map cache so updated VENUE_ALIASES are used
    _alias_map_cache.clear()

    updates: list[tuple[str, str, float, float, int]] = []

    for row_id, source_id, title, passage_text, char_offset in rows:
        entry = corpus.get(title)
        if entry is None:
            stats["unchanged"] += 1
            continue

        file_path = CORPUS_ROOT / entry["file_path"]
        if not file_path.exists():
            stats["unchanged"] += 1
            continue

        primary_cities = entry.get("primary_cities", "")

        if str(file_path) not in text_cache:
            text_cache[str(file_path)] = file_path.read_text(
                encoding="utf-8", errors="replace"
            )
        full_text = text_cache[str(file_path)]

        result = geocode_passage(
            full_text, passage_text, venues,
            primary_cities=primary_cities,
        )
        if result is None:
            stats["unchanged"] += 1
            continue

        updates.append((
            result["venue_id"],
            result["venue_name"],
            result["lat"],
            result["lon"],
            row_id,
        ))
        stats["geocoded"] += 1

    if write and updates:
        conn.executemany(
            "UPDATE sensory_evidence "
            "SET venue_id=?, venue_name=?, lat=?, lon=? "
            "WHERE id=?",
            updates,
        )
        conn.commit()

    conn.close()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true",
                        help="Write venue_id values to the database")
    args = parser.parse_args()

    stats = backfill(write=args.write)

    print(f"Fiction passages without venue : {stats['total_unlinked']:,}")
    print(f"Newly geocoded                 : {stats['geocoded']:,}")
    print(f"Could not link                 : {stats['unchanged']:,}")
    if stats["total_unlinked"] > 0:
        pct = 100 * stats["geocoded"] / stats["total_unlinked"]
        print(f"Link rate                      : {pct:.1f}%")
    if args.write:
        print("✓ Venue IDs written to sensory.db")
    else:
        print("(Dry run — pass --write to commit changes)")


if __name__ == "__main__":
    main()
