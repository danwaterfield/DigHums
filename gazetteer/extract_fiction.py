#!/usr/bin/env python3
"""
Extract sensory passages from the fiction corpus into sensory.db.

Reuses extract_sensory.extract_from_text and the existing corpus.py
metadata loading. Run after extract_sensory.py has already populated
the non-fiction sources.

Usage:
    python3 gazetteer/extract_fiction.py            # dry run
    python3 gazetteer/extract_fiction.py --write    # write to sensory.db
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "burney-attribution" / "scripts"))

from sensory_db    import init_db, DB_PATH_DEFAULT
from extract_sensory import extract_from_text, load_venues, strip_gutenberg
from corpus import load_metadata, load_work_text, get_project_paths

VENUES_PATH = Path(__file__).parent / "venues.csv"


def run(write: bool = False):
    paths   = get_project_paths()
    works   = load_metadata(paths["metadata"])
    venues  = load_venues(VENUES_PATH)
    conn    = init_db(DB_PATH_DEFAULT)

    # Load corpus_dates for date_min/date_max (text_year from corpus_dates.csv)
    dates: dict[tuple[str,str], int] = {}
    dates_path = Path(__file__).parent / "corpus_dates.csv"
    if dates_path.exists():
        with open(dates_path, newline="") as f:
            for row in csv.DictReader(f):
                key = (row["author"], row["title"])
                if row.get("text_year"):
                    dates[key] = int(row["text_year"])

    total = 0
    for work in works:
        text = strip_gutenberg(load_work_text(work, paths["processed"]))
        key  = (work.author, work.title)
        ty   = dates.get(key, work.year)
        sid  = f"fiction_{work.author}_{work.title}".replace(" ", "_")[:60]

        rows = extract_from_text(
            text=text,
            source_id=sid,
            source_type="fiction",
            author=work.author,
            title=work.title,
            pub_year=work.year,
            date_min=ty - 5,
            date_max=ty + 5,
            venues=venues,
            conn=conn,
            write=write,
        )
        geocoded = sum(1 for r in rows if r["venue_id"])
        print(f"  {work.author:20s} {work.title:35s}  "
              f"{len(rows):4d} passages  {geocoded:3d} geocoded")
        total += len(rows)

    print(f"\nTotal fiction passages: {total}")
    if not write:
        print("(dry run — pass --write to persist)")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    run(write=args.write)
