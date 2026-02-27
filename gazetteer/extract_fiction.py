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

    # Load corpus_dates for date_min/date_max and primary_cities.
    # Key on title alone (all titles are unique across the corpus).
    # Columns used: setting_period_start, setting_period_end, primary_cities.
    dates: dict[str, tuple[int, int]] = {}
    primary_cities_map: dict[str, str] = {}
    dates_path = Path(__file__).parent / "corpus_dates.csv"
    if dates_path.exists():
        with open(dates_path, newline="") as f:
            for row in csv.DictReader(f):
                title = row["title"]
                start = row.get("setting_period_start", "").strip()
                end   = row.get("setting_period_end",   "").strip()
                if start and end:
                    dates[title] = (int(start), int(end))
                primary_cities_map[title] = row.get("primary_cities", "")

    total = 0
    for work in works:
        try:
            text = strip_gutenberg(load_work_text(work, paths["processed"]))
        except (FileNotFoundError, OSError) as e:
            print(f"  [missing] {work.author} / {work.title}: {e}")
            continue

        sid  = f"fiction_{work.author}_{work.title}".replace(" ", "_")[:60]

        if work.title in dates:
            date_min, date_max = dates[work.title]
        else:
            date_min = work.year - 5
            date_max = work.year + 5

        primary_cities = primary_cities_map.get(work.title, "")

        rows = extract_from_text(
            text=text,
            source_id=sid,
            source_type="fiction",
            author=work.author,
            title=work.title,
            pub_year=work.year,
            date_min=date_min,
            date_max=date_max,
            venues=venues,
            conn=conn,
            write=write,
            primary_cities=primary_cities,
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
