#!/usr/bin/env python3
"""
Load events.csv, event_venues.csv, event_instances.csv into sensory.db.

Usage:
    python3 gazetteer/extract_events.py          # dry run
    python3 gazetteer/extract_events.py --write  # write to sensory.db
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sensory_db import init_db, DB_PATH_DEFAULT

EVENTS_PATH          = Path(__file__).parent / "events.csv"
EVENT_VENUES_PATH    = Path(__file__).parent / "event_venues.csv"
EVENT_INSTANCES_PATH = Path(__file__).parent / "event_instances.csv"


def _coerce(row: dict) -> dict:
    """Convert empty strings to None and numeric strings to int/float."""
    out = {}
    for k, v in row.items():
        if v == "":
            out[k] = None
        else:
            for conv in (int, float):
                try:
                    out[k] = conv(v)
                    break
                except ValueError:
                    pass
            else:
                out[k] = v
    return out


def run(db_path=DB_PATH_DEFAULT, write: bool = False) -> None:
    conn = init_db(db_path)

    with open(EVENTS_PATH, newline="", encoding="utf-8") as f:
        events = [_coerce(r) for r in csv.DictReader(f)]

    with open(EVENT_VENUES_PATH, newline="", encoding="utf-8") as f:
        event_venues = [_coerce(r) for r in csv.DictReader(f)]

    with open(EVENT_INSTANCES_PATH, newline="", encoding="utf-8") as f:
        event_instances = [_coerce(r) for r in csv.DictReader(f)]

    if write:
        for row in events:
            conn.execute("""
                INSERT OR IGNORE INTO events
                (event_id, name, category, month_start, month_end, day_of_week,
                 time_bands, year_start, year_end, recurrence,
                 smell_load, noise_load, crowd_load, visual_load,
                 calendar_break, month_start_ns, notes, sources)
                VALUES
                (:event_id, :name, :category, :month_start, :month_end, :day_of_week,
                 :time_bands, :year_start, :year_end, :recurrence,
                 :smell_load, :noise_load, :crowd_load, :visual_load,
                 :calendar_break, :month_start_ns, :notes, :sources)
            """, row)

        for row in event_venues:
            conn.execute("""
                INSERT OR IGNORE INTO event_venues (event_id, venue_id)
                VALUES (:event_id, :venue_id)
            """, row)

        for row in event_instances:
            conn.execute("""
                INSERT OR IGNORE INTO event_instances
                (instance_id, event_id, year, month, day, source_id, notes)
                VALUES
                (:instance_id, :event_id, :year, :month, :day, :source_id, :notes)
            """, row)

        conn.commit()

    print(f"  {len(events):3d} events")
    print(f"  {len(event_venues):3d} event-venue links")
    print(f"  {len(event_instances):3d} event instances")
    if not write:
        print("(dry run — pass --write to persist)")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    run(write=args.write)
