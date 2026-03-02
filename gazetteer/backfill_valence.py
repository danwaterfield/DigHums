"""
Backfill the valence column for passages where it is NULL or empty.

Usage:
    python3 gazetteer/backfill_valence.py           # dry run, print stats
    python3 gazetteer/backfill_valence.py --write   # write to sensory.db
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sensory_db import init_db, DB_PATH_DEFAULT
from sensory_lexicon import tag_valence

VALID_VALENCES = {"pleasant", "unpleasant", "neutral"}


def backfill(write: bool = False, db_path: Path = DB_PATH_DEFAULT) -> dict:
    """Tag unvalenced passages and optionally write to DB.

    Returns stats dict with keys: total_unvalenced, tagged, skipped.
    Does NOT overwrite rows that already have a legal valence value.
    """
    conn = init_db(db_path)
    rows = conn.execute(
        "SELECT id, text, valence FROM sensory_evidence "
        "WHERE valence IS NULL OR valence = '' OR valence NOT IN "
        "('pleasant', 'unpleasant', 'neutral')"
    ).fetchall()

    stats = {"total_unvalenced": len(rows), "tagged": 0, "skipped": 0}
    updates: list[tuple[str, int]] = []

    for row_id, text, existing_valence in rows:
        # Extra guard: skip if row somehow slipped through with a valid valence
        if existing_valence in VALID_VALENCES:
            stats["skipped"] += 1
            continue
        valence = tag_valence(text or "")
        updates.append((valence, row_id))
        stats["tagged"] += 1

    if write and updates:
        conn.executemany(
            "UPDATE sensory_evidence SET valence = ? WHERE id = ?", updates
        )
        conn.commit()

    conn.close()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true",
                        help="Write updated valence values to the database")
    args = parser.parse_args()

    stats = backfill(write=args.write)

    print(f"Passages without valence : {stats['total_unvalenced']:,}")
    print(f"Passages tagged          : {stats['tagged']:,}")
    print(f"Passages skipped         : {stats['skipped']:,}")
    if args.write:
        print("✓ Valence values written to sensory.db")
    else:
        print("(Dry run — pass --write to commit changes)")


if __name__ == "__main__":
    main()
