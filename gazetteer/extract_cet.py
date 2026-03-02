"""
Parse the Met Office HadCET monthly temperature file and load into
the env_temperature table.

Source: Met Office Hadley Centre Central England Temperature (HadCET)
        https://www.metoffice.gov.uk/hadobs/hadcet/
Data file: gazetteer/sources/environmental/cet_monthly.txt

Usage:
    python3 gazetteer/extract_cet.py            # dry run, print stats
    python3 gazetteer/extract_cet.py --write    # write to sensory.db
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sensory_db import init_db, DB_PATH_DEFAULT

CET_PATH = Path(__file__).parent / "sources" / "environmental" / "cet_monthly.txt"
CORPUS_START = 1660
CORPUS_END   = 1820
MISSING      = -99.9


def parse_cet(path: Path) -> list[tuple[int, int, float]]:
    """Parse the CET file and return (year, month, temp_c) tuples.

    Only returns rows for CORPUS_START–CORPUS_END.
    Skips months marked as missing (-99.9).
    Month 0 is used for the annual mean (stored as month=None in DB).
    """
    records: list[tuple[int, int, float]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or not stripped[0].isdigit():
                continue
            parts = stripped.split()
            if len(parts) < 13:
                continue
            try:
                year = int(parts[0])
            except ValueError:
                continue
            if year < CORPUS_START or year > CORPUS_END:
                continue
            # Monthly values (indices 1-12)
            for m_idx, col in enumerate(parts[1:13], start=1):
                try:
                    temp = float(col)
                except ValueError:
                    continue
                if abs(temp - MISSING) < 0.01:
                    continue
                records.append((year, m_idx, temp))
            # Annual mean (index 13, stored as month=0 sentinel → None in DB)
            if len(parts) >= 14:
                try:
                    annual = float(parts[13])
                    if abs(annual - MISSING) >= 0.01:
                        records.append((year, 0, annual))
                except ValueError:
                    pass
    return records


def load(write: bool = False, db_path: Path = DB_PATH_DEFAULT,
         cet_path: Path = CET_PATH) -> dict:
    """Parse CET data and optionally load into env_temperature table.

    Returns stats dict with keys: parsed, inserted, skipped.
    """
    records = parse_cet(cet_path)
    stats = {"parsed": len(records), "inserted": 0, "skipped": 0}

    conn = init_db(db_path)

    for year, month_raw, temp_c in records:
        month = None if month_raw == 0 else month_raw
        existing = conn.execute(
            "SELECT 1 FROM env_temperature WHERE year=? AND month IS ?",
            (year, month),
        ).fetchone()
        if existing:
            stats["skipped"] += 1
            continue
        if write:
            conn.execute(
                "INSERT INTO env_temperature (year, month, temp_c) VALUES (?,?,?)",
                (year, month, temp_c),
            )
            stats["inserted"] += 1

    if write:
        conn.commit()
    conn.close()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true",
                        help="Write temperature records to the database")
    args = parser.parse_args()

    stats = load(write=args.write)

    print(f"Records parsed ({CORPUS_START}–{CORPUS_END}) : {stats['parsed']:,}")
    print(f"Already present (skipped)          : {stats['skipped']:,}")
    if args.write:
        print(f"Inserted                           : {stats['inserted']:,}")
        print("✓ Temperature data written to sensory.db")
    else:
        print("(Dry run — pass --write to commit changes)")


if __name__ == "__main__":
    main()
