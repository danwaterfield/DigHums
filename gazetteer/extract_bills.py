"""
Parse the Death by Numbers Bills of Mortality CSV and load annual aggregate
burial totals into the env_mortality table.

Source: Death by Numbers project (deathbynumbers.org)
        https://github.com/chnm/bom — generalbills-parishes CSV
Data file: gazetteer/sources/environmental/bills_of_mortality_generalbills.csv

The CSV encodes annual London-wide burial data (c.1700–1752 from the Laxton
collection), with per-parish columns and summary totals for:
  - "Buried in the 97 Parishes within the Walls"
  - "Buried in the Parishes without the Walls"
  - "Buried in the 13 out-Parishes in Middlesex and Surrey"
  - "Buried in the Parishes in the City and Liberties of Westminster"

We extract the annual total for ALL_LONDON (sum of all four groups) per bill
year (using End Year).

Usage:
    python3 gazetteer/extract_bills.py           # dry run, print stats
    python3 gazetteer/extract_bills.py --write   # write to sensory.db
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sensory_db import init_db, DB_PATH_DEFAULT

BILLS_PATH = (
    Path(__file__).parent / "sources" / "environmental"
    / "bills_of_mortality_generalbills.csv"
)

# Column suffixes we want to sum for the ALL_LONDON total.
# Each of these is preceded by `is_missing` and `is_illegible` columns.
TOTAL_COLUMNS = [
    "Buried in the 97 Parishes within the Walls",
    "Buried in the Parishes without the Walls",
    "Buried in the 13 out-Parishes in Middlesex and Surrey",
    "Buried in the Parishes in the City and Liberties of Westminster",
]


def _safe_int(val: str) -> int | None:
    """Return int if val is a non-empty digit string, else None."""
    v = val.strip()
    if v and v.isdigit():
        return int(v)
    return None


def parse_bills(path: Path) -> list[tuple[int, int]]:
    """Return list of (year, total_burials) from the bills CSV.

    Uses End Year column.  Rows with missing total columns are skipped.
    Only returns rows whose year falls within 1636–1752.
    """
    records: list[tuple[int, int]] = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)

        # Locate column indices
        end_year_idx: int | None = None
        total_idxs: list[int] = []

        for i, h in enumerate(headers):
            if h.strip() == "End Year":
                end_year_idx = i
            for tc in TOTAL_COLUMNS:
                if h.strip() == tc:
                    total_idxs.append(i)

        if end_year_idx is None:
            raise ValueError("Could not locate 'End Year' column in bills CSV")

        for row in reader:
            if not row:
                continue
            # End Year
            year_val = row[end_year_idx].strip() if end_year_idx < len(row) else ""
            if not year_val.isdigit():
                continue
            year = int(year_val)

            total = 0
            any_found = False
            for idx in total_idxs:
                if idx >= len(row):
                    continue
                v = _safe_int(row[idx])
                if v is not None:
                    total += v
                    any_found = True

            if not any_found:
                continue
            records.append((year, total))

    return records


def load(write: bool = False, db_path: Path = DB_PATH_DEFAULT,
         bills_path: Path = BILLS_PATH) -> dict:
    """Parse Bills of Mortality CSV and optionally load into env_mortality.

    Returns stats dict with keys: parsed, inserted, skipped.
    """
    records = parse_bills(bills_path)
    stats = {"parsed": len(records), "inserted": 0, "skipped": 0}

    conn = init_db(db_path)

    for year, burials in records:
        existing = conn.execute(
            "SELECT 1 FROM env_mortality WHERE year=? AND month IS NULL AND parish=?",
            (year, "ALL_LONDON"),
        ).fetchone()
        if existing:
            stats["skipped"] += 1
            continue
        if write:
            conn.execute(
                "INSERT INTO env_mortality (year, month, parish, burials) VALUES (?,?,?,?)",
                (year, None, "ALL_LONDON", burials),
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
                        help="Write mortality records to the database")
    args = parser.parse_args()

    stats = load(write=args.write)

    print(f"Annual bills parsed : {stats['parsed']:,}")
    print(f"Already present     : {stats['skipped']:,}")
    if args.write:
        print(f"Inserted            : {stats['inserted']:,}")
        print("✓ Mortality data written to sensory.db")
    else:
        print("(Dry run — pass --write to commit changes)")


if __name__ == "__main__":
    main()
