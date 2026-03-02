"""
Insert hand-curated decade-level coal/smoke burden estimates into env_smoke.

Estimates derived from:
  - Brimblecombe, P. (1987). The Big Smoke: A History of Air Pollution in
    London since Medieval Times. Methuen.
  - Cavert, W.M. (2016). The Smoke of London: Energy and Environment in the
    Early Modern City. Cambridge University Press.

The SO2 index is normalised 0–1 relative to the peak (1820 estimate).
Wind modifier logic (east of LON048 = 1.3×, west of Hyde Park = 0.7×) is
applied at render time in the time map, not stored here.

Usage:
    python3 gazetteer/insert_smoke_data.py           # dry run
    python3 gazetteer/insert_smoke_data.py --write   # write to sensory.db
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sensory_db import init_db, DB_PATH_DEFAULT

# Decade-level estimates for 1660–1820.
# coal_tons_k: approximate annual coal consumption by London (thousands of tons)
# so2_index:   normalised 0–1 (1.0 = peak 1820)
# Source notes follow Brimblecombe 1987 ch.4-5 and Cavert 2016 ch.1-2.
SMOKE_DATA = [
    (1660,  200, 0.15, "Brimblecombe 1987; Cavert 2016"),
    (1670,  220, 0.17, "Brimblecombe 1987"),
    (1680,  250, 0.19, "Brimblecombe 1987; post-Fire rebuild increased coal use"),
    (1690,  280, 0.22, "Brimblecombe 1987"),
    (1700,  320, 0.25, "Cavert 2016; ~600,000 chaldrons Newcastle coal"),
    (1710,  360, 0.28, "Cavert 2016"),
    (1720,  400, 0.31, "Brimblecombe 1987"),
    (1730,  450, 0.35, "Brimblecombe 1987"),
    (1740,  500, 0.39, "Brimblecombe 1987"),
    (1750,  560, 0.44, "Cavert 2016; ~800,000 chaldrons"),
    (1760,  620, 0.49, "Brimblecombe 1987"),
    (1770,  700, 0.55, "Brimblecombe 1987"),
    (1780,  780, 0.61, "Brimblecombe 1987; early industrialisation"),
    (1790,  880, 0.69, "Brimblecombe 1987"),
    (1800,  990, 0.78, "Brimblecombe 1987; coal trade statistics"),
    (1810, 1120, 0.88, "Brimblecombe 1987"),
    (1820, 1270, 1.00, "Brimblecombe 1987; peak pre-railway era"),
]


def load(write: bool = False, db_path: Path = DB_PATH_DEFAULT) -> dict:
    """Insert smoke data into env_smoke.  Skips existing rows."""
    conn = init_db(db_path)
    inserted = 0
    skipped = 0

    for decade_start, coal_tons_k, so2_index, source in SMOKE_DATA:
        existing = conn.execute(
            "SELECT 1 FROM env_smoke WHERE decade_start = ?", (decade_start,)
        ).fetchone()
        if existing:
            skipped += 1
            continue
        if write:
            conn.execute(
                "INSERT INTO env_smoke (decade_start, coal_tons_k, so2_index, source) "
                "VALUES (?,?,?,?)",
                (decade_start, coal_tons_k, so2_index, source),
            )
            inserted += 1

    if write:
        conn.commit()
    conn.close()
    return {"total": len(SMOKE_DATA), "inserted": inserted, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true",
                        help="Write smoke estimates to the database")
    args = parser.parse_args()

    stats = load(write=args.write)
    print(f"Decade estimates   : {stats['total']}")
    print(f"Already present    : {stats['skipped']}")
    if args.write:
        print(f"Inserted           : {stats['inserted']}")
        print("✓ Smoke data written to sensory.db")
    else:
        print("(Dry run — pass --write to commit changes)")


if __name__ == "__main__":
    main()
