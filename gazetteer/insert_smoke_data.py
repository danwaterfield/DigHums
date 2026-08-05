"""
Insert hand-curated decade-level coal/smoke burden estimates into env_smoke.

Estimates derived from:
  - Brimblecombe, P. (1987). The Big Smoke: A History of Air Pollution in
    London since Medieval Times. Methuen.
  - Cavert, W.M. (2016). The Smoke of London: Energy and Environment in the
    Early Modern City. Cambridge University Press.
  - Smil, V. (2017). Energy and Civilization: A History. MIT Press.
    Combustion-efficiency data used to derive per-decade emission factors:
    open hearths (~10-12% thermal efficiency, high particulate yield) vs.
    improved grates and Rumford-era fireplaces (~18-22% efficiency, lower
    emissions per ton of coal burned).

The SO2 index is normalised 0–1 relative to the peak effective burden (1820).
Effective burden = coal_tons_k × emission_factor, where emission_factor
captures the declining pollution per ton as combustion technology improved.

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
# coal_tons_k:     approximate annual coal consumption by London (thousands of tons)
# emission_factor: relative pollution per ton (1.0 = 1820 baseline); higher values
#                  reflect less efficient combustion in earlier decades (Smil 2017).
#                  1660s open hearths: ~10-12% thermal efficiency → 1.80×
#                  1700s improved grates spreading: ~14% → 1.50×
#                  1750s better grates widespread: ~16% → 1.18×
#                  1790s Rumford fireplace (1796): ~20% → 1.03×
#                  1820 reference baseline: ~22% → 1.00×
# so2_index:       normalised 0–1, computed as (coal_tons_k × emission_factor) / peak
# Source notes follow Brimblecombe 1987 ch.4-5, Cavert 2016 ch.1-2, and Smil 2017.

# (decade_start, coal_tons_k, emission_factor, so2_index, source)
SMOKE_DATA = [
    (1660,  200, 1.80, 0.28, "Brimblecombe 1987; Cavert 2016; Smil 2017 (open hearth)"),
    (1670,  220, 1.75, 0.30, "Brimblecombe 1987; Smil 2017"),
    (1680,  250, 1.70, 0.33, "Brimblecombe 1987; post-Fire rebuild; Smil 2017"),
    (1690,  280, 1.60, 0.35, "Brimblecombe 1987; Smil 2017"),
    (1700,  320, 1.50, 0.38, "Cavert 2016; ~600k chaldrons; Smil 2017 (improved grates)"),
    (1710,  360, 1.42, 0.40, "Cavert 2016; Smil 2017"),
    (1720,  400, 1.35, 0.43, "Brimblecombe 1987; Smil 2017"),
    (1730,  450, 1.28, 0.45, "Brimblecombe 1987; Smil 2017"),
    (1740,  500, 1.22, 0.48, "Brimblecombe 1987; Smil 2017"),
    (1750,  560, 1.18, 0.52, "Cavert 2016; ~800k chaldrons; Smil 2017"),
    (1760,  620, 1.14, 0.56, "Brimblecombe 1987; Smil 2017"),
    (1770,  700, 1.10, 0.61, "Brimblecombe 1987; Smil 2017"),
    (1780,  780, 1.06, 0.65, "Brimblecombe 1987; Smil 2017 (early industrialisation)"),
    (1790,  880, 1.03, 0.71, "Brimblecombe 1987; Smil 2017 (Rumford era)"),
    (1800,  990, 1.01, 0.79, "Brimblecombe 1987; Smil 2017"),
    (1810, 1120, 1.00, 0.88, "Brimblecombe 1987; Smil 2017"),
    (1820, 1270, 1.00, 1.00, "Brimblecombe 1987; peak pre-railway era"),
]


def load(write: bool = False, db_path: Path = DB_PATH_DEFAULT) -> dict:
    """Insert or update smoke data in env_smoke."""
    conn = init_db(db_path)

    # Ensure emission_factor column exists (migration for existing DBs)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(env_smoke)").fetchall()]
    if "emission_factor" not in cols:
        conn.execute("ALTER TABLE env_smoke ADD COLUMN emission_factor REAL")

    inserted = 0
    updated = 0
    skipped = 0

    for decade_start, coal_tons_k, emission_factor, so2_index, source in SMOKE_DATA:
        existing = conn.execute(
            "SELECT so2_index, emission_factor FROM env_smoke WHERE decade_start = ?",
            (decade_start,),
        ).fetchone()
        if existing:
            old_so2, old_ef = existing
            if old_so2 != so2_index or old_ef != emission_factor:
                if write:
                    conn.execute(
                        "UPDATE env_smoke SET coal_tons_k=?, emission_factor=?, "
                        "so2_index=?, source=? WHERE decade_start=?",
                        (coal_tons_k, emission_factor, so2_index, source, decade_start),
                    )
                updated += 1
            else:
                skipped += 1
            continue
        if write:
            conn.execute(
                "INSERT INTO env_smoke "
                "(decade_start, coal_tons_k, emission_factor, so2_index, source) "
                "VALUES (?,?,?,?,?)",
                (decade_start, coal_tons_k, emission_factor, so2_index, source),
            )
            inserted += 1

    if write:
        conn.commit()
    conn.close()
    return {"total": len(SMOKE_DATA), "inserted": inserted,
            "updated": updated, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true",
                        help="Write smoke estimates to the database")
    args = parser.parse_args()

    stats = load(write=args.write)
    print(f"Decade estimates   : {stats['total']}")
    print(f"Unchanged          : {stats['skipped']}")
    print(f"Need update        : {stats['updated']}")
    if args.write:
        print(f"Inserted           : {stats['inserted']}")
        print(f"Updated            : {stats['updated']}")
        print("✓ Smoke data written to sensory.db")
    else:
        print(f"Would insert       : {stats['inserted']}")
        print("(Dry run — pass --write to commit changes)")


if __name__ == "__main__":
    main()
