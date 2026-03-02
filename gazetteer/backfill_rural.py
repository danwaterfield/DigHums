"""
Assign pseudo-venue RUR001 (Rural England) or RUR002 (Continental Gothic) to
fiction passages whose text has no venue_id and whose corpus entry lists no
London or Bath primary city.

Pseudo-venues:
  RUR001 Rural England   — English / Celtic rural settings (Scotland, Ireland, etc.)
  RUR002 Continental Gothic — Italian / French / Spanish / fantastical settings

The DB stores author as a lowercase last name (e.g. "radcliffe", "beckford"),
while corpus_dates.csv uses CamelCase full names ("AnnRadcliffe").  The helper
_db_author() extracts the lowercase last word from the CamelCase form.

Usage:
    python3 gazetteer/backfill_rural.py           # dry run
    python3 gazetteer/backfill_rural.py --write   # write to sensory.db
"""

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sensory_db import init_db, DB_PATH_DEFAULT

CORPUS_CSV  = Path(__file__).parent / "corpus_dates.csv"

# Keywords that indicate continental / fantastical (→ RUR002) rather than
# rural English / Celtic (→ RUR001).  Matched case-insensitively.
CONTINENTAL_KEYWORDS = [
    "italy", "italian", "sicil", "france", "french", "spain", "spanish",
    "arabia", "fantasy", "europe", "european", "medieval", "orient",
    "gothic",
]


def _classify(primary_cities: str) -> str:
    """Return 'RUR001' or 'RUR002' based on primary_cities string."""
    lc = primary_cities.lower()
    for kw in CONTINENTAL_KEYWORDS:
        if kw in lc:
            return "RUR002"
    return "RUR001"


def _has_urban(primary_cities: str) -> bool:
    """Return True if London or Bath are named in primary_cities."""
    lc = primary_cities.lower()
    return "london" in lc or "bath" in lc


def _db_author(corpus_author: str) -> str:
    """
    Convert CamelCase corpus author name to DB lowercase last name.
    'AnnRadcliffe' → 'radcliffe', 'MGLewis' → 'lewis'.
    Falls back to full lowercase if no CamelCase parts are found.
    """
    parts = re.findall(r"[A-Z][a-z]*", corpus_author)
    return parts[-1].lower() if parts else corpus_author.lower()


def _normalize_title(title: str) -> str:
    """Strip leading 'The ' or 'A ' for fuzzy title matching."""
    for prefix in ("the ", "a "):
        if title.lower().startswith(prefix):
            return title[len(prefix):]
    return title


def load_rural_texts(corpus_csv: Path = CORPUS_CSV) -> list[dict]:
    """
    Return list of dicts {db_author, title, venue_id} for texts whose
    primary_cities contain no London or Bath.
    """
    results = []
    with open(corpus_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cities = row.get("primary_cities", "")
            if not cities or _has_urban(cities):
                continue
            results.append({
                "db_author": _db_author(row["author"]),
                "title":     row["title"],
                "venue_id":  _classify(cities),
            })
    return results


def backfill(
    write: bool = False,
    db_path: Path = DB_PATH_DEFAULT,
    corpus_csv: Path = CORPUS_CSV,
) -> dict:
    """
    Assign RUR001/RUR002 to unlinked fiction passages.
    Only passages that currently have venue_id IS NULL are updated.
    Uses fuzzy title matching (strips leading 'The '/'A ').
    """
    rural_texts = load_rural_texts(corpus_csv)
    conn = init_db(db_path)
    updated = 0
    skipped = 0

    # Build in-DB title lookup: db_author → [(db_title, db_title_normalized)]
    for entry in rural_texts:
        db_author   = entry["db_author"]
        title_norm  = _normalize_title(entry["title"]).lower()
        venue_id    = entry["venue_id"]
        venue_name  = "Rural England" if venue_id == "RUR001" else "Continental Gothic"

        # Fetch candidate rows: match by author, source_type, NULL venue_id
        candidates = conn.execute(
            """
            SELECT id, title FROM sensory_evidence
            WHERE  source_type = 'fiction'
            AND    lower(author) = ?
            AND    venue_id IS NULL
            """,
            (db_author,),
        ).fetchall()

        matched = [
            row_id for row_id, db_title in candidates
            if _normalize_title(db_title).lower() == title_norm
        ]

        if not matched:
            skipped += 1
            continue

        if write:
            placeholders = ",".join("?" * len(matched))
            conn.execute(
                f"""
                UPDATE sensory_evidence
                SET    venue_id   = ?,
                       venue_name = ?
                WHERE  id IN ({placeholders})
                AND    venue_id IS NULL
                """,
                [venue_id, venue_name] + matched,
            )
        updated += len(matched)

    if write:
        conn.commit()
    conn.close()
    return {
        "texts_processed": len(rural_texts),
        "texts_skipped":   skipped,
        "passages_updated": updated,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--write", action="store_true",
                        help="Write updates to the database")
    args = parser.parse_args()

    stats = backfill(write=args.write)
    print(f"Rural texts found  : {stats['texts_processed']}")
    print(f"Texts with no rows : {stats['texts_skipped']}")
    if args.write:
        print(f"Passages updated   : {stats['passages_updated']}")
        print("✓ Rural venue assignment written to sensory.db")
    else:
        print(f"Passages to update : {stats['passages_updated']}")
        print("(Dry run — pass --write to commit changes)")


if __name__ == "__main__":
    main()
