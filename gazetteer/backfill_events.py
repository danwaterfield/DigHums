"""
Backfill event_id for passages that have a venue_id but no event_id.

For each passage with a venue_id:
  1. Find all events linked to that venue via the event_venues table.
  2. Filter by temporal overlap (passage date range vs event year_start/year_end).
  3. For irregular / one_off events, additionally require that at least one
     event_instance year falls within the passage date range.
  4. Disambiguate multiple candidates with keyword heuristics; fall back to
     the event with the smallest (earliest) event_id (primary recurring event).

Usage:
    python3 gazetteer/backfill_events.py           # dry run, print stats
    python3 gazetteer/backfill_events.py --write   # write to sensory.db
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sensory_db import init_db, DB_PATH_DEFAULT


# ── Keyword heuristics ─────────────────────────────────────────────────────────
# Maps event_id -> list of regex patterns. If any pattern matches the passage
# text, that event is promoted to the top of the candidate list.

EVENT_KEYWORDS: dict[str, list[re.Pattern]] = {
    "EVT010": [re.compile(p, re.IGNORECASE) for p in [
        r"\bBartholomew\b", r"\bfair\b",
    ]],
    "EVT001": [re.compile(p, re.IGNORECASE) for p in [
        r"\bcattle\b", r"\bmarket\b", r"\bslaughter\b", r"\bdrover\b",
    ]],
    "EVT020": [re.compile(p, re.IGNORECASE) for p in [
        r"\bfrost\b", r"\bfrozen\b", r"\bice\b", r"\bfrost fair\b",
    ]],
    "EVT030": [re.compile(p, re.IGNORECASE) for p in [
        r"\bexecution\b", r"\bhanging\b", r"\bgallows\b", r"\btyburn\b",
    ]],
    "EVT031": [re.compile(p, re.IGNORECASE) for p in [
        r"\bexecution\b", r"\bhanging\b", r"\bgallows\b", r"\bnewgate\b",
    ]],
    "EVT050": [re.compile(p, re.IGNORECASE) for p in [
        r"\blord mayor\b", r"\bmayor\b", r"\bprocession\b", r"\bshow\b",
    ]],
    "EVT051": [re.compile(p, re.IGNORECASE) for p in [
        r"\bgunpowder\b", r"\bfirework\b", r"\bfireworks\b", r"\bbonfire\b",
        r"\bpope night\b",
    ]],
    "EVT061": [re.compile(p, re.IGNORECASE) for p in [
        r"\briots?\b", r"\bgordon\b", r"\bburning\b", r"\bmob\b",
    ]],
}


def _temporal_overlap(
    p_min: int | None, p_max: int | None,
    e_start: int | None, e_end: int | None,
) -> bool:
    """True if [p_min, p_max] overlaps with [e_start, e_end]."""
    if p_min is None or p_max is None or e_start is None or e_end is None:
        return True  # can't rule it out; be inclusive
    return p_min <= e_end and p_max >= e_start


def _instance_in_range(
    instance_years: list[int],
    p_min: int | None,
    p_max: int | None,
) -> bool:
    """True if any instance year falls within the passage date range."""
    if not instance_years:
        return False
    if p_min is None or p_max is None:
        return True
    return any(p_min <= yr <= p_max for yr in instance_years)


def _best_event(
    candidates: list[dict],  # dicts with event_id, recurrence, year_start, year_end
    instance_map: dict[str, list[int]],  # event_id -> [years]
    text: str,
    p_min: int | None,
    p_max: int | None,
) -> str | None:
    """Return the best matching event_id from the candidate list, or None."""
    # Filter by temporal overlap
    filtered = []
    for ev in candidates:
        recurrence = ev["recurrence"]
        if recurrence in ("irregular", "one_off"):
            years = instance_map.get(ev["event_id"], [])
            if not _instance_in_range(years, p_min, p_max):
                continue
        else:
            if not _temporal_overlap(p_min, p_max, ev["year_start"], ev["year_end"]):
                continue
        filtered.append(ev)

    if not filtered:
        return None
    if len(filtered) == 1:
        return filtered[0]["event_id"]

    # Multiple candidates — try keyword heuristics
    for ev in filtered:
        patterns = EVENT_KEYWORDS.get(ev["event_id"], [])
        if any(pat.search(text) for pat in patterns):
            return ev["event_id"]

    # Fall back to event with the smallest (first) event_id
    return sorted(filtered, key=lambda e: e["event_id"])[0]["event_id"]


def backfill(write: bool = False, db_path: Path = DB_PATH_DEFAULT) -> dict:
    """Fill event_id for passages that have venue_id but no event_id.

    Returns stats dict with keys: total_eligible, matched, unchanged.
    """
    conn = init_db(db_path)

    rows = conn.execute(
        "SELECT id, venue_id, text, date_min, date_max "
        "FROM sensory_evidence "
        "WHERE venue_id IS NOT NULL AND venue_id != '' "
        "  AND (event_id IS NULL OR event_id = '')"
    ).fetchall()

    stats = {"total_eligible": len(rows), "matched": 0, "unchanged": 0}

    if not rows:
        conn.close()
        return stats

    # Build venue -> [event candidates] map
    venue_events: dict[str, list[dict]] = {}
    for event_id, venue_id in conn.execute("SELECT event_id, venue_id FROM event_venues"):
        ev_row = conn.execute(
            "SELECT event_id, recurrence, year_start, year_end "
            "FROM events WHERE event_id = ?", (event_id,)
        ).fetchone()
        if ev_row:
            d = {
                "event_id":   ev_row[0],
                "recurrence": ev_row[1],
                "year_start": ev_row[2],
                "year_end":   ev_row[3],
            }
            venue_events.setdefault(venue_id, []).append(d)

    # Build event_id -> [instance years] map
    instance_map: dict[str, list[int]] = {}
    for event_id, year in conn.execute(
        "SELECT event_id, year FROM event_instances WHERE year IS NOT NULL"
    ):
        instance_map.setdefault(event_id, []).append(year)

    updates: list[tuple[str, int]] = []

    for row_id, venue_id, text, date_min, date_max in rows:
        candidates = venue_events.get(venue_id, [])
        if not candidates:
            stats["unchanged"] += 1
            continue

        best = _best_event(candidates, instance_map, text or "", date_min, date_max)
        if best is None:
            stats["unchanged"] += 1
            continue

        updates.append((best, row_id))
        stats["matched"] += 1

    if write and updates:
        conn.executemany(
            "UPDATE sensory_evidence SET event_id = ? WHERE id = ?", updates
        )
        conn.commit()

    conn.close()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true",
                        help="Write event_id values to the database")
    args = parser.parse_args()

    stats = backfill(write=args.write)

    print(f"Passages eligible (has venue, no event) : {stats['total_eligible']:,}")
    print(f"Events matched                          : {stats['matched']:,}")
    print(f"Could not match                         : {stats['unchanged']:,}")
    if stats["total_eligible"] > 0:
        pct = 100 * stats["matched"] / stats["total_eligible"]
        print(f"Match rate                              : {pct:.1f}%")
    if args.write:
        print("✓ Event IDs written to sensory.db")
    else:
        print("(Dry run — pass --write to commit changes)")


if __name__ == "__main__":
    main()
