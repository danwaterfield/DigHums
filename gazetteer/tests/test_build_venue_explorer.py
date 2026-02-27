import csv
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from build_venue_explorer import load_data

VENUES_PATH = Path(__file__).parent.parent / "venues.csv"
DB_PATH     = Path(__file__).parent.parent / "sensory.db"


def test_all_venues_present():
    """All 74 venues from venues.csv appear in the output."""
    with open(VENUES_PATH, newline="") as f:
        expected_ids = {row["id"] for row in csv.DictReader(f)}
    venues = load_data(VENUES_PATH, DB_PATH)
    actual_ids = {v["id"] for v in venues}
    assert actual_ids == expected_ids


def test_evidence_counts_match_db():
    """Total evidence count matches sensory_evidence WHERE venue_id IS NOT NULL."""
    conn = sqlite3.connect(DB_PATH)
    db_count = conn.execute(
        "SELECT COUNT(*) FROM sensory_evidence WHERE venue_id IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    venues = load_data(VENUES_PATH, DB_PATH)
    total = sum(len(v["evidence"]) for v in venues)
    assert total == db_count


def test_evidence_sorted_by_date_min():
    """Evidence within each venue is sorted by date_min ascending."""
    venues = load_data(VENUES_PATH, DB_PATH)
    for v in venues:
        dates = [e["date_min"] for e in v["evidence"] if e["date_min"]]
        assert dates == sorted(dates), f"Unsorted evidence at {v['id']}"


def test_venues_have_required_fields():
    """Each venue dict has id, name, lat, lon, evidence list."""
    venues = load_data(VENUES_PATH, DB_PATH)
    for v in venues:
        for field in ("id", "name", "lat", "lon", "evidence"):
            assert field in v, f"Missing field '{field}' in venue {v.get('id')}"
        assert isinstance(v["evidence"], list)
