# gazetteer/tests/test_seed_booksellers.py
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

BOOKSELLERS_PATH = Path(__file__).parent.parent / "booksellers.csv"
LOCATIONS_PATH = Path(__file__).parent.parent / "bookseller_locations.csv"


def test_booksellers_csv_exists():
    assert BOOKSELLERS_PATH.exists()


def test_booksellers_has_required_columns():
    with open(BOOKSELLERS_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    for col in ("bookseller_id", "name", "sign", "type", "active_min", "active_max", "notes"):
        assert col in row, f"Missing column: {col}"


def test_locations_csv_exists():
    assert LOCATIONS_PATH.exists()


def test_locations_has_required_columns():
    with open(LOCATIONS_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    for col in ("bookseller_id", "venue_id", "date_min", "date_max", "address_detail", "source"):
        assert col in row, f"Missing column: {col}"


def test_millar_present():
    with open(BOOKSELLERS_PATH, newline="", encoding="utf-8") as f:
        ids = {row["bookseller_id"] for row in csv.DictReader(f)}
    assert "BS_MILLAR" in ids


def test_millar_has_location():
    with open(LOCATIONS_PATH, newline="", encoding="utf-8") as f:
        bs_ids = {row["bookseller_id"] for row in csv.DictReader(f)}
    assert "BS_MILLAR" in bs_ids


def test_location_venue_ids_valid():
    """All venue_ids in locations reference real venues or new bookseller venues."""
    venues_path = Path(__file__).parent.parent / "venues.csv"
    with open(venues_path, newline="", encoding="utf-8") as f:
        venue_ids = {row["id"] for row in csv.DictReader(f)}
    with open(LOCATIONS_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = row["venue_id"]
            if vid:  # Edinburgh booksellers may have empty venue_id
                assert vid in venue_ids, f"Unknown venue_id: {vid}"


def test_no_duplicate_bookseller_ids():
    with open(BOOKSELLERS_PATH, newline="", encoding="utf-8") as f:
        ids = [row["bookseller_id"] for row in csv.DictReader(f)]
    assert len(ids) == len(set(ids))
