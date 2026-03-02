"""Tests for extract_bills.py"""
import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from sensory_db import init_db
from extract_bills import parse_bills, load

# ── Minimal CSV fixture ────────────────────────────────────────────────────────
# Uses only the columns that parse_bills() looks for, by exact name.
# Columns: End Year | <4 burial totals each followed by is_missing, is_illegible>

FIELDNAMES = [
    "End Year", "im_ey", "il_ey",
    "Buried in the 97 Parishes within the Walls",              "im_w", "il_w",
    "Buried in the Parishes without the Walls",                "im_ow", "il_ow",
    "Buried in the 13 out-Parishes in Middlesex and Surrey",   "im_ms", "il_ms",
    "Buried in the Parishes in the City and Liberties of Westminster", "im_we", "il_we",
]


def _row(end_year, w, ow, ms, west):
    return {
        "End Year": str(end_year), "im_ey": "", "il_ey": "",
        "Buried in the 97 Parishes within the Walls":              str(w),    "im_w": "",  "il_w": "",
        "Buried in the Parishes without the Walls":                str(ow),   "im_ow": "", "il_ow": "",
        "Buried in the 13 out-Parishes in Middlesex and Surrey":   str(ms),   "im_ms": "", "il_ms": "",
        "Buried in the Parishes in the City and Liberties of Westminster": str(west), "im_we": "", "il_we": "",
    }


@pytest.fixture
def sample_bills_file(tmp_path):
    p = tmp_path / "test_bills.csv"
    rows = [
        _row(1701, 2691, 4783, 5440, 3631),  # total = 16545
        _row(1702, 2512, 5902, 6921, 4146),  # total = 19481
        _row(1703, "",   5000, 4000, 3000),   # empty 'within walls' → sum = 12000
    ]
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)
    return p


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_parse_returns_correct_years(sample_bills_file):
    records = parse_bills(sample_bills_file)
    years = [r[0] for r in records]
    assert 1701 in years
    assert 1702 in years


def test_parish_aggregation_sums_four_groups(sample_bills_file):
    records = parse_bills(sample_bills_file)
    r1701 = next((r for r in records if r[0] == 1701), None)
    assert r1701 is not None
    assert r1701[1] == 2691 + 4783 + 5440 + 3631  # = 16545


def test_row_with_missing_column_still_aggregates_present(sample_bills_file):
    """Row 1703 has an empty 'within walls' value — should sum the others."""
    records = parse_bills(sample_bills_file)
    r1703 = next((r for r in records if r[0] == 1703), None)
    # 0 + 5000 + 4000 + 3000 = 12000
    assert r1703 is not None
    assert r1703[1] == 5000 + 4000 + 3000


def test_load_writes_to_db(tmp_path, sample_bills_file):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    stats = load(write=True, db_path=db_path, bills_path=sample_bills_file)
    assert stats["inserted"] >= 2


def test_load_is_idempotent(tmp_path, sample_bills_file):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    load(write=True, db_path=db_path, bills_path=sample_bills_file)
    stats2 = load(write=True, db_path=db_path, bills_path=sample_bills_file)
    assert stats2["inserted"] == 0
