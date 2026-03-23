import csv
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from build_venue_explorer import load_data, build

VENUES_PATH = Path(__file__).parent.parent / "venues.csv"
DB_PATH     = Path(__file__).parent.parent / "sensory.db"


def test_all_venues_present():
    """All 73 venues from venues.csv appear in the output."""
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
        # SQLite ORDER BY date_min places NULLs first, so null entries appear before
        # dated entries rather than interleaved. We filter them out here and only
        # assert that the non-null dates are in ascending order.
        dates = [e["date_min"] for e in v["evidence"] if e["date_min"]]
        assert dates == sorted(dates), f"Unsorted evidence at {v['id']}"


def test_venues_have_required_fields():
    """Each venue dict has id, name, lat, lon, evidence list."""
    venues = load_data(VENUES_PATH, DB_PATH)
    for v in venues:
        for field in ("id", "name", "lat", "lon", "evidence"):
            assert field in v, f"Missing field '{field}' in venue {v.get('id')}"
        assert isinstance(v["evidence"], list)


def test_no_orphaned_venue_ids():
    """All venue_ids in sensory_evidence exist in venues.csv."""
    with open(VENUES_PATH, newline="") as f:
        csv_ids = {row["id"] for row in csv.DictReader(f)}
    conn = sqlite3.connect(DB_PATH)
    try:
        db_ids = {row[0] for row in conn.execute(
            "SELECT DISTINCT venue_id FROM sensory_evidence WHERE venue_id IS NOT NULL"
        )}
    finally:
        conn.close()
    orphans = db_ids - csv_ids
    assert not orphans, f"Orphaned venue_ids in DB not in venues.csv: {orphans}"


def test_legal_pill_present():
    """Generated HTML must contain a Legal source filter pill."""
    html = (Path(__file__).parent.parent / "venue_explorer.html").read_text()
    assert 'data-v="legal"' in html


def test_new_source_pills_present():
    """Generated HTML must contain pills for newly supported documentary source types."""
    html = (Path(__file__).parent.parent / "venue_explorer.html").read_text()
    assert 'data-v="newspaper"' in html
    assert 'data-v="parish"' in html
    assert 'data-v="institutional"' in html


def test_legal_badge_colour_present():
    """Generated HTML must define the src-legal CSS class."""
    html = (Path(__file__).parent.parent / "venue_explorer.html").read_text()
    assert "src-legal" in html


def test_new_badge_colours_present():
    """Generated HTML must define CSS classes for new documentary source types."""
    html = (Path(__file__).parent.parent / "venue_explorer.html").read_text()
    assert "src-newspaper" in html
    assert "src-parish" in html
    assert "src-institutional" in html


def test_valence_pip_in_js():
    """renderCard must include valence pip logic."""
    html = (Path(__file__).parent.parent / "venue_explorer.html").read_text()
    assert "valence-pip" in html


def test_time_map_link_present():
    """Generated HTML must contain a link to sensory_time_map.html."""
    html = (Path(__file__).parent.parent / "venue_explorer.html").read_text()
    assert "sensory_time_map.html" in html


import tempfile

def test_build_generates_valid_html():
    """Running build() produces an HTML file containing all expected landmarks."""
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        out = Path(f.name)
    from build_venue_explorer import build
    build(VENUES_PATH, DB_PATH, out)
    html = out.read_text(encoding="utf-8")
    assert "const VENUES" in html, "Missing VENUES constant"
    assert "__VENUES_DATA__" not in html, "Placeholder __VENUES_DATA__ was not replaced"
    assert "Vauxhall" in html,     "Missing Vauxhall venue"
    assert "Ranelagh" in html,     "Missing Ranelagh venue"
    assert "leaflet" in html.lower(), "Missing Leaflet"
    assert len(html) > 20_000,    "HTML suspiciously short"
    out.unlink()


def test_bookseller_js_data_in_html():
    """Built HTML contains BOOKSELLERS data object."""
    build()
    html = (Path(__file__).parent.parent / "venue_explorer.html").read_text(encoding="utf-8")
    assert "BOOKSELLERS" in html
    assert "Andrew Millar" in html
