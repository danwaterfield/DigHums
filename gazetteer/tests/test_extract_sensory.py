import sqlite3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from sensory_db import init_db
from extract_sensory import extract_from_text, WINDOW_CHARS

VENUES = [
    {"id": "LON001", "name": "Vauxhall Spring Gardens",
     "lat": "51.4882", "lon": "-0.1228"},
]

def test_extract_finds_auditory(tmp_path):
    db = init_db(tmp_path / "t.db")
    text = ("We went to Vauxhall last Tuesday. The din of the orchestra "
            "was excessive, and the crowd suffocating.")
    rows = extract_from_text(
        text=text, source_id="test", source_type="diary",
        author="Test", title="Test", pub_year=1760,
        date_min=1758, date_max=1762, venues=VENUES, conn=db
    )
    assert len(rows) > 0
    modalities = {r["modality"] for r in rows}
    assert "auditory" in modalities

def test_extract_geocodes_vauxhall(tmp_path):
    db = init_db(tmp_path / "t.db")
    text = ("We went to Vauxhall last Tuesday. The din of the orchestra "
            "was excessive, and the crowd suffocating.")
    rows = extract_from_text(
        text=text, source_id="test", source_type="diary",
        author="Test", title="Test", pub_year=1760,
        date_min=1758, date_max=1762, venues=VENUES, conn=db
    )
    vauxhall_rows = [r for r in rows if r.get("venue_id") == "LON001"]
    assert len(vauxhall_rows) > 0

def test_extract_writes_to_db(tmp_path):
    db = init_db(tmp_path / "t.db")
    text = ("We went to Vauxhall last Tuesday. The din of the orchestra "
            "was excessive, and the crowd suffocating.")
    extract_from_text(
        text=text, source_id="test_src", source_type="diary",
        author="Test", title="Test", pub_year=1760,
        date_min=1758, date_max=1762, venues=VENUES, conn=db,
        write=True
    )
    count = db.execute(
        "SELECT COUNT(*) FROM sensory_evidence WHERE source_id='test_src'"
    ).fetchone()[0]
    assert count > 0
