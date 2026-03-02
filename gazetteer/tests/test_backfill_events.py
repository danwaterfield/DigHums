"""Tests for backfill_events.py."""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from sensory_db import init_db
from backfill_events import backfill


# ── DB factory ────────────────────────────────────────────────────────────────

def _make_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)

    # Sources
    conn.execute(
        "INSERT INTO sources (source_id, source_type, author, title) VALUES (?,?,?,?)",
        ("S1", "fiction", "Test", "Test Novel"),
    )

    # Events
    conn.executemany(
        "INSERT INTO events "
        "(event_id, name, category, time_bands, year_start, year_end, recurrence) "
        "VALUES (?,?,?,?,?,?,?)",
        [
            ("EVT001", "Smithfield Market",    "market",    "daytime", 1660, 1820, "weekly"),
            ("EVT010", "Bartholomew Fair",     "fair",      "daytime", 1660, 1820, "annual"),
            ("EVT020", "Thames Frost Fair",    "weather",   "all_day",    1660, 1820, "irregular"),
            ("EVT061", "Gordon Riots",         "civil",     "all_day",    1780, 1780, "one_off"),
        ],
    )

    # Event → venue links
    conn.executemany(
        "INSERT INTO event_venues (event_id, venue_id) VALUES (?,?)",
        [
            ("EVT001", "LON082"),   # Smithfield: both EVT001 and EVT010
            ("EVT010", "LON082"),
            ("EVT020", "LON048"),   # City of London: Frost Fair
            ("EVT061", "LON048"),   # City of London: Gordon Riots
        ],
    )

    # Event instances (for irregular / one_off)
    conn.executemany(
        "INSERT INTO event_instances (instance_id, event_id, year) VALUES (?,?,?)",
        [
            ("INS001", "EVT020", 1684),
            ("INS002", "EVT020", 1740),
            ("INS003", "EVT020", 1814),
            ("INS012", "EVT061", 1780),
        ],
    )

    conn.commit()
    conn.close()
    return db_path


def _insert_passage(db_path, row_id_hint, venue_id, text, date_min, date_max):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO sensory_evidence "
        "(source_id, source_type, author, title, modality, text, char_offset, "
        " venue_id, venue_name, lat, lon, date_min, date_max) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("S1", "fiction", "Test", "Test Novel", "olfactory",
         text, row_id_hint, venue_id, "Test Venue", 51.5, -0.1,
         date_min, date_max),
    )
    conn.commit()
    row_id = conn.execute(
        "SELECT id FROM sensory_evidence WHERE char_offset = ?", (row_id_hint,)
    ).fetchone()[0]
    conn.close()
    return row_id


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_smithfield_cattle_gets_evt001(tmp_path):
    """Passage with 'cattle' at Smithfield should be linked to EVT001."""
    db_path = _make_db(tmp_path)
    _insert_passage(db_path, 10, "LON082",
                    "The smell of cattle and slaughter was overpowering.",
                    1750, 1750)
    backfill(write=True, db_path=db_path)
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT event_id FROM sensory_evidence WHERE char_offset = 10"
    ).fetchone()
    conn.close()
    assert row[0] == "EVT001"


def test_smithfield_fair_gets_evt010(tmp_path):
    """Passage mentioning 'Bartholomew' at Smithfield should get EVT010."""
    db_path = _make_db(tmp_path)
    _insert_passage(db_path, 20, "LON082",
                    "The crowd at the Bartholomew fair was immense.",
                    1750, 1750)
    backfill(write=True, db_path=db_path)
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT event_id FROM sensory_evidence WHERE char_offset = 20"
    ).fetchone()
    conn.close()
    assert row[0] == "EVT010"


def test_gordon_riots_passage_gets_evt061(tmp_path):
    """Passage at LON048 dated 1780 mentioning riots -> EVT061."""
    db_path = _make_db(tmp_path)
    _insert_passage(db_path, 30, "LON048",
                    "The mob was burning houses in the riots.",
                    1780, 1780)
    backfill(write=True, db_path=db_path)
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT event_id FROM sensory_evidence WHERE char_offset = 30"
    ).fetchone()
    conn.close()
    assert row[0] == "EVT061"


def test_frost_fair_gets_evt020(tmp_path):
    """Passage at LON048 dated 1740 mentioning frost -> EVT020."""
    db_path = _make_db(tmp_path)
    _insert_passage(db_path, 40, "LON048",
                    "The Thames was frozen hard and a frost fair set up.",
                    1740, 1740)
    backfill(write=True, db_path=db_path)
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT event_id FROM sensory_evidence WHERE char_offset = 40"
    ).fetchone()
    conn.close()
    assert row[0] == "EVT020"


def test_passage_at_unlinked_venue_stays_null(tmp_path):
    """A passage at a venue not in event_venues should keep event_id=NULL."""
    db_path = _make_db(tmp_path)
    _insert_passage(db_path, 50, "LON001",  # Vauxhall — no events in this test DB
                    "The music at Vauxhall was enchanting.",
                    1750, 1750)
    backfill(write=True, db_path=db_path)
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT event_id FROM sensory_evidence WHERE char_offset = 50"
    ).fetchone()
    conn.close()
    assert row[0] is None


def test_frost_fair_outside_instance_year_stays_null(tmp_path):
    """Frost Fair at LON048 dated 1750 (no instance that year) -> no event."""
    db_path = _make_db(tmp_path)
    # No keyword matches either, but the irregular event has no instance in 1750
    _insert_passage(db_path, 60, "LON048",
                    "The river flowed peacefully through the city.",
                    1750, 1750)
    backfill(write=True, db_path=db_path)
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT event_id FROM sensory_evidence WHERE char_offset = 60"
    ).fetchone()
    conn.close()
    # Neither EVT020 (no 1750 instance) nor EVT061 (only 1780) should match
    assert row[0] is None


def test_dry_run_does_not_write(tmp_path):
    """Dry run should not modify any event_id."""
    db_path = _make_db(tmp_path)
    _insert_passage(db_path, 70, "LON082",
                    "The cattle at the market were lowing loudly.",
                    1750, 1750)
    backfill(write=False, db_path=db_path)
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT event_id FROM sensory_evidence WHERE char_offset = 70"
    ).fetchone()
    conn.close()
    assert row[0] is None
