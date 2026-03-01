import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import extract_events as ee
from sensory_db import init_db


def test_dry_run_prints_counts(tmp_path, capsys):
    """Dry run reads CSVs and prints counts without writing to DB."""
    ee.run(db_path=tmp_path / "test.db", write=False)
    out = capsys.readouterr().out
    assert "events" in out
    assert "dry run" in out.lower()


def test_write_loads_events(tmp_path):
    """--write inserts all events into DB."""
    ee.run(db_path=tmp_path / "test.db", write=True)
    conn = init_db(tmp_path / "test.db")
    count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    conn.close()
    assert count > 0


def test_smithfield_loaded(tmp_path):
    """EVT001 (Smithfield) is present after load."""
    ee.run(db_path=tmp_path / "test.db", write=True)
    conn = init_db(tmp_path / "test.db")
    row = conn.execute(
        "SELECT name, smell_load, recurrence FROM events WHERE event_id='EVT001'"
    ).fetchone()
    conn.close()
    assert row is not None
    assert "Smithfield" in row[0]
    assert float(row[1]) == 1.0
    assert row[2] == "weekly"


def test_event_venues_loaded(tmp_path):
    """event_venues join table is populated."""
    ee.run(db_path=tmp_path / "test.db", write=True)
    conn = init_db(tmp_path / "test.db")
    count = conn.execute("SELECT COUNT(*) FROM event_venues").fetchone()[0]
    conn.close()
    assert count > 0


def test_event_instances_loaded(tmp_path):
    """event_instances are loaded; INS001 (Great Frost 1684) present."""
    ee.run(db_path=tmp_path / "test.db", write=True)
    conn = init_db(tmp_path / "test.db")
    row = conn.execute(
        "SELECT year, month FROM event_instances WHERE instance_id='INS001'"
    ).fetchone()
    conn.close()
    assert row is not None
    assert row[0] == 1684
    assert row[1] == 1


def test_idempotent(tmp_path):
    """Running twice does not duplicate rows (INSERT OR IGNORE)."""
    ee.run(db_path=tmp_path / "test.db", write=True)
    ee.run(db_path=tmp_path / "test.db", write=True)
    conn = init_db(tmp_path / "test.db")
    count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    conn.close()
    # Count should be same as single run
    conn2 = init_db(tmp_path / "test2.db")
    ee.run(db_path=tmp_path / "test2.db", write=True)
    count2 = conn2.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    conn2.close()
    assert count == count2
