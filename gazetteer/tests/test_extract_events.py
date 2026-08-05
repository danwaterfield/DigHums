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
    """Running twice keeps a stable row count after a full refresh."""
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


def test_write_refreshes_existing_rows(tmp_path, monkeypatch):
    """A second write should replace stale event metadata with the current CSV contents."""
    events_path = tmp_path / "events.csv"
    event_venues_path = tmp_path / "event_venues.csv"
    event_instances_path = tmp_path / "event_instances.csv"

    events_path.write_text(
        "\n".join([
            "event_id,name,category,month_start,month_end,day_of_week,time_bands,year_start,year_end,recurrence,smell_load,noise_load,crowd_load,visual_load,calendar_break,month_start_ns,notes,sources",
            "EVT900,Test Event,weekly_market,,,Mon,morning,1700,1701,weekly,0.1,0.2,0.3,0.4,,,Original note,Source A",
        ]) + "\n",
        encoding="utf-8",
    )
    event_venues_path.write_text(
        "event_id,venue_id\nEVT900,LON001\n",
        encoding="utf-8",
    )
    event_instances_path.write_text(
        "instance_id,event_id,year,month,day,source_id,notes\nINS900,EVT900,1700,1,2,,Original instance\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(ee, "EVENTS_PATH", events_path)
    monkeypatch.setattr(ee, "EVENT_VENUES_PATH", event_venues_path)
    monkeypatch.setattr(ee, "EVENT_INSTANCES_PATH", event_instances_path)

    db_path = tmp_path / "test.db"
    ee.run(db_path=db_path, write=True)

    events_path.write_text(
        "\n".join([
            "event_id,name,category,month_start,month_end,day_of_week,time_bands,year_start,year_end,recurrence,smell_load,noise_load,crowd_load,visual_load,calendar_break,month_start_ns,notes,sources",
            "EVT900,Test Event,weekly_market,,,Fri,evening,1700,1701,weekly,0.1,0.2,0.3,0.4,,,Updated note,Source B",
        ]) + "\n",
        encoding="utf-8",
    )
    event_venues_path.write_text(
        "event_id,venue_id\nEVT900,LON002\n",
        encoding="utf-8",
    )
    event_instances_path.write_text(
        "instance_id,event_id,year,month,day,source_id,notes\nINS901,EVT900,1701,2,3,,Updated instance\n",
        encoding="utf-8",
    )

    ee.run(db_path=db_path, write=True)

    conn = init_db(db_path)
    event_row = conn.execute(
        "SELECT day_of_week, time_bands, notes, sources FROM events WHERE event_id='EVT900'"
    ).fetchone()
    venue_rows = conn.execute(
        "SELECT venue_id FROM event_venues WHERE event_id='EVT900'"
    ).fetchall()
    instance_rows = conn.execute(
        "SELECT instance_id, notes FROM event_instances WHERE event_id='EVT900'"
    ).fetchall()
    conn.close()

    assert event_row == ("Fri", "evening", "Updated note", "Source B")
    assert venue_rows == [("LON002",)]
    assert instance_rows == [("INS901", "Updated instance")]
