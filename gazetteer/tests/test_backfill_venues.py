"""
Tests for backfill_venues.py.

Uses a minimal in-memory-style SQLite DB and a temporary corpus entry.
"""
import csv
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from sensory_db import init_db
from backfill_venues import backfill


# ── Fixtures ──────────────────────────────────────────────────────────────────

VENUES = [
    {"id": "LON021", "name": "Tower of London",
     "lat": "51.5081", "lon": "-0.0759",
     "city": "London", "opened": "1078", "closed": "", "map_layer": "rocque_1746"},
    {"id": "LON001", "name": "Vauxhall Spring Gardens",
     "lat": "51.4882", "lon": "-0.1228",
     "city": "London", "opened": "1661", "closed": "1859", "map_layer": "rocque_1746"},
]

CORPUS_ENTRY = {
    "author": "Test Author",
    "title": "Tower Novel",
    "file_path": "",        # filled in per test
    "pub_year": "1750",
    "setting_period_start": "1750",
    "setting_period_end": "1750",
    "primary_cities": "London",
    "map_layer": "rocque_1746",
    "notes": "",
}

CORPUS_ENTRY_ITALIAN = {
    "author": "Ann Radcliffe",
    "title": "Italian Novel",
    "file_path": "",
    "pub_year": "1794",
    "setting_period_start": "1760",
    "setting_period_end": "1790",
    "primary_cities": "Italy",
    "map_layer": "rocque_1746",
    "notes": "",
}


def _write_corpus(tmp_path: Path, entries: list[dict]) -> Path:
    corpus_path = tmp_path / "corpus_dates.csv"
    fieldnames = ["author", "title", "file_path", "pub_year",
                  "setting_period_start", "setting_period_end",
                  "primary_cities", "map_layer", "notes"]
    with open(corpus_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(entries)
    return corpus_path


def _write_venues(tmp_path: Path) -> Path:
    venues_path = tmp_path / "venues.csv"
    fieldnames = ["id", "name", "lat", "lon", "city", "opened", "closed", "map_layer"]
    with open(venues_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(VENUES)
    return venues_path


def _make_db(tmp_path: Path, passages: list[tuple]) -> Path:
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    conn.executemany(
        "INSERT INTO sources (source_id, source_type, author, title) VALUES (?,?,?,?)",
        [("S1", "fiction", "Test Author", "Tower Novel"),
         ("S2", "fiction", "Ann Radcliffe", "Italian Novel")],
    )
    conn.executemany(
        "INSERT INTO sensory_evidence "
        "(source_id, source_type, author, title, modality, text, char_offset) "
        "VALUES (?,?,?,?,?,?,?)",
        passages,
    )
    conn.commit()
    conn.close()
    return db_path


# ── Helper: patch backfill module paths ───────────────────────────────────────

import backfill_venues as bv_module


@pytest.fixture(autouse=True)
def patch_paths(tmp_path, monkeypatch):
    # Write a novel text file
    novel_file = tmp_path / "tower_novel.txt"
    novel_file.write_text(
        "She gazed with admiration upon the Tower of London as their boat "
        "passed beneath its ancient walls. The Thames was crowded with vessels.",
        encoding="utf-8",
    )

    italian_file = tmp_path / "italian_novel.txt"
    italian_file.write_text(
        "She entered the ancient piazza, overwhelmed by the stench of the crowd.",
        encoding="utf-8",
    )

    entry_london = dict(CORPUS_ENTRY, file_path=str(novel_file.relative_to(tmp_path.parent / tmp_path.name / "..") if False else novel_file))
    entry_italian = dict(CORPUS_ENTRY_ITALIAN, file_path=str(italian_file))

    corpus_path = _write_corpus(tmp_path, [entry_london, entry_italian])
    venues_path = _write_venues(tmp_path)

    monkeypatch.setattr(bv_module, "CORPUS_CSV", corpus_path)
    monkeypatch.setattr(bv_module, "VENUES_CSV", venues_path)
    monkeypatch.setattr(bv_module, "CORPUS_ROOT", Path("/"))


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_tower_of_london_passage_gets_lon021(tmp_path, patch_paths):
    """A passage containing 'Tower of London' should be geocoded to LON021."""
    novel_file = tmp_path / "tower_novel.txt"
    passages = [
        ("S1", "fiction", "Test Author", "Tower Novel",
         "crowd", "passed beneath its ancient walls", 60),
    ]
    db_path = _make_db(tmp_path, passages)
    stats = backfill(write=True, db_path=db_path)

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT venue_id FROM sensory_evidence WHERE char_offset = 60"
    ).fetchone()
    conn.close()

    assert stats["geocoded"] == 1
    assert row[0] == "LON021"


def test_italian_setting_stays_null(tmp_path, patch_paths):
    """A passage in an Italian-set novel should not be linked to a London venue."""
    passages = [
        ("S2", "fiction", "Ann Radcliffe", "Italian Novel",
         "crowd", "overwhelmed by the stench of the crowd", 40),
    ]
    db_path = _make_db(tmp_path, passages)
    stats = backfill(write=True, db_path=db_path)

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT venue_id FROM sensory_evidence WHERE char_offset = 40"
    ).fetchone()
    conn.close()

    # No London venue should match an Italian-set passage
    assert row[0] is None


def test_dry_run_does_not_write(tmp_path, patch_paths):
    """Dry run should not modify any venue_id."""
    novel_file = tmp_path / "tower_novel.txt"
    passages = [
        ("S1", "fiction", "Test Author", "Tower Novel",
         "crowd", "passed beneath its ancient walls", 60),
    ]
    db_path = _make_db(tmp_path, passages)
    backfill(write=False, db_path=db_path)

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT venue_id FROM sensory_evidence WHERE char_offset = 60"
    ).fetchone()
    conn.close()

    assert row[0] is None
