import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from sensory_db import init_db
from backfill_valence import backfill


def _make_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    conn.executemany(
        "INSERT INTO sources (source_id, source_type, author, title) VALUES (?,?,?,?)",
        [("S1", "fiction", "Test Author", "Test Novel")],
    )
    conn.executemany(
        "INSERT INTO sensory_evidence "
        "(source_id, source_type, author, title, modality, text, char_offset, valence) "
        "VALUES (?,?,?,?,?,?,?,?)",
        [
            # Unpleasant passage — no valence yet
            ("S1", "fiction", "Test Author", "Test Novel",
             "olfactory", "The stench from the kennel was overpowering.", 0, None),
            # Pleasant passage — no valence yet
            ("S1", "fiction", "Test Author", "Test Novel",
             "auditory", "Soft music and fragrant roses filled the room.", 100, None),
            # Already has valid valence — must NOT be overwritten
            ("S1", "fiction", "Test Author", "Test Novel",
             "crowd", "The mob jostled in the street.", 200, "unpleasant"),
            # Neutral passage — no valence yet
            ("S1", "fiction", "Test Author", "Test Novel",
             "visual", "The crowd moved slowly along the street.", 300, None),
        ],
    )
    conn.commit()
    conn.close()
    return db_path


def test_tags_unpleasant_passage(tmp_path):
    db_path = _make_db(tmp_path)
    backfill(write=True, db_path=db_path)
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT valence FROM sensory_evidence WHERE char_offset = 0"
    ).fetchone()
    conn.close()
    assert row[0] == "unpleasant"


def test_tags_pleasant_passage(tmp_path):
    db_path = _make_db(tmp_path)
    backfill(write=True, db_path=db_path)
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT valence FROM sensory_evidence WHERE char_offset = 100"
    ).fetchone()
    conn.close()
    assert row[0] == "pleasant"


def test_skips_existing_valence(tmp_path):
    db_path = _make_db(tmp_path)
    stats = backfill(write=True, db_path=db_path)
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT valence FROM sensory_evidence WHERE char_offset = 200"
    ).fetchone()
    conn.close()
    # The row already had 'unpleasant' and must keep it unchanged
    assert row[0] == "unpleasant"
    # It was not included in the backfill set (already valid)
    assert stats["total_unvalenced"] == 3  # only the 3 NULL rows were selected


def test_idempotent(tmp_path):
    db_path = _make_db(tmp_path)
    backfill(write=True, db_path=db_path)
    stats2 = backfill(write=True, db_path=db_path)
    # Second run should find nothing left to tag
    assert stats2["total_unvalenced"] == 0
    assert stats2["tagged"] == 0


def test_dry_run_does_not_write(tmp_path):
    db_path = _make_db(tmp_path)
    backfill(write=False, db_path=db_path)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT valence FROM sensory_evidence WHERE valence IS NULL"
    ).fetchall()
    conn.close()
    # All three NULL rows should still be NULL after dry run
    assert len(rows) == 3
