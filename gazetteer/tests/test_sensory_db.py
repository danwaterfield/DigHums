import sqlite3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from sensory_db import init_db, DB_PATH_DEFAULT

def test_schema_creates_tables(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "sensory_evidence" in tables
    assert "sources" in tables
    conn.close()

def test_sensory_evidence_columns(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    cols = {r[1] for r in conn.execute(
        "PRAGMA table_info(sensory_evidence)"
    ).fetchall()}
    for expected in ("venue_id", "modality", "source_type", "author",
                     "pub_year", "date_min", "date_max", "text", "confidence"):
        assert expected in cols, f"missing column: {expected}"
    conn.close()
