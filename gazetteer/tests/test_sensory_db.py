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

def test_valence_column_exists(tmp_path):
    """init_db() must create the valence column on a fresh DB."""
    conn = init_db(tmp_path / "v.db")
    cols = {row[1] for row in conn.execute("PRAGMA table_info(sensory_evidence)")}
    assert "valence" in cols

def test_valence_column_migrated(tmp_path):
    """init_db() adds valence column to an existing DB that lacks it."""
    import sqlite3
    old_ddl = """
    CREATE TABLE sensory_evidence (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id TEXT NOT NULL,
        modality TEXT NOT NULL,
        text TEXT NOT NULL,
        char_offset INTEGER,
        UNIQUE(source_id, char_offset, modality)
    );
    CREATE TABLE sources (
        source_id TEXT PRIMARY KEY, source_type TEXT NOT NULL,
        author TEXT NOT NULL, title TEXT NOT NULL
    );
    """
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(old_ddl)
    conn.close()
    conn2 = init_db(db_path)
    cols = {row[1] for row in conn2.execute("PRAGMA table_info(sensory_evidence)")}
    assert "valence" in cols
