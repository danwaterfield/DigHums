"""SQLite schema and connection helper for the sensory evidence store."""

import sqlite3
from pathlib import Path

DB_PATH_DEFAULT = Path(__file__).parent / "sensory.db"

DDL = """
CREATE TABLE IF NOT EXISTS sources (
    source_id   TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    author      TEXT NOT NULL,
    title       TEXT NOT NULL,
    pub_year    INTEGER,
    date_min    INTEGER,
    date_max    INTEGER,
    file_path   TEXT,
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS sensory_evidence (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   TEXT NOT NULL REFERENCES sources(source_id),
    venue_id    TEXT,
    venue_name  TEXT,
    lat         REAL,
    lon         REAL,
    source_type TEXT NOT NULL,
    author      TEXT NOT NULL,
    title       TEXT NOT NULL,
    pub_year    INTEGER,
    date_min    INTEGER,
    date_max    INTEGER,
    modality    TEXT NOT NULL,
    text        TEXT NOT NULL,
    context     TEXT,
    char_offset INTEGER,
    pos         REAL,
    confidence  REAL DEFAULT 1.0,
    notes       TEXT,
    valence     TEXT,
    UNIQUE(source_id, char_offset, modality)
);

CREATE INDEX IF NOT EXISTS idx_venue    ON sensory_evidence(venue_id);
CREATE INDEX IF NOT EXISTS idx_modality ON sensory_evidence(modality);
CREATE INDEX IF NOT EXISTS idx_year     ON sensory_evidence(pub_year);
CREATE INDEX IF NOT EXISTS idx_source_type ON sensory_evidence(source_type);
"""

def init_db(path: Path = DB_PATH_DEFAULT) -> sqlite3.Connection:
    """Create tables if not present; return open connection.

    The default database is created next to this module file
    (gazetteer/sensory.db). Foreign key enforcement is enabled on every
    connection — callers that open their own connections must also run
    PRAGMA foreign_keys = ON.
    """
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    # Migrate: add any columns that pre-date the current schema, before the
    # DDL script runs (which creates indexes that depend on these columns).
    cols = {row[1] for row in conn.execute("PRAGMA table_info(sensory_evidence)")}
    if cols:  # table already exists — apply incremental migrations
        for col, defn in [
            ("venue_id",    "TEXT"),
            ("venue_name",  "TEXT"),
            ("lat",         "REAL"),
            ("lon",         "REAL"),
            ("source_type", "TEXT NOT NULL DEFAULT ''"),
            ("author",      "TEXT NOT NULL DEFAULT ''"),
            ("title",       "TEXT NOT NULL DEFAULT ''"),
            ("pub_year",    "INTEGER"),
            ("date_min",    "INTEGER"),
            ("date_max",    "INTEGER"),
            ("context",     "TEXT"),
            ("pos",         "REAL"),
            ("confidence",  "REAL DEFAULT 1.0"),
            ("notes",       "TEXT"),
            ("valence",     "TEXT"),
        ]:
            if col not in cols:
                conn.execute(
                    f"ALTER TABLE sensory_evidence ADD COLUMN {col} {defn}"
                )
        conn.commit()
    conn.executescript(DDL)
    conn.commit()
    return conn
