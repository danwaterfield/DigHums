import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from build_comparison import NONFICTION_TYPES, build, load_data


VENUES_PATH = Path(__file__).parent.parent / "venues.csv"
DB_PATH = Path(__file__).parent.parent / "sensory.db"


def test_extended_nonfiction_types_registered():
    assert {"newspaper", "parish", "institutional"}.issubset(NONFICTION_TYPES)


def test_build_outputs_new_badge_classes():
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        out = Path(f.name)
    try:
        build(VENUES_PATH, DB_PATH, out)
        html = out.read_text(encoding="utf-8")
    finally:
        out.unlink(missing_ok=True)

    assert "src-newspaper" in html
    assert "src-parish" in html
    assert "src-institutional" in html


def test_new_nonfiction_types_land_in_nonfiction_bucket(tmp_path):
    venues_path = tmp_path / "venues.csv"
    venues_path.write_text(
        "\n".join([
            "id,name,lat,lon,enclosure,building_type,material,capacity",
            "LON999,Test Venue,51.5000,-0.1000,open,street,stone,small",
        ]),
        encoding="utf-8",
    )

    db_path = tmp_path / "comparison.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE sensory_evidence (
                venue_id TEXT,
                source_type TEXT,
                author TEXT,
                title TEXT,
                pub_year INTEGER,
                date_min INTEGER,
                date_max INTEGER,
                modality TEXT,
                text TEXT,
                context TEXT,
                valence TEXT,
                divergence TEXT
            )
            """
        )
        for source_type in ("newspaper", "parish", "institutional"):
            conn.execute(
                """
                INSERT INTO sensory_evidence
                (venue_id, source_type, author, title, pub_year, date_min, date_max,
                 modality, text, context, valence, divergence)
                VALUES (?, ?, 'Author', 'Title', 1780, 1780, 1780,
                        'auditory', 'Passage', 'noise', 'neutral', NULL)
                """,
                ("LON999", source_type),
            )
        conn.commit()
    finally:
        conn.close()

    venues = load_data(venues_path, db_path)
    assert len(venues) == 1
    assert [ev["source_type"] for ev in venues[0]["nonfiction"]] == [
        "newspaper", "parish", "institutional",
    ]
