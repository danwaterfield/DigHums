import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import extract_sensory
from extract_sensory import extract_from_text, WINDOW_CHARS, normalize_source_text
from sensory_db import init_db

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

def test_extract_includes_valence(tmp_path):
    db = init_db(tmp_path / "t.db")
    text = ("We went to Vauxhall last Tuesday. The stench was insupportable "
            "and the din quite unbearable.")
    rows = extract_from_text(
        text=text, source_id="test_valence", source_type="diary",
        author="Test", title="Test", pub_year=1760,
        date_min=1758, date_max=1762, venues=VENUES, conn=db
    )
    assert len(rows) > 0
    for r in rows:
        assert "valence" in r
        assert r["valence"] in ("pleasant", "neutral", "unpleasant")

def test_extract_writes_valence_to_db(tmp_path):
    db = init_db(tmp_path / "t.db")
    text = ("We went to Vauxhall. The stench was insupportable.")
    extract_from_text(
        text=text, source_id="test_val_write", source_type="diary",
        author="Test", title="Test", pub_year=1760,
        date_min=1758, date_max=1762, venues=VENUES, conn=db,
        write=True
    )
    rows = db.execute(
        "SELECT valence FROM sensory_evidence WHERE source_id='test_val_write'"
    ).fetchall()
    assert len(rows) > 0
    assert rows[0][0] in ("pleasant", "neutral", "unpleasant")


def test_run_uses_primary_cities_from_catalog(tmp_path, monkeypatch):
    db_path = tmp_path / "sensory.db"
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "bath_guide.txt").write_text(
        "In Bath we repaired to the Pump, where the heat and crowd were oppressive.",
        encoding="utf-8",
    )

    catalog_path = tmp_path / "sources_catalog.csv"
    catalog_path.write_text(
        "\n".join([
            "source_id,source_type,author,title,pub_year,date_min,date_max,primary_cities,file_path,notes",
            "bath_guide,topography,GuideAuthor,Bath Guide,1780,1780,1780,Bath,bath_guide.txt,",
        ]),
        encoding="utf-8",
    )

    venues_path = tmp_path / "venues.csv"
    venues_path.write_text(
        "\n".join([
            "id,name,lat,lon",
            "BAT003,The Pump Room,51.3814,-2.3594",
        ]),
        encoding="utf-8",
    )

    monkeypatch.setattr(extract_sensory, "SOURCES_DIR", sources_dir)
    monkeypatch.setattr(extract_sensory, "CATALOG_PATH", catalog_path)
    monkeypatch.setattr(extract_sensory, "VENUES_PATH", venues_path)
    monkeypatch.setattr(extract_sensory, "DB_PATH_DEFAULT", db_path)

    extract_sensory.run(write=True)

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT venue_id FROM sensory_evidence WHERE source_id='bath_guide'"
        ).fetchall()
    finally:
        conn.close()

    assert rows
    assert any(row[0] == "BAT003" for row in rows)


def test_normalize_source_text_rewrites_common_ocr_glyphs():
    text = "The noiſe in the co\u00adffee houſe and the \ufb01re alarm."
    normalized = normalize_source_text(text)
    assert "noise" in normalized
    assert "coffee" in normalized
    assert "fire" in normalized
