import sqlite3
import sys
import textwrap
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyproj")

ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

import export_analysis_tables as export_tables


def _write_csv(path: Path, text: str) -> None:
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def _build_minimal_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE sources (
            source_id TEXT PRIMARY KEY,
            source_type TEXT,
            author TEXT,
            title TEXT,
            pub_year INTEGER,
            date_min INTEGER,
            date_max INTEGER,
            file_path TEXT,
            notes TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE sensory_evidence (
            id INTEGER PRIMARY KEY,
            source_id TEXT,
            venue_id TEXT,
            venue_name TEXT,
            lat REAL,
            lon REAL,
            source_type TEXT,
            author TEXT,
            title TEXT,
            pub_year INTEGER,
            date_min INTEGER,
            date_max INTEGER,
            modality TEXT,
            text TEXT,
            context TEXT,
            char_offset INTEGER,
            pos REAL,
            confidence REAL,
            notes TEXT,
            valence TEXT,
            event_id TEXT,
            divergence REAL
        )
        """
    )
    return conn


def test_export_tables_merge_and_aggregate(tmp_path, monkeypatch):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    (processed_dir / "TestNovel.txt").write_text(
        "The crowd and music at Vauxhall were dazzling and loud.",
        encoding="utf-8",
    )

    nonfiction_dir = tmp_path / "sources"
    nonfiction_dir.mkdir()
    (nonfiction_dir / "bath_guide.txt").write_text(
        "The Pump Room in Bath was bright and crowded.",
        encoding="utf-8",
    )

    metadata_csv = tmp_path / "metadata_v2.csv"
    _write_csv(
        metadata_csv,
        """
        author,title,volume,file_path,genre,notes
        TestAuthor,Test Novel,1,TestNovel.txt,fiction,Modeled fiction sample
        """,
    )

    corpus_dates_csv = tmp_path / "corpus_dates.csv"
    _write_csv(
        corpus_dates_csv,
        """
        title,primary_cities,map_layer,setting_period_start,setting_period_end,notes
        Test Novel,London,rocque,1750,1760,Fiction setting window
        """,
    )

    sources_catalog_csv = tmp_path / "sources_catalog.csv"
    _write_csv(
        sources_catalog_csv,
        """
        source_id,source_type,author,title,pub_year,date_min,date_max,primary_cities,file_path,notes
        bath_guide,topography,GuideAuthor,Bath Guide,1780,1780,1780,Bath,bath_guide.txt,Documentary sample
        """,
    )

    venues_csv = tmp_path / "venues.csv"
    _write_csv(
        venues_csv,
        """
        id,name,city,lat,lon,opened,closed,notes,map_layer,tier,enclosure,building_type,material,capacity,hw_ratio
        LON001,Vauxhall Spring Gardens,London,51.4882,-0.1228,1660,1859,,rocque,1,open,garden,mixed,1000,1.5
        BAT001,The Pump Room,Bath,51.3814,-2.3594,1700,1830,,bath,1,enclosed,assembly,stone,400,1.1
        """,
    )

    db_path = tmp_path / "sensory.db"
    conn = _build_minimal_db(db_path)
    try:
        fiction_source_id = "fiction_TestAuthor_Test_Novel"
        conn.executemany(
            "INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    fiction_source_id,
                    "fiction",
                    "TestAuthor",
                    "Test Novel",
                    1755,
                    1755,
                    1755,
                    None,
                    "From DB",
                ),
                (
                    "bath_guide",
                    "topography",
                    "GuideAuthor",
                    "Bath Guide",
                    1780,
                    1780,
                    1780,
                    None,
                    "From DB",
                ),
            ],
        )
        conn.executemany(
            "INSERT INTO sensory_evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    1,
                    fiction_source_id,
                    "LON001",
                    "Vauxhall Spring Gardens",
                    51.4882,
                    -0.1228,
                    "fiction",
                    "TestAuthor",
                    "Test Novel",
                    1755,
                    1755,
                    1755,
                    "crowd",
                    "The crowd surged at Vauxhall.",
                    "The crowd surged at Vauxhall.",
                    0,
                    0.2,
                    0.95,
                    "",
                    "unpleasant",
                    None,
                    0.3,
                ),
                (
                    2,
                    "bath_guide",
                    "BAT001",
                    "The Pump Room",
                    51.3814,
                    -2.3594,
                    "topography",
                    "GuideAuthor",
                    "Bath Guide",
                    1780,
                    1780,
                    1780,
                    "visual",
                    "The Pump Room was bright.",
                    "The Pump Room was bright.",
                    12,
                    0.6,
                    0.9,
                    "",
                    "pleasant",
                    None,
                    0.1,
                ),
            ],
        )
        conn.commit()

        monkeypatch.setattr(export_tables, "METADATA_V2_CSV", metadata_csv)
        monkeypatch.setattr(export_tables, "CORPUS_DATES_CSV", corpus_dates_csv)
        monkeypatch.setattr(export_tables, "SOURCES_CSV", sources_catalog_csv)
        monkeypatch.setattr(export_tables, "VENUES_CSV", venues_csv)
        monkeypatch.setattr(export_tables, "PROCESSED_DIR", processed_dir)
        monkeypatch.setattr(export_tables, "NONFICTION_DIR", nonfiction_dir)

        work_metadata = export_tables.build_work_metadata(conn)
        evidence_points = export_tables.build_evidence_points(conn, work_metadata)
        work_feature_panel = export_tables.build_work_feature_panel(work_metadata, evidence_points)
        position_bins = export_tables.build_position_bins(work_metadata, evidence_points, bins=5)
    finally:
        conn.close()

    fiction_row = work_metadata.loc[work_metadata["source_id"] == fiction_source_id].iloc[0]
    nonfiction_row = work_metadata.loc[work_metadata["source_id"] == "bath_guide"].iloc[0]

    assert fiction_row["file_path"] == "TestNovel.txt"
    assert fiction_row["primary_cities"] == "London"
    assert fiction_row["source_domain"] == "fiction"
    assert fiction_row["source_family"] == "fiction"
    assert fiction_row["volume_count"] == 1
    assert int(fiction_row["word_count"]) > 0

    assert nonfiction_row["file_path"] == "bath_guide.txt"
    assert nonfiction_row["primary_cities"] == "Bath"
    assert nonfiction_row["source_domain"] == "documentary"
    assert nonfiction_row["source_family"] == "descriptive"
    assert int(nonfiction_row["word_count"]) > 0

    assert set(evidence_points["city"]) == {"London", "Bath"}
    assert set(evidence_points["venue_family"]) == {"leisure"}
    assert set(evidence_points["source_family"]) == {"fiction", "descriptive"}
    assert evidence_points["has_geocode"].all()
    assert evidence_points["x_m"].notna().all()
    assert evidence_points["y_m"].notna().all()

    feature_row = work_feature_panel.loc[work_feature_panel["source_id"] == fiction_source_id].iloc[0]
    assert feature_row["evidence_total"] == 1
    assert feature_row["modality_crowd_count"] == 1
    assert feature_row["city_london_count"] == 1
    assert feature_row["evidence_rate_per_10k_words"] > 0

    assert len(position_bins) == 10
    fiction_bins = position_bins.loc[position_bins["source_id"] == fiction_source_id]
    assert fiction_bins["evidence_total"].sum() == 1
    assert fiction_bins["modality_crowd_count"].sum() == 1
