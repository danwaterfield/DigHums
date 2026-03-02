"""Tests for backfill_rural.py"""
import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from sensory_db import init_db
from backfill_rural import backfill, load_rural_texts, _db_author, _classify


# ── Minimal corpus_dates.csv fixture ─────────────────────────────────────────
# Uses CamelCase author names as in the real corpus_dates.csv.
# The DB stores authors as lowercase last-names ("radcliffe", "lewis").

CORPUS_FIELDS = [
    "author", "title", "file_path", "pub_year",
    "setting_period_start", "setting_period_end",
    "primary_cities", "map_layer", "notes",
]


def _corpus_row(author, title, primary_cities):
    return {
        "author": author, "title": title,
        "file_path": "", "pub_year": 1800,
        "setting_period_start": "", "setting_period_end": "",
        "primary_cities": primary_cities,
        "map_layer": "rocque_1746", "notes": "",
    }


@pytest.fixture
def sample_corpus(tmp_path):
    p = tmp_path / "corpus_dates.csv"
    rows = [
        _corpus_row("AnnScott",       "Rural Novel",    "Scotland"),
        _corpus_row("MariaRossi",     "Gothic Novel",   "Italy"),
        _corpus_row("JohnLondon",     "London Novel",   "London"),
        _corpus_row("MargaretBath",   "Bath Novel",     "Bath;Somerset"),
        _corpus_row("ElizaMixed",     "Mixed Novel",    "London;France"),
        _corpus_row("PatrickIreland", "Ireland Novel",  "Ireland"),
        _corpus_row("WilliamArabia",  "Arabian Tale",   "Arabia (fantasy)"),
    ]
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CORPUS_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return p


# ── Unit tests for helpers ────────────────────────────────────────────────────

def test_db_author_camelcase():
    assert _db_author("AnnRadcliffe") == "radcliffe"
    assert _db_author("MGLewis")      == "lewis"
    assert _db_author("WilliamBeckford") == "beckford"


def test_classify_continental():
    assert _classify("Italy")         == "RUR002"
    assert _classify("France")        == "RUR002"
    assert _classify("Arabia (fantasy)") == "RUR002"
    assert _classify("Sicily/Italy")  == "RUR002"


def test_classify_rural():
    assert _classify("Scotland")  == "RUR001"
    assert _classify("Ireland")   == "RUR001"
    assert _classify("Wales")     == "RUR001"


# ── Integration tests ─────────────────────────────────────────────────────────

def test_load_excludes_london_and_bath(sample_corpus):
    """Texts whose primary_cities includes London or Bath are excluded."""
    texts = load_rural_texts(sample_corpus)
    db_authors = [t["db_author"] for t in texts]
    assert "london" not in db_authors   # JohnLondon
    assert "bath" not in db_authors     # MargaretBath
    assert "mixed" not in db_authors    # ElizaMixed (has London)


def test_load_includes_rural_and_continental(sample_corpus):
    texts = load_rural_texts(sample_corpus)
    db_authors = [t["db_author"] for t in texts]
    assert "scott" in db_authors   # Scotland → RUR001
    assert "rossi"    in db_authors   # Italy → RUR002
    assert "ireland"  in db_authors   # Ireland → RUR001
    assert "arabia"   in db_authors   # Arabia → RUR002


def test_italy_classified_as_rur002(sample_corpus):
    texts = load_rural_texts(sample_corpus)
    t = next(t for t in texts if t["db_author"] == "rossi")
    assert t["venue_id"] == "RUR002"


def test_scotland_classified_as_rur001(sample_corpus):
    texts = load_rural_texts(sample_corpus)
    t = next(t for t in texts if t["db_author"] == "scott")
    assert t["venue_id"] == "RUR001"


def test_backfill_writes_venue_id(tmp_path, sample_corpus):
    """Passages with NULL venue_id for a rural text get RUR001."""
    db_path = tmp_path / "test.db"
    # DB uses lowercase last-name author ("scott") matching _db_author("AnnMcAlpine")
    conn = init_db(db_path)
    conn.execute(
        "INSERT INTO sources VALUES (?,?,?,?,?,?,?,?,?)",
        ("src1", "fiction", "scott", "Rural Novel", 1800, 1795, 1800, "", ""),
    )
    conn.execute(
        "INSERT INTO sensory_evidence "
        "(source_id, source_type, author, title, pub_year, date_min, date_max, "
        "modality, text) VALUES (?,?,?,?,?,?,?,?,?)",
        ("src1", "fiction", "scott", "Rural Novel", 1800, 1795, 1800,
         "sight", "The Scottish hills loomed dark."),
    )
    conn.commit()
    conn.close()

    stats = backfill(write=True, db_path=db_path, corpus_csv=sample_corpus)
    assert stats["passages_updated"] >= 1

    conn2 = init_db(db_path)
    row = conn2.execute(
        "SELECT venue_id FROM sensory_evidence WHERE author = 'scott'"
    ).fetchone()
    conn2.close()
    assert row is not None
    assert row[0] == "RUR001"


def test_backfill_skips_already_linked(tmp_path, sample_corpus):
    """Passages already linked to a venue are not overwritten."""
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    conn.execute(
        "INSERT INTO sources VALUES (?,?,?,?,?,?,?,?,?)",
        ("src2", "fiction", "rossi", "Gothic Novel", 1790, 1785, 1790, "", ""),
    )
    conn.execute(
        "INSERT INTO sensory_evidence "
        "(source_id, source_type, author, title, pub_year, date_min, date_max, "
        "modality, text, venue_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("src2", "fiction", "rossi", "Gothic Novel", 1790, 1785, 1790,
         "sight", "The castle battlements.", "LON001"),
    )
    conn.commit()
    conn.close()

    backfill(write=True, db_path=db_path, corpus_csv=sample_corpus)
    # LON001 should not be overwritten
    conn2 = init_db(db_path)
    row = conn2.execute(
        "SELECT venue_id FROM sensory_evidence WHERE author = 'rossi'"
    ).fetchone()
    conn2.close()
    assert row[0] == "LON001"


def test_backfill_dry_run_makes_no_changes(tmp_path, sample_corpus):
    """Dry run does not modify the database."""
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    conn.execute(
        "INSERT INTO sources VALUES (?,?,?,?,?,?,?,?,?)",
        ("src3", "fiction", "ireland", "Ireland Novel", 1800, 1795, 1800, "", ""),
    )
    conn.execute(
        "INSERT INTO sensory_evidence "
        "(source_id, source_type, author, title, pub_year, date_min, date_max, "
        "modality, text) VALUES (?,?,?,?,?,?,?,?,?)",
        ("src3", "fiction", "ireland", "Ireland Novel", 1800, 1795, 1800,
         "smell", "The peat bogs of Ireland."),
    )
    conn.commit()
    conn.close()

    stats = backfill(write=False, db_path=db_path, corpus_csv=sample_corpus)
    assert stats["passages_updated"] >= 1

    conn2 = init_db(db_path)
    row = conn2.execute(
        "SELECT venue_id FROM sensory_evidence WHERE author = 'ireland'"
    ).fetchone()
    conn2.close()
    assert row[0] is None   # no change in dry run
