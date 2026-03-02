"""Tests for compute_divergence.py"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from sensory_db import init_db
from compute_divergence import compute, _compute_label, _is_silent


# ── Unit tests for helpers ────────────────────────────────────────────────────

def test_silence_detection():
    assert _is_silent("The room was silent.")
    assert _is_silent("A deep hush fell over the crowd.")
    assert _is_silent("All was quiet and still.")
    assert not _is_silent("The crowd roared and cheered.")


def test_compute_label_confirms_smell():
    loads = {"smell_load": 1.0, "noise_load": 0.3, "visual_load": 0.4, "crowd_load": 0.5}
    assert _compute_label("olfactory", "unpleasant", "The stench was appalling.", loads) == "confirms"


def test_compute_label_diverges_smell():
    loads = {"smell_load": 0.9, "noise_load": 0.3, "visual_load": 0.4, "crowd_load": 0.5}
    assert _compute_label("olfactory", "pleasant", "The air smelled of roses.", loads) == "diverges"


def test_compute_label_confirms_sound():
    loads = {"smell_load": 0.3, "noise_load": 0.9, "visual_load": 0.4, "crowd_load": 0.5}
    assert _compute_label("auditory", "unpleasant", "A deafening roar filled the street.", loads) == "confirms"


def test_compute_label_diverges_sound_silence():
    loads = {"smell_load": 0.3, "noise_load": 0.9, "visual_load": 0.4, "crowd_load": 0.5}
    assert _compute_label("auditory", "neutral", "All was silent and still.", loads) == "diverges"


def test_compute_label_none_for_low_load():
    loads = {"smell_load": 0.3, "noise_load": 0.2, "visual_load": 0.1, "crowd_load": 0.2}
    assert _compute_label("olfactory", "unpleasant", "A faint odour.", loads) is None


def test_compute_label_none_for_neutral_valence():
    loads = {"smell_load": 0.9, "noise_load": 0.5, "visual_load": 0.4, "crowd_load": 0.5}
    assert _compute_label("olfactory", "neutral", "There was a smell.", loads) is None


# ── Integration test ──────────────────────────────────────────────────────────

def _insert_evidence(conn, row_id, source_id, event_id, venue_id, modality, valence, text):
    conn.execute(
        "INSERT OR IGNORE INTO sensory_evidence "
        "(id, source_id, source_type, author, title, modality, text, "
        " valence, venue_id, event_id) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (row_id, source_id, "fiction", "testauthor", "Test Novel",
         modality, text, valence, venue_id, event_id),
    )


def test_compute_writes_confirms_and_diverges(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)

    # Insert source and event
    conn.execute(
        "INSERT INTO sources VALUES (?,?,?,?,?,?,?,?,?)",
        ("src1", "fiction", "testauthor", "Test Novel", 1750, 1748, 1752, "", ""),
    )
    conn.execute(
        "INSERT INTO events (event_id, name, category, time_bands, recurrence, "
        "smell_load, noise_load, visual_load, crowd_load) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        ("EVT_TEST", "Test Market", "weekly_market", "morning", "weekly",
         0.9, 0.8, 0.5, 0.7),
    )

    # Passage 1: unpleasant olfactory → confirms (high smell_load)
    _insert_evidence(conn, 1, "src1", "EVT_TEST", "LON001",
                     "olfactory", "unpleasant", "The stench was overwhelming.")
    # Passage 2: pleasant olfactory → diverges (high smell_load but pleasant)
    _insert_evidence(conn, 2, "src1", "EVT_TEST", "LON001",
                     "olfactory", "pleasant", "The air smelled of roses.")
    # Passage 3: neutral olfactory → undecided
    _insert_evidence(conn, 3, "src1", "EVT_TEST", "LON001",
                     "olfactory", "neutral", "There was a smell.")
    # Passage 4: auditory silence at noisy event → diverges
    _insert_evidence(conn, 4, "src1", "EVT_TEST", "LON001",
                     "auditory", "neutral", "All was silent and still.")

    conn.commit()
    conn.close()

    stats = compute(write=True, db_path=db_path)
    assert stats["confirms"] == 1
    assert stats["diverges"] == 2
    assert stats["updated"]  == 3

    conn2 = init_db(db_path)
    rows = conn2.execute(
        "SELECT id, divergence FROM sensory_evidence ORDER BY id"
    ).fetchall()
    conn2.close()

    div_map = {r[0]: r[1] for r in rows}
    assert div_map[1] == "confirms"
    assert div_map[2] == "diverges"
    assert div_map[3] is None
    assert div_map[4] == "diverges"


def test_compute_is_idempotent(tmp_path):
    """Running twice does not overwrite existing labels."""
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    conn.execute(
        "INSERT INTO sources VALUES (?,?,?,?,?,?,?,?,?)",
        ("src1", "fiction", "testauthor", "Test Novel", 1750, 1748, 1752, "", ""),
    )
    conn.execute(
        "INSERT INTO events (event_id, name, category, time_bands, recurrence, "
        "smell_load, noise_load, visual_load, crowd_load) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        ("EVT_TEST2", "Test Market", "weekly_market", "morning", "weekly",
         0.9, 0.5, 0.5, 0.5),
    )
    conn.execute(
        "INSERT INTO sensory_evidence "
        "(source_id, source_type, author, title, modality, text, "
        " valence, venue_id, event_id) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        ("src1", "fiction", "testauthor", "Test Novel",
         "olfactory", "Stench everywhere.", "unpleasant", "LON001", "EVT_TEST2"),
    )
    conn.commit()
    conn.close()

    stats1 = compute(write=True, db_path=db_path)
    assert stats1["confirms"] == 1
    assert stats1["updated"]  == 1

    stats2 = compute(write=True, db_path=db_path)
    assert stats2["updated"]  == 0   # already set, skipped
    assert stats2["passages_examined"] == 0
