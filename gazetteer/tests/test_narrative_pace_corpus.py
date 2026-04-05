"""Tests for narrative_pace_corpus.py — corpus loading module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from narrative_pace_corpus import concatenate_volumes, load_novel_texts, load_corpus

# ---------------------------------------------------------------------------
# concatenate_volumes
# ---------------------------------------------------------------------------

def test_concatenate_volumes_single_returns_text_unchanged():
    text = "Hello world this is a single volume."
    combined, boundaries = concatenate_volumes([text])
    assert combined == text
    assert boundaries == []


def test_concatenate_volumes_single_no_boundaries():
    combined, boundaries = concatenate_volumes(["some text"])
    assert len(boundaries) == 0


def test_concatenate_volumes_two_vols_concatenates():
    vol1 = "First volume text."
    vol2 = "Second volume text."
    combined, boundaries = concatenate_volumes([vol1, vol2])
    assert vol1 in combined
    assert vol2 in combined


def test_concatenate_volumes_two_vols_one_boundary():
    vol1 = "First volume text."
    vol2 = "Second volume text."
    combined, boundaries = concatenate_volumes([vol1, vol2])
    assert len(boundaries) == 1


def test_concatenate_volumes_boundary_normalised_between_zero_and_one():
    vol1 = "First volume text."
    vol2 = "Second volume text."
    combined, boundaries = concatenate_volumes([vol1, vol2])
    assert 0.0 < boundaries[0] < 1.0


def test_concatenate_volumes_boundary_position_correct():
    vol1 = "a" * 100
    vol2 = "b" * 100
    combined, boundaries = concatenate_volumes([vol1, vol2])
    # vol1 ends at char 100, total is ~200; boundary should be ~0.5
    assert abs(boundaries[0] - 0.5) < 0.05


def test_concatenate_volumes_three_vols_two_boundaries():
    texts = ["Volume one text.", "Volume two text.", "Volume three text."]
    combined, boundaries = concatenate_volumes(texts)
    assert len(boundaries) == 2


def test_concatenate_volumes_boundaries_ascending():
    texts = ["a" * 100, "b" * 100, "c" * 100]
    combined, boundaries = concatenate_volumes(texts)
    assert boundaries[0] < boundaries[1]


def test_concatenate_volumes_boundary_at_vol1_end():
    vol1 = "a" * 100
    vol2 = "b" * 200
    vol3 = "c" * 100
    combined, boundaries = concatenate_volumes([vol1, vol2, vol3])
    total = len(combined)
    # boundary[0] should be near 100/total
    assert abs(boundaries[0] - 100 / total) < 0.05


def test_concatenate_volumes_empty_list_returns_empty():
    combined, boundaries = concatenate_volumes([])
    assert combined == ""
    assert boundaries == []


# ---------------------------------------------------------------------------
# load_novel_texts
# ---------------------------------------------------------------------------

PROCESSED_ROOT = Path(__file__).parent.parent.parent / "burney-attribution" / "data" / "processed"


def _make_row(author, title, year, genre, volume, file_path, notes=""):
    return {
        "author": author,
        "title": title,
        "year": year,
        "genre": genre,
        "volume": volume,
        "file_path": file_path,
        "notes": notes,
    }


def test_load_novel_texts_single_volume_evelina():
    rows = [_make_row("burney", "Evelina", "1778", "epistolary", "", "burney/Evelina.txt")]
    result = load_novel_texts(rows, PROCESSED_ROOT)
    assert len(result) == 1
    novel = result[0]
    assert novel["author"] == "burney"
    assert novel["title"] == "Evelina"
    assert novel["year"] == "1778"
    assert novel["genre"] == "epistolary"
    assert len(novel["text"]) > 0
    assert novel["volume_boundaries"] == []
    assert novel["word_count"] > 0


def test_load_novel_texts_novel_id_format():
    rows = [_make_row("burney", "Evelina", "1778", "epistolary", "", "burney/Evelina.txt")]
    result = load_novel_texts(rows, PROCESSED_ROOT)
    assert result[0]["id"] == "burney_evelina_1778"


def test_load_novel_texts_title_snake_case_with_spaces():
    rows = [_make_row("austen", "Sense and Sensibility", "1811", "domestic", "", "austen/SenseAndSensibility.txt")]
    result = load_novel_texts(rows, PROCESSED_ROOT)
    assert result[0]["id"] == "austen_sense_and_sensibility_1811"


def test_load_novel_texts_multi_volume_cecilia():
    rows = [
        _make_row("burney", "Cecilia", "1782", "domestic", "1", "burney/CeciliaVol1.txt"),
        _make_row("burney", "Cecilia", "1782", "domestic", "2", "burney/CeciliaVol2.txt"),
        _make_row("burney", "Cecilia", "1782", "domestic", "3", "burney/CeciliaVol3.txt"),
    ]
    result = load_novel_texts(rows, PROCESSED_ROOT)
    assert len(result) == 1
    novel = result[0]
    assert novel["title"] == "Cecilia"
    assert len(novel["volume_boundaries"]) == 2
    assert novel["word_count"] > 0


def test_load_novel_texts_multi_volume_boundaries_sorted():
    rows = [
        _make_row("burney", "Cecilia", "1782", "domestic", "1", "burney/CeciliaVol1.txt"),
        _make_row("burney", "Cecilia", "1782", "domestic", "2", "burney/CeciliaVol2.txt"),
        _make_row("burney", "Cecilia", "1782", "domestic", "3", "burney/CeciliaVol3.txt"),
    ]
    result = load_novel_texts(rows, PROCESSED_ROOT)
    bounds = result[0]["volume_boundaries"]
    assert bounds == sorted(bounds)
    for b in bounds:
        assert 0.0 < b < 1.0


def test_load_novel_texts_multiple_novels_sorted_by_year_then_title():
    rows = [
        _make_row("austen", "Emma", "1815", "domestic", "", "austen/Emma.txt"),
        _make_row("burney", "Evelina", "1778", "epistolary", "", "burney/Evelina.txt"),
        _make_row("austen", "Sense and Sensibility", "1811", "domestic", "", "austen/SenseAndSensibility.txt"),
    ]
    result = load_novel_texts(rows, PROCESSED_ROOT)
    years = [int(r["year"]) for r in result]
    assert years == sorted(years)


def test_load_novel_texts_word_count_matches_text():
    rows = [_make_row("burney", "Evelina", "1778", "epistolary", "", "burney/Evelina.txt")]
    result = load_novel_texts(rows, PROCESSED_ROOT)
    novel = result[0]
    assert novel["word_count"] == len(novel["text"].split())


def test_load_novel_texts_multi_volume_word_count():
    rows = [
        _make_row("burney", "Cecilia", "1782", "domestic", "1", "burney/CeciliaVol1.txt"),
        _make_row("burney", "Cecilia", "1782", "domestic", "2", "burney/CeciliaVol2.txt"),
        _make_row("burney", "Cecilia", "1782", "domestic", "3", "burney/CeciliaVol3.txt"),
    ]
    result = load_novel_texts(rows, PROCESSED_ROOT)
    novel = result[0]
    assert novel["word_count"] == len(novel["text"].split())


# ---------------------------------------------------------------------------
# load_corpus
# ---------------------------------------------------------------------------

METADATA_PATH = Path(__file__).parent.parent.parent / "burney-attribution" / "data" / "metadata_v2.csv"


def test_load_corpus_returns_list():
    result = load_corpus(METADATA_PATH, PROCESSED_ROOT)
    assert isinstance(result, list)


def test_load_corpus_non_empty():
    result = load_corpus(METADATA_PATH, PROCESSED_ROOT)
    assert len(result) > 0


def test_load_corpus_fewer_entries_than_rows():
    """Multi-volume works should be collapsed: novels < CSV rows."""
    result = load_corpus(METADATA_PATH, PROCESSED_ROOT)
    # metadata_v2.csv has 53 rows but several multi-volume works
    assert len(result) < 53


def test_load_corpus_each_entry_has_required_keys():
    result = load_corpus(METADATA_PATH, PROCESSED_ROOT)
    required = {"id", "author", "title", "year", "genre", "text", "word_count", "volume_boundaries"}
    for novel in result:
        assert required.issubset(novel.keys()), f"Missing keys in {novel.get('id')}"


def test_load_corpus_all_texts_non_empty():
    result = load_corpus(METADATA_PATH, PROCESSED_ROOT)
    for novel in result:
        assert len(novel["text"]) > 0, f"Empty text for {novel['id']}"


def test_load_corpus_sorted_by_year_then_title():
    result = load_corpus(METADATA_PATH, PROCESSED_ROOT)
    pairs = [(int(r["year"]), r["title"]) for r in result]
    assert pairs == sorted(pairs)


def test_load_corpus_default_paths(monkeypatch, tmp_path):
    """load_corpus() works when called with explicit paths (path coverage)."""
    result = load_corpus(METADATA_PATH, PROCESSED_ROOT)
    assert len(result) > 0


def test_load_corpus_clarissa_collapsed():
    """Clarissa (9 vols) should appear as exactly one entry."""
    result = load_corpus(METADATA_PATH, PROCESSED_ROOT)
    clarissa = [r for r in result if r["title"] == "Clarissa"]
    assert len(clarissa) == 1
    assert len(clarissa[0]["volume_boundaries"]) == 8


def test_load_corpus_evelina_no_boundaries():
    """Evelina (single vol) should have no volume boundaries."""
    result = load_corpus(METADATA_PATH, PROCESSED_ROOT)
    evelina = [r for r in result if r["title"] == "Evelina"]
    assert len(evelina) == 1
    assert evelina[0]["volume_boundaries"] == []
