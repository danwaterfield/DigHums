"""Tests for extract_old_bailey.py"""

import json
import sys
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from sensory_db import init_db
from extract_old_bailey import (
    parse_date, alias_slug, extract_trial, fetch_page
)


def test_parse_date():
    assert parse_date(17650116) == 1765
    assert parse_date(17001231) == 1700
    assert parse_date(18200101) == 1820


def test_alias_slug():
    assert alias_slug("Newgate Prison") == "newgate_prison"
    assert alias_slug("King's Bench") == "king_s_bench"
    assert alias_slug("St Giles / Seven Dials") == "st_giles_seven_dials"


def test_extract_trial_finds_sensory_terms():
    text = ("The prisoner was seen near Newgate. The stench of the gaol "
            "was insupportable, and the din of the crowd overwhelming.")
    rows = extract_trial(
        text=text,
        venue_id="LON074",
        venue_name="Newgate Prison",
        lat=51.5153, lon=-0.1017,
        reference="t17650116-1",
        year=1765,
    )
    assert len(rows) > 0
    modalities = {r["modality"] for r in rows}
    assert "olfactory" in modalities or "auditory" in modalities


def test_extract_trial_sets_venue_directly():
    text = "The stench near the Sessions House was intolerable."
    rows = extract_trial(
        text=text,
        venue_id="LON080",
        venue_name="Old Bailey Courthouse",
        lat=51.5151, lon=-0.1017,
        reference="t17700601-2",
        year=1770,
    )
    assert all(r["venue_id"] == "LON080" for r in rows)
    assert all(r["source_type"] == "legal" for r in rows)


def test_extract_trial_valence_is_unpleasant():
    text = "The stench of Newgate was insupportable."
    rows = extract_trial(
        text=text,
        venue_id="LON074", venue_name="Newgate Prison",
        lat=51.5153, lon=-0.1017,
        reference="t17650116-3", year=1765,
    )
    assert len(rows) > 0
    assert all(r["valence"] == "unpleasant" for r in rows)


def test_extract_trial_source_id_format():
    text = "The mob jostled near the Fleet."
    rows = extract_trial(
        text=text,
        venue_id="LON075", venue_name="Fleet Prison",
        lat=51.5144, lon=-0.1048,
        reference="t17800501-5", year=1780,
    )
    assert len(rows) > 0
    assert all(r["source_id"] == "old_bailey_t17800501-5" for r in rows)


def test_fetch_page_uses_cache(tmp_path):
    """fetch_page() reads cache file and does not call urlopen."""
    cache_data = {"records": [], "total": 0}
    cache_file = tmp_path / "test_p0.json"
    cache_file.write_text(json.dumps(cache_data), encoding="utf-8")

    with patch("urllib.request.urlopen") as mock_url:
        result = fetch_page("Newgate", 0, cache_file)
        mock_url.assert_not_called()

    assert result == cache_data


def test_fetch_page_writes_cache(tmp_path):
    """fetch_page() writes response to cache on first fetch."""
    cache_file = tmp_path / "newgate_p0.json"
    fake_api_data = {"hits": {"total": 0, "hits": []}}
    fake_response = json.dumps(fake_api_data).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.read.return_value = fake_response
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with patch("time.sleep"):
            result = fetch_page("Newgate", 0, cache_file)

    assert cache_file.exists()
    assert result == fake_api_data


def test_ingest_single_trial(tmp_path):
    """End-to-end: fake API response → rows in sensory_evidence."""
    import extract_old_bailey as eob

    fake_trial = {
        "hits": {
            "total": 1,
            "hits": [
                {
                    "_source": {
                        "idkey": "t17650116-99",
                        "text": "Near Newgate. The stench of the place was insupportable "
                                "and the din of the condemned dreadful.",
                    }
                }
            ],
        }
    }

    db = init_db(tmp_path / "test.db")

    with patch.object(eob, "fetch_page", return_value=fake_trial):
        with patch.object(eob, "CACHE_DIR", tmp_path):
            eob.ingest_venue(
                venue={"id": "LON074", "name": "Newgate Prison",
                       "lat": "51.5153", "lon": "-0.1017"},
                aliases=[("Newgate", None)],
                conn=db,
                write=True,
            )

    count = db.execute(
        "SELECT COUNT(*) FROM sensory_evidence WHERE source_type='legal'"
    ).fetchone()[0]
    assert count > 0
