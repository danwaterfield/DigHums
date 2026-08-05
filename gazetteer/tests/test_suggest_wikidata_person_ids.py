import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from suggest_wikidata_person_ids import (
    build_query_variants,
    candidate_score,
    is_high_confidence,
    parse_year_span,
)


def test_parse_year_span_handles_simple_ranges():
    assert parse_year_span("1752-1840") == (1752, 1840)
    assert parse_year_span("c. 1741-1821") == (1741, 1821)
    assert parse_year_span("born 1726") == (1726, None)


def test_build_query_variants_adds_suffixless_and_first_last_forms():
    assert build_query_variants("Charles Burney Jr") == [
        "Charles Burney Jr",
        "Charles Burney",
    ]
    assert build_query_variants("Hester Thrale Piozzi") == [
        "Hester Thrale Piozzi",
        "Hester Piozzi",
    ]


def test_candidate_score_prefers_exact_name_and_date_matches():
    entity = {
        "labels": {"en": {"value": "Frances Burney"}},
        "aliases": {"en": [{"value": "Fanny Burney"}]},
        "claims": {
            "P569": [{"mainsnak": {"datavalue": {"value": {"time": "+1752-06-13T00:00:00Z"}}}}],
            "P570": [{"mainsnak": {"datavalue": {"value": {"time": "+1840-01-06T00:00:00Z"}}}}],
        },
    }
    score, reasons = candidate_score(
        "Frances Burney",
        1752,
        1840,
        ["Frances Burney"],
        entity,
    )
    assert score >= 19
    assert "exact_label" in reasons
    assert "full_date_match" in reasons
    assert is_high_confidence(1752, 1840, reasons) is True
