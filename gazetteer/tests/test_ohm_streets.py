# gazetteer/tests/test_ohm_streets.py
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))
from build_sensory_time_map import (
    perpendicular_distance,
    douglas_peucker,
    parse_ohm_year,
    parse_ohm_response,
)


def test_perpendicular_distance_on_line():
    # Point exactly on the line between (0,0)→(1,0) should have distance 0
    assert perpendicular_distance((0.5, 0), (0, 0), (1, 0)) == pytest.approx(0.0)


def test_perpendicular_distance_off_line():
    # Point at (0,1) perpendicular to line (0,0)→(1,0) has distance 1
    assert perpendicular_distance((0, 1), (0, 0), (1, 0)) == pytest.approx(1.0)


def test_douglas_peucker_removes_collinear():
    # Three collinear points — middle one should be removed
    pts = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0)]
    result = douglas_peucker(pts, epsilon=0.0001)
    assert result == [(0.0, 0.0), (1.0, 0.0)]


def test_douglas_peucker_keeps_deviant():
    # Middle point deviates significantly — must be kept
    pts = [(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)]
    result = douglas_peucker(pts, epsilon=0.0001)
    assert len(result) == 3


def test_parse_ohm_year_full():
    assert parse_ohm_year("1746-01-01") == 1746


def test_parse_ohm_year_partial():
    assert parse_ohm_year("1746") == 1746


def test_parse_ohm_year_ancient():
    assert parse_ohm_year("0045") == 45


def test_parse_ohm_year_none():
    assert parse_ohm_year("") is None
    assert parse_ohm_year(None) is None


def test_parse_ohm_response_filters_by_date():
    fake_response = {
        "elements": [
            # pre-1820 road — keep
            {"type": "way", "id": 1,
             "tags": {"highway": "primary", "start_date": "1746"},
             "geometry": [{"lat": 51.51, "lon": -0.12}, {"lat": 51.52, "lon": -0.13}]},
            # post-1820 road — discard
            {"type": "way", "id": 2,
             "tags": {"highway": "primary", "start_date": "1850"},
             "geometry": [{"lat": 51.51, "lon": -0.12}, {"lat": 51.52, "lon": -0.13}]},
            # ended before project period — discard
            {"type": "way", "id": 3,
             "tags": {"highway": "primary", "start_date": "1600", "end_date": "1640"},
             "geometry": [{"lat": 51.51, "lon": -0.12}, {"lat": 51.52, "lon": -0.13}]},
        ]
    }
    result = parse_ohm_response(fake_response)
    assert len(result) == 1
    assert result[0]["p"] == [[51.51, -0.12], [51.52, -0.13]]
    assert result[0]["s"] == 1746
    assert result[0]["e"] is None
    assert result[0]["t"] == "primary"


def test_parse_ohm_response_t_field_present():
    """Each segment must have a 't' (highway type) key."""
    fake_response = {
        "elements": [
            {"type": "way", "id": 1,
             "tags": {"highway": "residential", "start_date": "1700"},
             "geometry": [{"lat": 51.51, "lon": -0.12}, {"lat": 51.52, "lon": -0.13}]},
            {"type": "way", "id": 2,
             "tags": {"start_date": "1700"},   # no highway tag
             "geometry": [{"lat": 51.51, "lon": -0.12}, {"lat": 51.52, "lon": -0.13}]},
        ]
    }
    result = parse_ohm_response(fake_response)
    assert len(result) == 2
    assert result[0]["t"] == "residential"
    assert result[1]["t"] == ""   # missing highway tag -> empty string


def test_parse_ohm_response_skips_single_point_ways():
    fake_response = {
        "elements": [
            {"type": "way", "id": 1,
             "tags": {"highway": "primary", "start_date": "1746"},
             "geometry": [{"lat": 51.51, "lon": -0.12}]},  # only 1 point
        ]
    }
    result = parse_ohm_response(fake_response)
    assert result == []
