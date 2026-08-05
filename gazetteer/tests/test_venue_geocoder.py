import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from venue_geocoder import geocode_passage

VENUES = [
    {"id": "LON001", "name": "Vauxhall Spring Gardens",
     "lat": "51.4882", "lon": "-0.1228"},
    {"id": "LON006", "name": "Theatre Royal Drury Lane",
     "lat": "51.5133", "lon": "-0.1226"},
    {"id": "BAT003", "name": "The Pump Room",
     "lat": "51.3814", "lon": "-2.3594"},
]

def test_vauxhall_passage():
    text = ("The company proceeded to Vauxhall, where the illuminations "
            "were remarkably brilliant and the music loud.")
    result = geocode_passage(text, "illuminations were remarkably brilliant", VENUES)
    assert result is not None
    assert result["venue_id"] == "LON001"

def test_no_venue_returns_none():
    text = "The mud was deep and the fog considerable."
    result = geocode_passage(text, "mud was deep", VENUES)
    assert result is None

def test_drury_lane_passage():
    text = "We secured a box at Drury Lane and the noise of the pit was deafening."
    result = geocode_passage(text, "noise of the pit was deafening", VENUES)
    assert result is not None
    assert result["venue_id"] == "LON006"


def test_city_filtered_supplement_alias_for_bath():
    text = "We reached Bath by noon, and at the Pump the air felt oppressively warm."
    result = geocode_passage(
        text,
        "air felt oppressively warm",
        VENUES,
        primary_cities="Bath",
    )
    assert result is not None
    assert result["venue_id"] == "BAT003"


def test_anchor_pos_bypasses_passage_lookup():
    text = "We secured a box at Drury Lane and the noise of the pit was deafening."
    anchor_pos = text.index("noise")
    result = geocode_passage(
        text,
        "not actually present",
        VENUES,
        anchor_pos=anchor_pos,
    )
    assert result is not None
    assert result["venue_id"] == "LON006"
