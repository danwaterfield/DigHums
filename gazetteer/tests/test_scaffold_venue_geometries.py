from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).parent.parent))

from scaffold_venue_geometries import active_window, provisional_rows


def test_active_window_intersects_dates():
    assert active_window(1742, 1803, 1740, 1790) == (1742, 1790)
    assert active_window(1783, None, 1791, 1820) == (1791, 1820)
    assert active_window(None, 1783, 1791, 1820) is None


def test_provisional_rows_skips_existing_and_respects_city():
    venues = [
        {
            "id": "LON001",
            "city": "London",
            "lat": "51.4882",
            "lon": "-0.1228",
            "opened": "1661",
            "closed": "",
            "map_layer": "rocque_1746;horwood_1799",
        },
        {
            "id": "BAT001",
            "city": "Bath",
            "lat": "51.3812",
            "lon": "-2.3587",
            "opened": "1708",
            "closed": "1820",
            "map_layer": "wood_1736;harcourt_1794",
        },
    ]
    existing = [
        {
            "venue_id": "LON001",
            "year_start": "1740",
            "year_end": "1790",
            "source_map": "rocque_1746",
        }
    ]

    rows = provisional_rows(venues, existing, city="London")
    assert len(rows) == 1
    row = rows[0]
    assert row["venue_id"] == "LON001"
    assert row["source_map"] == "horwood_1799"
    assert row["year_start"] == 1791
    assert row["year_end"] == 1820
    assert row["confidence"] == "provisional"
