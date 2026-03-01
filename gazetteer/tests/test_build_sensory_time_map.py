import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


REPO_ROOT = Path(__file__).parent.parent.parent
HTML_PATH = REPO_ROOT / "gazetteer" / "sensory_time_map.html"


@pytest.fixture(scope="module")
def html():
    """Build the time map and return HTML content."""
    result = subprocess.run(
        [sys.executable, "gazetteer/build_sensory_time_map.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return HTML_PATH.read_text(encoding="utf-8")


def test_html_generated(html):
    assert "<html" in html
    assert "Sensory Time Map" in html


def test_year_slider_present(html):
    assert 'id="year-slider"' in html
    assert 'min="1660"' in html
    assert 'max="1820"' in html


def test_month_pills_present(html):
    for month in ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]:
        assert month in html


def test_band_pills_present(html):
    for band in ["Dawn", "Morning", "Midday", "Afternoon", "Evening", "Night"]:
        assert band in html


def test_dow_pills_present(html):
    for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
        assert day in html


def test_events_json_embedded(html):
    assert "EVT001" in html
    assert "EVT030" in html
    assert "Smithfield" in html


def test_venues_json_embedded(html):
    assert "LON001" in html
    assert "Vauxhall" in html


def test_literary_toggle_present(html):
    assert "literary" in html.lower()
    assert "toggle" in html.lower() or "lit-toggle" in html


def test_compute_intensity_function(html):
    assert "computeIntensity" in html


def test_leaflet_included(html):
    assert "leaflet" in html.lower()


def test_venue_explorer_link(html):
    assert "venue_explorer.html" in html
