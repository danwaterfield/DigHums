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
    assert "Smithfield" in html
    assert "Tyburn" in html


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


# ── C5: Environmental layer tests ─────────────────────────────────────────────

def test_cet_data_embedded(html):
    """CET_DATA constant is present and contains at least one year of data."""
    assert "CET_DATA" in html
    # 1684 is the Frost Fair year; Jan temp was -3.0 — verify it appears
    assert "1684" in html


def test_temperature_indicator_present(html):
    """Temperature gauge elements are rendered in the controls bar."""
    assert 'id="temp-badge"' in html
    assert "temp-badge" in html
    # Tooltip / title references HadCET or Met Office
    assert "HadCET" in html or "Met Office" in html


def test_mortality_data_embedded(html):
    """MORTALITY_DATA constant is present and contains bills-of-mortality totals."""
    assert "MORTALITY_DATA" in html
    assert 'id="mort-badge"' in html


def test_smoke_data_embedded(html):
    """SMOKE_DATA_ENV constant is present and contains decade estimates."""
    assert "SMOKE_DATA_ENV" in html
    assert "so2_index" in html


def test_smoke_overlay_present(html):
    """The smoke haze overlay div is rendered in the map."""
    assert 'id="smoke-overlay"' in html


def test_tier_toggle_present(html):
    """Tier view toggle button is present."""
    assert 'id="tier-toggle"' in html


def test_venues_include_tier(html):
    """Venue data includes the tier field."""
    assert '"tier"' in html


def test_street_network_embedded(html):
    """STREET_NETWORK constant is baked into the HTML."""
    assert "STREET_NETWORK" in html
    # Should be a non-empty array
    import re
    import json
    m = re.search(r'const STREET_NETWORK = (\[.*?\]);', html, re.DOTALL)
    assert m is not None
    data = json.loads(m.group(1))
    assert len(data) > 100  # OHM has 2000+ pre-1820 London streets
    # New format: {p: [[lat,lon],...], s: start_yr|null, e: end_yr|null}
    entry = data[0]
    assert isinstance(entry, dict), "STREET_NETWORK entries must be objects with p/s/e keys"
    assert "p" in entry, "Entry must have 'p' (polyline) key"
    assert isinstance(entry["p"], list)
    assert len(entry["p"][0]) == 2  # first point is [lat, lon]


def test_particle_canvas_present(html):
    assert 'id="particle-canvas"' in html

def test_particle_engine_present(html):
    assert "particleFrame" in html
    assert "startParticles" in html
    assert "stopParticles" in html
    assert "venueIntensityCache" in html


def test_smoke_mode_field_builder(html):
    assert "_buildSmokefield" in html
    assert "windDx" in html
    assert "enclosureFactor" in html
    assert "so2_index" in html


def test_flow_mode_field_builder(html):
    assert "_buildFlowField" in html
    assert "MODALITY_COLOURS" in html
    assert "attractStrength" in html


def test_network_mode_field_builder(html):
    assert "_buildNetworkField" in html
    assert "STREET_NETWORK" in html
    assert "segT" in html
    assert "_networkStep" in html


def test_particle_mode_buttons_present(html):
    assert 'data-pmode="off"'     in html
    assert 'data-pmode="smoke"'   in html
    assert 'data-pmode="flow"'    in html
    assert 'data-pmode="network"' in html

def test_particle_mode_js_handler(html):
    assert "data-pmode" in html
    assert "particleMode" in html


def test_venue_opened_closed_in_js(html):
    """Venues JSON must include opened and closed fields."""
    # LON001 has opened=1661 in venues.csv
    assert '"opened": 1661' in html or '"opened":1661' in html


def test_zone_data_embedded(html):
    """ZONE_DATA constant must be present and contain zone names."""
    assert 'const ZONE_DATA' in html
    assert 'Smithfield' in html


def test_zone_inference_functions(html):
    """Zone baseline inference functions must be present."""
    assert 'function pointInPolygon' in html
    assert 'function getZoneForPoint' in html
    assert 'function computeZoneBaseline' in html


def test_zone_point_in_polygon_present(html):
    """pointInPolygon must use GeoJSON [lon,lat] ring order."""
    # GeoJSON order: xi = ring[i][0] (longitude), yi = ring[i][1] (latitude)
    assert 'xi = ring[i][0]' in html
    assert 'yi = ring[i][1]' in html


def test_zone_river_proximity_interpolated(html):
    """river_proximity must be interpolated like other numeric fields."""
    assert 'p0.river_proximity + t * (p1.river_proximity - p0.river_proximity)' in html


def test_zone_env_modifiers_present(html):
    """computeZoneBaseline must apply all four environmental modifiers."""
    assert 'river_proximity' in html        # modifier 1
    assert 'industrial_intensity' in html   # modifier 2
    assert 'frost fair' in html             # modifier 3
    assert "=== 'narrow'" in html           # modifier 4


def test_zone_fill_layer(html):
    """Zone fill Leaflet layer must be present."""
    assert 'zoneFillLayer' in html
    assert 'updateZoneFills' in html


def test_click_anywhere_handler(html):
    """Map click-anywhere handler and renderLocationPanel must be present."""
    assert 'function renderLocationPanel' in html
    assert "map.on('click'" in html
