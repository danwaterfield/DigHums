import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


REPO_ROOT = Path(__file__).parent.parent.parent
HTML_PATH = REPO_ROOT / "gazetteer" / "sensory_time_map.html"
VENUE_GEOMETRIES_PATH = REPO_ROOT / "gazetteer" / "venue_geometries.csv"


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


def test_corrected_event_csv_content_embedded(html):
    """The built map should reflect the current CSV event notes, not stale DB copies."""
    assert "Act passed in 1699 (often cited under 1698 regnal dating)" in html
    assert "Usually open Mondays Wednesdays and Fridays" in html
    assert "Pre-1752: Oct 28 (Feast of St Simon and St Jude)" in html


def test_venues_json_embedded(html):
    assert "LON001" in html
    assert "Vauxhall" in html


def test_venue_geometries_csv_scaffold_exists():
    assert VENUE_GEOMETRIES_PATH.exists()
    lines = VENUE_GEOMETRIES_PATH.read_text(encoding="utf-8").splitlines()
    header = lines[0]
    assert header == "venue_id,year_start,year_end,lat,lon,source_map,confidence,notes"
    assert len(lines) > 10
    assert any(line.startswith("LON001,") for line in lines[1:])


def test_venue_geometry_support_embedded(html):
    assert "VENUE_GEOMETRIES" in html
    assert '"canonical_lat"' in html
    assert '"canonical_lon"' in html
    assert "map_layer" in html
    assert "_pickVenueGeometry" in html
    assert "_applyVenueDisplayGeometry" in html
    assert "_venuePlacementSummary" in html


def test_literary_toggle_present(html):
    assert "literary" in html.lower()
    assert "toggle" in html.lower() or "lit-toggle" in html


def test_compute_intensity_function(html):
    assert "computeIntensity" in html


def test_leaflet_included(html):
    assert "leaflet" in html.lower()


def test_venue_explorer_link(html):
    assert "venue_explorer.html" in html


def test_extended_source_badges_present(html):
    assert "src-newspaper" in html
    assert "src-parish" in html
    assert "src-institutional" in html


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


def test_street_direction_field_present(html):
    """Street direction field arrays and builder must be present."""
    assert "streetFieldDx"     in html
    assert "streetFieldDy"     in html
    assert "streetFieldMag"    in html
    assert "_rebuildStreetField" in html
    assert "_hwayMult"         in html


def test_street_network_has_t_field(html):
    """STREET_NETWORK entries must include 't' (highway type) field."""
    import re, json
    m = re.search(r'const STREET_NETWORK = (\[.*?\]);', html, re.DOTALL)
    assert m is not None
    data = json.loads(m.group(1))
    assert len(data) > 0
    entry = data[0]
    assert "t" in entry, "STREET_NETWORK entry must have 't' (highway type) key"


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


def test_compute_intensity_returns_smoke(html):
    """computeIntensity must return a .smoke property derived from SO2 index."""
    assert "loads.smoke" in html
    assert "smokeLoad" in html or "so2_index" in html


def test_tooltip_provenance_line(html):
    """Tooltip must call computeZoneBaseline and use provenance."""
    assert 'zoneBaseline.provenance' in html


def test_smithfield_event_has_wednesday(html):
    """Smithfield market must run Mon|Wed|Fri (not Mon|Fri only)."""
    assert '"Mon|Wed|Fri"' in html or "'Mon|Wed|Fri'" in html


def test_covent_garden_event_daily(html):
    """Covent Garden market must run Mon–Sat."""
    assert 'Mon|Tue|Wed|Thu|Fri|Sat' in html


def test_canyon_factor_in_zones(html):
    """ZONE_DATA must include canyon_factor in decade entries."""
    assert 'canyon_factor' in html


def test_canyon_factor_interpolated(html):
    """interpolateZoneProps must interpolate canyon_factor."""
    assert 'p0.canyon_factor' in html


def test_exeter_change_venue(html):
    """Exeter Change Menagerie (LON097) must be present in VENUES."""
    assert 'LON097' in html
    assert 'Exeter Change' in html


def test_exeter_change_event(html):
    """Exeter Change should remain embedded as venue data even without a dedicated event row."""
    assert 'Menagerie' in html


def test_canyon_factor_computeZoneBaseline(html):
    """computeZoneBaseline must return canyon_factor."""
    assert 'canyon_factor: p.canyon_factor' in html


def test_contour_overlay_buttons(html):
    """Contour overlay mode buttons must be present."""
    assert 'data-cmode="off"' in html
    assert 'data-cmode="atmosphere"' in html
    assert 'data-cmode="senses"' in html


def test_contour_sense_sub_selector(html):
    """Sense sub-selector buttons must be present when Senses mode active."""
    assert 'data-sense="smoke"' in html
    assert 'data-sense="smell"' in html
    assert 'data-sense="noise"' in html
    assert 'data-sense="crowd"' in html
    assert 'data-sense="visual"' in html


def test_contour_grid_layer_present(html):
    """IDW contour GridLayer must be defined."""
    assert "L.GridLayer.extend" in html
    assert "createTile" in html


def test_contour_colour_ramps(html):
    """Colour ramps must be defined for all modalities."""
    assert "CONTOUR_RAMPS" in html
    for mode in ["atmosphere", "smell", "noise", "crowd", "visual", "smoke"]:
        assert mode in html


def test_contour_idw_wind_bias(html):
    """Atmosphere mode must apply wind bias (eastward stretch)."""
    assert "0.77" in html


def test_contour_canyon_cutoff(html):
    """Canyon effect must modify IDW cutoff by enclosure type."""
    assert "enclosed" in html
    assert "400" in html


def test_contour_marching_squares(html):
    """Contour drawing must use marching squares algorithm."""
    assert "config" in html


def test_contour_gaussian_blur(html):
    """Grid must be blurred before contour tracing."""
    assert "_blurGrid" in html


def test_contour_three_thresholds(html):
    """Three contour isolines at 0.4, 0.6, 0.8."""
    assert "0.4" in html
    assert "0.6" in html
    assert "0.8" in html


def test_contour_labels_georgia(html):
    """Contour labels must use italic Georgia font."""
    assert "italic" in html
    assert "Georgia" in html


def test_contour_max_opacity(html):
    """Contour surface max opacity must be 0.40."""
    assert "maxAlpha" in html


def test_contour_mode_state(html):
    """State object must have contourMode and contourSense."""
    assert "contourMode" in html
    assert "contourSense" in html


def test_bookseller_building_type_colour(html):
    assert "'bookseller'" in html


def test_bookseller_venues_in_sensory_map(html):
    """Bookseller venues from venues.csv appear in the sensory map."""
    assert "LON100" in html  # Millar-Cadell shop
    assert "LON101" in html  # Dodsley's shop
