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
    assert "enclosureFactor" in html
    assert "so2_index" in html
    # Street-channeled smoke: all three blend components present
    assert "streetFieldMag" in html
    assert "streetFieldDx" in html
    assert "WIND_DX * 0.3" in html


def test_flow_mode_field_builder(html):
    assert "_buildFlowField" in html
    assert "MODALITY_PROFILE" in html
    assert "attractStrength" in html
    # Street channeling for noise and smell
    assert "streetFieldDy" in html
    assert "modalTotals" in html


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


def test_particle_mode_labels_renamed(html):
    """Smoke→Atmosphere, Flow→Senses in button labels."""
    assert "Atmosphere" in html
    assert "Senses" in html


def test_particle_legend_present(html):
    assert 'id="particle-legend"' in html


def test_particle_mode_js_handler(html):
    assert "data-pmode" in html
    assert "particleMode" in html


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


def test_modality_profiles_present(html):
    """MODALITY_PROFILE constant must define all five modality profiles."""
    assert "MODALITY_PROFILE" in html
    for modal in ("smoke", "smell", "noise", "crowd", "visual"):
        assert f"'{modal}'" in html or f'"{modal}"' in html


def test_per_modality_alpha_curves(html):
    """particleFrame must have per-modality alpha curves."""
    assert "p.modality === 'smoke'" in html
    assert "p.modality === 'noise'" in html
    assert "p.modality === 'smell'" in html


def test_particle_trail_fade(html):
    """particleFrame must use destination-out compositing for trails."""
    assert "destination-out" in html
    assert "globalCompositeOperation" in html


def test_smoke_dissipation(html):
    """Smoke particles render as velocity-aligned line segments (not fixed-radius arcs)."""
    assert "Math.hypot(p.vx, p.vy)" in html
    assert "pCtx.lineCap = 'round'" in html
    assert "pCtx.moveTo(p.px - p.vx * 3" in html


def test_modality_radius_used(html):
    """Particles must use prof.radius for lineWidth, not a fixed value."""
    assert "drawRadius" in html
    assert "prof.radius" in html


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


def test_tooltip_provenance_line(html):
    """Tooltip must call computeZoneBaseline and use provenance."""
    assert 'zoneBaseline.provenance' in html


def test_zone_aware_particles(html):
    """Smoke field builder must use zone industrial_intensity to scale particles."""
    assert 'zoneProps.industrial_intensity' in html


# ── Market schedule, canyon factor, and Exeter Change ─────────────────────────

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


def test_canyon_factor_used_in_smokefield(html):
    """_buildSmokefield must use venueCanyonFactor."""
    assert 'venueCanyonFactor' in html
    assert 'canyonFactor' in html


def test_canyon_factor_in_flowfield(html):
    """_buildFlowField must use _venueCanyonFactor for smell."""
    assert '_venueCanyonFactor' in html


def test_exeter_change_venue(html):
    """Exeter Change Menagerie (LON097) must be present in VENUES."""
    assert 'LON097' in html
    assert 'Exeter Change' in html


def test_exeter_change_event(html):
    """EVT070 Exeter Change Menagerie event must be present."""
    assert 'EVT070' in html
    assert 'Menagerie' in html


def test_canyon_factor_computeZoneBaseline(html):
    """computeZoneBaseline must return canyon_factor."""
    assert 'canyon_factor: p.canyon_factor' in html


def test_noise_rings_renderer(html):
    """Noise ring state and renderers must be present."""
    assert '_updateNoiseRings' in html
    assert '_drawNoiseRings' in html
    assert '_noiseRingPool' in html


def test_smell_halos_renderer(html):
    """Smell halo renderer must be present."""
    assert '_drawSmellHalos' in html


def test_crowd_circles_renderer(html):
    """Crowd circle renderer must be present."""
    assert '_drawCrowdCircles' in html


def test_visual_glow_renderer(html):
    """Visual glow renderer must be present."""
    assert '_drawVisualGlow' in html


def test_senses_mode_compositor(html):
    """All four per-modality renderer calls must appear in particleFrame."""
    assert '_updateNoiseRings' in html
    assert '_drawNoiseRings'   in html
    assert '_drawSmellHalos'   in html
    assert '_drawCrowdCircles' in html
    assert '_drawVisualGlow'   in html
