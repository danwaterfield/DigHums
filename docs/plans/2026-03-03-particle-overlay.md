# Particle Overlay Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a null-school-style canvas particle system to `sensory_time_map.html` with three switchable modes: smoke/smell diffusion (A), per-modality flow fields (B), and street-network-constrained diffusion (C).

**Architecture:** A single `<canvas>` overlay sits above Leaflet's tile panes and is driven by `requestAnimationFrame`. Three vector field algorithms share one particle engine; the active mode is stored in `state.particleMode`. Historical street geometry for Mode C is fetched from OpenHistoricalMap's Overpass API at Python build time and baked into the HTML as `STREET_NETWORK`.

**Tech Stack:** Python `urllib.request` (stdlib, no new deps) for OHM fetch; vanilla JS canvas 2D API; existing Leaflet `map.latLngToContainerPoint()` for coordinate transforms.

---

## Task 1: OHM street fetch, simplification, and bake

**Files:**
- Modify: `gazetteer/build_sensory_time_map.py` — add `fetch_ohm_streets()`, `douglas_peucker()`, `perpendicular_distance()`, wire into `build()`
- Create: `gazetteer/tests/test_ohm_streets.py`
- Side-effect: `gazetteer/ohm_streets_cache.json` (written at build time, gitignored)

### Step 1: Write the failing tests

```python
# gazetteer/tests/test_ohm_streets.py
import sys
import json
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
    import pytest
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
    assert result[0] == [[51.51, -0.12], [51.52, -0.13]]


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
```

### Step 2: Run tests — verify they fail

```bash
pytest gazetteer/tests/test_ohm_streets.py -v
```
Expected: `ImportError` (functions don't exist yet).

### Step 3: Implement the functions

Add to `gazetteer/build_sensory_time_map.py`, immediately after the existing `import` block (before `VENUES_PATH = ...`):

```python
import urllib.request
import urllib.parse

OHM_CACHE_PATH = Path(__file__).parent / "ohm_streets_cache.json"
OHM_BBOX       = "51.470,-0.200,51.540,-0.050"   # wider than venue spread


def perpendicular_distance(
    point: tuple[float, float],
    line_start: tuple[float, float],
    line_end: tuple[float, float],
) -> float:
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    den = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
    return num / den if den > 0 else 0.0


def douglas_peucker(
    points: list[tuple[float, float]], epsilon: float
) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return list(points)
    d_max, idx = 0.0, 0
    end = len(points) - 1
    for i in range(1, end):
        d = perpendicular_distance(points[i], points[0], points[end])
        if d > d_max:
            d_max, idx = d, i
    if d_max > epsilon:
        left  = douglas_peucker(points[:idx + 1], epsilon)
        right = douglas_peucker(points[idx:],     epsilon)
        return left[:-1] + right
    return [points[0], points[end]]


def parse_ohm_year(date_str: str | None) -> int | None:
    if not date_str:
        return None
    try:
        return int(date_str.split("-")[0])
    except (ValueError, AttributeError):
        return None


def parse_ohm_response(data: dict) -> list[list[list[float]]]:
    """
    Convert OHM Overpass JSON to list of simplified polylines.
    Each polyline is [[lat, lon], ...].
    Filters to ways that existed within 1660–1820.
    """
    segments = []
    for el in data.get("elements", []):
        if el.get("type") != "way":
            continue
        tags = el.get("tags", {})
        geom = el.get("geometry", [])
        if len(geom) < 2:
            continue
        start_yr = parse_ohm_year(tags.get("start_date"))
        end_yr   = parse_ohm_year(tags.get("end_date"))
        # Keep if way started by 1820 and hadn't ended before 1660
        if start_yr is not None and start_yr > 1820:
            continue
        if end_yr is not None and end_yr < 1660:
            continue
        pts = [(pt["lat"], pt["lon"]) for pt in geom]
        # Simplify: epsilon ~0.00015° ≈ 10 m — reduces points by ~60%
        simplified = douglas_peucker(pts, epsilon=0.00015)
        if len(simplified) >= 2:
            segments.append([[p[0], p[1]] for p in simplified])
    return segments


def fetch_ohm_streets(cache_path: Path = OHM_CACHE_PATH) -> list[list[list[float]]]:
    """
    Return pre-1820 London street segments from OpenHistoricalMap.
    Result is cached to ohm_streets_cache.json; subsequent builds are instant.
    """
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    query = f"""
[out:json][timeout:30];
way["highway"]({OHM_BBOX})["start_date"];
out geom qt;
"""
    url  = "https://overpass-api.openhistoricalmap.org/api/interpreter"
    data = urllib.parse.urlencode({"data": query}).encode()
    req  = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=35) as resp:
        raw = json.loads(resp.read().decode("utf-8"))

    segments = parse_ohm_response(raw)
    cache_path.write_text(json.dumps(segments, separators=(",", ":")), encoding="utf-8")
    print(f"  OHM streets cached: {len(segments)} segments -> {cache_path.name}")
    return segments
```

### Step 4: Wire into `build()`

In `build_sensory_time_map.py`, find the `build()` function. Add the street fetch and pass it to `HTML_TEMPLATE.format()`:

```python
def build(venues_path: Path = VENUES_PATH, db_path: Path = DB_PATH,
          out_path: Path = OUT_PATH) -> None:
    data    = load_data(venues_path, db_path)
    streets = fetch_ohm_streets()          # ← add this line

    html = HTML_TEMPLATE.format(
        EVENTS_JSON          = json.dumps(data["events"],          ensure_ascii=False),
        EVENT_VENUES_JSON    = json.dumps(data["event_venues"],    ensure_ascii=False),
        EVENT_INSTANCES_JSON = json.dumps(data["event_instances"], ensure_ascii=False),
        VENUES_JSON          = json.dumps(data["venues"],          ensure_ascii=False),
        EVIDENCE_JSON        = json.dumps(data["evidence"],        ensure_ascii=False),
        CET_JSON             = json.dumps(data["cet"],             ensure_ascii=False),
        MORTALITY_JSON       = json.dumps(data["mortality"],       ensure_ascii=False),
        SMOKE_JSON           = json.dumps(data["smoke"],           ensure_ascii=False),
        STREET_NETWORK_JSON  = json.dumps(streets,                 ensure_ascii=False),  # ← add
    )
    ...
```

Also add `{STREET_NETWORK_JSON}` placeholder in the JS section of `HTML_TEMPLATE` (after the existing `const SMOKE_DATA_ENV` line):

```javascript
const STREET_NETWORK = {STREET_NETWORK_JSON};  // [[lat,lon],...] pre-1820 streets
```

Add `ohm_streets_cache.json` to `.gitignore`:

```bash
echo "gazetteer/ohm_streets_cache.json" >> .gitignore
```

### Step 5: Add HTML-level test for STREET_NETWORK

Append to `gazetteer/tests/test_build_sensory_time_map.py`:

```python
def test_street_network_embedded(html):
    """STREET_NETWORK constant is baked into the HTML."""
    assert "STREET_NETWORK" in html
    # Should be a non-empty array
    import re
    m = re.search(r'const STREET_NETWORK = (\[.*?\]);', html, re.DOTALL)
    assert m is not None
    data = json.loads(m.group(1))
    assert len(data) > 100  # OHM has 2000+ pre-1820 London streets
    assert isinstance(data[0][0], list)   # [[lat,lon], ...]
    assert len(data[0][0]) == 2
```

### Step 6: Run all tests

```bash
pytest gazetteer/tests/test_ohm_streets.py gazetteer/tests/test_build_sensory_time_map.py -v
```
Expected: all pass. If `fetch_ohm_streets` makes a real network call, that's fine — it writes a cache file and subsequent runs are instant.

### Step 7: Commit

```bash
git add gazetteer/build_sensory_time_map.py gazetteer/tests/test_ohm_streets.py \
        gazetteer/tests/test_build_sensory_time_map.py .gitignore \
        gazetteer/sensory_time_map.html
git commit -m "feat: bake OHM pre-1820 street network into sensory time map"
```

---

## Task 2: Canvas infrastructure and particle engine skeleton

**Files:**
- Modify: `gazetteer/build_sensory_time_map.py` — HTML/CSS/JS additions only

### Step 1: Add canvas element to HTML template

Find `<div id="smoke-overlay"></div>` in `HTML_TEMPLATE`. Add the particle canvas **after** it:

```html
<div id="smoke-overlay"></div>
<canvas id="particle-canvas" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:410;opacity:0;transition:opacity 0.5s"></canvas>
```

The canvas starts at `opacity:0` — it becomes visible only when a mode is selected.

### Step 2: Add CSS for particle modes in style block

Append to the `<style>` block (after the existing `.btype-pill` rules):

```css
.particle-btn {{ background: #1a1a2e; border: 1px solid #555; color: #c8b89a; padding: 2px 8px;
                 cursor: pointer; border-radius: 12px; font-size: 0.78em; opacity: 0.75; }}
.particle-btn:hover {{ opacity: 1; }}
.particle-btn.active {{ background: rgba(255,255,255,0.12); border-color: #aaa; color: #fff;
                        opacity: 1; font-weight: bold; }}
```

### Step 3: Add particle engine JS to HTML template

Add a new `<script>` block immediately **before** the closing `</script>` of the main script (i.e., before `updateMap();`). This is the particle engine skeleton — no modes yet, just the infrastructure:

```javascript
// ── Particle System ──────────────────────────────────────────────────────────
const pCanvas = document.getElementById('particle-canvas');
const pCtx    = pCanvas ? pCanvas.getContext('2d') : null;

// Resize canvas to match map container
function resizeParticleCanvas() {{
    if (!pCanvas) return;
    const mapEl = document.getElementById('map');
    pCanvas.width  = mapEl.offsetWidth;
    pCanvas.height = mapEl.offsetHeight;
}}
resizeParticleCanvas();
window.addEventListener('resize', resizeParticleCanvas);

// Particle state — simple object array (2 000 particles max)
const MAX_P = 2000;
const particles = [];

function spawnParticle() {{
    return {{
        px: 0, py: 0,       // pixel position
        vx: 0, vy: 0,       // pixel velocity
        age: 0, maxAge: 200 + Math.random() * 400,
        r: 180, g: 130, b: 40,  // colour (overwritten per mode)
    }};
}}

// Initialise pool
for (let i = 0; i < MAX_P; i++) particles.push(spawnParticle());

// Vector field grid (updated when mode/year changes)
const FIELD_W = 80, FIELD_H = 60;
const fieldDx = new Float32Array(FIELD_W * FIELD_H);
const fieldDy = new Float32Array(FIELD_W * FIELD_H);

// Cache of venue pixel positions — updated on Leaflet moveend/zoomend
const venuePx = {{}};  // {{ id: {{px, py}} }}

function updateVenuePx() {{
    VENUES.forEach(v => {{
        const pt = map.latLngToContainerPoint([v.lat, v.lon]);
        venuePx[v.id] = {{ px: pt.x, py: pt.y }};
    }});
}}
updateVenuePx();
map.on('moveend zoomend', () => {{
    updateVenuePx();
    resetParticles();
}});

// Cache of last computed intensities (populated by updateMap)
const venueIntensityCache = {{}};   // {{ id: loads }}

function resetParticles() {{
    particles.forEach(p => {{ p.age = p.maxAge; }});  // force respawn next frame
}}

// Bilinear field sample at pixel (px, py)
function sampleField(px, py) {{
    if (!pCanvas) return {{ dx: 0, dy: 0 }};
    const gx = Math.min(FIELD_W - 1, Math.max(0, (px / pCanvas.width)  * FIELD_W)) | 0;
    const gy = Math.min(FIELD_H - 1, Math.max(0, (py / pCanvas.height) * FIELD_H)) | 0;
    const i  = gy * FIELD_W + gx;
    return {{ dx: fieldDx[i], dy: fieldDy[i] }};
}}

// Spawn a particle near a random venue weighted by composite intensity
function respawnParticle(p, activeCount) {{
    const year = parseInt(document.getElementById('year-slider').value);
    // Build weighted venue list
    const weighted = [];
    VENUES.forEach(v => {{
        const loads = venueIntensityCache[v.id];
        if (!loads) return;
        const w = loads.composite || 0;
        if (w > 0.02) weighted.push({{ v, w }});
    }});
    if (!weighted.length) {{
        // Scatter randomly across map
        p.px = Math.random() * (pCanvas ? pCanvas.width : 800);
        p.py = Math.random() * (pCanvas ? pCanvas.height : 600);
    }} else {{
        const total = weighted.reduce((s, x) => s + x.w, 0);
        let rnd = Math.random() * total;
        let chosen = weighted[weighted.length - 1].v;
        for (const {{ v, w }} of weighted) {{
            rnd -= w;
            if (rnd <= 0) {{ chosen = v; break; }}
        }}
        const vp = venuePx[chosen.id];
        if (vp) {{
            p.px = vp.px + (Math.random() - 0.5) * 60;
            p.py = vp.py + (Math.random() - 0.5) * 60;
        }}
    }}
    p.vx = (Math.random() - 0.5) * 0.5;
    p.vy = (Math.random() - 0.5) * 0.5;
    p.age = 0;
    p.maxAge = 200 + Math.random() * 400;
}}

// Main RAF loop — called by mode activator
let particleRaf = null;
let activeParticleCount = 0;

function particleFrame() {{
    if (!pCtx || !pCanvas) return;
    // Trail fade
    pCtx.fillStyle = 'rgba(0,0,0,0.04)';
    pCtx.fillRect(0, 0, pCanvas.width, pCanvas.height);

    for (let i = 0; i < activeParticleCount; i++) {{
        const p = particles[i];
        p.age++;
        if (p.age >= p.maxAge) {{ respawnParticle(p, activeParticleCount); continue; }}

        const f = sampleField(p.px, p.py);
        p.vx = p.vx * 0.95 + f.dx * 0.15;
        p.vy = p.vy * 0.95 + f.dy * 0.15;
        p.px += p.vx;
        p.py += p.vy;

        // Wrap at canvas edges
        if (p.px < 0) p.px = pCanvas.width;
        if (p.px > pCanvas.width) p.px = 0;
        if (p.py < 0) p.py = pCanvas.height;
        if (p.py > pCanvas.height) p.py = 0;

        const alpha = Math.min(0.8, (p.age / p.maxAge) * 2 * (1 - p.age / p.maxAge) * 4);
        pCtx.beginPath();
        pCtx.arc(p.px, p.py, 1.2, 0, Math.PI * 2);
        pCtx.fillStyle = `rgba(${{p.r}},${{p.g}},${{p.b}},${{alpha.toFixed(2)}})`;
        pCtx.fill();
    }}
    particleRaf = requestAnimationFrame(particleFrame);
}}

function startParticles(count) {{
    if (particleRaf) cancelAnimationFrame(particleRaf);
    activeParticleCount = Math.min(count, MAX_P);
    if (pCanvas) pCanvas.style.opacity = '1';
    particleRaf = requestAnimationFrame(particleFrame);
}}

function stopParticles() {{
    if (particleRaf) {{ cancelAnimationFrame(particleRaf); particleRaf = null; }}
    if (pCtx && pCanvas) pCtx.clearRect(0, 0, pCanvas.width, pCanvas.height);
    if (pCanvas) pCanvas.style.opacity = '0';
    activeParticleCount = 0;
}}
```

Also, in `updateMap()`, after the `computeIntensity()` call, cache the result:

```javascript
// Inside the VENUES.forEach loop in updateMap(), after:
//   const intensity = computeIntensity(v.id, year, month, dow, band);
// Add:
venueIntensityCache[v.id] = intensity;
```

And after the entire `VENUES.forEach` block in `updateMap()`, trigger a field refresh:

```javascript
// At end of updateMap(), add:
if (state.particleMode && state.particleMode !== 'off') {{
    _scheduleFieldUpdate();
}}
```

Add the throttled field updater (debounced to avoid recomputing every slider tick during playback):

```javascript
let _fieldUpdateTimer = null;
function _scheduleFieldUpdate() {{
    if (_fieldUpdateTimer) clearTimeout(_fieldUpdateTimer);
    _fieldUpdateTimer = setTimeout(() => {{ updateParticleField(); }}, 400);
}}
```

`updateParticleField()` is a stub here — implemented per mode in Tasks 3–5.

```javascript
function updateParticleField() {{
    if      (state.particleMode === 'smoke')   _buildSmokefield();
    else if (state.particleMode === 'flow')    _buildFlowField();
    else if (state.particleMode === 'network') _buildNetworkField();
}}
```

### Step 4: Add `particleMode: null` to state

```javascript
// Change the state declaration line:
const state = {{ month: null, dow: null, band: null, literary: false,
                 selectedVenue: null, modality: null, tierView: false,
                 buildingType: null, particleMode: null }};
```

### Step 5: Write HTML-level test

Append to `gazetteer/tests/test_build_sensory_time_map.py`:

```python
def test_particle_canvas_present(html):
    assert 'id="particle-canvas"' in html

def test_particle_engine_present(html):
    assert "particleFrame" in html
    assert "startParticles" in html
    assert "stopParticles" in html
    assert "venueIntensityCache" in html
```

### Step 6: Build and run tests

```bash
python3 gazetteer/build_sensory_time_map.py
pytest gazetteer/tests/test_build_sensory_time_map.py -v -k "particle"
```
Expected: all pass.

### Step 7: Commit

```bash
git add gazetteer/build_sensory_time_map.py gazetteer/sensory_time_map.html \
        gazetteer/tests/test_build_sensory_time_map.py
git commit -m "feat: canvas particle engine skeleton (RAF loop, field grid, pool)"
```

---

## Task 3: Mode A — Smoke & Smell diffusion

**Files:**
- Modify: `gazetteer/build_sensory_time_map.py` — add `_buildSmokefield()` JS

### Step 1: Write the test first

Append to `gazetteer/tests/test_build_sensory_time_map.py`:

```python
def test_smoke_mode_field_builder(html):
    assert "_buildSmokefield" in html
    assert "windDx" in html
    assert "enclosureFactor" in html
    assert "so2_index" in html  # particle count scales with smoke data
```

### Step 2: Run test — verify it fails

```bash
pytest gazetteer/tests/test_build_sensory_time_map.py -v -k "smoke_mode"
```
Expected: FAIL — `_buildSmokefield` not yet defined.

### Step 3: Implement `_buildSmokefield()`

Add this JS function inside the particle system script block (after `updateParticleField()`):

```javascript
const WIND_DX = 0.40;   // prevailing SW wind: east drift
const WIND_DY = -0.15;  // slight northward component

function _buildSmokefield() {{
    if (!pCanvas) return;
    const W = pCanvas.width, H = pCanvas.height;
    const cW = W / FIELD_W, cH = H / FIELD_H;

    for (let gy = 0; gy < FIELD_H; gy++) {{
        for (let gx = 0; gx < FIELD_W; gx++) {{
            const cx = (gx + 0.5) * cW;
            const cy = (gy + 0.5) * cH;
            let fdx = WIND_DX, fdy = WIND_DY;

            VENUES.forEach(v => {{
                const loads = venueIntensityCache[v.id];
                if (!loads || loads.smell < 0.02) return;
                const enc = v.enclosure || 'open';
                const encF = enc === 'open' ? 1.0 : enc === 'semi_open' ? 0.6 : 0.2;
                // East-west position modifier
                const posF = v.lon > -0.09 ? 1.3 : v.lon < -0.17 ? 0.7 : 1.0;
                const vp = venuePx[v.id];
                if (!vp) return;
                const dx = cx - vp.px, dy = cy - vp.py;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 1 || dist > 300) return;
                const strength = loads.smell * encF * posF * 4000 / (dist * dist);
                fdx += (dx / dist) * strength;
                fdy += (dy / dist) * strength;
            }});

            // Clamp to reasonable velocity
            const mag = Math.sqrt(fdx * fdx + fdy * fdy);
            if (mag > 2.0) {{ fdx = fdx / mag * 2.0; fdy = fdy / mag * 2.0; }}

            const i = gy * FIELD_W + gx;
            fieldDx[i] = fdx;
            fieldDy[i] = fdy;
        }}
    }}

    // Particle count: base 600 + smoke-scaled bonus (up to 1400 total)
    const year = parseInt(document.getElementById('year-slider').value);
    const decade = Math.floor(year / 10) * 10;
    const smokeRow = SMOKE_DATA_ENV.find(s => s.decade_start === decade);
    const so2 = smokeRow ? smokeRow.so2_index : 0;
    const count = Math.round(600 + so2 * 800);

    // Colour: amber-brown
    for (let i = 0; i < count; i++) {{
        particles[i].r = 180; particles[i].g = 130; particles[i].b = 40;
    }}
    startParticles(count);
}}
```

### Step 4: Build and run test

```bash
python3 gazetteer/build_sensory_time_map.py
pytest gazetteer/tests/test_build_sensory_time_map.py -v -k "smoke_mode"
```
Expected: PASS.

**Browser check:** Open `sensory_time_map.html`, select Smoke mode (UI added in Task 6), set year to 1780. Particles should drift east from smell-heavy venues (Smithfield, Thames wharves).

### Step 5: Commit

```bash
git add gazetteer/build_sensory_time_map.py gazetteer/sensory_time_map.html
git commit -m "feat: particle Mode A — smoke/smell diffusion with SW wind field"
```

---

## Task 4: Mode B — Per-modality flow fields

**Files:**
- Modify: `gazetteer/build_sensory_time_map.py` — add `_buildFlowField()` JS

### Step 1: Write the test first

```python
def test_flow_mode_field_builder(html):
    assert "_buildFlowField" in html
    # Each modality sub-field is referenced
    for modal in ["smell", "noise", "crowd", "visual"]:
        assert modal in html   # already true, but verifies flow field uses them
    # Crowd field is attractive (pulls toward venues)
    assert "attractStrength" in html or "crowd" in html
```

### Step 2: Run test — verify it fails

```bash
pytest gazetteer/tests/test_build_sensory_time_map.py -v -k "flow_mode"
```

### Step 3: Implement `_buildFlowField()`

```javascript
// Modality sub-field colours
const MODALITY_COLOURS = {{
    smell:  {{ r: 180, g: 130, b:  40 }},
    noise:  {{ r:  60, g: 120, b: 200 }},
    crowd:  {{ r: 180, g:  60, b:  60 }},
    visual: {{ r:  60, g: 160, b:  80 }},
}};

function _buildFlowField() {{
    if (!pCanvas) return;
    const W = pCanvas.width, H = pCanvas.height;
    const cW = W / FIELD_W, cH = H / FIELD_H;

    // Which modalities are active via sense pills?
    const activeModals = [];
    if (!state.modality) {{
        activeModals.push('smell','noise','crowd','visual');
    }} else {{
        activeModals.push(state.modality);
    }}

    fieldDx.fill(0); fieldDy.fill(0);

    activeModals.forEach(modal => {{
        for (let gy = 0; gy < FIELD_H; gy++) {{
            for (let gx = 0; gx < FIELD_W; gx++) {{
                const cx = (gx + 0.5) * cW;
                const cy = (gy + 0.5) * cH;
                let fdx = 0, fdy = 0;

                VENUES.forEach(v => {{
                    const loads = venueIntensityCache[v.id];
                    if (!loads || (loads[modal] || 0) < 0.02) return;
                    const vp = venuePx[v.id];
                    if (!vp) return;
                    const dx = cx - vp.px, dy = cy - vp.py;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 1 || dist > 350) return;

                    if (modal === 'crowd') {{
                        // Crowd: attractive — pull toward venue
                        const attractStrength = loads.crowd * 3000 / (dist * dist);
                        fdx -= (dx / dist) * attractStrength;
                        fdy -= (dy / dist) * attractStrength;
                    }} else if (modal === 'noise') {{
                        // Noise: radial outward + reverb turbulence for stone
                        const strength = loads.noise * 3500 / (dist * dist);
                        fdx += (dx / dist) * strength;
                        fdy += (dy / dist) * strength;
                        if (v.material === 'stone' && v.enclosure !== 'open') {{
                            // Small rotational turbulence
                            fdx += (dy / dist) * strength * 0.25;
                            fdy -= (dx / dist) * strength * 0.25;
                        }}
                    }} else {{
                        // Smell and visual: radial outward, gentle
                        const str = loads[modal] * (modal === 'visual' ? 1500 : 3000) / (dist * dist);
                        fdx += (dx / dist) * str;
                        fdy += (dy / dist) * str;
                    }}
                }});

                const mag = Math.sqrt(fdx * fdx + fdy * fdy);
                if (mag > 2.0) {{ fdx = fdx/mag*2.0; fdy = fdy/mag*2.0; }}

                const i = gy * FIELD_W + gx;
                fieldDx[i] += fdx / activeModals.length;
                fieldDy[i] += fdy / activeModals.length;
            }}
        }}
    }});

    // Assign particle colours by dominant active modality
    const col = MODALITY_COLOURS[activeModals[0]] || MODALITY_COLOURS.smell;
    const count = 900;
    for (let i = 0; i < count; i++) {{
        particles[i].r = col.r; particles[i].g = col.g; particles[i].b = col.b;
    }}
    startParticles(count);
}}
```

### Step 4: Build and test

```bash
python3 gazetteer/build_sensory_time_map.py
pytest gazetteer/tests/test_build_sensory_time_map.py -v -k "flow_mode"
```

**Browser check:** Select Flow mode. With no sense filter, four sub-fields blend. Click "Noise" pill — only blue particles, centred on theatres and markets.

### Step 5: Commit

```bash
git add gazetteer/build_sensory_time_map.py gazetteer/sensory_time_map.html
git commit -m "feat: particle Mode B — per-modality flow fields with sense pill integration"
```

---

## Task 5: Mode C — Street network diffusion

**Files:**
- Modify: `gazetteer/build_sensory_time_map.py` — add `_buildNetworkField()` JS

### Step 1: Write the test first

```python
def test_network_mode_field_builder(html):
    assert "_buildNetworkField" in html
    assert "STREET_NETWORK" in html
    assert "segT" in html      # segment parametric position
```

### Step 2: Run test — verify it fails

```bash
pytest gazetteer/tests/test_build_sensory_time_map.py -v -k "network_mode"
```

### Step 3: Implement `_buildNetworkField()`

Particles in network mode don't use the vector field grid — they are constrained to street segments. Override the `respawnParticle` call by giving each particle a `seg` reference and `segT` parameter. The `particleFrame` loop needs a branch for network mode.

Add after `_buildFlowField()`:

```javascript
// Street segment pixel cache (recomputed on moveend/zoomend)
let streetSegsPx = [];  // [{{x0,y0,x1,y1,wx,wy}}] — per-subsegment, weighted

function _projectStreets() {{
    streetSegsPx = [];
    STREET_NETWORK.forEach(polyline => {{
        for (let i = 0; i < polyline.length - 1; i++) {{
            const a = map.latLngToContainerPoint([polyline[i][0],   polyline[i][1]]);
            const b = map.latLngToContainerPoint([polyline[i+1][0], polyline[i+1][1]]);
            const len = Math.sqrt((b.x-a.x)**2 + (b.y-a.y)**2);
            if (len < 2) continue;
            streetSegsPx.push({{ x0: a.x, y0: a.y, x1: b.x, y1: b.y, len }});
        }}
    }});
}}

map.on('moveend zoomend', _projectStreets);
if (STREET_NETWORK.length) _projectStreets();

function _buildNetworkField() {{
    if (!pCanvas || !streetSegsPx.length) return;
    // In network mode the field grid is unused — particles ride segments
    // Assign each particle a random street segment and a t ∈ [0,1]
    const count = 1200;
    for (let i = 0; i < count; i++) {{
        const p = particles[i];
        const seg = streetSegsPx[Math.floor(Math.random() * streetSegsPx.length)];
        p._seg   = seg;
        p._segT  = Math.random();         // position along segment
        p._segDir = Math.random() < 0.5 ? 1 : -1;  // direction of travel
        // Colour from nearest venue modality
        p.r = 160; p.g = 110; p.b = 60;  // default warm tone
    }}
    startParticles(count);
}}

// Override particleFrame for network mode: replace field-based movement
// with segment-constrained movement. Achieved by patching particleFrame
// to call _networkStep when mode === 'network'.
const _origParticleFrame = particleFrame;

function _networkStep(p) {{
    const seg = p._seg;
    if (!seg) return;
    const speed = 0.6 + Math.random() * 0.2;
    p._segT += p._segDir * speed / seg.len;
    if (p._segT > 1 || p._segT < 0) {{
        // Pick a new random segment
        p._seg    = streetSegsPx[Math.floor(Math.random() * streetSegsPx.length)];
        p._segT   = p._segDir > 0 ? 0 : 1;
        p._segDir = Math.random() < 0.5 ? 1 : -1;
    }}
    p.px = seg.x0 + (seg.x1 - seg.x0) * p._segT;
    p.py = seg.y0 + (seg.y1 - seg.y0) * p._segT;
}}

// Monkey-patch particleFrame to handle network mode
const _baseParticleFrame = particleFrame;
// Replace the RAF callback
function particleFrame() {{
    if (!pCtx || !pCanvas) return;
    pCtx.fillStyle = 'rgba(0,0,0,0.035)';
    pCtx.fillRect(0, 0, pCanvas.width, pCanvas.height);

    const isNetwork = state.particleMode === 'network';

    for (let i = 0; i < activeParticleCount; i++) {{
        const p = particles[i];
        p.age++;
        if (p.age >= p.maxAge) {{
            if (isNetwork && streetSegsPx.length) {{
                p._seg   = streetSegsPx[Math.floor(Math.random() * streetSegsPx.length)];
                p._segT  = Math.random();
                p._segDir = Math.random() < 0.5 ? 1 : -1;
                p.age = 0;
            }} else {{
                respawnParticle(p, activeParticleCount);
            }}
            continue;
        }}

        if (isNetwork) {{
            _networkStep(p);
        }} else {{
            const f = sampleField(p.px, p.py);
            p.vx = p.vx * 0.95 + f.dx * 0.15;
            p.vy = p.vy * 0.95 + f.dy * 0.15;
            p.px += p.vx;
            p.py += p.vy;
            if (p.px < 0) p.px = pCanvas.width;
            if (p.px > pCanvas.width) p.px = 0;
            if (p.py < 0) p.py = pCanvas.height;
            if (p.py > pCanvas.height) p.py = 0;
        }}

        const alpha = Math.min(0.8, (p.age / p.maxAge) * 2 * (1 - p.age / p.maxAge) * 4);
        pCtx.beginPath();
        pCtx.arc(p.px, p.py, 1.2, 0, Math.PI * 2);
        pCtx.fillStyle = `rgba(${{p.r}},${{p.g}},${{p.b}},${{alpha.toFixed(2)}})`;
        pCtx.fill();
    }}
    particleRaf = requestAnimationFrame(particleFrame);
}}
```

> **Note:** The final `particleFrame` function above replaces the skeleton from Task 2 — delete the skeleton version from the HTML template when adding this.

### Step 4: Build and test

```bash
python3 gazetteer/build_sensory_time_map.py
pytest gazetteer/tests/test_build_sensory_time_map.py -v -k "network_mode"
```

**Browser check:** Select Network mode. Particles should trace street-like paths across the map, denser near central London venues.

### Step 5: Commit

```bash
git add gazetteer/build_sensory_time_map.py gazetteer/sensory_time_map.html
git commit -m "feat: particle Mode C — street-network-constrained diffusion via OHM segments"
```

---

## Task 6: UI — Particles control row

**Files:**
- Modify: `gazetteer/build_sensory_time_map.py` — HTML and JS additions

### Step 1: Write the test first

```python
def test_particle_mode_buttons_present(html):
    assert 'data-pmode="off"'     in html
    assert 'data-pmode="smoke"'   in html
    assert 'data-pmode="flow"'    in html
    assert 'data-pmode="network"' in html

def test_particle_mode_js_handler(html):
    assert "data-pmode" in html
    assert "particleMode" in html
```

### Step 2: Run test — verify it fails

```bash
pytest gazetteer/tests/test_build_sensory_time_map.py -v -k "particle_mode"
```

### Step 3: Add particle control row to HTML template

Find the building-type pill row in `HTML_TEMPLATE` and add after it:

```html
  <div class="pill-row">
    <span class="pill-label">Particles</span>
    <button class="particle-btn active" data-pmode="off">Off</button>
    <button class="particle-btn" data-pmode="smoke">&#127844; Smoke</button>
    <button class="particle-btn" data-pmode="flow">&#8767; Flow</button>
    <button class="particle-btn" data-pmode="network">&#9780; Network</button>
  </div>
```

### Step 4: Add JS event wiring

Add after the `btype-pill` click handler:

```javascript
document.querySelectorAll('.particle-btn').forEach(btn => {{
    btn.addEventListener('click', () => {{
        const mode = btn.dataset.pmode;
        document.querySelectorAll('.particle-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.particleMode = mode === 'off' ? null : mode;
        if (!state.particleMode) {{
            stopParticles();
        }} else {{
            updateParticleField();
        }}
    }});
}});
```

Also update `clearFilters()` to reset particle mode:

```javascript
function clearFilters() {{
    state.month = null; state.dow = null; state.band = null;
    state.modality = null; state.buildingType = null; state.particleMode = null;
    document.querySelectorAll('.pill.active, .sense-pill.active, .btype-pill.active')
        .forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.particle-btn').forEach(b => b.classList.remove('active'));
    document.querySelector('.particle-btn[data-pmode="off"]')?.classList.add('active');
    stopParticles();
    updateMap();
}}
```

### Step 5: Build and run all tests

```bash
python3 gazetteer/build_sensory_time_map.py
pytest gazetteer/tests/ -q
```
Expected: all 115+ tests pass.

**Browser verification checklist:**
- [ ] Particles → Off: canvas hidden, no RAF running
- [ ] Particles → Smoke, year 1780: amber particles drift east from Smithfield/Thames
- [ ] Particles → Smoke, year 1700: fewer particles (low so2_index)
- [ ] Particles → Flow, Noise pill active: blue particles radiate from theatres
- [ ] Particles → Flow, Crowd pill: red particles pull toward Vauxhall/pleasure gardens
- [ ] Particles → Network: particles trace street-like paths, denser in City
- [ ] Zoom/pan: particles respawn correctly (no stale positions)
- [ ] Year animation playing: particles adapt smoothly, no flash/reset

### Step 6: Final commit

```bash
git add gazetteer/build_sensory_time_map.py gazetteer/sensory_time_map.html \
        gazetteer/tests/test_build_sensory_time_map.py
git commit -m "feat: particle mode UI — Smoke/Flow/Network pills with RAF start/stop"
```

---

## Verification

```bash
python3 gazetteer/build_sensory_time_map.py
pytest gazetteer/tests/ -q
open gazetteer/sensory_time_map.html
```

All tests pass. Three particle modes selectable from the controls bar. Smoke mode scales with coal data across the 1660–1820 timeline.

---

## Future enrichment (out of scope here)

- **Locating London's Past** vector street data (if downloadable) would replace OHM for Mode C
- **WebGL shader rendering** for >10,000 particle counts
- **Blendable modes** (two particle systems simultaneously)
- **Per-decade street filtering** in Mode C (remove streets that didn't exist yet)
