# Contour Surface Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the particle animation system and CSS smoke overlay with an IDW-interpolated contour surface (smooth gradient wash + sparse isolines).

**Architecture:** Custom Leaflet `L.GridLayer` renders IDW-interpolated sensory intensity per 256x256 tile. Three contour isolines (0.4, 0.6, 0.8) are traced via marching squares and drawn as engraving-style lines. Two modes: Atmosphere (0.7×smoke + 0.3×smell, wind-biased) and Senses (per-modality with sub-selector). Replaces ~900 lines of particle system JS.

**Tech Stack:** Leaflet.js (already loaded), vanilla JS Canvas API, no new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-16-contour-surface-design.md`

---

## Chunk 1: Remove particle system and smoke overlay

### Task 1: Add `.smoke` to `computeIntensity()` and write tests

**Files:**
- Modify: `gazetteer/build_sensory_time_map.py:1205-1288`
- Modify: `gazetteer/tests/test_build_sensory_time_map.py`

- [ ] **Step 1: Write failing test for `.smoke` property**

Add to `gazetteer/tests/test_build_sensory_time_map.py`:

```python
def test_compute_intensity_returns_smoke(html):
    """computeIntensity must return a .smoke property derived from SO2 index."""
    assert "loads.smoke" in html
    assert "smokeLoad" in html or "so2_index" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest gazetteer/tests/test_build_sensory_time_map.py::test_compute_intensity_returns_smoke -v`
Expected: FAIL

- [ ] **Step 3: Add `.smoke` to `computeIntensity()` in the HTML_TEMPLATE**

In `gazetteer/build_sensory_time_map.py`, inside `computeIntensity()` (around line 1272-1288), add before the `loads.composite` line:

```javascript
    // Smoke load: SO2 index × enclosure modifier × zone industrial intensity
    const decade = Math.floor(year / 10) * 10;
    const smokeRow = SMOKE_DATA_ENV.find(s => s.decade_start === decade)
                  || SMOKE_DATA_ENV[SMOKE_DATA_ENV.length - 1];
    const so2 = smokeRow ? smokeRow.so2_index : 0;
    const zoneRaw = _venue ? getZoneForPoint(_venue.lat, _venue.lon) : null;
    const zoneInterp = zoneRaw ? interpolateZoneProps(zoneRaw, year) : null;
    const indIntensity = zoneInterp ? (zoneInterp.industrial_intensity || 0.3) : 0.3;
    loads.smoke = Math.min(1, so2 * smokeMult * indIntensity);
```

Keep `composite` as the existing 4-component average (smell, noise, crowd, visual) — smoke is excluded to preserve heatmap/tooltip backward compatibility.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest gazetteer/tests/test_build_sensory_time_map.py::test_compute_intensity_returns_smoke -v`
Expected: PASS

- [ ] **Step 5: Run full test suite to check nothing broke**

Run: `pytest gazetteer/tests/test_build_sensory_time_map.py -v`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add gazetteer/build_sensory_time_map.py gazetteer/tests/test_build_sensory_time_map.py
git commit -m "feat: add .smoke property to computeIntensity()"
```

### Task 2: Remove particle system JS, CSS, and HTML

**Files:**
- Modify: `gazetteer/build_sensory_time_map.py`

This is the largest single change. Remove these blocks from `HTML_TEMPLATE`:

- [ ] **Step 1: Remove particle CSS** (lines ~416-420, ~474-475)

Remove `.particle-btn` CSS rules and the night-mode `.particle-btn` rule.

- [ ] **Step 2: Remove `#smoke-overlay` CSS** (lines ~453-459)

Remove the `#smoke-overlay` CSS block. Keep `#map {{ flex: 1; position: relative; }}` — it's on the same line but belongs to the map, not the overlay.

- [ ] **Step 3: Remove particle HTML elements** (lines ~583-600, ~619, ~624)

Remove:
- The particle pill-row div (buttons: Off/Atmosphere/Senses/Network)
- The `#particle-legend` div
- The `#smoke-overlay` div
- The `<canvas id="particle-canvas">` element

- [ ] **Step 4: Remove `smokeOverlay` JS and smoke overlay opacity logic**

Remove:
- `const smokeOverlay = document.getElementById('smoke-overlay');` (line ~875)
- The smoke overlay opacity lines in `updateEnvIndicators()` (lines ~919-922)

- [ ] **Step 5: Remove particle button event listeners** (lines ~1753-1767)

Remove the `document.querySelectorAll('.particle-btn')` click handler block.

- [ ] **Step 6: Remove `_scheduleFieldUpdate()` call in `updateMap()`** (line ~1532)

Remove the call but keep surrounding `updateMap()` logic intact.

- [ ] **Step 7: Remove the entire particle system block** (lines ~1866-3045)

This is the bulk removal — everything from `const pCanvas` through `_scheduleFieldUpdate`. Approximately 1180 lines. Includes:
- Canvas setup, resize handler
- `MAX_P`, particles array, `MODALITY_PROFILE`
- `spawnParticle`, `resetParticles`, field arrays
- `updateVenuePx`, pan/zoom handlers for particles
- `sampleField`, `sampleCanyon`
- `respawnParticle`
- All per-modality renderers (`_updateNoiseRings`, `_drawNoiseRings`, `_drawSmellHalos`, `_rebuildCrowdStreetState`, `_drawCrowdDensity`, `_drawVisualGlow`)
- `particleFrame` (main animation loop)
- `startParticles`, `stopParticles`
- `updateParticleField`, `_buildSmokefield`, `_buildFlowField`
- `streetSegsPx`, `_hwayMult`, `_rebuildStreetField`, `_applyVenueCanyonAnchors`
- `_networkStep`, `_buildNetworkField`
- `_fieldUpdateTimer`, `_scheduleFieldUpdate`

- [ ] **Step 8: Remove `particleMode` from state object and clean up pan/zoom handlers**

Find the `state` object declaration and remove `particleMode` property. Search for all references to `state.particleMode` and remove them.

Also search for `movestart`, `zoomstart`, `moveend`, `zoomend` event handlers that reference particle functions (`_projectStreets`, `updateParticleField`, `resetParticles`, `pCtx.clearRect`). Remove the particle-specific logic from these handlers but keep any non-particle map logic in them.

**Note**: `STREET_NETWORK` data and `STREET_NETWORK_JSON` template variable are **retained** in the build output — do not remove them.

- [ ] **Step 9: Verify all smoke-overlay references are gone**

Search the HTML_TEMPLATE for any remaining references to `smoke-overlay`, `smokeOverlay`, or the smoke overlay opacity setting. The smoke *gauge* (bar + percentage in env bar) should remain — it reads `SMOKE_DATA_ENV` independently and is unrelated to the overlay.

- [ ] **Step 10: Rebuild and verify**

Run: `python3 gazetteer/build_sensory_time_map.py`
Expected: builds without error, HTML output is valid

- [ ] **Step 11: Commit**

```bash
git add gazetteer/build_sensory_time_map.py gazetteer/sensory_time_map.html
git commit -m "refactor: remove particle system and smoke overlay"
```

**Note**: `sensory_time_map.html` is a checked-in generated file — include it in commits whenever the build script changes.

### Task 3: Update tests — remove particle assertions, verify clean removal

**Files:**
- Modify: `gazetteer/tests/test_build_sensory_time_map.py`

- [ ] **Step 1: Remove particle-specific test functions**

Remove these test functions:
- `test_smoke_overlay_present` (~line 144)
- `test_particle_canvas_present` (~line 177)
- `test_particle_engine_present` (~line 180)
- `test_particle_mode_buttons_present` (~line 225)
- `test_particle_mode_labels_renamed` (~line 232)
- `test_particle_legend_present` (~line 238)
- `test_particle_mode_js_handler` (~line 242)
- Test at ~line 268 (MODALITY_PROFILE constant)
- Test at ~line 275 (per-modality alpha curves)
- `test_particle_trail_fade` (~line 281)
- Test at ~line 288 (smoke particle rendering)
- `test_zone_aware_particles` (~line 356)
- Test at ~line 384 (_buildSmokefield canyon)
- Test at ~line 390 (_buildFlowField canyon)
- Test at ~line 433 (per-modality renderer calls)

Also remove/update any tests that reference `_buildSmokefield`, `_buildFlowField`, `_buildNetworkField`, `particleFrame`, `MODALITY_PROFILE`, `pmode`, `particle-canvas`, `smoke-overlay`, or `particle-legend`.

- [ ] **Step 2: Run test suite**

Run: `pytest gazetteer/tests/test_build_sensory_time_map.py -v`
Expected: all remaining tests pass

- [ ] **Step 3: Rebuild HTML and verify it loads**

Run: `python3 gazetteer/build_sensory_time_map.py`
Open `gazetteer/sensory_time_map.html` in browser — map should load, venues should appear, heatmap toggle should work. No JS console errors.

- [ ] **Step 4: Commit**

```bash
git add gazetteer/tests/test_build_sensory_time_map.py
git commit -m "test: remove particle system assertions"
```

---

## Chunk 2: Build the IDW contour surface

### Task 4: Add contour surface UI controls

**Files:**
- Modify: `gazetteer/build_sensory_time_map.py` (HTML_TEMPLATE — CSS and HTML sections)
- Modify: `gazetteer/tests/test_build_sensory_time_map.py`

- [ ] **Step 1: Write failing tests for new UI**

Add to `gazetteer/tests/test_build_sensory_time_map.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest gazetteer/tests/test_build_sensory_time_map.py::test_contour_overlay_buttons -v`
Expected: FAIL

- [ ] **Step 3: Add overlay pill CSS**

In the CSS section of `HTML_TEMPLATE`, add styles for `.contour-btn` (same pattern as the existing `.overlay-btn` or the removed `.particle-btn`):

```css
.contour-btn {{ background: #f4f1eb; border: 1px solid #d8d4cc; color: #5c5850; padding: 2px 8px;
               cursor: pointer; border-radius: 3px; font-size: 0.69em; font-weight: 500; opacity: 0.75; }}
.contour-btn:hover {{ opacity: 1; }}
.contour-btn.active {{ background: #1e3c6e; border-color: #1e3c6e; color: #fff;
                      opacity: 1; font-weight: 600; }}
.sense-btn {{ background: #f4f1eb; border: 1px solid #d8d4cc; color: #5c5850; padding: 2px 8px;
             cursor: pointer; border-radius: 3px; font-size: 0.69em; font-weight: 500; opacity: 0.75; }}
.sense-btn:hover {{ opacity: 1; }}
.sense-btn.active {{ background: #3a5a2a; border-color: #3a5a2a; color: #fff;
                    opacity: 1; font-weight: 600; }}
#sense-row {{ display: none; }}
```

- [ ] **Step 4: Add overlay pill HTML**

Where the particle pill-row was (before the Layer pill-row), add:

```html
<div class="pill-row">
  <span class="pill-label">Overlay</span>
  <button class="contour-btn active" data-cmode="off">Off</button>
  <button class="contour-btn" data-cmode="atmosphere">Atmosphere</button>
  <button class="contour-btn" data-cmode="senses">Senses</button>
</div>
<div class="pill-row" id="sense-row">
  <span class="pill-label">Sense</span>
  <button class="sense-btn active" data-sense="smoke">Smoke</button>
  <button class="sense-btn" data-sense="smell">Smell</button>
  <button class="sense-btn" data-sense="noise">Noise</button>
  <button class="sense-btn" data-sense="crowd">Crowd</button>
  <button class="sense-btn" data-sense="visual">Visual</button>
</div>
```

- [ ] **Step 5: Add `contourMode` and `contourSense` to JS state object**

In the `state` object declaration, add:

```javascript
contourMode: 'off',      // 'off' | 'atmosphere' | 'senses'
contourSense: 'smoke',   // 'smoke' | 'smell' | 'noise' | 'crowd' | 'visual'
```

- [ ] **Step 6: Add button event listeners**

```javascript
document.querySelectorAll('.contour-btn').forEach(btn => {{
    btn.addEventListener('click', () => {{
        document.querySelectorAll('.contour-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.contourMode = btn.dataset.cmode;
        document.getElementById('sense-row').style.display =
            state.contourMode === 'senses' ? 'flex' : 'none';
        if (contourLayer) {{
            if (state.contourMode === 'off') map.removeLayer(contourLayer);
            else {{ if (!map.hasLayer(contourLayer)) map.addLayer(contourLayer); contourLayer.redraw(); }}
        }}
    }});
}});

document.querySelectorAll('.sense-btn').forEach(btn => {{
    btn.addEventListener('click', () => {{
        document.querySelectorAll('.sense-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.contourSense = btn.dataset.sense;
        if (contourLayer && state.contourMode === 'senses') contourLayer.redraw();
    }});
}});
```

- [ ] **Step 7: Run tests**

Run: `pytest gazetteer/tests/test_build_sensory_time_map.py -v`
Expected: all pass including new tests

- [ ] **Step 8: Commit**

```bash
git add gazetteer/build_sensory_time_map.py gazetteer/tests/test_build_sensory_time_map.py
git commit -m "feat: add contour overlay UI controls"
```

### Task 5: Implement IDW GridLayer with colour ramps

**Files:**
- Modify: `gazetteer/build_sensory_time_map.py` (HTML_TEMPLATE — JS section)
- Modify: `gazetteer/tests/test_build_sensory_time_map.py`

- [ ] **Step 1: Write failing tests**

```python
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
    assert "windBias" in html or "anisotropic" in html or "1.3" in html


def test_contour_canyon_cutoff(html):
    """Canyon effect must modify IDW cutoff by enclosure type."""
    assert "enclosed" in html
    # 400m cutoff for enclosed venues
    assert "400" in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest gazetteer/tests/test_build_sensory_time_map.py::test_contour_grid_layer_present -v`
Expected: FAIL

- [ ] **Step 3: Add colour ramp definitions**

Add to JS section of HTML_TEMPLATE:

```javascript
// ── Contour surface colour ramps ──
const CONTOUR_RAMPS = {{
    atmosphere: [[245,235,220],[200,170,100],[140,95,40],[50,30,10]],
    smell:      [[245,235,220],[210,170,90],[180,110,50],[140,70,30]],
    noise:      [[230,240,250],[150,180,210],[80,120,170],[20,40,80]],
    crowd:      [[245,235,220],[220,160,140],[180,80,60],[130,20,20]],
    visual:     [[230,240,225],[160,185,140],[100,130,80],[40,70,25]],
    smoke:      [[245,235,220],[180,170,155],[120,105,85],[50,40,30]],
}};

function sampleRamp(ramp, t) {{
    // t in [0,1] → interpolated [r,g,b] from ramp (array of 4 RGB triples)
    t = Math.max(0, Math.min(1, t));
    const n = ramp.length - 1;
    const i = Math.min(Math.floor(t * n), n - 1);
    const f = t * n - i;
    return [
        Math.round(ramp[i][0] + f * (ramp[i+1][0] - ramp[i][0])),
        Math.round(ramp[i][1] + f * (ramp[i+1][1] - ramp[i][1])),
        Math.round(ramp[i][2] + f * (ramp[i+1][2] - ramp[i][2])),
    ];
}}
```

- [ ] **Step 4: Implement the IDW GridLayer**

Add the core `L.GridLayer` implementation. This is the main rendering engine:

```javascript
// ── IDW Contour GridLayer ──
let contourLayer = null;

const ContourSurface = L.GridLayer.extend({{
    createTile: function(coords) {{
        const tile = document.createElement('canvas');
        const tileSize = this.getTileSize();
        tile.width = tileSize.x;
        tile.height = tileSize.y;
        const ctx = tile.getContext('2d');

        // Determine grid resolution based on zoom
        const zoom = coords.z;
        const gridW = zoom >= 15 ? 64 : 256;
        const gridH = zoom >= 15 ? 64 : 256;
        const cellW = tileSize.x / gridW;
        const cellH = tileSize.y / gridH;

        // Get active ramp and modality
        const mode = state.contourMode;
        if (mode === 'off') return tile;
        const rampKey = mode === 'atmosphere' ? 'atmosphere' : state.contourSense;
        const ramp = CONTOUR_RAMPS[rampKey] || CONTOUR_RAMPS.atmosphere;

        // IDW helper: compute one pass for a given modality
        const _idwPass = (gridOut, modality, applyWindBias) => {{
            for (let gy = 0; gy < gridH; gy++) {{
                for (let gx = 0; gx < gridW; gx++) {{
                    const px = coords.x * tileSize.x + (gx + 0.5) * cellW;
                    const py = coords.y * tileSize.y + (gy + 0.5) * cellH;
                    const latlng = map.unproject([px, py], zoom);

                    let wSum = 0, vSum = 0;

                    VENUES.forEach(v => {{
                        const cache = venueIntensityCache[v.id];
                        if (!cache) return;
                        const intensity = cache[modality] || 0;
                        if (intensity <= 0.01) return;

                        let dist = latlng.distanceTo(L.latLng(v.lat, v.lon));

                        // Wind bias: pixel east of venue → dist scaled down (reaches further)
                        if (applyWindBias) {{
                            if (v.lon < latlng.lng) dist *= 0.77;
                            else if (v.lon > latlng.lng) dist *= 1.43;
                        }}

                        const enc = v.enclosure || 'open';
                        const cutoff = enc === 'enclosed' ? 400
                                     : enc === 'semi_open' ? 600 : 800;
                        if (dist > cutoff) return;

                        const w = 1.0 / Math.max(dist, 1) ** 2;
                        wSum += w;
                        vSum += w * intensity;
                    }});

                    gridOut[gy * gridW + gx] = wSum > 0 ? vSum / wSum : 0;
                }}
            }}
        }};

        // Build IDW grid
        const grid = new Float32Array(gridW * gridH);
        const maxAlpha = 0.40;

        if (mode === 'atmosphere') {{
            // Two separate IDW passes: smoke (with wind bias) + smell (without)
            const smokeGrid = new Float32Array(gridW * gridH);
            const smellGrid = new Float32Array(gridW * gridH);
            _idwPass(smokeGrid, 'smoke', true);
            _idwPass(smellGrid, 'smell', false);
            for (let i = 0; i < grid.length; i++) {{
                grid[i] = 0.7 * smokeGrid[i] + 0.3 * smellGrid[i];
            }}
        }} else {{
            // Senses mode: single pass for chosen modality, no wind bias
            _idwPass(grid, state.contourSense, false);
        }}

        // Render gradient wash with bilinear interpolation
        const imgData = ctx.createImageData(tileSize.x, tileSize.y);
        for (let py2 = 0; py2 < tileSize.y; py2++) {{
            for (let px2 = 0; px2 < tileSize.x; px2++) {{
                // Map pixel back to grid coords (with 0.5 offset for cell centres)
                const gxf = (px2 / tileSize.x) * gridW - 0.5;
                const gyf = (py2 / tileSize.y) * gridH - 0.5;
                const gx0 = Math.max(0, Math.floor(gxf));
                const gy0 = Math.max(0, Math.floor(gyf));
                const gx1 = Math.min(gridW - 1, gx0 + 1);
                const gy1 = Math.min(gridH - 1, gy0 + 1);
                const fx = gxf - gx0, fy = gyf - gy0;

                // Bilinear interpolation of grid values
                const val = (1-fx)*(1-fy) * grid[gy0*gridW+gx0]
                          +    fx *(1-fy) * grid[gy0*gridW+gx1]
                          + (1-fx)*   fy  * grid[gy1*gridW+gx0]
                          +    fx *   fy  * grid[gy1*gridW+gx1];

                const rgb = sampleRamp(ramp, val);
                const alpha = Math.round(val * maxAlpha * 255);
                const idx = (py2 * tileSize.x + px2) * 4;
                imgData.data[idx]     = rgb[0];
                imgData.data[idx + 1] = rgb[1];
                imgData.data[idx + 2] = rgb[2];
                imgData.data[idx + 3] = alpha;
            }}
        }}
        ctx.putImageData(imgData, 0, 0);

        // Draw contour isolines
        this._drawContours(ctx, grid, gridW, gridH, cellW, cellH, ramp);

        return tile;
    }},

    _drawContours: function(ctx, grid, gridW, gridH, cellW, cellH, ramp) {{
        const thresholds = [
            {{ val: 0.4, dash: [4, 3], width: 0.8, alpha: 0.45 }},
            {{ val: 0.6, dash: [4, 3], width: 1.0, alpha: 0.55 }},
            {{ val: 0.8, dash: [],     width: 1.2, alpha: 0.70 }},
        ];

        thresholds.forEach(th => {{
            const rgb = sampleRamp(ramp, th.val);
            ctx.strokeStyle = `rgba(${{rgb[0]}},${{rgb[1]}},${{rgb[2]}},${{th.alpha}})`;
            ctx.lineWidth = th.width;
            ctx.setLineDash(th.dash);
            ctx.beginPath();

            // Simple marching squares
            for (let gy = 0; gy < gridH - 1; gy++) {{
                for (let gx = 0; gx < gridW - 1; gx++) {{
                    const tl = grid[gy * gridW + gx];
                    const tr = grid[gy * gridW + gx + 1];
                    const bl = grid[(gy+1) * gridW + gx];
                    const br = grid[(gy+1) * gridW + gx + 1];

                    const config = (tl >= th.val ? 8 : 0)
                                 | (tr >= th.val ? 4 : 0)
                                 | (br >= th.val ? 2 : 0)
                                 | (bl >= th.val ? 1 : 0);

                    if (config === 0 || config === 15) continue;

                    // Interpolation helpers
                    const lerp = (a, b) => a === b ? 0.5 : (th.val - a) / (b - a);
                    const cx0 = gx * cellW, cy0 = gy * cellH;
                    const cx1 = (gx+1) * cellW, cy1 = (gy+1) * cellH;

                    const top    = [cx0 + lerp(tl, tr) * cellW, cy0];
                    const right  = [cx1, cy0 + lerp(tr, br) * cellH];
                    const bottom = [cx0 + lerp(bl, br) * cellW, cy1];
                    const left   = [cx0, cy0 + lerp(tl, bl) * cellH];

                    const segments = [];
                    // 16-case marching squares lookup
                    switch (config) {{
                        case 1: case 14: segments.push([left, bottom]); break;
                        case 2: case 13: segments.push([bottom, right]); break;
                        case 3: case 12: segments.push([left, right]); break;
                        case 4: case 11: segments.push([top, right]); break;
                        case 5: segments.push([left, top], [bottom, right]); break;
                        case 6: case 9:  segments.push([top, bottom]); break;
                        case 7: case 8:  segments.push([left, top]); break;
                        case 10: segments.push([left, bottom], [top, right]); break;
                    }}

                    segments.forEach(([a, b]) => {{
                        ctx.moveTo(a[0], a[1]);
                        ctx.lineTo(b[0], b[1]);
                    }});
                }}
            }}

            ctx.stroke();
        }});

        ctx.setLineDash([]);

        // Contour labels
        this._drawContourLabels(ctx, grid, gridW, gridH, cellW, cellH, ramp);
    }},

    _drawContourLabels: function(ctx, grid, gridW, gridH, cellW, cellH, ramp) {{
        const thresholds = [0.4, 0.6, 0.8];
        ctx.font = 'italic 9px Georgia';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        // Shared across all thresholds to avoid cross-threshold overlap
        const placedLabels = [];
        const minDist2 = 200 * 200;

        thresholds.forEach(th => {{
            const rgb = sampleRamp(ramp, th);
            ctx.fillStyle = `rgba(${{rgb[0]}},${{rgb[1]}},${{rgb[2]}},0.7)`;

            for (let gy = 0; gy < gridH - 1; gy += 4) {{
                for (let gx = 0; gx < gridW - 1; gx += 4) {{
                    const v = grid[gy * gridW + gx];
                    const vr = grid[gy * gridW + gx + 1];
                    if ((v < th) !== (vr < th)) {{
                        const x = (gx + 0.5) * cellW;
                        const y = (gy + 0.5) * cellH;
                        const tooClose = placedLabels.some(([lx, ly]) => {{
                            const dx = x - lx, dy = y - ly;
                            return dx * dx + dy * dy < minDist2;
                        }});
                        if (!tooClose) {{
                            ctx.fillText(th.toFixed(1), x, y);
                            placedLabels.push([x, y]);
                        }}
                    }}
                }}
            }}
        }});
    }},
}});
```

- [ ] **Step 5: Instantiate the layer and hook into `updateMap()`**

After the GridLayer definition, add:

```javascript
contourLayer = new ContourSurface({{ opacity: 1, zIndex: 400 }});
// Don't add to map initially (mode starts as 'off')
```

In `updateMap()`, after `venueIntensityCache` is populated, add:

```javascript
if (state.contourMode !== 'off' && contourLayer && map.hasLayer(contourLayer)) {{
    contourLayer.redraw();
}}
```

- [ ] **Step 6: Run tests**

Run: `pytest gazetteer/tests/test_build_sensory_time_map.py -v`
Expected: all pass

- [ ] **Step 7: Rebuild and visual check**

Run: `python3 gazetteer/build_sensory_time_map.py`
Open in browser. Click "Atmosphere" button — should see sepia gradient wash with contour lines. Click "Senses" → "Noise" — should see blue gradient. Click "Off" — surface disappears.

- [ ] **Step 8: Commit**

```bash
git add gazetteer/build_sensory_time_map.py gazetteer/tests/test_build_sensory_time_map.py
git commit -m "feat: IDW contour surface with colour ramps and isolines"
```

### Task 6: Add Gaussian blur for contour smoothing

**Files:**
- Modify: `gazetteer/build_sensory_time_map.py` (HTML_TEMPLATE — JS)

- [ ] **Step 1: Add blur function before `_drawContours`**

```javascript
_blurGrid: function(grid, w, h, sigma) {{
    // 1D Gaussian blur, applied horizontally then vertically
    const kernel = [];
    const radius = Math.ceil(sigma * 2);
    let kSum = 0;
    for (let i = -radius; i <= radius; i++) {{
        const g = Math.exp(-(i * i) / (2 * sigma * sigma));
        kernel.push(g);
        kSum += g;
    }}
    kernel.forEach((_, i, a) => a[i] /= kSum);

    const tmp = new Float32Array(w * h);
    // Horizontal pass
    for (let y = 0; y < h; y++) {{
        for (let x = 0; x < w; x++) {{
            let sum = 0;
            for (let k = 0; k < kernel.length; k++) {{
                const sx = Math.min(w - 1, Math.max(0, x + k - radius));
                sum += grid[y * w + sx] * kernel[k];
            }}
            tmp[y * w + x] = sum;
        }}
    }}
    // Vertical pass
    for (let y = 0; y < h; y++) {{
        for (let x = 0; x < w; x++) {{
            let sum = 0;
            for (let k = 0; k < kernel.length; k++) {{
                const sy = Math.min(h - 1, Math.max(0, y + k - radius));
                sum += tmp[sy * w + x] * kernel[k];
            }}
            grid[y * w + x] = sum;
        }}
    }}
}},
```

- [ ] **Step 2: Call blur before drawing contours in `createTile`**

In `createTile`, after the IDW grid loop and before the gradient wash rendering, add:

```javascript
// Smooth grid before contour tracing
const sigma = zoom >= 15 ? 1 : 2;
this._blurGrid(grid, gridW, gridH, sigma);
```

- [ ] **Step 3: Rebuild and verify smoother contours**

Run: `python3 gazetteer/build_sensory_time_map.py`
Check in browser — contour lines should be smoother, less jagged.

- [ ] **Step 4: Commit**

```bash
git add gazetteer/build_sensory_time_map.py
git commit -m "feat: Gaussian blur for smoother contour isolines"
```

---

## Chunk 3: Integration and final polish

### Task 7: Night mode and pointer-events integration

**Files:**
- Modify: `gazetteer/build_sensory_time_map.py` (HTML_TEMPLATE)

- [ ] **Step 1: Add pointer-events: none to contour layer CSS**

Add to CSS:

```css
.contour-surface {{ pointer-events: none; }}
```

And set the className on the GridLayer instantiation:

```javascript
contourLayer = new ContourSurface({{ opacity: 1, zIndex: 400, className: 'contour-surface' }});
```

- [ ] **Step 2: Add night-mode styling for contour buttons**

Find the existing night-mode CSS block and add:

```css
#controls.night-ctrl .contour-btn {{ background: #1a2a3a; border-color: #333; color: #6080a0; }}
#controls.night-ctrl .contour-btn.active {{ background: #1e3c6e; color: #fff; }}
#controls.night-ctrl .sense-btn {{ background: #1a2a3a; border-color: #333; color: #6080a0; }}
#controls.night-ctrl .sense-btn.active {{ background: #3a5a2a; color: #fff; }}
```

- [ ] **Step 3: Rebuild and test night mode**

Run: `python3 gazetteer/build_sensory_time_map.py`
Check in browser — toggle night mode, verify contour surface works correctly and buttons are styled.

- [ ] **Step 4: Commit**

```bash
git add gazetteer/build_sensory_time_map.py
git commit -m "fix: contour surface pointer-events and night mode styling"
```

### Task 8: Final test suite update and validation

**Files:**
- Modify: `gazetteer/tests/test_build_sensory_time_map.py`

- [ ] **Step 1: Add comprehensive contour surface tests**

```python
def test_contour_marching_squares(html):
    """Contour drawing must use marching squares algorithm."""
    assert "marching" in html.lower() or "config" in html


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
    assert "0.40" in html or "maxAlpha" in html


def test_contour_mode_state(html):
    """State object must have contourMode and contourSense."""
    assert "contourMode" in html
    assert "contourSense" in html
```

- [ ] **Step 2: Run full test suite**

Run: `pytest gazetteer/tests/test_build_sensory_time_map.py -v`
Expected: all pass

- [ ] **Step 3: Run all gazetteer tests**

Run: `pytest gazetteer/tests/ -v`
Expected: all pass

- [ ] **Step 4: Final rebuild and browser validation**

Run: `python3 gazetteer/build_sensory_time_map.py`

Browser checklist:
- Map loads without JS errors
- Venue markers appear and are clickable
- Decade slider changes environmental data
- Smoke gauge updates per decade
- "Atmosphere" mode: sepia wash appears, heavier to the east, contour lines visible
- "Senses" mode: sub-selector appears; each sense shows its own colour ramp
- "Off" mode: surface disappears
- Heatmap toggle still works independently
- Night mode: buttons and surface work correctly
- Time-of-day tint renders above the contour surface

- [ ] **Step 5: Commit**

```bash
git add gazetteer/tests/test_build_sensory_time_map.py
git commit -m "test: add comprehensive contour surface test suite"
```
