# Contour Surface: IDW Smoke & Sensory Visualisation

**Date**: 2026-03-16
**Status**: Approved (pending implementation)

## Summary

Replace the particle animation system and CSS smoke overlay in
`sensory_time_map.html` with a multi-modal IDW-interpolated contour surface
rendered via a custom Leaflet `L.GridLayer`. The surface displays the same
per-venue sensory intensity data the particles conveyed, but as a smooth
gradient wash with sparse engraving-style isolines — period-appropriate,
analytically precise, and legible.

Motivated by Smil's work on pre-industrial combustion efficiency: the updated
`so2_index` values in `env_smoke` (computed as `coal_tons_k × emission_factor`,
where `emission_factor` captures declining pollution per ton as combustion
technology improved) produce physically grounded smoke burden estimates that the
contour surface renders as a spatial field.

## What gets removed

~800-900 lines of JS comprising the particle system:

- Particle canvas element and RAF loop (`particleRaf`, `_animateParticles`)
- Street direction field (`_rebuildStreetField`, `streetFieldDx/Dy/Mag`,
  `_hwayMult`)
- `MODALITY_PROFILE` definitions and per-modality alpha curves
- `_buildSmokefield`, `_buildFlowfield`, `_buildNetworkfield`
- Particle pill buttons (Off/Atmosphere/Senses/Network) and particle legend div
- `#smoke-overlay` CSS radial gradient div
- `_scheduleFieldUpdate`, `_projectStreets`, `updateParticleField`

**Also removed**: the Network contour mode. Contour surfaces cannot
meaningfully constrain to street geometry, and the network visualisation's
value was primarily aesthetic rather than analytical. Street network data
(`STREET_NETWORK`) is retained in the build output for potential future use.

**Retained**: The heatmap toggle stays. All venue markers,
`computeIntensity()`, `venueIntensityCache`, and the evidence system are
unchanged.

## Data model changes

### New `.smoke` property on `computeIntensity()`

The existing `computeIntensity()` returns `{smell, noise, crowd, visual,
composite}`. A new `.smoke` property will be added, computed as:

```
smokeLoad = so2_index × smokeMult × industrial_intensity
```

Where:
- `so2_index` comes from `SMOKE_DATA_ENV` for the current decade
- `smokeMult` is the existing enclosure modifier (open=1.0, semi_open=0.8,
  enclosed=0.4)
- `industrial_intensity` is obtained by adding an `interpolateZoneProps()`
  call inside `computeIntensity()` using the venue's lat/lon (the same zone
  lookup the particle system uses externally at lines 2620-2628)

This separates smoke from smell (currently smoke is folded into `.smell` via
`smokeBoost`). The existing `.smell` computation keeps its `smokeBoost`
contribution for backward compatibility with the heatmap and venue tooltips.

The existing `composite` property remains a 4-component average
`(smell + noise + crowd + visual) / 4`, excluding `.smoke`, so that existing
heatmap and tooltip behaviour is unaffected.

## Rendering approach

Custom Canvas `L.GridLayer` with IDW interpolation. No external dependencies.

### IDW computation

Each tile canvas computes inverse-distance-weighted interpolation from active
venues. Tile resolution adapts to zoom level:

- **Zoom < 15**: full 256x256 grid (65,536 pixels per tile)
- **Zoom >= 15**: 64x64 grid upscaled to 256x256 with bilinear interpolation
  (4,096 pixels per tile — ~16x faster)

**Input**: `venueIntensityCache[v.id]` — the per-venue intensity object with
`.smell`, `.noise`, `.crowd`, `.visual`, `.smoke`, `.composite` loads.

**Per pixel**: standard IDW with power parameter p = 2:

```
value(x,y) = sum(w_i × intensity_i) / sum(w_i)
where w_i = 1 / max(distance_i, 1)^2
```

**Distance cutoff**: venues beyond 800m (in map units) are excluded from each
pixel's computation. This improves performance and produces more realistic
localised fields — a tannery in Bermondsey should not contribute to readings
at Hyde Park.

**Wind bias** (Atmosphere mode only): the distance calculation applies an
anisotropic scaling factor based on the venue-to-pixel bearing relative to the
prevailing SW wind:

- Pixels east/downwind of a venue: distance scaled by 0.77 (1/1.3), making the
  venue's influence reach further east
- Pixels west/upwind of a venue: distance scaled by 1.43 (1/0.7), reducing
  westward influence

This matches the existing wind modifier (east of LON048 = 1.3×, west of Hyde
Park = 0.7×).

**Canyon effect**: the distance cutoff is modified by enclosure type:

- `enclosed`: cutoff reduced to 400m (pollution concentrated locally)
- `semi_open`: cutoff reduced to 600m
- `open`: full 800m cutoff

**Output**: a 0–1 intensity value per pixel, mapped to the active mode's
colour ramp.

**Tile boundary continuity**: tiles use world coordinates (lat/lng projected to
pixel space) for all distance calculations, ensuring identical IDW values at
shared boundary pixels. No padding or overlap needed.

### Atmosphere mode composition

The Atmosphere contour uses a weighted blend matching the old particle system:

```
atmosphere(x,y) = 0.7 × idw(smoke) + 0.3 × idw(smell)
```

Both components are computed as separate IDW passes, then blended. The smoke
component applies wind bias; the smell component does not (smell sources are
more localised and less wind-dependent).

### Contour isolines

Three isolines at normalised thresholds **0.4, 0.6, 0.8**, traced by scanning
the computed grid for threshold crossings (marching squares). A light Gaussian
blur (sigma=2) is applied to the grid before tracing to smooth jagged edges. At zoom >= 15 (64x64 grid), sigma
scales to 1 to avoid over-blurring at the lower resolution.

| Threshold | Stroke | Width | Style |
|-----------|--------|-------|-------|
| 0.4 | low opacity | 0.8px | dashed (4,3) |
| 0.6 | medium opacity | 1.0px | dashed (4,3) |
| 0.8 | higher opacity | 1.2px | solid |

Colour matches the active modality. Labels show the threshold value (e.g.,
"0.4", "0.6", "0.8") in italic Georgia, placed once per contour segment
(approximately every 200px of contour length), with collision avoidance by
skipping labels that would overlap a previously placed one.

### Colour ramps per modality

| Mode | Gradient (transparent to dense) | Max opacity |
|------|--------------------------------|-------------|
| Atmosphere | cream, ochre, umber, charcoal | 0.40 |
| Smell | cream, warm amber, burnt sienna | 0.40 |
| Noise | pale ice, steel blue, dark navy | 0.40 |
| Crowd | cream, salmon, deep crimson | 0.40 |
| Visual | pale sage, olive, dark forest | 0.40 |
| Smoke | cream, grey-brown, charcoal | 0.40 |

Max opacity capped at 0.40 so the basemap always shows through.

## Modes and UI

### Mode mapping (particle system to contour surface)

| Old particle mode | New contour mode | Notes |
|---|---|---|
| Atmosphere (70% smoke, 30% smell) | **Atmosphere** | 0.7×smoke + 0.3×smell IDW, wind-biased |
| Senses (per-modality fields) | **Senses** | Per-modality surfaces from venueIntensityCache |
| Network (street-constrained) | *Removed* | Cannot constrain contours to street geometry |

### Controls

The particle pill row becomes:

```
Overlay:  Off | Atmosphere | Senses
```

When Senses is active, a sub-selector appears:

```
Sense:  Smoke | Smell | Noise | Crowd | Visual
```

Same `pill-row` styling as existing buttons. The particle legend div is
removed. The heatmap toggle stays on its own row. The smoke gauge in the env
bar stays (shows the decade's SO2 index).

### Pointer events

The contour surface tiles have `pointer-events: none` set via CSS on the
GridLayer container, matching the behaviour of the removed `#smoke-overlay`.
Venue markers remain clickable.

## Decade slider behaviour

Snap with redraw. No crossfade animation — the data is genuinely decadal.

1. `updateMap()` fires, recomputing `venueIntensityCache`
2. `L.GridLayer.redraw()` invalidates all tiles
3. Tiles recompute IDW from new intensity values (~200ms)

## Layer stacking order

Bottom to top:

1. Leaflet tile pane (basemap)
2. **Contour surface** (`L.GridLayer`, z-index 400) — replaces `#smoke-overlay`
   which had z-index 400
3. Time-of-day tint (`#tod-tint`, z-index 401) — renders above the contour
   surface, tinting it with time-of-day colour
4. Heatmap canvas (`#heatmap-canvas`, z-index 403)
5. Night overlay (`#night-overlay`, z-index 404)
6. Venue markers (Leaflet marker pane)

## Scholarly grounding

- SO2 index values incorporate emission factors per decade derived from Smil
  (2017), *Energy and Civilization*, capturing declining pollution per ton as
  combustion technology improved (open hearths at 1.80× in 1660s down to 1.0×
  by 1820). The `emission_factor` column is stored in `env_smoke` alongside
  `so2_index` for scholarly transparency.
- Wind bias (SW prevailing) documented by Evelyn (1661), Grosley (c.1765),
  Le Blanc (c.1740), confirming eastward smoke drift.
- Canyon concentration effect documented by BHO sources: narrow pre-Fire lanes
  with H/W > 2-4 trapped smoke and smell.
- Coal consumption estimates from Brimblecombe (1987) and Cavert (2016).

## Files affected

- `gazetteer/build_sensory_time_map.py` — the `HTML_TEMPLATE` string (bulk of
  changes: remove particle JS, add IDW GridLayer JS, update CSS, update UI
  pills, add `.smoke` to `computeIntensity()`)
- `gazetteer/sensory_time_map.html` — regenerated output
- `gazetteer/tests/test_build_sensory_time_map.py` — update tests: remove
  particle-specific assertions, add contour surface assertions

## Risks and mitigations

- **Performance**: IDW at 100 venues across ~8 tiles is ~200ms at standard
  zoom. At zoom >= 15 (20-30 tiles), the grid downsamples to 64x64 and
  upscales, keeping total computation under 300ms.
- **Contour tracing quality**: Simple marching squares on the grid may produce
  jagged isolines. Mitigation: Gaussian blur (sigma=2) applied to the grid
  before tracing.
- **Removing particles is irreversible in UX terms**: The particle system code
  is in git history if ever needed again. The contour surface preserves all the
  same information (per-modality intensities, wind bias, canyon effects) in a
  more legible presentation.
