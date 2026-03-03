# Particle Overlay Design — Null-School-Style Sensory Flows
**Date:** 2026-03-03
**Status:** Approved
**Scope:** `gazetteer/build_sensory_time_map.py` only

---

## Context

The sensory time map currently shows venues as circle markers whose radius and colour encode composite sensory intensity. The user wants a "null school"-style animated particle overlay that makes sensory diffusion visible — particles flowing across the map driven by smell, sound, crowd pressure, and the 18th-century street plan.

Null school (earth.nullschool.net) renders thousands of particles tracing continuous paths through a vector field, creating emergent flow patterns that no static map can convey. This design applies the same principle to 18th-century London sensory data.

---

## Decisions Made

| Question | Decision |
|----------|----------|
| Rendering | `<canvas>` overlay — not SVG or Leaflet markers |
| Particle count | High (thousands) — canvas enables this |
| Year transitions | Particles continuously adapt; no reset on year change |
| Mode switching | Single canvas, one mode at a time (Approach A) |
| Street network source | OpenHistoricalMap Overpass API, baked at build time |

---

## Architecture

### Canvas Setup

A single `<canvas>` element is injected into `#map` above `#smoke-overlay` and below Leaflet's tile panes. It fills 100% of the map div. On Leaflet `moveend`/`zoomend` events every particle's pixel position is recalculated from its stored lat/lon. The canvas is `pointer-events: none` so Leaflet interaction is unaffected.

### Particle Loop

`requestAnimationFrame` drives a continuous loop at ~60fps. Each frame:
1. Paint with `rgba(0,0,0,0.04)` rect — creates trailing fade without full clear
2. Advance each particle by its velocity vector; draw a 1–2px circle
3. Age each particle; when it exceeds `maxAge` (random 200–600 frames), respawn near a venue weighted by current intensity

### Field Update

`updateField()` computes a `W × H` grid of `{dx, dy}` vectors in pixel space, stored as a flat `Float32Array`. Particles sample the grid bilinearly. Field recomputes lazily (~2/sec during playback) so particles drift smoothly toward the new state rather than snapping.

---

## The Three Modes

### Mode A — Smoke & Smell Diffusion

Vector field = prevailing SW wind + radial emission from smell venues.

```
field[p] = windVector(dx=+0.4, dy=-0.2) + Σ venues: (smellIntensity × enclosureFactor × falloff(dist))
```

- `enclosureFactor`: open=1.0, semi_open=0.6, enclosed=0.2
- `falloff`: inverse-square, capped at 300px radius
- Colour: amber-brown `rgba(180, 130, 40, α)`
- Particle count scales with current decade's `so2_index` (more particles = denser industrial haze)
- Position modifier: venues east of City of London (lon > −0.09) emit 1.3× particles; west of Hyde Park (lon < −0.17) emit 0.7×

### Mode B — Per-Modality Flow Fields

Four simultaneous colour-coded sub-fields on the same canvas:

| Modality | Colour | Field logic |
|----------|--------|-------------|
| Smell | `rgba(180,130,40,α)` | radial outward, SW wind drift |
| Noise | `rgba(60,120,200,α)` | radial outward; stone enclosure adds turbulence (reverb) |
| Crowd | `rgba(180,60,60,α)` | pulls *toward* high-crowd venues |
| Visual | `rgba(60,160,80,α)` | slow ambient drift, weak field |

Active sense filter pills (existing Smell/Noise/Crowd/Visual buttons) constrain which sub-fields are drawn. All four active gives the full layered effect.

### Mode C — Street Network Diffusion

Particles are constrained to pre-1820 street segments from OpenHistoricalMap, baked into the HTML at build time as `STREET_NETWORK`. Each particle:
1. Picks a venue-pair with flow weight = `1 / OSRM_distance` (shorter = more particles)
2. Walks the straight-line approximation between them, wrapping at endpoints
3. Colour follows dominant modality of the source venue

Street data is fetched in the Python build script, simplified with Douglas-Peucker, and stored as a constant. Streets are filtered to `start_date ≤ 1820` and `end_date ≥ 1660`.

---

## Street Network Data Source

**Primary:** OpenHistoricalMap Overpass API
`https://overpass-api.openhistoricalmap.org/api/interpreter`

Query (executed at build time):
```
way["highway"](51.480,-0.175,51.530,-0.065)["start_date"];
out geom qt;
```

Then filter in Python to ways where `start_date ≤ 1820`. Result: ~2,391 pre-1820 streets covering central London, many dated to Rocque's 1746 survey.

**Future enrichment sources:**
- [Locating London's Past](https://www.locatinglondon.org/) — digitised Rocque 1746 street network
- [Layers of London](https://www.layersoflondon.org/) — georectified historical maps with street GIS
- National Library of Scotland 1848–51 OS maps — post-period but captures most pre-1820 streets

---

## UI

A **Particles** control row added to the controls bar:

```
[ Particles: ◉ Smoke  ◯ Flow  ◯ Network  ◯ Off ]
```

- Radio-style pills: only one mode active at a time, or Off
- Inserted below the building-type filter row
- No additional controls — particle density and speed derive from existing year/intensity state

---

## Implementation Sequence

1. **Fetch and bake street network** — Python build script queries OHM Overpass, simplifies geometry, outputs `STREET_NETWORK` JS constant
2. **Canvas infrastructure** — inject canvas, handle resize/reproject on Leaflet events, wire RAF loop
3. **Mode A: Smoke/Smell** — wind vector field + venue emission, smoke-scaled particle count
4. **Mode B: Per-modality flows** — four sub-fields, sense-filter integration
5. **Mode C: Street network** — venue-pair weighting, segment-constrained particle walk
6. **UI: Particles control row** — pills wired to mode state, Off kills RAF loop

---

## Files Modified

| File | Change |
|------|--------|
| `gazetteer/build_sensory_time_map.py` | All changes — OHM fetch, canvas JS, mode logic, UI |

No other files affected.
