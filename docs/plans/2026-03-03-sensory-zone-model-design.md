# Sensory Zone Model — Design
**Date:** 2026-03-03
**Status:** Approved
**Scope:** `gazetteer/build_sensory_time_map.py` + new `gazetteer/zones.json`

---

## Context

The sensory time map currently computes experience only at venue point-sources. Standing 80 metres from Billingsgate on a hot July morning yields no smell unless you are directly on a venue marker. The user wants a model that answers "if I were standing on this street at this time of year, what would I experience?" — combining zone-level ambient character, environmental physics, and venue point-sources, with provenance surfaced in the UI.

---

## Decisions Made

| Question | Decision |
|----------|----------|
| Opened/closed venue dates | Fix first, independently, before zone work |
| Zone granularity | ~20 hand-defined GeoJSON polygons, editorial not cartographic |
| Inference approach | Three-layer: zone ambient + environmental modifiers + venue point-sources |
| Visual rendering | Zone fills (5–8% opacity wash) + zone-aware particle density |
| Provenance display | Tooltip provenance line + panel breakdown on demand |

---

## Task 0: Venue Opened/Closed Dates (quick win first)

`venues.csv` has `opened` and `closed` fields that are never passed to JavaScript. Venues should be hidden before their opening date and after their closing date.

**Change:** In `load_data()` in `build_sensory_time_map.py`, add `"opened"` and `"closed"` to the venue dict. In `updateMap()` JS, skip venues where `year < v.opened || year > v.closed` (treating null as no constraint).

---

## Zone Layer

### Geographic Definition

~20 named zones stored in `gazetteer/zones.json` as a GeoJSON FeatureCollection. Each Feature is a polygon covering a district of central London (1660–1820 extent), with properties:

```json
{
  "name": "Smithfield & Newgate",
  "decades": {
    "1660": { "smell_base": 0.7, "noise_base": 0.5, "crowd_density": 0.6,
               "river_proximity": 0.1, "industrial_intensity": 0.4,
               "street_character": "narrow", "building_height": "medium" },
    "1700": { ... },
    ...
    "1820": { ... }
  }
}
```

**Proposed zones (subject to editorial refinement):**
1. City of London (dense commercial core)
2. Thames Waterfront East (Billingsgate, docks, wharves)
3. Thames Waterfront West (Westminster Bridge, Lambeth)
4. Southwark / Bankside
5. Covent Garden & Strand
6. Westminster & Whitehall
7. St James's & Pall Mall
8. Mayfair & Piccadilly
9. Smithfield & Newgate
10. Fleet Street & Ludgate
11. Moorfields & Finsbury
12. St Giles & Seven Dials
13. Spitalfields & Stepney
14. East End & Wapping (docks, industry)
15. Hyde Park & Kensington
16. Green Park & St James's Park
17. Vauxhall & Lambeth (south)
18. Holborn & Inns of Court
19. Cheapside & Royal Exchange
20. Clerkenwell

### Zone Properties

| Property | Type | Drives |
|----------|------|--------|
| `smell_base` | 0–1 | Ambient smell baseline |
| `noise_base` | 0–1 | Ambient noise baseline |
| `crowd_density` | 0–1 | Ambient crowd baseline |
| `river_proximity` | 0–1 | Heat-amplified Thames smell |
| `industrial_intensity` | 0–1 | Coal/tanning/brewing smell, scales with SO2 |
| `street_character` | narrow/medium/broad | Sound channelling factor |
| `building_height` | low/medium/tall | Enclosure, sight-line factor |

Properties vary by decade — e.g. East End industrial intensity rises from 0.3 in 1660s to 0.8 in 1810s as manufacturing expands.

---

## Inference Engine

For any location (lat/lon) and time (year, month), the sensory baseline is computed in three layers:

### Layer 1 — Zone Ambient
Look up the zone containing the point. Interpolate between the two nearest decade entries. Result: `{smell, noise, crowd, visual}` baseline.

### Layer 2 — Environmental Modifiers

Applied to zone baseline:

```
river_smell_boost = river_proximity × max(0, (temp_c - 10) / 20)
  — Thames smell rises above 10°C, peaks at ~30°C

smoke_boost = industrial_intensity × (so2_index / max_so2)
  — East zones get ×1.3 in prevailing SW wind

frost_crowd_boost = crowd_density × 0.3  if temp_c < -2
  — frozen Thames draws spectators (frost fair conditions)

street_channelling = noise × (street_character === 'narrow' ? 1.4 : 1.0)
  — narrow streets amplify and contain sound
```

### Layer 3 — Venue Point-Sources
Existing `computeIntensity()` output for venues within ~300px, attenuated by distance and zone street_character. Combined additively with zone baseline (capped at 1.0 per modality).

### Provenance Tracking

Each computed value carries a source tag:
- `"zone"` — from zone ambient properties
- `"env"` — from environmental modifier (CET/SO2)
- `"venue"` — from venue point-source evidence

---

## "Click Anywhere" Mode

Users can click any point on the map (not just venue markers). The side panel shows the inferred sensory estimate for that location with provenance breakdown:

```
SMITHFIELD & NEWGATE · 1750 · July

Smell     ████████░░  high
  zone character (market district) · amplified by July heat · confirmed in 3 passages

Noise     █████░░░░░  moderate
  zone character (narrow streets, stone) · no nearby venue evidence

Crowd     ██████░░░░  moderate–high
  market district baseline · 2 passages (Spectator 1712, Smollett 1771)

Visual    ███░░░░░░░  low–moderate
  enclosed street character · overcast (inferred from CET)
```

---

## Visual Rendering

### Zone Fills
A Leaflet GeoJSON layer renders zone polygons with:
- Fill colour = dominant sense colour for that zone (amber=smell, blue=noise, red=crowd, green=visual)
- Fill opacity = 0.05–0.08 (barely-there wash — does not compete with markers)
- No stroke (zone boundaries invisible unless zoomed in)
- Updates on year change (opacity shifts as zone character shifts)

### Zone-Aware Particles
In Smoke mode: particle spawn rate in each zone scales with `industrial_intensity × so2_index`. High-industrial zones (East End, Smithfield) visibly denser.
In Flow mode: crowd-dense zones generate more particles; narrow street_character increases particle speed and channel direction.

### Tooltip Enhancement
Add a provenance line below the existing modality percentages:

```
Vauxhall Gardens
open · garden · outdoor · vast
smell 45% · noise 30% · crowd 60%
📖 23 passages
~ zone: pleasure garden district · SW wind carries coal smoke
```

---

## Files

| File | Change |
|------|--------|
| `gazetteer/zones.json` | New — zone polygon definitions |
| `gazetteer/build_sensory_time_map.py` | Load zones.json, bake as JS constant, inference engine, zone fills, particle changes, click-anywhere mode, tooltip provenance, venue opened/closed |

No other files affected.

---

## Implementation Sequence

0. Venue opened/closed dates (independent, quick)
1. Create `zones.json` with polygon definitions and decade properties
2. Bake zones into HTML as `ZONE_DATA` constant
3. Zone fill rendering (Leaflet GeoJSON layer)
4. Inference engine JS (`computeZoneBaseline`, `applyEnvModifiers`)
5. Click-anywhere mode (map click handler, panel render)
6. Zone-aware particles (smoke density, flow channelling)
7. Tooltip provenance line
