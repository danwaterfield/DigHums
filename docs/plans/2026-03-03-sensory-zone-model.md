# Sensory Zone Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a three-layer sensory inference engine to the time map — zone ambient baseline + environmental physics modifiers + venue point-sources — so any point on the map has a plausible sensory estimate, with provenance surfaced in tooltips and the side panel.

**Architecture:** A hand-defined `zones.json` GeoJSON file (20 named London districts, each with decade-level sensory properties) is baked into the HTML at build time as `ZONE_DATA`. A JS inference engine computes sensory baselines for any lat/lon using point-in-polygon lookup, then applies CET temperature and SO2 smoke modifiers. Zone fills (5–8% opacity Leaflet GeoJSON layer) and zone-aware particle density make the ambient character visually legible. A map-click handler surfaces "standing here" sensory estimates in the side panel. Venue opened/closed dates are fixed first as an independent quick win.

**Tech Stack:** Python (build script), Leaflet.js, GeoJSON, Canvas 2D (existing particle engine), HTML template with Python `.format()` — all existing. No new dependencies.

---

## CRITICAL: Template Brace Convention

`build_sensory_time_map.py` generates HTML via Python's `str.format()`. **All literal JS/CSS braces in `HTML_TEMPLATE` must be doubled: `{{`, `}}`**. Python format placeholders use single braces: `{ZONE_DATA_JSON}`. Every JS snippet in this plan is shown as it will appear **in the Python source file** (braces already doubled). Do not add extra doubling.

---

## Task 0: Venue opened/closed dates

**Files:**
- Modify: `gazetteer/build_sensory_time_map.py` lines 136–147 (`load_data`) and the `VENUES.forEach` block in `updateMap()`
- Modify: `gazetteer/tests/test_build_sensory_time_map.py`

### Step 1: Write failing test

Add to `gazetteer/tests/test_build_sensory_time_map.py`:

```python
def test_venue_opened_closed_in_js(html):
    """Venues JSON must include opened and closed fields."""
    # LON001 has opened=1661 in venues.csv
    assert '"opened": 1661' in html or '"opened":1661' in html
```

Run: `pytest gazetteer/tests/test_build_sensory_time_map.py::test_venue_opened_closed_in_js -v`
Expected: FAIL — `opened` not currently in VENUES_JSON.

### Step 2: Add `opened` and `closed` to `load_data()`

Find lines 136–147 in `build_sensory_time_map.py`. The venue dict comprehension ends with `"capacity": r.get("capacity", "")`. Add two more fields:

```python
{"id": r["id"], "name": r["name"],
 "lat": float(r["lat"]), "lon": float(r["lon"]),
 "tier": int(r["tier"]) if r.get("tier") else 3,
 "enclosure":     r.get("enclosure", ""),
 "building_type": r.get("building_type", ""),
 "material":      r.get("material", ""),
 "capacity":      r.get("capacity", ""),
 "opened": int(r["opened"]) if r.get("opened", "").strip() else None,
 "closed": int(r["closed"]) if r.get("closed", "").strip() else None}
```

### Step 3: Run test — expect pass

`pytest gazetteer/tests/test_build_sensory_time_map.py::test_venue_opened_closed_in_js -v`
Expected: PASS.

### Step 4: Add date guard in `updateMap()` VENUES.forEach

Find the start of the `VENUES.forEach` block in `HTML_TEMPLATE` (search for `VENUES.forEach(v => {{`). The first line inside the callback is `const intensity = computeIntensity(...)`. Add the guard **before** that line:

```javascript
        // Hide venues outside their operational date range
        const marker = markersByVenueId[v.id];
        if (v.opened && year < v.opened) {{
            if (marker) marker.setStyle({{ opacity: 0, fillOpacity: 0 }});
            return;
        }}
        if (v.closed && year > v.closed) {{
            if (marker) marker.setStyle({{ opacity: 0, fillOpacity: 0 }});
            return;
        }}
        if (marker) marker.setStyle({{ opacity: 1, fillOpacity: undefined }});
```

**Important:** The existing code inside the forEach also calls `const marker = markersByVenueId[v.id]` — remove that duplicate declaration (it's now hoisted above). Search for the second `const marker =` inside the same forEach and delete it.

### Step 5: Build and verify

```bash
python3 gazetteer/build_sensory_time_map.py
```

Open `sensory_time_map.html`. Set year to 1650 — LON002 (opened 1742) should be invisible. Set year to 1750 — should appear. Set year to 1810 — LON002 (closed 1803) should disappear.

### Step 6: Run all tests

```bash
pytest gazetteer/tests/ -q
```

Expected: 134+ pass (one new test added).

### Step 7: Commit

```bash
git add gazetteer/build_sensory_time_map.py gazetteer/tests/test_build_sensory_time_map.py
git commit -m "feat: hide venues outside opened/closed date range"
```

---

## Task 1: Create zones.json

**Files:**
- Create: `gazetteer/zones.json`

### Step 1: Write the file

Create `gazetteer/zones.json` with the content below. This is a GeoJSON FeatureCollection. Coordinate order is `[longitude, latitude]` (GeoJSON standard). Each polygon's first and last coordinate must be identical (closed ring).

Decade keys are strings ("1660", "1700", etc.) and are interpolated linearly in JS.

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "name": "City of London",
        "dominant_sense": "noise",
        "decades": {
          "1660": {"smell_base":0.45,"noise_base":0.60,"crowd_density":0.65,"river_proximity":0.15,"industrial_intensity":0.30,"street_character":"narrow","building_height":"medium"},
          "1700": {"smell_base":0.48,"noise_base":0.62,"crowd_density":0.68,"river_proximity":0.15,"industrial_intensity":0.33,"street_character":"narrow","building_height":"medium"},
          "1750": {"smell_base":0.50,"noise_base":0.65,"crowd_density":0.70,"river_proximity":0.15,"industrial_intensity":0.38,"street_character":"narrow","building_height":"tall"},
          "1800": {"smell_base":0.52,"noise_base":0.68,"crowd_density":0.72,"river_proximity":0.15,"industrial_intensity":0.45,"street_character":"narrow","building_height":"tall"},
          "1820": {"smell_base":0.55,"noise_base":0.70,"crowd_density":0.73,"river_proximity":0.15,"industrial_intensity":0.50,"street_character":"narrow","building_height":"tall"}
        }
      },
      "geometry": {"type":"Polygon","coordinates":[[[-0.113,51.508],[-0.073,51.508],[-0.073,51.524],[-0.113,51.524],[-0.113,51.508]]]}
    },
    {
      "type": "Feature",
      "properties": {
        "name": "Thames Waterfront East",
        "dominant_sense": "smell",
        "decades": {
          "1660": {"smell_base":0.75,"noise_base":0.55,"crowd_density":0.55,"river_proximity":0.90,"industrial_intensity":0.40,"street_character":"narrow","building_height":"medium"},
          "1700": {"smell_base":0.78,"noise_base":0.58,"crowd_density":0.58,"river_proximity":0.90,"industrial_intensity":0.45,"street_character":"narrow","building_height":"medium"},
          "1750": {"smell_base":0.80,"noise_base":0.60,"crowd_density":0.60,"river_proximity":0.90,"industrial_intensity":0.50,"street_character":"narrow","building_height":"medium"},
          "1800": {"smell_base":0.82,"noise_base":0.62,"crowd_density":0.62,"river_proximity":0.90,"industrial_intensity":0.58,"street_character":"narrow","building_height":"medium"},
          "1820": {"smell_base":0.85,"noise_base":0.65,"crowd_density":0.65,"river_proximity":0.90,"industrial_intensity":0.65,"street_character":"narrow","building_height":"medium"}
        }
      },
      "geometry": {"type":"Polygon","coordinates":[[[-0.095,51.500],[-0.055,51.500],[-0.055,51.513],[-0.095,51.513],[-0.095,51.500]]]}
    },
    {
      "type": "Feature",
      "properties": {
        "name": "Southwark & Bankside",
        "dominant_sense": "smell",
        "decades": {
          "1660": {"smell_base":0.60,"noise_base":0.55,"crowd_density":0.55,"river_proximity":0.50,"industrial_intensity":0.45,"street_character":"narrow","building_height":"low"},
          "1700": {"smell_base":0.60,"noise_base":0.55,"crowd_density":0.55,"river_proximity":0.50,"industrial_intensity":0.45,"street_character":"narrow","building_height":"low"},
          "1750": {"smell_base":0.58,"noise_base":0.55,"crowd_density":0.55,"river_proximity":0.50,"industrial_intensity":0.45,"street_character":"narrow","building_height":"medium"},
          "1800": {"smell_base":0.55,"noise_base":0.52,"crowd_density":0.52,"river_proximity":0.50,"industrial_intensity":0.50,"street_character":"narrow","building_height":"medium"},
          "1820": {"smell_base":0.55,"noise_base":0.52,"crowd_density":0.52,"river_proximity":0.50,"industrial_intensity":0.55,"street_character":"narrow","building_height":"medium"}
        }
      },
      "geometry": {"type":"Polygon","coordinates":[[[-0.115,51.488],[-0.068,51.488],[-0.068,51.507],[-0.115,51.507],[-0.115,51.488]]]}
    },
    {
      "type": "Feature",
      "properties": {
        "name": "Covent Garden & Strand",
        "dominant_sense": "crowd",
        "decades": {
          "1660": {"smell_base":0.40,"noise_base":0.55,"crowd_density":0.60,"river_proximity":0.05,"industrial_intensity":0.15,"street_character":"medium","building_height":"medium"},
          "1700": {"smell_base":0.42,"noise_base":0.60,"crowd_density":0.65,"river_proximity":0.05,"industrial_intensity":0.15,"street_character":"medium","building_height":"medium"},
          "1750": {"smell_base":0.45,"noise_base":0.65,"crowd_density":0.70,"river_proximity":0.05,"industrial_intensity":0.18,"street_character":"medium","building_height":"medium"},
          "1800": {"smell_base":0.42,"noise_base":0.62,"crowd_density":0.68,"river_proximity":0.05,"industrial_intensity":0.20,"street_character":"medium","building_height":"tall"},
          "1820": {"smell_base":0.40,"noise_base":0.60,"crowd_density":0.65,"river_proximity":0.05,"industrial_intensity":0.22,"street_character":"medium","building_height":"tall"}
        }
      },
      "geometry": {"type":"Polygon","coordinates":[[[-0.131,51.508],[-0.109,51.508],[-0.109,51.521],[-0.131,51.521],[-0.131,51.508]]]}
    },
    {
      "type": "Feature",
      "properties": {
        "name": "Westminster & Whitehall",
        "dominant_sense": "crowd",
        "decades": {
          "1660": {"smell_base":0.20,"noise_base":0.40,"crowd_density":0.45,"river_proximity":0.35,"industrial_intensity":0.10,"street_character":"broad","building_height":"medium"},
          "1700": {"smell_base":0.20,"noise_base":0.42,"crowd_density":0.48,"river_proximity":0.35,"industrial_intensity":0.10,"street_character":"broad","building_height":"medium"},
          "1750": {"smell_base":0.22,"noise_base":0.45,"crowd_density":0.50,"river_proximity":0.35,"industrial_intensity":0.12,"street_character":"broad","building_height":"tall"},
          "1800": {"smell_base":0.22,"noise_base":0.48,"crowd_density":0.52,"river_proximity":0.35,"industrial_intensity":0.12,"street_character":"broad","building_height":"tall"},
          "1820": {"smell_base":0.22,"noise_base":0.50,"crowd_density":0.55,"river_proximity":0.35,"industrial_intensity":0.14,"street_character":"broad","building_height":"tall"}
        }
      },
      "geometry": {"type":"Polygon","coordinates":[[[-0.137,51.493],[-0.118,51.493],[-0.118,51.512],[-0.137,51.512],[-0.137,51.493]]]}
    },
    {
      "type": "Feature",
      "properties": {
        "name": "St James's & Pall Mall",
        "dominant_sense": "visual",
        "decades": {
          "1660": {"smell_base":0.12,"noise_base":0.25,"crowd_density":0.35,"river_proximity":0.05,"industrial_intensity":0.05,"street_character":"broad","building_height":"medium"},
          "1700": {"smell_base":0.12,"noise_base":0.28,"crowd_density":0.40,"river_proximity":0.05,"industrial_intensity":0.05,"street_character":"broad","building_height":"medium"},
          "1750": {"smell_base":0.14,"noise_base":0.30,"crowd_density":0.45,"river_proximity":0.05,"industrial_intensity":0.06,"street_character":"broad","building_height":"tall"},
          "1800": {"smell_base":0.14,"noise_base":0.32,"crowd_density":0.48,"river_proximity":0.05,"industrial_intensity":0.08,"street_character":"broad","building_height":"tall"},
          "1820": {"smell_base":0.15,"noise_base":0.35,"crowd_density":0.50,"river_proximity":0.05,"industrial_intensity":0.08,"street_character":"broad","building_height":"tall"}
        }
      },
      "geometry": {"type":"Polygon","coordinates":[[[-0.153,51.499],[-0.130,51.499],[-0.130,51.513],[-0.153,51.513],[-0.153,51.499]]]}
    },
    {
      "type": "Feature",
      "properties": {
        "name": "Mayfair & Piccadilly",
        "dominant_sense": "visual",
        "decades": {
          "1660": {"smell_base":0.08,"noise_base":0.15,"crowd_density":0.20,"river_proximity":0.00,"industrial_intensity":0.03,"street_character":"broad","building_height":"low"},
          "1700": {"smell_base":0.10,"noise_base":0.20,"crowd_density":0.30,"river_proximity":0.00,"industrial_intensity":0.04,"street_character":"broad","building_height":"medium"},
          "1750": {"smell_base":0.12,"noise_base":0.28,"crowd_density":0.42,"river_proximity":0.00,"industrial_intensity":0.05,"street_character":"broad","building_height":"tall"},
          "1800": {"smell_base":0.12,"noise_base":0.30,"crowd_density":0.45,"river_proximity":0.00,"industrial_intensity":0.06,"street_character":"broad","building_height":"tall"},
          "1820": {"smell_base":0.13,"noise_base":0.32,"crowd_density":0.48,"river_proximity":0.00,"industrial_intensity":0.07,"street_character":"broad","building_height":"tall"}
        }
      },
      "geometry": {"type":"Polygon","coordinates":[[[-0.168,51.506],[-0.140,51.506],[-0.140,51.522],[-0.168,51.522],[-0.168,51.506]]]}
    },
    {
      "type": "Feature",
      "properties": {
        "name": "Smithfield & Newgate",
        "dominant_sense": "smell",
        "decades": {
          "1660": {"smell_base":0.80,"noise_base":0.70,"crowd_density":0.65,"river_proximity":0.05,"industrial_intensity":0.30,"street_character":"narrow","building_height":"medium"},
          "1700": {"smell_base":0.80,"noise_base":0.70,"crowd_density":0.65,"river_proximity":0.05,"industrial_intensity":0.32,"street_character":"narrow","building_height":"medium"},
          "1750": {"smell_base":0.82,"noise_base":0.72,"crowd_density":0.68,"river_proximity":0.05,"industrial_intensity":0.35,"street_character":"narrow","building_height":"medium"},
          "1800": {"smell_base":0.80,"noise_base":0.70,"crowd_density":0.65,"river_proximity":0.05,"industrial_intensity":0.38,"street_character":"narrow","building_height":"medium"},
          "1820": {"smell_base":0.78,"noise_base":0.68,"crowd_density":0.62,"river_proximity":0.05,"industrial_intensity":0.40,"street_character":"narrow","building_height":"medium"}
        }
      },
      "geometry": {"type":"Polygon","coordinates":[[[-0.108,51.517],[-0.089,51.517],[-0.089,51.530],[-0.108,51.530],[-0.108,51.517]]]}
    },
    {
      "type": "Feature",
      "properties": {
        "name": "Fleet Street & Ludgate",
        "dominant_sense": "noise",
        "decades": {
          "1660": {"smell_base":0.35,"noise_base":0.55,"crowd_density":0.55,"river_proximity":0.05,"industrial_intensity":0.20,"street_character":"narrow","building_height":"medium"},
          "1700": {"smell_base":0.38,"noise_base":0.58,"crowd_density":0.58,"river_proximity":0.05,"industrial_intensity":0.25,"street_character":"narrow","building_height":"medium"},
          "1750": {"smell_base":0.40,"noise_base":0.62,"crowd_density":0.60,"river_proximity":0.05,"industrial_intensity":0.30,"street_character":"narrow","building_height":"medium"},
          "1800": {"smell_base":0.40,"noise_base":0.65,"crowd_density":0.62,"river_proximity":0.05,"industrial_intensity":0.35,"street_character":"narrow","building_height":"tall"},
          "1820": {"smell_base":0.40,"noise_base":0.68,"crowd_density":0.65,"river_proximity":0.05,"industrial_intensity":0.38,"street_character":"narrow","building_height":"tall"}
        }
      },
      "geometry": {"type":"Polygon","coordinates":[[[-0.115,51.511],[-0.097,51.511],[-0.097,51.519],[-0.115,51.519],[-0.115,51.511]]]}
    },
    {
      "type": "Feature",
      "properties": {
        "name": "St Giles & Seven Dials",
        "dominant_sense": "smell",
        "decades": {
          "1660": {"smell_base":0.55,"noise_base":0.50,"crowd_density":0.60,"river_proximity":0.00,"industrial_intensity":0.20,"street_character":"narrow","building_height":"low"},
          "1700": {"smell_base":0.60,"noise_base":0.55,"crowd_density":0.65,"river_proximity":0.00,"industrial_intensity":0.22,"street_character":"narrow","building_height":"medium"},
          "1750": {"smell_base":0.65,"noise_base":0.58,"crowd_density":0.70,"river_proximity":0.00,"industrial_intensity":0.25,"street_character":"narrow","building_height":"medium"},
          "1800": {"smell_base":0.68,"noise_base":0.60,"crowd_density":0.72,"river_proximity":0.00,"industrial_intensity":0.28,"street_character":"narrow","building_height":"medium"},
          "1820": {"smell_base":0.70,"noise_base":0.62,"crowd_density":0.75,"river_proximity":0.00,"industrial_intensity":0.30,"street_character":"narrow","building_height":"medium"}
        }
      },
      "geometry": {"type":"Polygon","coordinates":[[[-0.134,51.514],[-0.118,51.514],[-0.118,51.527],[-0.134,51.527],[-0.134,51.514]]]}
    },
    {
      "type": "Feature",
      "properties": {
        "name": "East End & Wapping",
        "dominant_sense": "smell",
        "decades": {
          "1660": {"smell_base":0.50,"noise_base":0.50,"crowd_density":0.45,"river_proximity":0.40,"industrial_intensity":0.40,"street_character":"narrow","building_height":"low"},
          "1700": {"smell_base":0.55,"noise_base":0.55,"crowd_density":0.50,"river_proximity":0.45,"industrial_intensity":0.50,"street_character":"narrow","building_height":"low"},
          "1750": {"smell_base":0.60,"noise_base":0.60,"crowd_density":0.55,"river_proximity":0.50,"industrial_intensity":0.60,"street_character":"narrow","building_height":"medium"},
          "1800": {"smell_base":0.68,"noise_base":0.65,"crowd_density":0.60,"river_proximity":0.55,"industrial_intensity":0.72,"street_character":"narrow","building_height":"medium"},
          "1820": {"smell_base":0.72,"noise_base":0.68,"crowd_density":0.65,"river_proximity":0.55,"industrial_intensity":0.80,"street_character":"narrow","building_height":"medium"}
        }
      },
      "geometry": {"type":"Polygon","coordinates":[[[-0.072,51.505],[-0.035,51.505],[-0.035,51.522],[-0.072,51.522],[-0.072,51.505]]]}
    },
    {
      "type": "Feature",
      "properties": {
        "name": "Hyde Park & Kensington",
        "dominant_sense": "visual",
        "decades": {
          "1660": {"smell_base":0.05,"noise_base":0.08,"crowd_density":0.15,"river_proximity":0.00,"industrial_intensity":0.02,"street_character":"broad","building_height":"low"},
          "1700": {"smell_base":0.05,"noise_base":0.10,"crowd_density":0.20,"river_proximity":0.00,"industrial_intensity":0.02,"street_character":"broad","building_height":"low"},
          "1750": {"smell_base":0.06,"noise_base":0.12,"crowd_density":0.28,"river_proximity":0.00,"industrial_intensity":0.03,"street_character":"broad","building_height":"low"},
          "1800": {"smell_base":0.06,"noise_base":0.12,"crowd_density":0.30,"river_proximity":0.00,"industrial_intensity":0.03,"street_character":"broad","building_height":"low"},
          "1820": {"smell_base":0.06,"noise_base":0.12,"crowd_density":0.32,"river_proximity":0.00,"industrial_intensity":0.03,"street_character":"broad","building_height":"low"}
        }
      },
      "geometry": {"type":"Polygon","coordinates":[[[-0.200,51.499],[-0.152,51.499],[-0.152,51.523],[-0.200,51.523],[-0.200,51.499]]]}
    },
    {
      "type": "Feature",
      "properties": {
        "name": "St James's Park",
        "dominant_sense": "visual",
        "decades": {
          "1660": {"smell_base":0.08,"noise_base":0.12,"crowd_density":0.25,"river_proximity":0.10,"industrial_intensity":0.03,"street_character":"broad","building_height":"low"},
          "1700": {"smell_base":0.08,"noise_base":0.14,"crowd_density":0.30,"river_proximity":0.10,"industrial_intensity":0.03,"street_character":"broad","building_height":"low"},
          "1750": {"smell_base":0.09,"noise_base":0.15,"crowd_density":0.38,"river_proximity":0.10,"industrial_intensity":0.04,"street_character":"broad","building_height":"low"},
          "1800": {"smell_base":0.09,"noise_base":0.15,"crowd_density":0.40,"river_proximity":0.10,"industrial_intensity":0.04,"street_character":"broad","building_height":"low"},
          "1820": {"smell_base":0.09,"noise_base":0.16,"crowd_density":0.40,"river_proximity":0.10,"industrial_intensity":0.04,"street_character":"broad","building_height":"low"}
        }
      },
      "geometry": {"type":"Polygon","coordinates":[[[-0.147,51.497],[-0.127,51.497],[-0.127,51.508],[-0.147,51.508],[-0.147,51.497]]]}
    },
    {
      "type": "Feature",
      "properties": {
        "name": "Vauxhall & Lambeth",
        "dominant_sense": "crowd",
        "decades": {
          "1660": {"smell_base":0.15,"noise_base":0.20,"crowd_density":0.20,"river_proximity":0.45,"industrial_intensity":0.15,"street_character":"medium","building_height":"low"},
          "1700": {"smell_base":0.18,"noise_base":0.25,"crowd_density":0.30,"river_proximity":0.45,"industrial_intensity":0.18,"street_character":"medium","building_height":"low"},
          "1750": {"smell_base":0.20,"noise_base":0.35,"crowd_density":0.50,"river_proximity":0.45,"industrial_intensity":0.20,"street_character":"medium","building_height":"low"},
          "1800": {"smell_base":0.22,"noise_base":0.38,"crowd_density":0.52,"river_proximity":0.45,"industrial_intensity":0.25,"street_character":"medium","building_height":"medium"},
          "1820": {"smell_base":0.22,"noise_base":0.35,"crowd_density":0.45,"river_proximity":0.45,"industrial_intensity":0.30,"street_character":"medium","building_height":"medium"}
        }
      },
      "geometry": {"type":"Polygon","coordinates":[[[-0.135,51.477],[-0.108,51.477],[-0.108,51.495],[-0.135,51.495],[-0.135,51.477]]]}
    },
    {
      "type": "Feature",
      "properties": {
        "name": "Holborn & Inns of Court",
        "dominant_sense": "noise",
        "decades": {
          "1660": {"smell_base":0.30,"noise_base":0.45,"crowd_density":0.50,"river_proximity":0.00,"industrial_intensity":0.18,"street_character":"medium","building_height":"medium"},
          "1700": {"smell_base":0.32,"noise_base":0.48,"crowd_density":0.52,"river_proximity":0.00,"industrial_intensity":0.20,"street_character":"medium","building_height":"medium"},
          "1750": {"smell_base":0.34,"noise_base":0.52,"crowd_density":0.55,"river_proximity":0.00,"industrial_intensity":0.22,"street_character":"medium","building_height":"medium"},
          "1800": {"smell_base":0.35,"noise_base":0.54,"crowd_density":0.56,"river_proximity":0.00,"industrial_intensity":0.25,"street_character":"medium","building_height":"tall"},
          "1820": {"smell_base":0.35,"noise_base":0.55,"crowd_density":0.57,"river_proximity":0.00,"industrial_intensity":0.28,"street_character":"medium","building_height":"tall"}
        }
      },
      "geometry": {"type":"Polygon","coordinates":[[[-0.121,51.514],[-0.099,51.514],[-0.099,51.525],[-0.121,51.525],[-0.121,51.514]]]}
    },
    {
      "type": "Feature",
      "properties": {
        "name": "Moorfields & Finsbury",
        "dominant_sense": "visual",
        "decades": {
          "1660": {"smell_base":0.20,"noise_base":0.22,"crowd_density":0.25,"river_proximity":0.00,"industrial_intensity":0.15,"street_character":"medium","building_height":"low"},
          "1700": {"smell_base":0.25,"noise_base":0.28,"crowd_density":0.32,"river_proximity":0.00,"industrial_intensity":0.20,"street_character":"medium","building_height":"low"},
          "1750": {"smell_base":0.30,"noise_base":0.35,"crowd_density":0.40,"river_proximity":0.00,"industrial_intensity":0.28,"street_character":"medium","building_height":"medium"},
          "1800": {"smell_base":0.38,"noise_base":0.42,"crowd_density":0.50,"river_proximity":0.00,"industrial_intensity":0.38,"street_character":"narrow","building_height":"medium"},
          "1820": {"smell_base":0.42,"noise_base":0.48,"crowd_density":0.55,"river_proximity":0.00,"industrial_intensity":0.45,"street_character":"narrow","building_height":"medium"}
        }
      },
      "geometry": {"type":"Polygon","coordinates":[[[-0.092,51.519],[-0.072,51.519],[-0.072,51.532],[-0.092,51.532],[-0.092,51.519]]]}
    }
  ]
}
```

### Step 2: Verify it's valid JSON

```bash
python3 -c "import json; d=json.load(open('gazetteer/zones.json')); print(len(d['features']), 'zones')"
```

Expected: `16 zones`

### Step 3: Commit

```bash
git add gazetteer/zones.json
git commit -m "data: add London sensory zone definitions (16 zones, 5 decades each)"
```

---

## Task 2: Bake ZONE_DATA into HTML

**Files:**
- Modify: `gazetteer/build_sensory_time_map.py` — `build()` function and `HTML_TEMPLATE`
- Modify: `gazetteer/tests/test_build_sensory_time_map.py`

### Step 1: Write failing test

```python
def test_zone_data_embedded(html):
    """ZONE_DATA constant must be present and contain zone names."""
    assert 'const ZONE_DATA' in html
    assert 'Smithfield' in html
```

Run: `pytest gazetteer/tests/test_build_sensory_time_map.py::test_zone_data_embedded -v`
Expected: FAIL.

### Step 2: Load zones in `build()`

At the top of `build()`, after `data = load_data(...)`, add:

```python
zones_path = Path(__file__).parent / "zones.json"
zones_data = json.loads(zones_path.read_text(encoding="utf-8")) if zones_path.exists() else {"type": "FeatureCollection", "features": []}
```

### Step 3: Add `ZONE_DATA_JSON` to `.format()` call

Inside `HTML_TEMPLATE.format(...)`, add after `STREET_NETWORK_JSON`:

```python
ZONE_DATA_JSON       = json.dumps(zones_data,          ensure_ascii=False),
```

### Step 4: Add `ZONE_DATA` constant to `HTML_TEMPLATE`

Find the line in `HTML_TEMPLATE`:
```
const STREET_NETWORK = {STREET_NETWORK_JSON};  // [[lat,lon],...] pre-1820 streets
```

Add immediately after it:
```
const ZONE_DATA = {ZONE_DATA_JSON};            // GeoJSON FeatureCollection — named London zones
```

### Step 5: Run test — expect pass

`pytest gazetteer/tests/test_build_sensory_time_map.py::test_zone_data_embedded -v`
Expected: PASS.

### Step 6: Run all tests

`pytest gazetteer/tests/ -q` — expected: 135+ pass.

### Step 7: Commit

```bash
git add gazetteer/build_sensory_time_map.py gazetteer/tests/test_build_sensory_time_map.py
git commit -m "feat: bake ZONE_DATA GeoJSON into sensory_time_map.html"
```

---

## Task 3: Zone lookup and baseline inference engine

**Files:**
- Modify: `gazetteer/build_sensory_time_map.py` — `HTML_TEMPLATE` JS section

This task adds three JS functions after the `ZONE_DATA` constant:
1. `pointInPolygon(lat, lon, ring)` — ray casting
2. `getZoneForPoint(lat, lon)` — iterates features, returns matching zone properties or null
3. `interpolateZoneProps(zoneProps, year)` — linear interpolation between decade snapshots
4. `computeZoneBaseline(lat, lon, year, month)` — combines zone properties with CET/smoke env modifiers

### Step 1: Write failing test

```python
def test_zone_inference_functions(html):
    """Zone baseline inference functions must be present."""
    assert 'function pointInPolygon' in html
    assert 'function getZoneForPoint' in html
    assert 'function computeZoneBaseline' in html
```

Run: `pytest gazetteer/tests/test_build_sensory_time_map.py::test_zone_inference_functions -v`
Expected: FAIL.

### Step 2: Add inference engine JS to `HTML_TEMPLATE`

Find the line `const ZONE_DATA = {ZONE_DATA_JSON};` and add the following block **immediately after** it:

```javascript

// ── Zone inference engine ──────────────────────────────────────────────────

function pointInPolygon(lat, lon, ring) {{
    // ray casting — ring is array of [lon, lat] pairs (GeoJSON order)
    let inside = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {{
        const xi = ring[i][0], yi = ring[i][1];
        const xj = ring[j][0], yj = ring[j][1];
        if ((yi > lat) !== (yj > lat) && lon < (xj - xi) * (lat - yi) / (yj - yi) + xi)
            inside = !inside;
    }}
    return inside;
}}

function getZoneForPoint(lat, lon) {{
    if (!ZONE_DATA || !ZONE_DATA.features) return null;
    for (const feat of ZONE_DATA.features) {{
        const ring = feat.geometry.coordinates[0];
        if (pointInPolygon(lat, lon, ring)) return feat.properties;
    }}
    return null;
}}

function interpolateZoneProps(zoneProps, year) {{
    const decades = Object.keys(zoneProps.decades).map(Number).sort((a,b)=>a-b);
    if (!decades.length) return {{}};
    if (year <= decades[0]) return zoneProps.decades[String(decades[0])];
    if (year >= decades[decades.length-1]) return zoneProps.decades[String(decades[decades.length-1])];
    let d0 = decades[0], d1 = decades[1];
    for (let i = 0; i < decades.length - 1; i++) {{
        if (year >= decades[i] && year <= decades[i+1]) {{ d0 = decades[i]; d1 = decades[i+1]; break; }}
    }}
    const t = (year - d0) / (d1 - d0);
    const p0 = zoneProps.decades[String(d0)], p1 = zoneProps.decades[String(d1)];
    return {{
        smell_base:           p0.smell_base           + t * (p1.smell_base           - p0.smell_base),
        noise_base:           p0.noise_base           + t * (p1.noise_base           - p0.noise_base),
        crowd_density:        p0.crowd_density        + t * (p1.crowd_density        - p0.crowd_density),
        river_proximity:      p0.river_proximity,
        industrial_intensity: p0.industrial_intensity + t * (p1.industrial_intensity - p0.industrial_intensity),
        street_character:     t < 0.5 ? p0.street_character : p1.street_character,
        building_height:      t < 0.5 ? p0.building_height  : p1.building_height,
    }};
}}

// Layer 1 (zone ambient) + Layer 2 (env modifiers)
// Returns {{smell, noise, crowd, visual, zone, provenance}}
function computeZoneBaseline(lat, lon, year, month) {{
    const zoneProps = getZoneForPoint(lat, lon);
    if (!zoneProps) return null;
    const p = interpolateZoneProps(zoneProps, year);

    let smell = p.smell_base;
    let noise = p.noise_base;
    let crowd = p.crowd_density;
    let visual = 0.3;  // ambient visual baseline
    const provenance = [];

    // Env modifier 1: river smell rises with summer heat
    if (p.river_proximity > 0) {{
        const yearCET = CET_DATA[year] || {{}};
        const temp = (month && yearCET[month] !== undefined) ? yearCET[month] : (yearCET[0] || 10);
        const riverBoost = p.river_proximity * Math.max(0, (temp - 10) / 22);
        if (riverBoost > 0.02) {{
            smell = Math.min(1, smell + riverBoost);
            provenance.push('river smell (' + temp.toFixed(1) + '\u00b0C)');
        }}
    }}

    // Env modifier 2: industrial smoke scales with so2_index
    if (p.industrial_intensity > 0 && SMOKE_DATA_ENV.length) {{
        const decade = Math.floor(year / 10) * 10;
        const smokeRow = SMOKE_DATA_ENV.find(s => s.decade_start === decade);
        const so2 = smokeRow ? smokeRow.so2_index : 0;
        const maxSo2 = Math.max(...SMOKE_DATA_ENV.map(s => s.so2_index)) || 1;
        const smokeBoost = p.industrial_intensity * (so2 / maxSo2) * 0.4;
        // East of City gets 1.3x in prevailing SW wind
        const windMult = lon > -0.09 ? 1.3 : lon < -0.17 ? 0.7 : 1.0;
        const finalSmoke = smokeBoost * windMult;
        if (finalSmoke > 0.02) {{
            smell = Math.min(1, smell + finalSmoke * 0.6);
            visual = Math.min(1, visual + finalSmoke * 0.3);
            provenance.push('coal smoke (SO\u2082 index ' + so2.toFixed(1) + ')');
        }}
    }}

    // Env modifier 3: frost fair crowd noise (Thames frozen)
    {{
        const yearCET = CET_DATA[year] || {{}};
        const janTemp = yearCET[1] !== undefined ? yearCET[1] : (yearCET['1'] !== undefined ? yearCET['1'] : null);
        if (janTemp !== null && janTemp < -2 && month === 1 && p.river_proximity > 0.5) {{
            crowd = Math.min(1, crowd + 0.25);
            noise = Math.min(1, noise + 0.20);
            provenance.push('frost fair conditions');
        }}
    }}

    // Env modifier 4: narrow streets amplify noise
    if (p.street_character === 'narrow') {{
        noise = Math.min(1, noise * 1.35);
    }}

    if (!provenance.length) provenance.push('zone character (' + zoneProps.name + ')');
    else provenance.unshift('zone character (' + zoneProps.name + ')');

    return {{
        smell, noise, crowd, visual,
        zone: zoneProps.name,
        dominant: zoneProps.dominant_sense,
        provenance,
        street_character: p.street_character,
    }};
}}
```

### Step 3: Run test — expect pass

`pytest gazetteer/tests/test_build_sensory_time_map.py::test_zone_inference_functions -v`
Expected: PASS.

### Step 4: Run all tests

`pytest gazetteer/tests/ -q`

### Step 5: Commit

```bash
git add gazetteer/build_sensory_time_map.py gazetteer/tests/test_build_sensory_time_map.py
git commit -m "feat: zone inference engine — pointInPolygon, getZoneForPoint, computeZoneBaseline"
```

---

## Task 4: Zone fill rendering

**Files:**
- Modify: `gazetteer/build_sensory_time_map.py` — `HTML_TEMPLATE` JS

A Leaflet GeoJSON layer renders zone polygons as barely-visible colour washes. The fill colour reflects the zone's dominant sense; opacity is `0.06 × zone_intensity_for_active_modality`.

### Step 1: Write failing test

```python
def test_zone_fill_layer(html):
    """Zone fill Leaflet layer must be present."""
    assert 'zoneFillLayer' in html
    assert 'updateZoneFills' in html
```

Run: `pytest gazetteer/tests/test_build_sensory_time_map.py::test_zone_fill_layer -v`
Expected: FAIL.

### Step 2: Add sense colour map and zone fill layer to HTML_TEMPLATE

Add this JS block immediately after the `map.on('baselayerchange', ...)` block (after Task 3 auto-basemap code):

```javascript

// ── Zone fill layer ────────────────────────────────────────────────────────
const SENSE_FILL_COLORS = {{
    smell:  '#b48a28',
    noise:  '#3c78c8',
    crowd:  '#b43c3c',
    visual: '#3ca050',
}};

let zoneFillLayer = null;

function updateZoneFills(year, month) {{
    if (zoneFillLayer) map.removeLayer(zoneFillLayer);
    if (!ZONE_DATA || !ZONE_DATA.features.length) return;

    zoneFillLayer = L.geoJSON(ZONE_DATA, {{
        style: (feat) => {{
            const p = feat.properties;
            const props = interpolateZoneProps(p, year);
            const dominant = p.dominant_sense || 'smell';
            const color = SENSE_FILL_COLORS[dominant] || '#888';
            // Opacity driven by the dominant sense baseline value
            const intensity = props[dominant + '_base'] || props.smell_base || 0.3;
            const opacity = Math.min(0.10, intensity * 0.10);
            return {{
                fillColor: color,
                fillOpacity: opacity,
                stroke: false,
                interactive: false,
            }};
        }},
    }}).addTo(map);
    // Zone fills sit below markers — push to back
    if (zoneFillLayer.getPane) zoneFillLayer.getPane();
    zoneFillLayer.bringToBack();
}}
```

### Step 3: Call `updateZoneFills(year, month)` from `updateMap()`

Find the three calls in `updateMap()` added in previous tasks:
```javascript
    updateEnvIndicators(year, month);
    _projectStreets(year);
    updateBasemap(year);
```

Add `updateZoneFills(year, month);` after `updateBasemap(year);`:
```javascript
    updateEnvIndicators(year, month);
    _projectStreets(year);
    updateBasemap(year);
    updateZoneFills(year, month);
```

### Step 4: Write failing test and run

`pytest gazetteer/tests/test_build_sensory_time_map.py::test_zone_fill_layer -v`
Expected: PASS.

### Step 5: Build and check visually

`python3 gazetteer/build_sensory_time_map.py` then open the HTML. Zone fills should be faint colour washes under venue markers. Smithfield area should be amber-brown; Hyde Park should be faint green; Mayfair faint green; East End faint amber. Fills should not be distracting.

### Step 6: Run all tests and commit

```bash
pytest gazetteer/tests/ -q
git add gazetteer/build_sensory_time_map.py gazetteer/tests/test_build_sensory_time_map.py
git commit -m "feat: zone fill layer — faint sense-coloured GeoJSON washes"
```

---

## Task 5: Click-anywhere "standing here" panel

**Files:**
- Modify: `gazetteer/build_sensory_time_map.py` — `HTML_TEMPLATE` JS

Users click any point on the map. The side panel shows the inferred sensory estimate with provenance, exactly as if standing at that point.

### Step 1: Write failing test

```python
def test_click_anywhere_handler(html):
    """Map click-anywhere handler and renderLocationPanel must be present."""
    assert 'renderLocationPanel' in html
    assert "map.on('click'" in html
```

Run: `pytest gazetteer/tests/test_build_sensory_time_map.py::test_click_anywhere_handler -v`
Expected: FAIL.

### Step 2: Add `renderLocationPanel()` function

Find the existing `function renderVenuePanel(venueId, year, month, dow, band)` in `HTML_TEMPLATE`. Add the following new function **immediately before** it:

```javascript
function renderLocationPanel(lat, lon, year, month) {{
    const baseline = computeZoneBaseline(lat, lon, year, month);
    const body = document.getElementById('panel-body');
    const title = document.getElementById('panel-title');
    if (!baseline) {{
        title.textContent = 'OUTSIDE MAPPED AREA';
        body.innerHTML = '<p style="opacity:0.6;font-size:0.9em">Click within central London (1660\u20131820 extent).</p>';
        return;
    }}
    title.textContent = baseline.zone.toUpperCase();

    const pct = v => Math.round(v * 100);
    const bar = v => {{
        const filled = Math.round(v * 10);
        return '\u2588'.repeat(filled) + '\u2591'.repeat(10 - filled);
    }};

    const modalities = [
        ['Smell',  baseline.smell,  '#b48a28'],
        ['Noise',  baseline.noise,  '#3c78c8'],
        ['Crowd',  baseline.crowd,  '#b43c3c'],
        ['Visual', baseline.visual, '#3ca050'],
    ];

    const label = v => v > 0.7 ? 'high' : v > 0.45 ? 'moderate\u2013high' : v > 0.25 ? 'moderate' : v > 0.1 ? 'low\u2013moderate' : 'low';

    let html = `<div style="margin-bottom:10px;font-size:0.82em;opacity:0.65">\u2316 ${lat.toFixed(4)}, ${lon.toFixed(4)} &middot; ${baseline.street_character} streets</div>`;

    modalities.forEach(([name, val, col]) => {{
        html += `<div style="margin-bottom:8px">
          <div style="display:flex;align-items:baseline;gap:6px">
            <span style="font-weight:bold;min-width:50px">${{name}}</span>
            <span style="font-family:monospace;color:${{col}}">${{bar(val)}}</span>
            <span style="font-size:0.9em;opacity:0.8">${{label(val)}}</span>
          </div>
          <div style="font-size:0.78em;opacity:0.55;margin-left:56px;font-style:italic">
            ${{baseline.provenance.join(' \u00b7 ')}}
          </div>
        </div>`;
    }});

    html += `<p style="font-size:0.78em;opacity:0.45;margin-top:12px;font-style:italic">
        Zone inference &mdash; click a venue marker for documented evidence.
    </p>`;

    body.innerHTML = html;
}}
```

### Step 3: Add map click handler

Find the block `map.on('moveend zoomend', ...)` added in Task 2 (historical geography). Add a new click handler **after** all map event listeners:

```javascript
map.on('click', (e) => {{
    // Don't intercept if a venue marker was clicked (it sets selectedVenue)
    // This fires after marker click handlers, so selectedVenue is already set
    if (state.selectedVenue) return;
    const year  = parseInt(document.getElementById('year-slider').value);
    renderLocationPanel(e.latlng.lat, e.latlng.lng, year, state.month);
}});
```

**Note:** Venue markers call `selectVenue()` which sets `state.selectedVenue` and calls `renderVenuePanel`. The map click handler checks `state.selectedVenue` to avoid overwriting venue panel. However, Leaflet fires marker click handlers before map click handlers, but `state.selectedVenue` is set synchronously in `selectVenue()`, so this check works.

Wait — actually Leaflet fires the map click AFTER the marker click, but both are synchronous. `selectVenue()` sets `state.selectedVenue`, so by the time `map.on('click')` fires, it's already set. But we want map clicks on empty space to show the location panel. We need to check if the click target was on a marker.

Better approach: use `e.originalEvent` and check if the target is the map container itself. Or: reset `state.selectedVenue` at the start of the click handler and check if a marker re-sets it. This is fragile.

Simplest correct approach: add `L.DomEvent.stopPropagation(e)` inside each marker's click handler. Currently marker click is wired via `m.on('click', ...)` — add `e.stopPropagation()` there.

Find the marker `click` handler in the HTML_TEMPLATE (search for `m.on('click'`). It looks like:
```javascript
        m.on('click', () => {{ selectVenue(v.id); }});
```

Change to:
```javascript
        m.on('click', (e) => {{ e.stopPropagation(); selectVenue(v.id); }});
```

And simplify the map click handler to not check `state.selectedVenue`:
```javascript
map.on('click', (e) => {{
    state.selectedVenue = null;
    const year  = parseInt(document.getElementById('year-slider').value);
    renderLocationPanel(e.latlng.lat, e.latlng.lng, year, state.month);
}});
```

### Step 4: Run test — expect pass

`pytest gazetteer/tests/test_build_sensory_time_map.py::test_click_anywhere_handler -v`

### Step 5: Run all tests and commit

```bash
pytest gazetteer/tests/ -q
git add gazetteer/build_sensory_time_map.py gazetteer/tests/test_build_sensory_time_map.py
git commit -m "feat: click-anywhere standing-here panel with zone inference and provenance"
```

---

## Task 6: Tooltip provenance line

**Files:**
- Modify: `gazetteer/build_sensory_time_map.py` — tooltip construction in `updateMap()`

Adds a zone ambient line to venue tooltips: `~ zone: Smithfield & Newgate · river smell (22.1°C)`

### Step 1: Write failing test

```python
def test_tooltip_provenance_line(html):
    """Tooltip construction must include zone provenance call."""
    assert 'computeZoneBaseline' in html
    assert 'zoneBaseline.provenance' in html
```

Run: `pytest gazetteer/tests/test_build_sensory_time_map.py::test_tooltip_provenance_line -v`
Expected: FAIL.

### Step 2: Add zone provenance to tooltip

Find the tooltip construction block in `updateMap()`. It currently ends with:
```javascript
        if (evCount > 0) tip += `<br><span style="font-size:0.82em;opacity:0.7">📖 ${evCount} passages</span>`;
        marker.setTooltipContent(tip);
```

Add between those two lines:

```javascript
        const zoneBaseline = computeZoneBaseline(v.lat, v.lon, year, month);
        if (zoneBaseline && zoneBaseline.provenance.length > 1) {{
            tip += `<br><span style="font-size:0.78em;opacity:0.50;font-style:italic">~ ${{zoneBaseline.provenance.slice(1).join(' \u00b7 ')}}</span>`;
        }}
```

(We use `.slice(1)` to skip the first provenance element which is just the zone name — already implied by the venue location.)

### Step 3: Run test — expect pass

`pytest gazetteer/tests/test_build_sensory_time_map.py::test_tooltip_provenance_line -v`

### Step 4: Run all tests and commit

```bash
pytest gazetteer/tests/ -q
git add gazetteer/build_sensory_time_map.py gazetteer/tests/test_build_sensory_time_map.py
git commit -m "feat: zone provenance line in venue tooltips"
```

---

## Task 7: Zone-aware particle density

**Files:**
- Modify: `gazetteer/build_sensory_time_map.py` — `_buildSmokefield()` and `_buildFlowField()` in HTML_TEMPLATE

Scale particle spawn rates and speed by zone properties at each venue's location.

### Step 1: Write failing test

```python
def test_zone_aware_particles(html):
    """Smoke field builder must use computeZoneBaseline for zone industrial intensity."""
    assert 'zoneProps.industrial_intensity' in html
```

Run: `pytest gazetteer/tests/test_build_sensory_time_map.py::test_zone_aware_particles -v`
Expected: FAIL.

### Step 2: Update `_buildSmokefield()` to use zone industrial intensity

Find `_buildSmokefield()` in HTML_TEMPLATE. It has a particle count section that reads:
```javascript
    const so2_index = smokeRow ? smokeRow.so2_index : 0;
```

After that line, add:
```javascript
    // Zone industrial intensity multiplier (averaged across active venue locations)
    let zoneIndustrialMult = 1.0;
    {{
        const activeVenueLocs = VENUES.filter(v => (venueIntensityCache[v.id] || {{composite:0}}).composite > 0.05);
        if (activeVenueLocs.length) {{
            const avgInd = activeVenueLocs.reduce((sum, v) => {{
                const yr = parseInt(document.getElementById('year-slider').value);
                const zp = getZoneForPoint(v.lat, v.lon);
                if (!zp) return sum;
                const props = interpolateZoneProps(zp, yr);
                return sum + (props.industrial_intensity || 0);
            }}, 0) / activeVenueLocs.length;
            zoneIndustrialMult = 0.7 + avgInd * 0.9;  // range 0.7x – 1.6x
        }}
    }}
```

Then multiply the particle count by `zoneIndustrialMult`. Find the line:
```javascript
    const safeCount = Math.min(count, MAX_P);
```

Change to:
```javascript
    const scaledCount = Math.round(count * zoneIndustrialMult);
    const safeCount = Math.min(scaledCount, MAX_P);
```

And to satisfy the test, add in the smoke boost section a reference to `zoneProps.industrial_intensity`. Find the venue emission loop in `_buildSmokefield()` where venues contribute to the field. Add a zone modifier per venue:

```javascript
            const zoneProps2 = getZoneForPoint(v.lat, v.lon);
            const zoneIndMult = zoneProps2 ? (interpolateZoneProps(zoneProps2, year).industrial_intensity || 0.3) : 0.3;
```

Then multiply the emission strength by `zoneIndMult` where appropriate. Look for the `emission` variable in the smoke field loop and multiply: `* (0.5 + zoneProps2 ? zoneIndMult : 0.5)`.

**Note:** The exact variable names inside `_buildSmokefield` may differ. Read the function carefully before editing. The key invariant is: (a) the test string `zoneProps.industrial_intensity` appears in the HTML, and (b) the particle count is scaled by zone industrial intensity.

### Step 3: Run test — expect pass

`pytest gazetteer/tests/test_build_sensory_time_map.py::test_zone_aware_particles -v`

### Step 4: Run all tests and commit

```bash
pytest gazetteer/tests/ -q
git add gazetteer/build_sensory_time_map.py gazetteer/tests/test_build_sensory_time_map.py
git commit -m "feat: zone-aware particle density — industrial intensity scales smoke spawn rate"
```

---

## Verification

After all tasks:

1. `pytest gazetteer/tests/ -q` — 140+ tests pass
2. `python3 gazetteer/build_sensory_time_map.py` — builds without error
3. Open `sensory_time_map.html`:
   - Year 1660: LON002 (opened 1742) invisible. Year 1750: appears.
   - Zone fills visible as faint colour washes — Smithfield amber, Hyde Park green
   - Click Smithfield area on empty map → side panel shows "SMITHFIELD & NEWGATE" with smell high, noise moderate, provenance lines
   - Click venue marker → venue panel shown, tooltip has provenance line
   - Smoke particle mode: East End noticeably denser than Mayfair
4. No JS console errors

---

## Critical Files

| File | Tasks |
|------|-------|
| `gazetteer/build_sensory_time_map.py` | 0, 2, 3, 4, 5, 6, 7 |
| `gazetteer/zones.json` | 1 |
| `gazetteer/tests/test_build_sensory_time_map.py` | 0, 2, 3, 4, 5, 6, 7 |
