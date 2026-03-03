#!/usr/bin/env python3
"""
Build the self-contained sensory time map HTML.

Reads venues.csv and sensory.db, writes sensory_time_map.html.

Usage:
    python3 gazetteer/build_sensory_time_map.py
    open gazetteer/sensory_time_map.html
"""

import csv
import json
import sqlite3
import urllib.request
import urllib.parse
from pathlib import Path

OHM_CACHE_PATH = Path(__file__).parent / "ohm_streets_cache.json"
OHM_BBOX       = "51.470,-0.200,51.540,-0.050"   # wider than venue spread


def perpendicular_distance(
    point: tuple,
    line_start: tuple,
    line_end: tuple,
) -> float:
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    den = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
    return num / den if den > 0 else 0.0


def douglas_peucker(
    points: list, epsilon: float
) -> list:
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


def parse_ohm_year(date_str) -> int:
    if not date_str:
        return None
    try:
        return int(str(date_str).split("-")[0])
    except (ValueError, AttributeError):
        return None


def parse_ohm_response(data: dict) -> list:
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


def fetch_ohm_streets(cache_path=None) -> list:
    """
    Return pre-1820 London street segments from OpenHistoricalMap.
    Result is cached to ohm_streets_cache.json; subsequent builds are instant.
    Returns [] gracefully if network is unavailable or cache is corrupt.
    """
    if cache_path is None:
        cache_path = OHM_CACHE_PATH
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, Exception):
            print(f"  OHM cache corrupt or unreadable — re-fetching.")

    query = f"""
[out:json][timeout:30];
way["highway"]({OHM_BBOX})["start_date"];
out geom qt;
"""
    url  = "https://overpass-api.openhistoricalmap.org/api/interpreter"
    try:
        data = urllib.parse.urlencode({"data": query}).encode()
        req  = urllib.request.Request(url, data=data, method="POST")
        req.add_header("User-Agent", "DigHums-SensoryMap/1.0")
        with urllib.request.urlopen(req, timeout=35) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        segments = parse_ohm_response(raw)
        cache_path.write_text(json.dumps(segments, separators=(",", ":")), encoding="utf-8")
        print(f"  OHM streets cached: {len(segments)} segments -> {cache_path.name}")
        return segments
    except Exception as exc:
        print(f"  WARNING: OHM fetch failed ({exc}). Building without street network.")
        return []


VENUES_PATH = Path(__file__).parent / "venues.csv"
DB_PATH     = Path(__file__).parent / "sensory.db"
OUT_PATH    = Path(__file__).parent / "sensory_time_map.html"


def load_data(venues_path: Path, db_path: Path) -> dict:
    with open(venues_path, newline="", encoding="utf-8") as f:
        venues = [
            {"id": r["id"], "name": r["name"],
             "lat": float(r["lat"]), "lon": float(r["lon"]),
             "tier": int(r["tier"]) if r.get("tier") else 3,
             "enclosure":     r.get("enclosure", ""),
             "building_type": r.get("building_type", ""),
             "material":      r.get("material", ""),
             "capacity":      r.get("capacity", "")}
            for r in csv.DictReader(f)
        ]

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        events          = [dict(r) for r in conn.execute("SELECT * FROM events")]
        event_venues    = [dict(r) for r in conn.execute("SELECT * FROM event_venues")]
        event_instances = [dict(r) for r in conn.execute("SELECT * FROM event_instances")]
        evidence        = [
            dict(r) for r in conn.execute("""
                SELECT venue_id, source_type, author, title, pub_year,
                       modality, text, valence
                FROM   sensory_evidence
                WHERE  venue_id IS NOT NULL AND venue_id != ''
                ORDER  BY date_min
            """)
        ]

        # ── Environmental layers ──────────────────────────────────────────────
        # CET: {year: {month_int: temp_c}} where 0 = annual mean
        cet: dict[int, dict[int, float]] = {}
        for row in conn.execute(
            "SELECT year, month, temp_c FROM env_temperature "
            "WHERE year BETWEEN 1660 AND 1820"
        ):
            yr = row["year"]
            mo = row["month"] if row["month"] is not None else 0
            cet.setdefault(yr, {})[mo] = row["temp_c"]

        # Mortality: {year: total_burials}
        mortality: dict[int, int] = {
            row["year"]: row["burials"]
            for row in conn.execute(
                "SELECT year, burials FROM env_mortality "
                "WHERE parish = 'ALL_LONDON' ORDER BY year"
            )
        }

        # Smoke: list of {decade_start, coal_tons_k, so2_index}
        smoke = [
            {"decade_start": row["decade_start"],
             "coal_tons_k":  row["coal_tons_k"],
             "so2_index":    row["so2_index"]}
            for row in conn.execute(
                "SELECT decade_start, coal_tons_k, so2_index FROM env_smoke "
                "ORDER BY decade_start"
            )
        ]
    finally:
        conn.close()
    return {
        "venues": venues,
        "events": events,
        "event_venues": event_venues,
        "event_instances": event_instances,
        "evidence": evidence,
        "cet": cet,
        "mortality": mortality,
        "smoke": smoke,
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sensory Time Map \u2014 London 1660\u20131820</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Georgia', serif; background: #f9f6f0; color: #2c2c2c; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }}
#controls {{ background: #1a1a2e; color: #e8e0d0; padding: 10px 16px; flex-shrink: 0; }}
#header-row {{ display: flex; align-items: center; gap: 16px; margin-bottom: 8px; flex-wrap: wrap; }}
.title {{ font-size: 1.1em; letter-spacing: 0.1em; font-weight: bold; }}
#year-control {{ display: flex; align-items: center; gap: 8px; }}
#year-slider {{ width: 200px; }}
#year-display {{ font-size: 1.2em; font-weight: bold; min-width: 3em; text-align: center; }}
.step-btn {{ background: #2a2a4e; border: 1px solid #555; color: #e8e0d0; padding: 2px 8px; cursor: pointer; border-radius: 3px; }}
.step-btn:hover {{ background: #3a3a6e; }}
#play-btn {{ background: #2a2a4e; border: 1px solid #555; color: #e8e0d0; padding: 2px 10px; cursor: pointer; border-radius: 3px; font-size: 0.85em; letter-spacing: 0.05em; }}
#play-btn:hover {{ background: #3a3a6e; }}
#play-btn.playing {{ background: #5a2a0a; border-color: #cc7744; color: #ffd0a0; }}
.nav-link {{ color: #c8b89a; text-decoration: none; font-size: 0.85em; margin-left: auto; }}
.nav-link:hover {{ text-decoration: underline; }}
.pill-row {{ display: flex; gap: 4px; flex-wrap: wrap; margin-bottom: 4px; align-items: center; }}
.pill-label {{ font-size: 0.75em; color: #aaa; margin-right: 4px; min-width: 3em; }}
.pill {{ background: #2a2a4e; border: 1px solid #555; color: #c8b89a; padding: 2px 8px; cursor: pointer; border-radius: 12px; font-size: 0.78em; }}
.pill:hover {{ background: #3a3a6e; }}
.pill.active {{ background: #8b1a1a; border-color: #cc4444; color: #fff; }}
#lit-toggle {{ background: #2a4a2a; border: 1px solid #555; color: #a8d8a8; padding: 2px 10px; cursor: pointer; border-radius: 12px; font-size: 0.78em; }}
#lit-toggle.active {{ background: #1a6a1a; border-color: #44cc44; color: #fff; }}
#main {{ display: flex; flex: 1; overflow: hidden; }}
#map {{ flex: 1; }}
#panel {{ width: 340px; flex-shrink: 0; background: #faf7f2; border-left: 1px solid #ddd; overflow-y: auto; display: flex; flex-direction: column; }}
#panel-header {{ display: flex; align-items: center; background: #2c2c2c; color: #e8e0d0; padding: 10px 14px; font-size: 0.85em; letter-spacing: 0.08em; flex-shrink: 0; }}
#panel-title {{ flex: 1; }}
#panel-body {{ padding: 10px; flex: 1; }}
.event-card {{ background: #fff; border: 1px solid #e0d8d0; border-radius: 4px; padding: 10px 12px; margin-bottom: 8px; }}
.event-card h3 {{ font-size: 0.9em; margin-bottom: 4px; }}
.event-meta {{ font-size: 0.75em; color: #777; margin-bottom: 6px; }}
.intensity-bars {{ display: grid; grid-template-columns: 4em 1fr; gap: 2px 6px; font-size: 0.72em; align-items: center; }}
.bar-track {{ background: #eee; border-radius: 2px; height: 6px; }}
.bar-fill {{ height: 6px; border-radius: 2px; }}
.bar-smell {{ background: #7c6f3e; }}
.bar-noise {{ background: #3e5c7c; }}
.bar-crowd {{ background: #7c3e3e; }}
.bar-visual {{ background: #3e7c4a; }}
.event-notes {{ font-size: 0.72em; color: #666; margin-top: 6px; font-style: italic; border-top: 1px solid #eee; padding-top: 4px; }}
.ev-sources {{ font-size: 0.68em; color: #999; margin-top: 3px; }}
.passage-card {{ background: #fff8f0; border: 1px solid #e0c8a8; border-radius: 4px; padding: 8px 10px; margin-bottom: 6px; font-size: 0.78em; }}
.passage-card .src-badge {{ display: inline-block; font-size: 0.7em; padding: 1px 5px; border-radius: 8px; margin-bottom: 4px; }}
.src-fiction    {{ background: #e0e8f0; color: #1a3a5c; }}
.src-diary      {{ background: #e8f0e0; color: #1a4a1a; }}
.src-topography {{ background: #f0e8d0; color: #5c3a1a; }}
.src-legal      {{ background: #f5e0e0; color: #8b1a1a; }}
.src-poetry     {{ background: #ede0f5; color: #3a1a5c; }}
.src-letters    {{ background: #e0f5f0; color: #1a5c4a; }}
.no-events {{ color: #999; font-style: italic; font-size: 0.85em; padding: 8px 0; }}
.sense-pill {{ background: #1e3a2a; border: 1px solid #555; color: #a8c8a8; padding: 2px 8px; cursor: pointer; border-radius: 12px; font-size: 0.78em; }}
.sense-pill:hover {{ background: #2a4a3a; }}
.btype-pill {{ background: #1a1a2e; border: 1px solid; padding: 2px 8px; cursor: pointer; border-radius: 12px; font-size: 0.76em; opacity: 0.75; transition: opacity 0.15s, background 0.15s; }}
.btype-pill:hover {{ opacity: 1; }}
.btype-pill.active {{ background: rgba(255,255,255,0.12); opacity: 1; font-weight: bold; }}
.particle-btn {{ background: #1a1a2e; border: 1px solid #555; color: #c8b89a; padding: 2px 8px;
                 cursor: pointer; border-radius: 12px; font-size: 0.78em; opacity: 0.75; }}
.particle-btn:hover {{ opacity: 1; }}
.particle-btn.active {{ background: rgba(255,255,255,0.12); border-color: #aaa; color: #fff;
                        opacity: 1; font-weight: bold; }}
.sense-pill[data-sense="smell"].active {{ background: #5c4a10; border-color: #c89230; color: #fff; }}
.sense-pill[data-sense="noise"].active {{ background: #10305c; border-color: #3a7acc; color: #fff; }}
.sense-pill[data-sense="crowd"].active {{ background: #5c1010; border-color: #cc3030; color: #fff; }}
.sense-pill[data-sense="visual"].active {{ background: #10401a; border-color: #30cc50; color: #fff; }}
.prose-summary {{ background: #f5efe4; border-left: 3px solid #8b6914; padding: 8px 12px; margin-bottom: 10px; font-size: 0.83em; line-height: 1.5; color: #3a2a08; border-radius: 0 4px 4px 0; }}
.season-chart {{ display: flex; gap: 2px; margin: 6px 0 2px; }}
.season-bar {{ flex: 1; height: 7px; border-radius: 1px; cursor: pointer; }}
.season-active {{ background: #8b6914; }}
.season-inactive {{ background: #e0d8cc; }}
.season-bar:hover {{ opacity: 0.7; }}
#milestone-label {{ font-size: 0.7em; color: #c8a060; margin-left: 6px; font-style: italic; white-space: nowrap; }}
.speed-btn {{ background: #2a2a4e; border: 1px solid #555; color: #c8b89a; padding: 1px 5px; cursor: pointer; border-radius: 3px; font-size: 0.72em; }}
.speed-btn:hover {{ background: #3a3a6e; }}
.speed-btn.active {{ background: #1a4a2a; border-color: #44aa55; color: #aaffaa; }}
#clear-btn {{ background: none; border: 1px solid #666; color: #a08870; padding: 2px 8px; cursor: pointer; border-radius: 3px; font-size: 0.78em; }}
#clear-btn:hover {{ color: #e8d0b0; border-color: #888; }}
#back-btn {{ background: none; border: none; color: #c8a870; font-size: 0.8em; cursor: pointer; padding: 0 8px 0 0; flex-shrink: 0; }}
#back-btn:hover {{ color: #fff; }}
/* ── Environmental indicators ── */
#env-bar {{ display: flex; align-items: center; gap: 14px; padding: 3px 0 2px; flex-wrap: wrap; }}
.env-gauge {{ display: flex; align-items: center; gap: 5px; font-size: 0.75em; color: #a8a090; }}
.env-gauge .env-label {{ color: #888; }}
.temp-badge {{ background: #1a2a3a; border: 1px solid #3a5a7a; color: #78b0d8; padding: 1px 7px; border-radius: 3px; font-size: 0.82em; font-family: monospace; min-width: 4.5em; text-align: center; transition: background 0.3s, color 0.3s; }}
.temp-badge.cold  {{ background: #0e2040; border-color: #4a88c0; color: #a8d8ff; }}
.temp-badge.frost {{ background: #081828; border-color: #80c0f8; color: #c8eeff; font-weight: bold; }}
.mort-badge {{ background: #2a1010; border: 1px solid #6a2020; color: #e09080; padding: 1px 7px; border-radius: 3px; font-size: 0.82em; font-family: monospace; min-width: 5em; text-align: center; }}
.smoke-gauge {{ display: flex; align-items: center; gap: 5px; font-size: 0.75em; color: #a89060; }}
.smoke-bar-track {{ width: 50px; height: 5px; background: #333; border-radius: 2px; }}
.smoke-bar-fill {{ height: 5px; background: linear-gradient(to right, #c8a050, #8b6020); border-radius: 2px; transition: width 0.4s; }}
#tier-toggle {{ background: #1a2a3a; border: 1px solid #444; color: #8090a8; padding: 1px 8px; cursor: pointer; border-radius: 12px; font-size: 0.75em; }}
#tier-toggle:hover {{ background: #253545; }}
#tier-toggle.active {{ background: #1a3a5a; border-color: #4488cc; color: #aaddff; }}
/* ── Smoke haze overlay on map ── */
#map {{ flex: 1; position: relative; }}
#smoke-overlay {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 400; opacity: 0; background: linear-gradient(to right, rgba(139,119,80,0) 10%, rgba(139,119,80,0.65) 90%); transition: opacity 0.6s ease; }}
/* ── Tier marker colours (used in tier-view mode) ── */
</style>
</head>
<body>
<div id="controls">
  <div id="header-row">
    <span class="title">SENSORY TIME MAP</span>
    <div id="year-control">
      <button class="step-btn" onclick="stepYear(-10)" title="-10 years">&#8592;</button>
      <button id="play-btn" onclick="togglePlay()" title="Animate through years">&#9654; Play</button>
      <span style="display:inline-flex;gap:2px;margin-left:2px">
        <button class="speed-btn" onclick="setSpeed(1000)" title="Slow">&#9664;&#9654;</button>
        <button class="speed-btn active" onclick="setSpeed(500)" title="Normal">&#9654;</button>
        <button class="speed-btn" onclick="setSpeed(200)" title="Fast">&#9654;&#9654;</button>
      </span>
      <input type="range" id="year-slider" min="1660" max="1820" value="1750" list="year-ticks" oninput="updateMap()">
      <datalist id="year-ticks">
        <option value="1688"></option><option value="1707"></option>
        <option value="1752"></option><option value="1775"></option>
        <option value="1780"></option><option value="1783"></option>
        <option value="1789"></option><option value="1803"></option>
      </datalist>
      <span id="year-display">1750</span>
      <span id="milestone-label"></span>
      <button class="step-btn" onclick="stepYear(10)" title="+10 years">&#8594;</button>
    </div>
    <button id="clear-btn" onclick="clearFilters()" title="Clear all filters">&#215; Clear</button>
    <a class="nav-link" href="venue_explorer.html">Browse evidence &#8594;</a>
  </div>
  <div class="pill-row">
    <span class="pill-label">Month</span>
    <button class="pill" data-group="month" data-val="1">Jan</button>
    <button class="pill" data-group="month" data-val="2">Feb</button>
    <button class="pill" data-group="month" data-val="3">Mar</button>
    <button class="pill" data-group="month" data-val="4">Apr</button>
    <button class="pill" data-group="month" data-val="5">May</button>
    <button class="pill" data-group="month" data-val="6">Jun</button>
    <button class="pill" data-group="month" data-val="7">Jul</button>
    <button class="pill" data-group="month" data-val="8">Aug</button>
    <button class="pill" data-group="month" data-val="9">Sep</button>
    <button class="pill" data-group="month" data-val="10">Oct</button>
    <button class="pill" data-group="month" data-val="11">Nov</button>
    <button class="pill" data-group="month" data-val="12">Dec</button>
  </div>
  <div class="pill-row">
    <span class="pill-label">Day</span>
    <button class="pill" data-group="dow" data-val="Mon">Mon</button>
    <button class="pill" data-group="dow" data-val="Tue">Tue</button>
    <button class="pill" data-group="dow" data-val="Wed">Wed</button>
    <button class="pill" data-group="dow" data-val="Thu">Thu</button>
    <button class="pill" data-group="dow" data-val="Fri">Fri</button>
    <button class="pill" data-group="dow" data-val="Sat">Sat</button>
    <button class="pill" data-group="dow" data-val="Sun">Sun</button>
    <button id="lit-toggle" onclick="toggleLiterary()">&#9776; Literary layer: OFF</button>
  </div>
  <div class="pill-row">
    <span class="pill-label">Time</span>
    <button class="pill" data-group="band" data-val="dawn">Dawn</button>
    <button class="pill" data-group="band" data-val="morning">Morning</button>
    <button class="pill" data-group="band" data-val="midday">Midday</button>
    <button class="pill" data-group="band" data-val="afternoon">Afternoon</button>
    <button class="pill" data-group="band" data-val="evening">Evening</button>
    <button class="pill" data-group="band" data-val="night">Night</button>
  </div>
  <div class="pill-row">
    <span class="pill-label">Sense</span>
    <button class="sense-pill" data-sense="smell">Smell</button>
    <button class="sense-pill" data-sense="noise">Noise</button>
    <button class="sense-pill" data-sense="crowd">Crowd</button>
    <button class="sense-pill" data-sense="visual">Visual</button>
    <button id="tier-toggle" onclick="toggleTierView()" title="Colour venues by economic tier">&#9632; Tier</button>
  </div>
  <div class="pill-row" id="btype-row">
    <span class="pill-label">Type</span>
    <button class="btype-pill" data-btype="garden" style="border-color:#4a7c4f;color:#4a7c4f">Garden</button>
    <button class="btype-pill" data-btype="park"   style="border-color:#4a7c4f;color:#4a7c4f">Park</button>
    <button class="btype-pill" data-btype="theatre"  style="border-color:#8b5e3c;color:#8b5e3c">Theatre</button>
    <button class="btype-pill" data-btype="assembly" style="border-color:#8b5e3c;color:#8b5e3c">Assembly</button>
    <button class="btype-pill" data-btype="church"   style="border-color:#6b5b8a;color:#6b5b8a">Church</button>
    <button class="btype-pill" data-btype="market"   style="border-color:#b8860b;color:#b8860b">Market</button>
    <button class="btype-pill" data-btype="square"   style="border-color:#7a7a7a;color:#7a7a7a">Square</button>
    <button class="btype-pill" data-btype="street"   style="border-color:#7a7a7a;color:#7a7a7a">Street</button>
    <button class="btype-pill" data-btype="prison"   style="border-color:#8b0000;color:#8b0000">Prison</button>
    <button class="btype-pill" data-btype="court"    style="border-color:#8b0000;color:#8b0000">Court</button>
    <button class="btype-pill" data-btype="execution" style="border-color:#8b0000;color:#8b0000">Execution</button>
    <button class="btype-pill" data-btype="district" style="border-color:#7a7a7a;color:#7a7a7a">District</button>
  </div>
  <div class="pill-row">
    <span class="pill-label">Particles</span>
    <button class="particle-btn active" data-pmode="off">Off</button>
    <button class="particle-btn" data-pmode="smoke">&#127844; Smoke</button>
    <button class="particle-btn" data-pmode="flow">&#8767; Flow</button>
    <button class="particle-btn" data-pmode="network">&#9780; Network</button>
  </div>
  <div class="pill-row" id="env-bar">
    <span class="env-gauge" title="Central England Temperature (HadCET / Met Office)">
      <span class="env-label">&#127783; Temp</span>
      <span id="temp-badge" class="temp-badge">&#8212;</span>
    </span>
    <span class="env-gauge" title="London burials — Bills of Mortality 1701&#8211;1752 (Death by Numbers)">
      <span class="env-label">&#9760; Burials</span>
      <span id="mort-badge" class="mort-badge">n/a</span>
    </span>
    <span class="smoke-gauge" title="Coal smoke burden (Brimblecombe 1987; Cavert 2016)">
      <span class="env-label">&#127844; Smoke</span>
      <div class="smoke-bar-track"><div class="smoke-bar-fill" id="smoke-bar" style="width:0%"></div></div>
      <span id="smoke-pct" style="font-size:0.82em;font-family:monospace;color:#c0a060">0%</span>
    </span>
  </div>
</div>
<div id="main">
  <div id="map">
    <div id="smoke-overlay"></div>
    <canvas id="particle-canvas" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:410;opacity:0;transition:opacity 0.5s"></canvas>
  </div>
  <div id="panel">
    <div id="panel-header">
      <button id="back-btn" style="display:none" onclick="backToGlobal()">&#8592; All</button>
      <span id="panel-title">ACTIVE EVENTS</span>
    </div>
    <div id="panel-body"></div>
  </div>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const EVENTS = {EVENTS_JSON};
const EVENT_VENUES = {EVENT_VENUES_JSON};
const EVENT_INSTANCES = {EVENT_INSTANCES_JSON};
const VENUES = {VENUES_JSON};
const EVIDENCE = {EVIDENCE_JSON};

// ── Environmental data ─────────────────────────────────────────────────────
const CET_DATA       = {CET_JSON};         // {{year: {{0:annual, 1:jan, ...12:dec}}}}
const MORTALITY_DATA = {MORTALITY_JSON};   // {{year: total_burials}}
const SMOKE_DATA_ENV = {SMOKE_JSON};       // [{{decade_start, so2_index, coal_tons_k}}]
const STREET_NETWORK = {STREET_NETWORK_JSON};  // [[lat,lon],...] pre-1820 streets

// Tier colour palette (1=impoverished → 5=aristocratic)
const TIER_COLORS = {{ 1:'#8b2020', 2:'#c07030', 3:'#5a7a4a', 4:'#3a6090', 5:'#9a7020' }};

// Building type border colours for inactive markers
const BUILDING_TYPE_COLORS = {{
    'garden':    '#4a7c4f',
    'park':      '#4a7c4f',
    'theatre':   '#8b5e3c',
    'assembly':  '#8b5e3c',
    'church':    '#6b5b8a',
    'street':    '#7a7a7a',
    'square':    '#7a7a7a',
    'district':  '#7a7a7a',
    'market':    '#b8860b',
    'prison':    '#8b0000',
    'court':     '#8b0000',
    'execution': '#8b0000',
}};

// Enclosure dash patterns: makes physical form legible on all markers
const ENCLOSURE_DASH = {{
    'open':      null,     // solid stroke
    'semi_open': '4 3',   // dashed
    'enclosed':  '1 3',   // dotted
}};

const EVENTS_BY_ID = Object.fromEntries(EVENTS.map(e => [e.event_id, e]));

const MILESTONES = {{
    1688: 'Glorious Revolution', 1707: 'Acts of Union',
    1752: 'Calendar Reform',     1775: 'American War begins',
    1780: 'Gordon Riots',        1783: 'Tyburn closes',
    1789: 'French Revolution',   1803: 'Ranelagh closes',
}};

let playSpeed = 500;

// Evidence count per venue (for literary badge)
const evidenceCountByVenue = {{}};
EVIDENCE.forEach(p => {{
    if (p.venue_id) evidenceCountByVenue[p.venue_id] = (evidenceCountByVenue[p.venue_id] || 0) + 1;
}});

const state = {{ month: null, dow: null, band: null, literary: false,
                 selectedVenue: null, modality: null, tierView: false,
                 buildingType: null, particleMode: null }};

// ── Smoke haze overlay ──────────────────────────────────────────────────────
// Applied once the map div is present; opacity driven by updateEnvIndicators.
const smokeOverlay = document.getElementById('smoke-overlay');

// ── Environmental indicator update ──────────────────────────────────────────
function updateEnvIndicators(year, month) {{
    // Temperature (HadCET)
    const yearCET = CET_DATA[year];
    const tempBadge = document.getElementById('temp-badge');
    if (yearCET !== undefined) {{
        const temp = (month !== null && yearCET[month] !== undefined)
            ? yearCET[month]
            : yearCET[0];  // 0 = annual mean
        if (temp !== undefined) {{
            tempBadge.textContent = temp.toFixed(1) + '\u00b0C';
            tempBadge.className = 'temp-badge' + (temp < 0 ? ' frost' : temp < 4 ? ' cold' : '');
            tempBadge.title = (temp < 0 ? '\u26ac Frost Fair conditions' : temp < 4 ? '\u26ac Cold' : '') +
                              ' | CET ' + year + (month ? '-' + String(month).padStart(2,'0') : '') +
                              ' | Source: Met Office HadCET';
        }}
    }} else {{
        tempBadge.textContent = '\u2014';
        tempBadge.className = 'temp-badge';
    }}

    // Mortality (Bills of Mortality)
    const mortBadge = document.getElementById('mort-badge');
    const burials = MORTALITY_DATA[year];
    if (burials !== undefined) {{
        mortBadge.textContent = burials.toLocaleString();
        mortBadge.title = 'London burials ' + year + ': ' + burials.toLocaleString() + ' | Source: Death by Numbers';
    }} else {{
        mortBadge.textContent = (year < 1701 || year > 1752) ? 'n/a' : '\u2014';
        mortBadge.title = year < 1701 || year > 1752 ? 'Bills of Mortality data: 1701\u20131752 only' : '';
    }}

    // Smoke (decade-level coal burden)
    const decadeStart = Math.floor(year / 10) * 10;
    const smokeRow = SMOKE_DATA_ENV.find(s => s.decade_start === decadeStart);
    const smokeBar  = document.getElementById('smoke-bar');
    const smokePct  = document.getElementById('smoke-pct');
    if (smokeRow) {{
        const pct = Math.round(smokeRow.so2_index * 100);
        if (smokeBar)  smokeBar.style.width  = pct + '%';
        if (smokePct)  smokePct.textContent  = pct + '%';
        // Haze overlay: hidden when so2_index = 0 (pre-coal or no data)
        if (smokeOverlay) smokeOverlay.style.opacity = smokeRow.so2_index > 0
            ? (smokeRow.so2_index * 0.28).toFixed(3) : '0';
    }} else {{
        if (smokeOverlay) smokeOverlay.style.opacity = '0';
    }}
}}

function toggleTierView() {{
    state.tierView = !state.tierView;
    const btn = document.getElementById('tier-toggle');
    if (btn) {{ btn.classList.toggle('active', state.tierView); }}
    updateMap();
}}

// Map setup
const map = L.map('map', {{ zoomControl: true }}).setView([51.508, -0.13], 13);
const baseLayers = {{
  'Modern (CartoDB)': L.tileLayer(
    'https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',
    {{ attribution: '&copy; OpenStreetMap contributors &copy; CARTO', maxZoom: 19 }}
  ).addTo(map),
  'Rocque 1746': L.tileLayer(
    'https://www.dhi.ac.uk/san/llptiles/molarocque/{{z}}/{{x}}/{{y}}.png',
    {{ attribution: 'Map tiles &copy; Museum of London Archaeology / DHI, based on John Rocque 1746',
       minZoom: 13, maxZoom: 15, maxNativeZoom: 15, opacity: 0.9 }}
  ),
  'Horwood 1792\u201399': L.tileLayer(
    'https://www.romanticlondon.org/horwoodplan/{{z}}/{{x}}/{{y}}.png',
    {{ attribution: 'Map tiles &copy; Romantic London project, based on Richard Horwood 1792\u201399',
       minZoom: 11, maxZoom: 17, maxNativeZoom: 16, opacity: 0.9 }}
  ),
}};
L.control.layers(baseLayers, {{}}, {{ collapsed: false }}).addTo(map);

// ── Procession routes ─────────────────────────────────────────────────────
// Shown as dashed polylines when the relevant event is temporally active.
// Coordinates trace the historical street route, not just the endpoint venues.
const PROCESSION_ROUTES = {{
  'EVT032': {{
    // Tyburn Procession: Newgate Prison → Holborn → St Giles → Oxford Street → Tyburn Tree
    // The condemned rode in a cart; the journey took up to 3 hours. Crowds lined the
    // whole route. Source: Old Bailey Online; Linebaugh London Hanged 1992.
    coords: [
      [51.5152, -0.1016],  // Newgate Prison (Old Bailey)
      [51.5173, -0.1028],  // Giltspur Street heading north
      [51.5189, -0.1065],  // Holborn Viaduct east end
      [51.5184, -0.1148],  // High Holborn
      [51.5165, -0.1264],  // St Giles Circus (Centrepoint)
      [51.5163, -0.1304],  // Oxford St / Tottenham Court Rd
      [51.5154, -0.1416],  // Oxford Circus
      [51.5136, -0.1497],  // Bond Street
      [51.5131, -0.1589],  // Marble Arch / Tyburn Tree
    ],
    color: '#8b1a1a', dashArray: '7 5', weight: 3,
    label: 'Tyburn Procession (1660\u20131783): Newgate \u2192 Marble Arch, 3 miles',
  }},
  'EVT050': {{
    // Lord Mayor\u2019s Show: Guildhall \u2192 Cheapside \u2192 Ludgate Hill \u2192 Fleet St \u2192 Temple Bar
    // Route varied by ward of new Mayor; this traces the core ceremonial spine.
    // Source: Lord Mayor\u2019s Show official history; Ian Visits.
    coords: [
      [51.5155, -0.0924],  // Guildhall
      [51.5131, -0.0890],  // Bank Junction / Mansion House
      [51.5135, -0.0955],  // Cheapside
      [51.5138, -0.0984],  // St Paul's Cathedral
      [51.5125, -0.1040],  // Ludgate Circus
      [51.5134, -0.1071],  // Fleet Street mid
      [51.5133, -0.1127],  // Temple Bar / Royal Courts
    ],
    color: '#1a3a8b', dashArray: '10 5', weight: 3,
    label: "Lord Mayor\u2019s Show: Guildhall \u2192 Temple Bar",
  }},
}};

const routeLines = {{}};
Object.entries(PROCESSION_ROUTES).forEach(([evtId, route]) => {{
    routeLines[evtId] = L.polyline(route.coords, {{
        color: route.color, weight: route.weight,
        opacity: 0.85, dashArray: route.dashArray,
    }}).bindTooltip(route.label, {{ sticky: true, direction: 'top' }});
}});

const markersByVenueId = {{}};
VENUES.forEach(v => {{
    const hasEvidence = (evidenceCountByVenue[v.id] || 0) > 0;
    const btColor   = BUILDING_TYPE_COLORS[v.building_type] || '#666';
    const dashArray = ENCLOSURE_DASH[v.enclosure] || null;
    const m = L.circleMarker([v.lat, v.lon], {{
        radius: 4,
        fillColor: '#aaa',
        color: hasEvidence ? btColor : '#888',
        fillOpacity: 0.35,
        weight: hasEvidence ? 1.5 : 0.8,
        dashArray: dashArray,
    }}).addTo(map);
    m.bindTooltip('', {{ permanent: false, direction: 'top' }});
    m.on('click', () => selectVenue(v.id));
    markersByVenueId[v.id] = m;
}});

// Sensory spread rings: pale fill circles showing approximate reach
// Smell max ~400m, noise max ~300m (period sources: Howard 1777; Defoe 1724)
const smellRings = {{}};
const noiseRings = {{}};
VENUES.forEach(v => {{
    smellRings[v.id] = L.circle([v.lat, v.lon], {{
        radius: 0, weight: 0,
        fillColor: '#9c8a3e', fillOpacity: 0.08, interactive: false,
    }}).addTo(map);
    noiseRings[v.id] = L.circle([v.lat, v.lon], {{
        radius: 0, weight: 0,
        fillColor: '#3e6a9c', fillOpacity: 0.07, interactive: false,
    }}).addTo(map);
}});

function computeIntensity(venueId, year, month, dow, band) {{
    const loads = {{smell: 0, noise: 0, crowd: 0, visual: 0}};

    // ── Architectural exposure lookup ──────────────────────────────────────
    const _venue = VENUES.find(v => v.id === venueId);
    const enc = _venue ? (_venue.enclosure || 'open') : 'open';
    const mat = _venue ? (_venue.material  || 'outdoor') : 'outdoor';
    // Exposure multipliers: how much of the external environment penetrates
    const thermalMult = enc === 'open' ? 1.0 : enc === 'semi_open' ? 0.6 : 0.2;
    const smokeMult   = enc === 'open' ? 1.0 : enc === 'semi_open' ? 0.8 : 0.4;
    // Stone reverberation boosts noise in enclosed/semi-open stone buildings
    const reverbBonus = (mat === 'stone' && enc !== 'open') ? 0.08
                      : (mat === 'timber' && enc === 'enclosed') ? 0.04 : 0;

    EVENT_VENUES.forEach(ev => {{
        if (ev.venue_id !== venueId) return;
        const evt = EVENTS_BY_ID[ev.event_id];
        if (!evt) return;

        // Irregular/one_off events only appear via EVENT_INSTANCES (specific years)
        if (evt.recurrence === 'irregular' || evt.recurrence === 'one_off') return;

        // Year range
        const ys = evt.year_start, ye = evt.year_end;
        if (ys !== null && year < ys) return;
        if (ye !== null && year > ye) return;

        // Month (with calendar reform)
        if (month !== null && evt.month_start !== null) {{
            let mStart = evt.month_start;
            if (evt.calendar_break && evt.month_start_ns && year >= evt.calendar_break) {{
                mStart = evt.month_start_ns;
            }}
            const mEnd = evt.month_end !== null ? evt.month_end : mStart;
            if (month < mStart || month > mEnd) return;
        }}

        // Day of week
        if (dow !== null && evt.day_of_week) {{
            const days = evt.day_of_week.split('|');
            if (!days.includes(dow)) return;
        }}

        // Time band
        if (band !== null && evt.time_bands) {{
            const bands = evt.time_bands.split('|');
            if (!bands.includes(band)) return;
        }}

        loads.smell  = Math.min(1, loads.smell  + (evt.smell_load  || 0));
        loads.noise  = Math.min(1, loads.noise  + (evt.noise_load  || 0));
        loads.crowd  = Math.min(1, loads.crowd  + (evt.crowd_load  || 0));
        loads.visual = Math.min(1, loads.visual + (evt.visual_load || 0));
    }});

    // Event instances for specific year
    EVENT_INSTANCES.forEach(inst => {{
        if (inst.year !== year) return;
        if (month !== null && inst.month !== null && inst.month !== month) return;
        const linked = EVENT_VENUES.some(ev => ev.event_id === inst.event_id && ev.venue_id === venueId);
        if (!linked) return;
        const evt = EVENTS_BY_ID[inst.event_id];
        if (!evt) return;
        loads.smell  = Math.min(1, loads.smell  + (evt.smell_load  || 0));
        loads.noise  = Math.min(1, loads.noise  + (evt.noise_load  || 0));
        loads.crowd  = Math.min(1, loads.crowd  + (evt.crowd_load  || 0));
        loads.visual = Math.min(1, loads.visual + (evt.visual_load || 0));
    }});

    // ── D5: Evidence density boost ────────────────────────────────────────
    // Count passages linked to this venue within ±15 years of current year.
    const evidenceAtVenue = EVIDENCE.filter(ev =>
        ev.venue_id === venueId
        && (ev.pub_year === null
            || (ev.pub_year >= year - 15 && ev.pub_year <= year + 15))
    ).length;
    // Logarithmic boost: +0.06 per passage (capped at +0.25)
    const densityBoost = Math.min(0.25, Math.log1p(evidenceAtVenue) * 0.05);

    // ── D5: Environmental modifiers ───────────────────────────────────────
    // Cold winter: monthly temp below 3°C boosts thermal load
    let thermalBoost = 0;
    if (month !== null) {{
        const yearData = CET_DATA[year] || CET_DATA[String(year)];
        const temp = yearData ? (yearData[month] ?? yearData[String(month)] ?? null) : null;
        if (temp !== null && temp < 3.0) {{
            // Scale: 3°C → 0, -3°C → 0.25
            thermalBoost = Math.min(0.25, Math.max(0, (3.0 - temp) / 24));
        }}
    }}
    // Smoke burden boosts olfactory baseline proportionally
    let smokeBoost = 0;
    if (SMOKE_DATA_ENV.length > 0) {{
        const decade = Math.floor(year / 10) * 10;
        const smokeRow = SMOKE_DATA_ENV.find(s => s.decade_start === decade)
                      || SMOKE_DATA_ENV[SMOKE_DATA_ENV.length - 1];
        smokeBoost = (smokeRow ? smokeRow.so2_index : 0) * 0.12;
    }}

    loads.smell  = Math.min(1, loads.smell  + densityBoost * 0.6 + smokeBoost * smokeMult);
    loads.noise  = Math.min(1, loads.noise  + densityBoost * 0.4 + reverbBonus);
    loads.crowd  = Math.min(1, loads.crowd  + densityBoost * 0.4);
    loads.visual = Math.min(1, loads.visual + densityBoost * 0.4 + thermalBoost * thermalMult);

    loads.composite = (loads.smell + loads.noise + loads.crowd + loads.visual) / 4;
    return loads;
}}

function intensityColour(c) {{
    if (c < 0.01) return '#aaaaaa';
    if (c < 0.3)  return '#f59e0b';
    if (c < 0.6)  return '#f97316';
    return '#dc2626';
}}

function bar(label, value, cls) {{
    const pct = Math.round(value * 100);
    return `<div class="bar-label">${{label}}</div>
            <div class="bar-track"><div class="bar-fill ${{cls}}" style="width:${{pct}}%"></div></div>`;
}}

function renderEventCard(evt, inst) {{
    const instNote = inst ? `<div class="ev-sources">Instance ${{inst.instance_id}}: ${{inst.notes || ''}}</div>` : '';
    return `<div class="event-card">
      <h3>${{evt.name}}</h3>
      <div class="event-meta">${{evt.category.replace(/_/g,' ')}} &middot; ${{evt.recurrence}}</div>
      <div class="intensity-bars">
        ${{bar('smell', evt.smell_load||0, 'bar-smell')}}
        ${{bar('noise', evt.noise_load||0, 'bar-noise')}}
        ${{bar('crowd', evt.crowd_load||0, 'bar-crowd')}}
        ${{bar('visual',evt.visual_load||0,'bar-visual')}}
      </div>
      ${{seasonChart(evt)}}
      ${{evt.notes ? `<div class="event-notes">${{evt.notes}}</div>` : ''}}
      ${{evt.sources ? `<div class="ev-sources">Sources: ${{evt.sources}}</div>` : ''}}
      ${{instNote}}
    </div>`;
}}

function renderPassageCard(p) {{
    const srcCls = 'src-' + (p.source_type||'fiction');
    return `<div class="passage-card">
      <span class="src-badge ${{srcCls}}">${{p.source_type}}</span>
      <strong>${{p.author}}</strong> &mdash; <em>${{p.title}}</em> (${{p.pub_year||''}})
      <div style="margin-top:4px;color:#333">${{p.text ? p.text.substring(0,200) : ''}}&hellip;</div>
    </div>`;
}}

function seasonChart(evt) {{
    const abbr = ['J','F','M','A','M','J','J','A','S','O','N','D'];
    const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    let ms = evt.month_start, me = evt.month_end;
    if (ms === null) {{
        return '<div class="season-chart">' + abbr.map((a, i) =>
            `<div class="season-bar season-active" title="Click to filter: ${{months[i]}}" onclick="setMonth(${{i+1}})"></div>`
        ).join('') + '</div>';
    }}
    if (me === null) me = ms;
    return '<div class="season-chart">' + abbr.map((a, i) => {{
        const active = (i + 1) >= ms && (i + 1) <= me;
        return `<div class="season-bar ${{active ? 'season-active' : 'season-inactive'}}" title="Click to filter: ${{months[i]}}" onclick="setMonth(${{i+1}})"></div>`;
    }}).join('') + '</div>';
}}

function proseSummary(venueId, evts, year) {{
    if (!evts.length) return '';
    const venue = VENUES.find(v => v.id === venueId);
    const place = venue ? venue.name : 'this place';
    const loads = {{smell: 0, noise: 0, crowd: 0, visual: 0}};
    evts.forEach(e => {{
        loads.smell  = Math.min(1, loads.smell  + (e.evt.smell_load  || 0));
        loads.noise  = Math.min(1, loads.noise  + (e.evt.noise_load  || 0));
        loads.crowd  = Math.min(1, loads.crowd  + (e.evt.crowd_load  || 0));
        loads.visual = Math.min(1, loads.visual + (e.evt.visual_load || 0));
    }});
    const ranked = Object.entries(loads).sort((a, b) => b[1] - a[1]).filter(x => x[1] > 0.05);
    if (!ranked.length) return '';
    const descs = {{
        smell: ['overwhelmed by powerful odours', 'heavy with the smell of livestock and refuse', 'thick with distinctive smells'],
        noise: ['filled with clamour and din', 'loud with shouts, bells, and the crush of people', 'alive with noise'],
        crowd: ['thronged with people', 'dense with crowds pressing through the streets', 'busy with passersby and traders'],
        visual: ['a striking and disordered spectacle', 'a remarkable sight', 'visually arresting'],
    }};
    const intensityWord = ranked[0][1] > 0.8 ? 'overwhelmingly' : ranked[0][1] > 0.5 ? 'markedly' : 'noticeably';
    let sentence = `In ${{year}}, standing at ${{place}}, you would have found it ${{intensityWord}} ${{descs[ranked[0][0]][0]}}`;
    if (ranked.length > 1 && ranked[1][1] > 0.3) {{
        sentence += `, and ${{descs[ranked[1][0]][1]}}`;
    }}
    sentence += '.';
    return `<div class="prose-summary">${{sentence}}</div>`;
}}

// Cache of last computed intensities (populated by updateMap)
const venueIntensityCache = {{}};

function updateMap() {{
    const year  = parseInt(document.getElementById('year-slider').value);
    const month = state.month;
    const dow   = state.dow;
    const band  = state.band;
    document.getElementById('year-display').textContent = year;
    updateEnvIndicators(year, month);

    // Milestone label
    const ml = document.getElementById('milestone-label');
    if (ml) ml.textContent = MILESTONES[year] ? '\u00b7 ' + MILESTONES[year] : '';

    // URL hash state (for sharing/bookmarking)
    const hashParts = ['y=' + year];
    if (month) hashParts.push('m=' + month);
    if (dow)   hashParts.push('dow=' + dow);
    if (band)  hashParts.push('band=' + band);
    history.replaceState(null, '', '#' + hashParts.join('&'));

    let activeEvents = [];

    VENUES.forEach(v => {{
        const intensity = computeIntensity(v.id, year, month, dow, band);
        venueIntensityCache[v.id] = intensity;
        const marker = markersByVenueId[v.id];
        if (!marker) return;

        // If a sense filter is active, drive display from that modality alone
        const displayLoad = state.modality ? (intensity[state.modality] || 0) : intensity.composite;
        const col = intensityColour(displayLoad);
        const r = displayLoad < 0.01 ? 4 : 4 + displayLoad * 14;

        marker.setRadius(r);
        const hasEvidence = (evidenceCountByVenue[v.id] || 0) > 0;
        const tierCol    = TIER_COLORS[v.tier] || '#888';
        const btColor    = BUILDING_TYPE_COLORS[v.building_type] || '#666';
        const dashArray  = ENCLOSURE_DASH[v.enclosure] || null;
        // Building-type filter dimming
        const btDimmed = state.buildingType && v.building_type !== state.buildingType;
        if (btDimmed) {{
            marker.setStyle({{
                fillColor: '#aaa', color: '#666',
                fillOpacity: 0.08, weight: 0.5, dashArray: dashArray,
            }});
        }} else if (displayLoad < 0.01) {{
            if (state.tierView) {{
                marker.setStyle({{
                    fillColor: tierCol, color: '#ffffff',
                    fillOpacity: 0.65, weight: 1.2, dashArray: dashArray,
                }});
            }} else {{
                marker.setStyle({{
                    fillColor: '#aaa', color: hasEvidence ? btColor : '#888',
                    fillOpacity: 0.3, weight: hasEvidence ? 1.5 : 0.8,
                    dashArray: dashArray,
                }});
            }}
        }} else {{
            marker.setStyle({{
                fillColor: col, color: state.tierView ? tierCol : col,
                fillOpacity: 0.78, weight: state.tierView ? 2 : 1,
                dashArray: dashArray,
            }});
        }}
        marker._intensity = intensity;

        // Rich tooltip: name + architectural metadata + per-modality breakdown + evidence count
        const evCount = evidenceCountByVenue[v.id] || 0;
        let tip = `<strong>${{v.name}}</strong>`;
        // Architectural metadata line
        const metaParts = [v.enclosure, v.building_type, v.material, v.capacity].filter(Boolean);
        if (metaParts.length) {{
            tip += `<br><span style="font-size:0.80em;opacity:0.55;font-style:italic">${{metaParts.join(' \u00b7 ')}}</span>`;
        }}
        if (intensity.composite > 0.01) {{
            const parts = [];
            if (intensity.smell  > 0.01) parts.push(`smell ${{Math.round(intensity.smell*100)}}%`);
            if (intensity.noise  > 0.01) parts.push(`noise ${{Math.round(intensity.noise*100)}}%`);
            if (intensity.crowd  > 0.01) parts.push(`crowd ${{Math.round(intensity.crowd*100)}}%`);
            if (intensity.visual > 0.01) parts.push(`visual ${{Math.round(intensity.visual*100)}}%`);
            if (parts.length) tip += '<br><span style="font-size:0.88em;opacity:0.85">' + parts.join(' &middot; ') + '</span>';
        }}
        if (evCount > 0) tip += `<br><span style="font-size:0.82em;opacity:0.7">&#128214; ${{evCount}} passages</span>`;
        marker.setTooltipContent(tip);

        // Spread rings
        if (smellRings[v.id]) smellRings[v.id].setRadius(intensity.smell * 400);
        if (noiseRings[v.id]) noiseRings[v.id].setRadius(intensity.noise * 300);

        if (intensity.composite > 0.01) {{
            const evts = getActiveEvents(v.id, year, month, dow, band);
            evts.forEach(e => {{
                if (!activeEvents.find(x => x.evt.event_id === e.evt.event_id && x.venueId === v.id))
                    activeEvents.push({{...e, venueId: v.id, venueName: v.name}});
            }});
        }}
    }});

    // Show/hide procession route polylines
    Object.entries(routeLines).forEach(([evtId, line]) => {{
        if (isEventActive(evtId, year, month, dow, band)) {{
            if (!map.hasLayer(line)) line.addTo(map);
        }} else {{
            if (map.hasLayer(line)) map.removeLayer(line);
        }}
    }});

    if (state.selectedVenue) {{
        renderVenuePanel(state.selectedVenue, year, month, dow, band);
    }} else {{
        renderGlobalPanel(activeEvents);
    }}

    if (state.particleMode && state.particleMode !== 'off') {{
        _scheduleFieldUpdate();
    }}
}}

function getActiveEvents(venueId, year, month, dow, band) {{
    const results = [];
    EVENT_VENUES.forEach(ev => {{
        if (ev.venue_id !== venueId) return;
        const evt = EVENTS_BY_ID[ev.event_id];
        if (!evt) return;
        // Irregular/one_off events only appear via EVENT_INSTANCES (specific years)
        if (evt.recurrence === 'irregular' || evt.recurrence === 'one_off') return;
        const ys = evt.year_start, ye = evt.year_end;
        if (ys !== null && year < ys) return;
        if (ye !== null && year > ye) return;
        if (month !== null && evt.month_start !== null) {{
            let mStart = evt.month_start;
            if (evt.calendar_break && evt.month_start_ns && year >= evt.calendar_break) mStart = evt.month_start_ns;
            const mEnd = evt.month_end !== null ? evt.month_end : mStart;
            if (month < mStart || month > mEnd) return;
        }}
        if (dow !== null && evt.day_of_week) {{
            if (!evt.day_of_week.split('|').includes(dow)) return;
        }}
        if (band !== null && evt.time_bands) {{
            if (!evt.time_bands.split('|').includes(band)) return;
        }}
        const inst = EVENT_INSTANCES.find(i => i.event_id === evt.event_id && i.year === year);
        results.push({{evt, inst: inst || null}});
    }});
    EVENT_INSTANCES.forEach(inst => {{
        if (inst.year !== year) return;
        if (month !== null && inst.month !== null && inst.month !== month) return;
        if (!EVENT_VENUES.some(ev => ev.event_id === inst.event_id && ev.venue_id === venueId)) return;
        const evt = EVENTS_BY_ID[inst.event_id];
        if (!evt || results.find(r => r.evt.event_id === inst.event_id)) return;
        results.push({{evt, inst}});
    }});
    return results;
}}

function renderGlobalPanel(activeEvents) {{
    document.getElementById('back-btn').style.display = 'none';
    document.getElementById('panel-title').textContent =
        activeEvents.length ? `ACTIVE EVENTS (${{activeEvents.length}})` : 'ACTIVE EVENTS';
    const body = document.getElementById('panel-body');
    if (!activeEvents.length) {{
        body.innerHTML = '<p class="no-events">No events active at this time. Try selecting a different month, day, or band.</p>';
        return;
    }}
    body.innerHTML = activeEvents.map(e =>
        `<div style="font-size:0.72em;color:#777;margin-bottom:2px">&#8250; ${{e.venueName}}</div>` +
        renderEventCard(e.evt, e.inst)
    ).join('');
}}

function renderVenuePanel(venueId, year, month, dow, band) {{
    const venue = VENUES.find(v => v.id === venueId);
    const body = document.getElementById('panel-body');
    const evCount = evidenceCountByVenue[venueId] || 0;
    const evBadge = evCount > 0 ? ` <span style="font-size:0.75em;opacity:0.7">&#128214; ${{evCount}}</span>` : '';
    document.getElementById('back-btn').style.display = '';
    document.getElementById('panel-title').innerHTML =
        (venue ? venue.name : 'VENUE').toUpperCase() + evBadge;

    const evts = getActiveEvents(venueId, year, month, dow, band);
    let html = proseSummary(venueId, evts, year);
    html += evts.length
        ? evts.map(e => renderEventCard(e.evt, e.inst)).join('')
        : '<p class="no-events">No institutional events active at this time.</p>';

    if (state.literary) {{
        const passages = EVIDENCE.filter(p => p.venue_id === venueId);
        if (passages.length) {{
            html += `<div style="margin:10px 0 4px;font-size:0.75em;letter-spacing:0.05em;color:#555">LITERARY EVIDENCE (${{passages.length}} passages)</div>`;
            html += passages.slice(0, 20).map(renderPassageCard).join('');
        }}
    }}

    body.innerHTML = html;
}}

function selectVenue(venueId) {{
    state.selectedVenue = venueId;
    const year  = parseInt(document.getElementById('year-slider').value);
    renderVenuePanel(venueId, year, state.month, state.dow, state.band);
}}

function toggleLiterary() {{
    state.literary = !state.literary;
    const btn = document.getElementById('lit-toggle');
    btn.textContent = '\u2630 Literary layer: ' + (state.literary ? 'ON' : 'OFF');
    btn.classList.toggle('active', state.literary);
    if (state.selectedVenue) {{
        const year = parseInt(document.getElementById('year-slider').value);
        renderVenuePanel(state.selectedVenue, year, state.month, state.dow, state.band);
    }}
}}

function stepYear(delta) {{
    const s = document.getElementById('year-slider');
    s.value = Math.max(1660, Math.min(1820, parseInt(s.value) + delta));
    updateMap();
}}

// ── Animated playback ────────────────────────────────────────────────────
let playInterval = null;

function togglePlay() {{
    const btn = document.getElementById('play-btn');
    if (playInterval) {{
        clearInterval(playInterval);
        playInterval = null;
        btn.innerHTML = '&#9654; Play';
        btn.classList.remove('playing');
        btn.title = 'Animate through years';
    }} else {{
        btn.innerHTML = '&#9646;&#9646; Pause';
        btn.classList.add('playing');
        btn.title = 'Pause';
        playInterval = setInterval(() => {{
            const s = document.getElementById('year-slider');
            const next = parseInt(s.value) + 1;
            if (next > 1820) {{
                clearInterval(playInterval);
                playInterval = null;
                btn.innerHTML = '&#9654; Play';
                btn.classList.remove('playing');
                return;
            }}
            s.value = next;
            updateMap();
        }}, playSpeed);
    }}
}}

// Check whether an event is temporally active under the current filter state
function isEventActive(evtId, year, month, dow, band) {{
    const evt = EVENTS_BY_ID[evtId];
    if (!evt) return false;
    const ys = evt.year_start, ye = evt.year_end;
    if (ys !== null && year < ys) return false;
    if (ye !== null && year > ye) return false;
    if (month !== null && evt.month_start !== null) {{
        let mStart = evt.month_start;
        if (evt.calendar_break && evt.month_start_ns && year >= evt.calendar_break) mStart = evt.month_start_ns;
        const mEnd = evt.month_end !== null ? evt.month_end : mStart;
        if (month < mStart || month > mEnd) return false;
    }}
    if (dow !== null && evt.day_of_week) {{
        if (!evt.day_of_week.split('|').includes(dow)) return false;
    }}
    if (band !== null && evt.time_bands) {{
        if (!evt.time_bands.split('|').includes(band)) return false;
    }}
    return true;
}}

document.querySelectorAll('.pill[data-group]').forEach(btn => {{
    btn.addEventListener('click', () => {{
        const group = btn.dataset.group;
        const val = btn.dataset.val;
        const wasActive = btn.classList.contains('active');
        document.querySelectorAll(`.pill[data-group="${{group}}"]`).forEach(b => b.classList.remove('active'));
        if (!wasActive) {{
            btn.classList.add('active');
            state[group] = group === 'month' ? parseInt(val) : val;
        }} else {{
            state[group] = null;
        }}
        updateMap();
    }});
}});

document.querySelectorAll('.sense-pill').forEach(btn => {{
    btn.addEventListener('click', () => {{
        const sense = btn.dataset.sense;
        const wasActive = btn.classList.contains('active');
        document.querySelectorAll('.sense-pill').forEach(b => b.classList.remove('active'));
        state.modality = wasActive ? null : sense;
        if (!wasActive) btn.classList.add('active');
        updateMap();
    }});
}});

document.querySelectorAll('.btype-pill').forEach(btn => {{
    btn.addEventListener('click', () => {{
        const btype = btn.dataset.btype;
        const wasActive = btn.classList.contains('active');
        document.querySelectorAll('.btype-pill').forEach(b => b.classList.remove('active'));
        state.buildingType = wasActive ? null : btype;
        if (!wasActive) btn.classList.add('active');
        updateMap();
    }});
}});

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

function clearFilters() {{
    if (_fieldUpdateTimer) {{ clearTimeout(_fieldUpdateTimer); _fieldUpdateTimer = null; }}
    state.month = null; state.dow = null; state.band = null; state.modality = null; state.buildingType = null;
    state.particleMode = null;
    document.querySelectorAll('.pill.active, .sense-pill.active, .btype-pill.active').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.particle-btn').forEach(b => b.classList.remove('active'));
    document.querySelector('.particle-btn[data-pmode="off"]')?.classList.add('active');
    stopParticles();
    updateMap();
}}

function setSpeed(ms) {{
    playSpeed = ms;
    const speeds = [1000, 500, 200];
    document.querySelectorAll('.speed-btn').forEach((b, i) =>
        b.classList.toggle('active', speeds[i] === ms));
    if (playInterval) {{
        clearInterval(playInterval);
        const btn = document.getElementById('play-btn');
        playInterval = setInterval(() => {{
            const s = document.getElementById('year-slider');
            const next = parseInt(s.value) + 1;
            if (next > 1820) {{
                clearInterval(playInterval); playInterval = null;
                btn.innerHTML = '&#9654; Play'; btn.classList.remove('playing');
                return;
            }}
            s.value = next; updateMap();
        }}, playSpeed);
    }}
}}

function backToGlobal() {{
    state.selectedVenue = null;
    updateMap();
}}

function setMonth(m) {{
    const wasActive = state.month === m;
    document.querySelectorAll('.pill[data-group="month"]').forEach(b => b.classList.remove('active'));
    if (!wasActive) {{
        state.month = m;
        const pill = document.querySelector(`.pill[data-group="month"][data-val="${{m}}"]`);
        if (pill) pill.classList.add('active');
    }} else {{
        state.month = null;
    }}
    updateMap();
}}

// ── Keyboard shortcuts ────────────────────────────────────────────────────
// ← / → : step year by ±1  |  Space : toggle play  |  1–9, 0 : set month
document.addEventListener('keydown', (e) => {{
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key === 'ArrowLeft')  {{ e.preventDefault(); stepYear(-1); return; }}
    if (e.key === 'ArrowRight') {{ e.preventDefault(); stepYear(1);  return; }}
    if (e.key === ' ' || e.key === 'Spacebar') {{ e.preventDefault(); togglePlay(); return; }}
    const n = parseInt(e.key);
    if (!isNaN(n)) setMonth(n === 0 ? 10 : n);
}});

// ── Restore state from URL hash ───────────────────────────────────────────
(function() {{
    const hash = location.hash.replace('#', '');
    if (!hash) return;
    const params = Object.fromEntries(hash.split('&').filter(p => p.includes('=')).map(p => p.split('=')));
    if (params.y) {{
        const y = parseInt(params.y);
        if (y >= 1660 && y <= 1820) document.getElementById('year-slider').value = y;
    }}
    if (params.m) {{
        const m = parseInt(params.m);
        if (m >= 1 && m <= 12) {{
            state.month = m;
            document.querySelector(`.pill[data-group="month"][data-val="${{m}}"]`)?.classList.add('active');
        }}
    }}
    if (params.dow) {{
        const days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
        if (days.includes(params.dow)) {{
            state.dow = params.dow;
            document.querySelector(`.pill[data-group="dow"][data-val="${{params.dow}}"]`)?.classList.add('active');
        }}
    }}
    if (params.band) {{
        const bands = ['dawn','morning','midday','afternoon','evening','night'];
        if (bands.includes(params.band)) {{
            state.band = params.band;
            document.querySelector(`.pill[data-group="band"][data-val="${{params.band}}"]`)?.classList.add('active');
        }}
    }}
}})();

// ── Particle System ──────────────────────────────────────────────────────────
const pCanvas = document.getElementById('particle-canvas');
const pCtx    = pCanvas ? pCanvas.getContext('2d') : null;

// Declare shared state before resizeParticleCanvas (called immediately below)
const venuePx = {{}};           // venue pixel positions, refreshed on move/resize
let particleRaf = null;         // RAF handle; null when stopped
let activeParticleCount = 0;    // how many particles are active this frame

// Resize canvas to match map container
function resizeParticleCanvas() {{
    if (!pCanvas) return;
    const mapEl = document.getElementById('map');
    pCanvas.width  = mapEl.offsetWidth;
    pCanvas.height = mapEl.offsetHeight;
    updateVenuePx();
    if (activeParticleCount > 0) resetParticles();
}}
resizeParticleCanvas();
window.addEventListener('resize', resizeParticleCanvas);

// Particle state — simple object array (2 000 particles max)
const MAX_P = 2000;
const particles = [];

function spawnParticle() {{
    return {{
        px: 0, py: 0,
        vx: 0, vy: 0,
        age: 0, maxAge: 200 + Math.random() * 400,
        r: 180, g: 130, b: 40,
    }};
}}

for (let i = 0; i < MAX_P; i++) particles.push(spawnParticle());

// Vector field grid
const FIELD_W = 80, FIELD_H = 60;
const fieldDx = new Float32Array(FIELD_W * FIELD_H);
const fieldDy = new Float32Array(FIELD_W * FIELD_H);

// Cache of venue pixel positions (declared early — see top of particle system block)
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

function resetParticles() {{
    particles.forEach(p => {{ p.age = p.maxAge; }});
}}

// Nearest-neighbour field sample at pixel (px, py)
function sampleField(px, py) {{
    if (!pCanvas) return {{ dx: 0, dy: 0 }};
    const gx = Math.min(FIELD_W - 1, Math.max(0, (px / pCanvas.width)  * FIELD_W)) | 0;
    const gy = Math.min(FIELD_H - 1, Math.max(0, (py / pCanvas.height) * FIELD_H)) | 0;
    const i  = gy * FIELD_W + gx;
    return {{ dx: fieldDx[i], dy: fieldDy[i] }};
}}

// Spawn a particle near a random venue weighted by composite intensity
function respawnParticle(p, activeCount) {{
    const weighted = [];
    VENUES.forEach(v => {{
        const loads = venueIntensityCache[v.id];
        if (!loads) return;
        const w = loads.composite || 0;
        if (w > 0.02) weighted.push({{ v, w }});
    }});
    if (!weighted.length) {{
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

// Main RAF loop (particleRaf and activeParticleCount declared early — see top of particle system block)
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
                // Weight segment selection by proximity to intensity-active venues
                const weighted = [];
                VENUES.forEach(v => {{
                    const loads = venueIntensityCache[v.id];
                    if (!loads || (loads.composite || 0) < 0.02) return;
                    const vp = venuePx[v.id];
                    if (vp) weighted.push({{ vp, w: loads.composite }});
                }});
                let chosenSeg;
                if (weighted.length) {{
                    // Pick a venue weighted by intensity
                    const total = weighted.reduce((s, x) => s + x.w, 0);
                    let rnd = Math.random() * total;
                    let chosenVp = weighted[weighted.length - 1].vp;
                    for (const {{ vp, w }} of weighted) {{ rnd -= w; if (rnd <= 0) {{ chosenVp = vp; break; }} }}
                    // Find nearest segment to chosen venue
                    let bestDist = Infinity;
                    streetSegsPx.forEach(seg => {{
                        const mx = (seg.x0 + seg.x1) * 0.5, my = (seg.y0 + seg.y1) * 0.5;
                        const d = (mx - chosenVp.px)**2 + (my - chosenVp.py)**2;
                        if (d < bestDist) {{ bestDist = d; chosenSeg = seg; }}
                    }});
                }}
                p._seg    = chosenSeg || streetSegsPx[Math.floor(Math.random() * streetSegsPx.length)];
                p._segT   = Math.random();
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

// Field dispatch + throttled update
function updateParticleField() {{
    if (Object.keys(venueIntensityCache).length === 0) return;
    if      (state.particleMode === 'smoke')   _buildSmokefield();
    else if (state.particleMode === 'flow')    _buildFlowField();
    else if (state.particleMode === 'network') _buildNetworkField();
}}

const WIND_DX = 0.40;   // prevailing SW wind: east drift
const WIND_DY = -0.15;  // slight northward component

function _buildSmokefield() {{
    if (!pCanvas) return;
    updateVenuePx();
    const W = pCanvas.width, H = pCanvas.height;
    const cW = W / FIELD_W, cH = H / FIELD_H;

    for (let gy = 0; gy < FIELD_H; gy++) {{
        for (let gx = 0; gx < FIELD_W; gx++) {{
            const cx = (gx + 0.5) * cW;
            const cy = (gy + 0.5) * cH;
            let windDx = WIND_DX, windDy = WIND_DY;

            VENUES.forEach(v => {{
                const loads = venueIntensityCache[v.id];
                if (!loads || loads.smell < 0.02) return;
                const enc = v.enclosure || 'open';
                const enclosureFactor = enc === 'open' ? 1.0 : enc === 'semi_open' ? 0.6 : 0.2;
                const posF = v.lon > -0.09 ? 1.3 : v.lon < -0.17 ? 0.7 : 1.0;
                const vp = venuePx[v.id];
                if (!vp) return;
                const dx = cx - vp.px, dy = cy - vp.py;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 1 || dist > 300) return;
                const strength = loads.smell * enclosureFactor * posF * 4000 / (dist * dist);
                windDx += (dx / dist) * strength;
                windDy += (dy / dist) * strength;
            }});

            const mag = Math.sqrt(windDx * windDx + windDy * windDy);
            if (mag > 2.0) {{ windDx = windDx / mag * 2.0; windDy = windDy / mag * 2.0; }}

            const i = gy * FIELD_W + gx;
            fieldDx[i] = windDx;
            fieldDy[i] = windDy;
        }}
    }}

    // Particle count: base 600 + smoke-scaled bonus (up to 1400 total)
    const year = parseInt(document.getElementById('year-slider').value);
    const decade = Math.floor(year / 10) * 10;
    const smokeRow = SMOKE_DATA_ENV.find(s => s.decade_start === decade);
    const so2_index = smokeRow ? smokeRow.so2_index : 0;
    const count = Math.round(600 + so2_index * 800);

    // Colour: amber-brown
    const safeCount = Math.min(count, MAX_P);
    for (let i = 0; i < safeCount; i++) {{
        particles[i].r = 180; particles[i].g = 130; particles[i].b = 40;
    }}
    startParticles(safeCount);
}}
const MODALITY_COLOURS = {{
    smell:  {{ r: 180, g: 130, b:  40 }},
    noise:  {{ r:  60, g: 120, b: 200 }},
    crowd:  {{ r: 180, g:  60, b:  60 }},
    visual: {{ r:  60, g: 160, b:  80 }},
}};

function _buildFlowField() {{
    if (!pCanvas) return;
    updateVenuePx();
    const W = pCanvas.width, H = pCanvas.height;
    const cW = W / FIELD_W, cH = H / FIELD_H;

    // Which modalities are active via sense pills?
    const activeModals = [];
    if (!state.modality) {{
        activeModals.push('smell', 'noise', 'crowd', 'visual');
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
                        const attractStrength = loads.crowd * 3000 / (dist * dist);
                        fdx -= (dx / dist) * attractStrength;
                        fdy -= (dy / dist) * attractStrength;
                    }} else if (modal === 'noise') {{
                        const strength = loads.noise * 3500 / (dist * dist);
                        fdx += (dx / dist) * strength;
                        fdy += (dy / dist) * strength;
                        if (v.material === 'stone' && (v.enclosure === 'enclosed' || v.enclosure === 'semi_open')) {{
                            fdx += (dy / dist) * strength * 0.25;
                            fdy -= (dx / dist) * strength * 0.25;
                        }}
                    }} else {{
                        const str = loads[modal] * (modal === 'visual' ? 1500 : 3000) / (dist * dist);
                        fdx += (dx / dist) * str;
                        fdy += (dy / dist) * str;
                    }}
                }});

                const i = gy * FIELD_W + gx;
                fieldDx[i] += fdx / activeModals.length;
                fieldDy[i] += fdy / activeModals.length;
            }}
        }}
    }});

    // Final clamp pass — bound the accumulated multi-modality field
    for (let ci = 0; ci < FIELD_W * FIELD_H; ci++) {{
        const fm = Math.sqrt(fieldDx[ci] * fieldDx[ci] + fieldDy[ci] * fieldDy[ci]);
        if (fm > 2.0) {{ fieldDx[ci] = fieldDx[ci] / fm * 2.0; fieldDy[ci] = fieldDy[ci] / fm * 2.0; }}
    }}

    // Blend particle colours across active modalities
    let blendR = 0, blendG = 0, blendB = 0;
    activeModals.forEach(m => {{
        const c = MODALITY_COLOURS[m] || MODALITY_COLOURS.smell;
        blendR += c.r; blendG += c.g; blendB += c.b;
    }});
    blendR = Math.round(blendR / activeModals.length);
    blendG = Math.round(blendG / activeModals.length);
    blendB = Math.round(blendB / activeModals.length);
    const count = 900;
    const safeCount = Math.min(count, MAX_P);
    for (let i = 0; i < safeCount; i++) {{
        particles[i].r = blendR; particles[i].g = blendG; particles[i].b = blendB;
    }}
    startParticles(safeCount);
}}
// Street segment pixel cache
let streetSegsPx = [];

function _projectStreets() {{
    streetSegsPx = [];
    if (!STREET_NETWORK || !STREET_NETWORK.length) return;
    STREET_NETWORK.forEach(polyline => {{
        for (let i = 0; i < polyline.length - 1; i++) {{
            const a = map.latLngToContainerPoint([polyline[i][0],   polyline[i][1]]);
            const b = map.latLngToContainerPoint([polyline[i+1][0], polyline[i+1][1]]);
            const len = Math.sqrt((b.x-a.x)**2 + (b.y-a.y)**2);
            if (len < 2) continue;
            streetSegsPx.push({{ x0: a.x, y0: a.y, x1: b.x, y1: b.y, len }});
        }}
    }});
    // Reset network particles so they resample the new projection
    if (state && state.particleMode === 'network') resetParticles();
}}

map.on('moveend zoomend', _projectStreets);
if (STREET_NETWORK && STREET_NETWORK.length) _projectStreets();

function _networkStep(p) {{
    const seg = p._seg;
    if (!seg) return;
    const speed = 0.6 + Math.random() * 0.2;
    p._segT += p._segDir * speed / seg.len;
    if (p._segT > 1 || p._segT < 0) {{
        if (!streetSegsPx.length) return;
        p._seg    = streetSegsPx[Math.floor(Math.random() * streetSegsPx.length)];
        p._segT   = p._segDir > 0 ? 0 : 1;
        p._segDir = Math.random() < 0.5 ? 1 : -1;
    }}
    const s = p._seg;
    p.px = s.x0 + (s.x1 - s.x0) * p._segT;
    p.py = s.y0 + (s.y1 - s.y0) * p._segT;
}}

function _buildNetworkField() {{
    if (!pCanvas || !streetSegsPx.length) return;
    updateVenuePx();
    const count = 1200;
    const safeCount = Math.min(count, MAX_P);
    for (let i = 0; i < safeCount; i++) {{
        const p = particles[i];
        p._seg    = streetSegsPx[Math.floor(Math.random() * streetSegsPx.length)];
        p._segT   = Math.random();
        p._segDir = Math.random() < 0.5 ? 1 : -1;
        p.r = 160; p.g = 110; p.b = 60;
    }}
    startParticles(safeCount);
}}

let _fieldUpdateTimer = null;
function _scheduleFieldUpdate() {{
    if (_fieldUpdateTimer) clearTimeout(_fieldUpdateTimer);
    _fieldUpdateTimer = setTimeout(() => {{ updateParticleField(); }}, 400);
}}

updateMap();
</script>
</body>
</html>
"""


def build(venues_path: Path = VENUES_PATH, db_path: Path = DB_PATH,
          out_path: Path = OUT_PATH) -> None:
    data = load_data(venues_path, db_path)

    streets = fetch_ohm_streets()

    html = HTML_TEMPLATE.format(
        EVENTS_JSON          = json.dumps(data["events"],          ensure_ascii=False),
        EVENT_VENUES_JSON    = json.dumps(data["event_venues"],    ensure_ascii=False),
        EVENT_INSTANCES_JSON = json.dumps(data["event_instances"], ensure_ascii=False),
        VENUES_JSON          = json.dumps(data["venues"],          ensure_ascii=False),
        EVIDENCE_JSON        = json.dumps(data["evidence"],        ensure_ascii=False),
        CET_JSON             = json.dumps(data["cet"],             ensure_ascii=False),
        MORTALITY_JSON       = json.dumps(data["mortality"],       ensure_ascii=False),
        SMOKE_JSON           = json.dumps(data["smoke"],           ensure_ascii=False),
        STREET_NETWORK_JSON  = json.dumps(streets,                 ensure_ascii=False),
    )

    out_path.write_text(html, encoding="utf-8")
    n_venues = len(data["venues"])
    n_events = len(data["events"])
    n_ev     = len(data["evidence"])
    print(f"Sensory time map -> {out_path}")
    print(f"  {n_venues} venues  {n_events} event types  {n_ev} evidence passages")


if __name__ == "__main__":
    build()
