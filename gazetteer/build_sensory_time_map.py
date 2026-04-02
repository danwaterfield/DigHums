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


# H/W canyon ratio lookup by OSM highway type.
# Based on 1667 Rebuilding Act street grades and urban morphology research.
# H/W < 1.0 = open/dispersed; H/W > 2.0 = canyon/retained
HW_RATIOS = {
    "motorway":      0.5,   # not historically relevant, but safe fallback
    "trunk":         0.7,
    "primary":       1.0,   # Grade 1 — Cheapside, Strand (~40ft W, ~40ft H)
    "secondary":     1.3,   # Grade 2 — Fleet St, Bow St (~30ft W)
    "tertiary":      1.6,   # Grade 3 — by-lanes (~16ft W)
    "residential":   1.9,   # Grade 3-4 — residential lanes
    "living_street": 2.2,   # narrow residential
    "unclassified":  1.4,
    "service":       2.0,   # mews, yard access
    "footway":       3.0,   # Grade 4 — courts/alleys (~8ft W)
    "path":          2.5,
    "steps":         3.5,
    "cycleway":      1.8,
    "track":         1.2,
    "":              1.2,   # unknown → middle estimate
}

def hw_for_highway(t: str) -> float:
    return HW_RATIOS.get(t, 1.2)


def parse_ohm_response(data: dict) -> list:
    """
    Convert OHM Overpass JSON to list of simplified polylines.
    Each entry is {"p": [[lat,lon],...], "s": start_yr|None, "e": end_yr|None, "h": hw_ratio}.
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
            hw_type = tags.get("highway", "")
            segments.append({
                "p": [[p[0], p[1]] for p in simplified],
                "s": start_yr,   # int or None
                "e": end_yr,     # int or None
                "t": hw_type,    # highway type for width proxy
                "h": hw_for_highway(hw_type),  # H/W canyon ratio
            })
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
            segments = json.loads(cache_path.read_text(encoding="utf-8"))
            # Backfill "h" for legacy cache entries built before canyon physics
            enriched = False
            for seg in segments:
                if "h" not in seg:
                    seg["h"] = hw_for_highway(seg.get("t", ""))
                    enriched = True
            if enriched:
                # Re-save cache with h field so next build is instant
                cache_path.write_text(json.dumps(segments, separators=(",", ":")), encoding="utf-8")
                print(f"  OHM cache enriched with canyon H/W ratios.")
            return segments
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
VENUE_GEOMETRIES_PATH = Path(__file__).parent / "venue_geometries.csv"
EVENTS_PATH = Path(__file__).parent / "events.csv"
EVENT_VENUES_PATH = Path(__file__).parent / "event_venues.csv"
EVENT_INSTANCES_PATH = Path(__file__).parent / "event_instances.csv"
DB_PATH     = Path(__file__).parent / "sensory.db"
OUT_PATH    = Path(__file__).parent / "sensory_time_map.html"


def _parse_csv_int(value: str):
    value = (value or "").strip()
    return int(value) if value else None


def _parse_csv_float(value: str):
    value = (value or "").strip()
    return float(value) if value else None


def _coerce_csv_row(row: dict[str, str]) -> dict:
    """Convert CSV row values to ints/floats/None where possible."""
    out = {}
    for key, value in row.items():
        value = (value or "").strip()
        if not value:
            out[key] = None
            continue
        for conv in (int, float):
            try:
                out[key] = conv(value)
                break
            except ValueError:
                pass
        else:
            out[key] = value
    return out


def load_venue_geometries(path: Path) -> dict[str, list[dict]]:
    """Load optional epoch-specific venue display geometries."""
    if not path.exists():
        return {}

    rows: dict[str, list[dict]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            venue_id = (raw.get("venue_id") or "").strip()
            lat = _parse_csv_float(raw.get("lat", ""))
            lon = _parse_csv_float(raw.get("lon", ""))
            if not venue_id or lat is None or lon is None:
                continue
            rows.setdefault(venue_id, []).append({
                "year_start": _parse_csv_int(raw.get("year_start", "")),
                "year_end": _parse_csv_int(raw.get("year_end", "")),
                "lat": lat,
                "lon": lon,
                "source_map": (raw.get("source_map") or "").strip() or None,
                "confidence": (raw.get("confidence") or "").strip() or None,
                "notes": (raw.get("notes") or "").strip() or None,
            })

    for geoms in rows.values():
        geoms.sort(key=lambda g: (
            ((g["year_end"] if g["year_end"] is not None else 1820) -
             (g["year_start"] if g["year_start"] is not None else 1660)),
            g["year_start"] if g["year_start"] is not None else 1660,
            g["source_map"] or "",
        ))
    return rows


def load_event_tables(
    events_path: Path,
    event_venues_path: Path,
    event_instances_path: Path,
) -> tuple[list[dict], list[dict], list[dict]] | None:
    """Load event tables from CSV when the source-of-truth files are available."""
    if not (events_path.exists() and event_venues_path.exists() and event_instances_path.exists()):
        return None

    with open(events_path, newline="", encoding="utf-8") as f:
        events = [_coerce_csv_row(r) for r in csv.DictReader(f)]
    with open(event_venues_path, newline="", encoding="utf-8") as f:
        event_venues = [_coerce_csv_row(r) for r in csv.DictReader(f)]
    with open(event_instances_path, newline="", encoding="utf-8") as f:
        event_instances = [_coerce_csv_row(r) for r in csv.DictReader(f)]
    return events, event_venues, event_instances


def load_data(
    venues_path: Path,
    db_path: Path,
    venue_geometries_path: Path = VENUE_GEOMETRIES_PATH,
    events_path: Path = EVENTS_PATH,
    event_venues_path: Path = EVENT_VENUES_PATH,
    event_instances_path: Path = EVENT_INSTANCES_PATH,
) -> dict:
    venue_geometries = load_venue_geometries(venue_geometries_path)
    with open(venues_path, newline="", encoding="utf-8") as f:
        venues = [
            {"id": r["id"], "name": r["name"],
             "lat": float(r["lat"]), "lon": float(r["lon"]),
             "canonical_lat": float(r["lat"]), "canonical_lon": float(r["lon"]),
             "tier": int(r["tier"]) if r.get("tier") else 3,
             "map_layer":      r.get("map_layer", ""),
             "enclosure":     r.get("enclosure", ""),
             "building_type": r.get("building_type", ""),
             "material":      r.get("material", ""),
             "capacity":      r.get("capacity", ""),
             "opened": int(r["opened"]) if r.get("opened", "").strip() else None,
             "closed": int(r["closed"]) if r.get("closed", "").strip() else None,
             "hw_ratio": float(r["hw_ratio"]) if r.get("hw_ratio", "").strip() else None,}
            for r in csv.DictReader(f)
        ]

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        event_tables = load_event_tables(events_path, event_venues_path, event_instances_path)
        if event_tables is None:
            events          = [dict(r) for r in conn.execute("SELECT * FROM events")]
            event_venues    = [dict(r) for r in conn.execute("SELECT * FROM event_venues")]
            event_instances = [dict(r) for r in conn.execute("SELECT * FROM event_instances")]
        else:
            events, event_venues, event_instances = event_tables
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
        "venue_geometries": venue_geometries,
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
:root {{
  /* -- Typography -- */
  --font-sans: 'Inter', system-ui, -apple-system, sans-serif;
  --font-serif: 'Georgia', 'Times New Roman', serif;
  --font-mono: 'SF Mono', 'Menlo', 'Monaco', monospace;
  --text-xs: 0.6875rem;
  --text-sm: 0.75rem;
  --text-base: 0.8125rem;
  --text-md: 0.875rem;
  --text-lg: 1rem;
  --leading-tight: 1.3;
  --leading-normal: 1.5;
  --tracking-wide: 0.04em;

  /* -- Colors — surface -- */
  --bg-page: #f5f3ef;
  --bg-surface: #ffffff;
  --bg-surface-alt: #f8f6f2;
  --bg-control: #f0ece6;
  --bg-control-hover: #e8e4dc;
  --bg-control-active: #1e3c6e;

  /* -- Colors — text -- */
  --text-primary: #1a1816;
  --text-secondary: #5c5850;
  --text-muted: #9c9890;
  --text-inverse: #ffffff;
  --text-link: #1e3c6e;

  /* -- Colors — border -- */
  --border-default: #d8d4cc;
  --border-subtle: #ece8e0;
  --border-strong: #9c9890;

  /* -- Colors — sense modalities -- */
  --smell-base: #b48a28;
  --smell-active: #c47a1e;
  --smell-bar: #7c6f3e;
  --noise-base: #3c78c8;
  --noise-active: #10305c;
  --noise-bar: #3e5c7c;
  --crowd-base: #b43c3c;
  --crowd-active: #b02a1a;
  --crowd-bar: #7c3e3e;
  --visual-base: #3ca050;
  --visual-active: #10401a;
  --visual-bar: #3e7c4a;

  /* -- Colors — semantic -- */
  --accent-primary: #1e3c6e;
  --accent-literary: #1a7d4a;
  --accent-heatmap: #5c3a98;
  --accent-milestone: #c47a1e;
  --accent-prose-border: #8b6914;
  --accent-prose-bg: #f5efe4;
  --accent-prose-text: #3a2a08;

  /* -- Colors — environmental -- */
  --temp-bg: #edf2f8;
  --temp-border: #c0d0e4;
  --temp-text: #1e3c6e;
  --mort-bg: #fef0ee;
  --mort-border: #f0c0b8;
  --mort-text: #b02a1a;
  --smoke-track: #ece8e0;
  --smoke-fill-start: #c8a050;
  --smoke-fill-end: #8b5a20;
  --smoke-pct: #c0a060;

  /* -- Colors — source badges -- */
  --src-fiction-bg: #e0e8f0; --src-fiction-text: #1a3a5c;
  --src-diary-bg: #e8f0e0; --src-diary-text: #1a4a1a;
  --src-topography-bg: #f0e8d0; --src-topography-text: #5c3a1a;
  --src-legal-bg: #f5e0e0; --src-legal-text: #8b1a1a;
  --src-poetry-bg: #ede0f5; --src-poetry-text: #3a1a5c;
  --src-letters-bg: #e0f5f0; --src-letters-text: #1a5c4a;
  --src-newspaper-bg: #f8ecd2; --src-newspaper-text: #8a5a11;
  --src-parish-bg: #d9f0ec; --src-parish-text: #0f615f;
  --src-institutional-bg: #e4e8f6; --src-institutional-text: #304c86;

  /* -- Spacing -- */
  --space-1: 0.25rem;
  --space-2: 0.5rem;
  --space-3: 0.75rem;
  --space-4: 1rem;
  --space-5: 1.25rem;

  /* -- Borders & Radii -- */
  --radius-sm: 4px;
  --radius-md: 6px;
  --radius-lg: 8px;
  --radius-pill: 9999px;

  /* -- Shadows -- */
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.06);
  --shadow-md: 0 2px 8px rgba(0,0,0,0.1);
  --shadow-lg: 0 4px 16px rgba(0,0,0,0.12);

  /* -- Transitions -- */
  --transition-fast: 0.15s ease;
  --transition-base: 0.25s ease;
  --transition-slow: 0.4s ease;

  /* -- Z-index scale -- */
  --z-tint: 401;
  --z-heatmap: 403;
  --z-night: 404;
  --z-legend: 500;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: var(--font-sans); background: var(--bg-page); color: var(--text-primary); display: flex; flex-direction: column; height: 100vh; overflow: hidden; font-size: var(--text-base); line-height: var(--leading-normal); }}
#controls {{ background: var(--bg-surface); border-bottom: 1px solid var(--border-default); padding: var(--space-2) var(--space-3); flex-shrink: 0; }}
#header-row {{ display: flex; align-items: center; gap: var(--space-4); margin-bottom: var(--space-2); flex-wrap: wrap; }}
.title {{ font-size: var(--text-md); font-weight: 700; letter-spacing: 0; color: var(--text-primary); }}
#year-control {{ display: flex; align-items: center; gap: var(--space-2); }}
#year-slider {{ width: 200px; accent-color: var(--accent-primary); }}
#year-display {{ font-size: var(--text-lg); font-weight: 700; min-width: 3em; text-align: center; color: var(--accent-primary); font-variant-numeric: tabular-nums; }}
.step-btn {{ background: var(--bg-control); border: 1px solid var(--border-default); color: var(--text-secondary); padding: 3px 10px; cursor: pointer; border-radius: var(--radius-sm); font-size: var(--text-sm); font-family: inherit; transition: background var(--transition-fast), color var(--transition-fast); }}
.step-btn:hover {{ background: var(--bg-control-hover); color: var(--text-primary); }}
.step-btn:focus-visible {{ outline: 2px solid var(--accent-primary); outline-offset: 1px; }}
#play-btn {{ background: var(--bg-control); border: 1px solid var(--border-default); color: var(--text-secondary); padding: 3px 12px; cursor: pointer; border-radius: var(--radius-sm); font-size: var(--text-sm); font-family: inherit; transition: background var(--transition-fast), color var(--transition-fast); }}
#play-btn:hover {{ background: var(--bg-control-hover); color: var(--text-primary); }}
#play-btn:focus-visible {{ outline: 2px solid var(--accent-primary); outline-offset: 1px; }}
#play-btn.playing {{ background: var(--accent-primary); border-color: var(--accent-primary); color: var(--text-inverse); }}
.view-tab {{ background: var(--bg-control); border: 1px solid var(--border-default); color: var(--text-secondary); padding: 3px 12px; border-radius: var(--radius-sm); font-size: var(--text-xs); font-weight: 500; text-decoration: none; display: inline-block; font-family: inherit; transition: background var(--transition-fast), color var(--transition-fast); }}
.view-tab:hover {{ background: var(--bg-control-hover); color: var(--text-primary); }}
.view-tab.active {{ background: var(--accent-primary); border-color: var(--accent-primary); color: var(--text-inverse); font-weight: 600; cursor: default; }}
.pill-row {{ display: flex; gap: var(--space-1); flex-wrap: wrap; margin-bottom: var(--space-1); align-items: center; }}
.pill-label {{ font-size: var(--text-xs); color: var(--text-muted); margin-right: var(--space-1); min-width: 3em; font-weight: 500; }}
.pill {{ background: var(--bg-control); border: 1px solid var(--border-default); color: var(--text-secondary); padding: 3px 10px; cursor: pointer; border-radius: var(--radius-sm); font-size: var(--text-xs); font-weight: 500; font-family: inherit; transition: background var(--transition-fast), color var(--transition-fast), border-color var(--transition-fast); }}
.pill:hover {{ background: var(--bg-control-hover); color: var(--text-primary); }}
.pill:focus-visible {{ outline: 2px solid var(--accent-primary); outline-offset: 1px; }}
.pill.active {{ background: var(--accent-primary); border-color: var(--accent-primary); color: var(--text-inverse); font-weight: 600; }}
#lit-toggle {{ background: var(--bg-control); border: 1px solid var(--border-default); color: var(--text-secondary); padding: 3px 12px; cursor: pointer; border-radius: var(--radius-sm); font-size: var(--text-xs); font-weight: 500; font-family: inherit; transition: background var(--transition-fast), color var(--transition-fast); }}
#lit-toggle:focus-visible {{ outline: 2px solid var(--accent-primary); outline-offset: 1px; }}
#lit-toggle.active {{ background: var(--accent-literary); border-color: var(--accent-literary); color: var(--text-inverse); }}
#main {{ display: flex; flex: 1; overflow: hidden; }}
#map {{ flex: 1; position: relative; }}
#panel {{ width: 360px; flex-shrink: 0; background: var(--bg-surface); border-left: 1px solid var(--border-default); overflow-y: auto; display: flex; flex-direction: column; }}
#panel-header {{ display: flex; align-items: center; background: var(--bg-surface-alt); border-bottom: 1px solid var(--border-subtle); color: var(--text-primary); padding: var(--space-2) var(--space-3); font-size: var(--text-xs); font-weight: 600; letter-spacing: var(--tracking-wide); text-transform: uppercase; flex-shrink: 0; }}
#panel-title {{ flex: 1; }}
#panel-body {{ padding: var(--space-3); flex: 1; }}
/* -- Event cards -- */
.event-card {{ background: var(--bg-surface); border: 1px solid var(--border-default); border-radius: var(--radius-md); padding: var(--space-3); margin-bottom: var(--space-2); transition: box-shadow var(--transition-fast); }}
.event-card:hover {{ box-shadow: var(--shadow-sm); }}
.event-card h3 {{ font-size: var(--text-md); margin-bottom: var(--space-1); font-weight: 600; }}
.event-meta {{ font-size: var(--text-sm); color: var(--text-muted); margin-bottom: var(--space-2); }}
/* -- Intensity bars -- */
.intensity-bars {{ display: grid; grid-template-columns: 4em 1fr; gap: 3px 8px; font-size: var(--text-sm); align-items: center; }}
.bar-track {{ background: var(--border-subtle); border-radius: var(--radius-sm); height: 6px; }}
.bar-fill {{ height: 6px; border-radius: var(--radius-sm); transition: width var(--transition-base); }}
.bar-smell {{ background: var(--smell-bar); }}
.bar-noise {{ background: var(--noise-bar); }}
.bar-crowd {{ background: var(--crowd-bar); }}
.bar-visual {{ background: var(--visual-bar); }}
/* -- Event notes / sources -- */
.event-notes {{ font-size: var(--text-sm); color: var(--text-secondary); margin-top: var(--space-2); font-style: italic; border-top: 1px solid var(--border-subtle); padding-top: var(--space-1); }}
.ev-sources {{ font-size: var(--text-xs); color: var(--text-muted); margin-top: var(--space-1); }}
/* -- Passage cards -- */
.passage-card {{ background: #fff8f0; border: 1px solid #e0c8a8; border-radius: var(--radius-md); padding: var(--space-2) var(--space-3); margin-bottom: var(--space-2); font-size: var(--text-sm); line-height: var(--leading-normal); }}
.passage-card .src-badge {{ display: inline-block; font-size: var(--text-xs); padding: 2px 8px; border-radius: var(--radius-pill); margin-bottom: var(--space-1); font-weight: 500; }}
.src-fiction    {{ background: var(--src-fiction-bg); color: var(--src-fiction-text); }}
.src-diary      {{ background: var(--src-diary-bg); color: var(--src-diary-text); }}
.src-topography {{ background: var(--src-topography-bg); color: var(--src-topography-text); }}
.src-legal      {{ background: var(--src-legal-bg); color: var(--src-legal-text); }}
.src-poetry     {{ background: var(--src-poetry-bg); color: var(--src-poetry-text); }}
.src-letters    {{ background: var(--src-letters-bg); color: var(--src-letters-text); }}
.src-newspaper  {{ background: var(--src-newspaper-bg); color: var(--src-newspaper-text); }}
.src-parish     {{ background: var(--src-parish-bg); color: var(--src-parish-text); }}
.src-institutional {{ background: var(--src-institutional-bg); color: var(--src-institutional-text); }}
.no-events {{ color: var(--text-muted); font-style: italic; font-size: var(--text-base); padding: var(--space-2) 0; }}
/* -- Sense pills -- */
.sense-pill {{ background: var(--bg-control); border: 1px solid var(--border-default); color: var(--text-secondary); padding: 3px 10px; cursor: pointer; border-radius: var(--radius-sm); font-size: var(--text-xs); font-weight: 500; font-family: inherit; transition: background var(--transition-fast), color var(--transition-fast), border-color var(--transition-fast); }}
.sense-pill:hover {{ background: var(--bg-control-hover); color: var(--text-primary); }}
.sense-pill:focus-visible {{ outline: 2px solid var(--accent-primary); outline-offset: 1px; }}
.sense-pill[data-sense="smell"].active {{ background: var(--smell-active); border-color: var(--smell-active); color: var(--text-inverse); }}
.sense-pill[data-sense="noise"].active {{ background: var(--noise-active); border-color: var(--noise-bar); color: var(--text-inverse); }}
.sense-pill[data-sense="crowd"].active {{ background: var(--crowd-active); border-color: var(--crowd-active); color: var(--text-inverse); }}
.sense-pill[data-sense="visual"].active {{ background: var(--visual-active); border-color: var(--visual-base); color: var(--text-inverse); }}
/* -- Building-type pills -- */
.btype-pill {{ background: var(--bg-control); border: 1px solid; padding: 3px 10px; cursor: pointer; border-radius: var(--radius-sm); font-size: var(--text-xs); font-weight: 500; opacity: 0.55; transition: opacity var(--transition-fast), background var(--transition-fast); font-family: inherit; }}
.btype-pill:hover {{ opacity: 1; }}
.btype-pill:focus-visible {{ outline: 2px solid var(--accent-primary); outline-offset: 1px; }}
.btype-pill.active {{ background: var(--bg-surface); opacity: 1; font-weight: 600; }}
/* -- Prose summary -- */
.prose-summary {{ background: var(--accent-prose-bg); border-left: 3px solid var(--accent-prose-border); padding: var(--space-2) var(--space-3); margin-bottom: var(--space-3); font-size: var(--text-base); line-height: var(--leading-normal); color: var(--accent-prose-text); border-radius: 0 var(--radius-sm) var(--radius-sm) 0; }}
/* -- Season chart -- */
.season-chart {{ display: flex; gap: 2px; margin: var(--space-2) 0 var(--space-1); }}
.season-bar {{ flex: 1; height: 8px; border-radius: 2px; cursor: pointer; transition: opacity var(--transition-fast); }}
.season-active {{ background: var(--accent-prose-border); }}
.season-inactive {{ background: var(--border-default); }}
.season-bar:hover {{ opacity: 0.7; }}
/* -- Milestone -- */
#milestone-label {{ font-size: var(--text-xs); color: var(--accent-milestone); margin-left: var(--space-2); font-style: italic; white-space: nowrap; font-weight: 500; }}
/* -- Speed buttons -- */
.speed-btn {{ background: var(--bg-control); border: 1px solid var(--border-default); color: var(--text-secondary); padding: 2px 6px; cursor: pointer; border-radius: var(--radius-sm); font-size: var(--text-xs); font-weight: 500; font-family: inherit; transition: background var(--transition-fast), color var(--transition-fast); }}
.speed-btn:hover {{ background: var(--bg-control-hover); color: var(--text-primary); }}
.speed-btn:focus-visible {{ outline: 2px solid var(--accent-primary); outline-offset: 1px; }}
.speed-btn.active {{ background: var(--temp-bg); border-color: #2563a8; color: var(--accent-primary); font-weight: 600; }}
/* -- Clear / Back buttons -- */
#clear-btn {{ background: none; border: 1px solid var(--border-default); color: var(--text-muted); padding: 3px 10px; cursor: pointer; border-radius: var(--radius-sm); font-size: var(--text-xs); font-weight: 500; font-family: inherit; transition: color var(--transition-fast), border-color var(--transition-fast); }}
#clear-btn:hover {{ color: var(--text-primary); border-color: var(--border-strong); }}
#clear-btn:focus-visible {{ outline: 2px solid var(--accent-primary); outline-offset: 1px; }}
#back-btn {{ background: none; border: none; color: var(--text-link); font-size: var(--text-sm); cursor: pointer; padding: 0 var(--space-2) 0 0; flex-shrink: 0; font-family: inherit; transition: color var(--transition-fast); }}
#back-btn:hover {{ color: var(--text-primary); }}
#back-btn:focus-visible {{ outline: 2px solid var(--accent-primary); outline-offset: 1px; }}
/* -- Environmental indicators -- */
#env-bar {{ display: flex; align-items: center; gap: var(--space-3); padding: var(--space-1) 0; flex-wrap: wrap; }}
.env-gauge {{ display: flex; align-items: center; gap: 5px; font-size: var(--text-xs); color: var(--text-muted); font-weight: 500; }}
.env-gauge .env-label {{ color: var(--text-muted); }}
.temp-badge {{ background: var(--temp-bg); border: 1px solid var(--temp-border); color: var(--temp-text); padding: 2px 8px; border-radius: var(--radius-sm); font-size: var(--text-sm); font-family: var(--font-mono); min-width: 4.5em; text-align: center; font-variant-numeric: tabular-nums; transition: background var(--transition-base), color var(--transition-base); }}
.temp-badge.cold  {{ background: #ddeeff; border-color: #7ab0d8; color: #1a3c70; }}
.temp-badge.frost {{ background: #081828; border-color: #80c0f8; color: #c8eeff; font-weight: bold; }}
.mort-badge {{ background: var(--mort-bg); border: 1px solid var(--mort-border); color: var(--mort-text); padding: 2px 8px; border-radius: var(--radius-sm); font-size: var(--text-sm); font-family: var(--font-mono); min-width: 5em; text-align: center; font-variant-numeric: tabular-nums; }}
.smoke-gauge {{ display: flex; align-items: center; gap: 5px; font-size: var(--text-xs); color: var(--text-muted); font-weight: 500; }}
.smoke-bar-track {{ width: 50px; height: 5px; background: var(--smoke-track); border-radius: var(--radius-sm); }}
.smoke-bar-fill {{ height: 5px; background: linear-gradient(to right, var(--smoke-fill-start), var(--smoke-fill-end)); border-radius: var(--radius-sm); transition: width var(--transition-slow); }}
/* -- Tier toggle -- */
#tier-toggle {{ background: #1a2a3a; border: 1px solid #444; color: #8090a8; padding: 2px 10px; cursor: pointer; border-radius: var(--radius-pill); font-size: var(--text-sm); font-family: inherit; transition: background var(--transition-fast), color var(--transition-fast); }}
#tier-toggle:hover {{ background: #253545; }}
#tier-toggle:focus-visible {{ outline: 2px solid var(--accent-primary); outline-offset: 1px; }}
#tier-toggle.active {{ background: #1a3a5a; border-color: #4488cc; color: #aaddff; }}
/* -- Overlay / contour / sense buttons -- */
.overlay-btn {{ background: var(--bg-control); border: 1px solid var(--border-default); color: var(--text-secondary); padding: 3px 10px; cursor: pointer; border-radius: var(--radius-sm); font-size: var(--text-xs); font-weight: 500; font-family: inherit; transition: background var(--transition-fast), color var(--transition-fast); }}
.overlay-btn:hover {{ background: var(--bg-control-hover); color: var(--text-primary); }}
.overlay-btn:focus-visible {{ outline: 2px solid var(--accent-primary); outline-offset: 1px; }}
.overlay-btn.active {{ background: var(--accent-heatmap); border-color: var(--accent-heatmap); color: var(--text-inverse); font-weight: 600; }}
.contour-btn {{ background: var(--bg-control); border: 1px solid var(--border-default); color: var(--text-secondary); padding: 3px 10px; cursor: pointer; border-radius: var(--radius-sm); font-size: var(--text-xs); font-weight: 500; opacity: 0.75; font-family: inherit; transition: background var(--transition-fast), color var(--transition-fast), opacity var(--transition-fast); }}
.contour-btn:hover {{ opacity: 1; }}
.contour-btn:focus-visible {{ outline: 2px solid var(--accent-primary); outline-offset: 1px; }}
.contour-btn.active {{ background: var(--accent-primary); border-color: var(--accent-primary); color: var(--text-inverse); opacity: 1; font-weight: 600; }}
.sense-btn {{ background: var(--bg-control); border: 1px solid var(--border-default); color: var(--text-secondary); padding: 3px 10px; cursor: pointer; border-radius: var(--radius-sm); font-size: var(--text-xs); font-weight: 500; opacity: 0.75; font-family: inherit; transition: background var(--transition-fast), color var(--transition-fast), opacity var(--transition-fast); }}
.sense-btn:hover {{ opacity: 1; }}
.sense-btn:focus-visible {{ outline: 2px solid var(--accent-primary); outline-offset: 1px; }}
.sense-btn.active {{ background: #3a5a2a; border-color: #3a5a2a; color: var(--text-inverse); opacity: 1; font-weight: 600; }}
#sense-row {{ display: none; }}
.contour-surface {{ pointer-events: none; }}
/* -- Heatmap overlay -- */
#heatmap-canvas {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: var(--z-heatmap); opacity: 0; transition: opacity var(--transition-slow); filter: blur(18px); }}
/* -- Time-of-day atmospheric tint -- */
#tod-tint {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: var(--z-tint); opacity: 0; transition: background 1.5s ease, opacity 1.5s ease; }}
/* -- Night mode overlay -- */
#night-overlay {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: var(--z-night); background: radial-gradient(ellipse at 50% 40%, rgba(0,5,20,0.45) 0%, rgba(0,5,20,0.65) 100%); opacity: 0; transition: opacity 1.4s; }}
/* -- Venue name labels -- */
.venue-label {{ background: none; border: none; white-space: nowrap; font-size: var(--text-xs); font-weight: 600; color: #1a1008; text-shadow: 0 0 3px rgba(255,255,240,0.95), 0 0 7px rgba(255,255,240,0.7); pointer-events: none; padding: 1px 0 0 6px; letter-spacing: 0.01em; }}
/* -- Night mode controls -- */
#map.night-mode .leaflet-tile-pane {{ filter: brightness(0.52) saturate(0.75) sepia(0.15); transition: filter 1.4s; }}
#controls.night-ctrl {{ background: #0d1520; border-bottom-color: #1a2a3a; }}
#controls.night-ctrl .title {{ color: #7ab8d8; }}
#controls.night-ctrl #year-display {{ color: #7ab8d8; }}
#controls.night-ctrl .pill-label {{ color: #4a6a88; }}
#controls.night-ctrl .pill, #controls.night-ctrl .step-btn, #controls.night-ctrl #play-btn,
#controls.night-ctrl .sense-pill, #controls.night-ctrl .speed-btn,
#controls.night-ctrl .overlay-btn {{ background: #0e1e30; border-color: #1e3048; color: #6a9abc; }}
#controls.night-ctrl .overlay-btn.active {{ background: #3a1e6a; border-color: #7a4ecf; color: #c8aaff; font-weight: 600; }}
#controls.night-ctrl .pill.active {{ background: #1a4870; border-color: #2a70b8; color: #aadcff; }}
#controls.night-ctrl .btype-pill {{ background: #0e1e30 !important; filter: brightness(1.4) saturate(1.3); }}
#controls.night-ctrl #milestone-label {{ color: #8aabcc; }}
#controls.night-ctrl .contour-btn {{ background: #1a2a3a; border-color: #333; color: #6080a0; }}
#controls.night-ctrl .contour-btn.active {{ background: #1e3c6e; color: #fff; }}
#controls.night-ctrl .sense-btn {{ background: #1a2a3a; border-color: #333; color: #6080a0; }}
#controls.night-ctrl .sense-btn.active {{ background: #3a5a2a; color: #fff; }}
/* -- Map legend -- */
#map-legend {{ position: absolute; bottom: 32px; left: 10px; z-index: var(--z-legend); background: rgba(255,255,255,0.95); border: 1px solid var(--border-default); border-radius: var(--radius-md); padding: var(--space-2) var(--space-3); font-size: var(--text-sm); min-width: 108px; box-shadow: var(--shadow-md); pointer-events: none; backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px); transition: background 0.8s, border-color 0.8s, color 0.8s; }}
.leg-title {{ font-weight: 600; margin-bottom: var(--space-1); font-size: var(--text-xs); letter-spacing: var(--tracking-wide); color: inherit; text-transform: uppercase; }}
.leg-row {{ display: flex; align-items: center; gap: 5px; margin-bottom: 2px; }}
.leg-dot {{ display: inline-block; border-radius: 50%; flex-shrink: 0; opacity: 0.88; }}
/* -- Tier marker colours (used in tier-view mode) -- */
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
    <button id="automap-btn" onclick="enableAutoBasemap()" title="Auto-switch base map to match year" style="background:#f4f1eb;border:1px solid #d8d4cc;color:#5c5850;padding:2px 8px;cursor:pointer;border-radius:3px;font-size:0.69em;font-weight:500;">Auto map</button>
  </div>
  <div class="pill-row">
    <span class="pill-label">View</span>
    <span class="view-tab active">Sensory Map</span>
    <a href="venue_explorer.html" class="view-tab">Evidence</a>
    <a href="narrative_map.html" class="view-tab">Narrative</a>
    <a href="comparison.html" class="view-tab">Comparison</a>
    <a href="sensory_timeline.html" class="view-tab">Timeline</a>
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
  <div class="pill-row">
    <span class="pill-label">Layer</span>
    <button class="overlay-btn" id="heatmap-btn" onclick="toggleHeatmap()">&#9638; Heatmap</button>
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
    <div id="tod-tint"></div>
    <canvas id="heatmap-canvas"></canvas>
    <div id="night-overlay"></div>
    <div id="map-legend"></div>
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
const VENUE_GEOMETRIES = {VENUE_GEOMETRIES_JSON};
const EVIDENCE = {EVIDENCE_JSON};

// ── Environmental data ─────────────────────────────────────────────────────
const CET_DATA       = {CET_JSON};         // {{year: {{0:annual, 1:jan, ...12:dec}}}}
const MORTALITY_DATA = {MORTALITY_JSON};   // {{year: total_burials}}
const SMOKE_DATA_ENV = {SMOKE_JSON};       // [{{decade_start, so2_index, coal_tons_k}}]
const STREET_NETWORK = {STREET_NETWORK_JSON};  // [[lat,lon],...] pre-1820 streets
const ZONE_DATA = {ZONE_DATA_JSON};            // GeoJSON FeatureCollection — named London zones

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
        river_proximity:      p0.river_proximity + t * (p1.river_proximity - p0.river_proximity),
        industrial_intensity: p0.industrial_intensity + t * (p1.industrial_intensity - p0.industrial_intensity),
        street_character:     t < 0.5 ? p0.street_character : p1.street_character,
        building_height:      t < 0.5 ? p0.building_height  : p1.building_height,
        canyon_factor:        p0.canyon_factor + t * (p1.canyon_factor - p0.canyon_factor),
    }};
}}

// Layer 1 (zone ambient) + Layer 2 (env modifiers)
// Returns {{smell, noise, crowd, visual, zone, dominant, provenance, street_character}} or null
function computeZoneBaseline(lat, lon, year, month) {{
    const zoneProps = getZoneForPoint(lat, lon);
    if (!zoneProps) return null;
    const p = interpolateZoneProps(zoneProps, year);

    let smell = p.smell_base;
    let noise = p.noise_base;
    let crowd = p.crowd_density;
    let visual = 0.3;
    const provenance = [];

    // Env modifier 1: river smell rises with summer heat
    if (p.river_proximity > 0) {{
        const yearCET = CET_DATA[year] || {{}};
        const temp = (month !== null && yearCET[month] !== undefined) ? yearCET[month] : (yearCET[0] || 10);
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
        const windMult = lon > -0.09 ? 1.3 : lon < -0.17 ? 0.7 : 1.0;
        const finalSmoke = smokeBoost * windMult;
        if (finalSmoke > 0.02) {{
            smell = Math.min(1, smell + finalSmoke * 0.6);
            visual = Math.min(1, visual + finalSmoke * 0.3);
            provenance.push('coal smoke (SO\u2082 ' + so2.toFixed(1) + ')');
        }}
    }}

    // Env modifier 3: frost fair crowd noise (January, Thames zones, very cold)
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
        canyon_factor: p.canyon_factor || 1.0,
    }};
}}

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
    'menagerie': '#7b4513',
    'bookseller': '#8B4513',
    'coffeehouse': '#6F4E37',
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

function getBasemapKeyForYear(year) {{
    return year < 1740 ? 'Modern (CartoDB)'
         : year < 1791 ? 'Rocque 1746'
         : 'Horwood 1792\u201399';
}}

function _matchesBasemapMapLayer(mapLayer, basemapKey) {{
    const ml = mapLayer || '';
    if (basemapKey === 'Rocque 1746') return ml.includes('rocque_1746');
    if (basemapKey === 'Horwood 1792\u201399') return ml.includes('horwood_1799');
    return true;
}}

function _pickVenueGeometry(v, year) {{
    const candidates = VENUE_GEOMETRIES[v.id] || [];
    let best = null;
    let bestSpan = Infinity;
    candidates.forEach(g => {{
        if (g.year_start !== null && year < g.year_start) return;
        if (g.year_end !== null && year > g.year_end) return;
        const spanStart = g.year_start !== null ? g.year_start : 1660;
        const spanEnd = g.year_end !== null ? g.year_end : 1820;
        const span = spanEnd - spanStart;
        if (!best || span < bestSpan) {{
            best = g;
            bestSpan = span;
        }}
    }});
    return best;
}}

function _distanceMeters(lat0, lon0, lat1, lon1) {{
    const phi = ((lat0 + lat1) / 2) * Math.PI / 180;
    const dx = (lon1 - lon0) * 111320 * Math.cos(phi);
    const dy = (lat1 - lat0) * 110540;
    return Math.sqrt(dx * dx + dy * dy);
}}

function _venuePlacementSummary(v, year, shortForm=false) {{
    const basemapKey = getBasemapKeyForYear(year);
    const shift = _distanceMeters(v.canonical_lat, v.canonical_lon, v.lat, v.lon);
    if (v.display_mode === 'epoch') {{
        const parts = [];
        parts.push(v.display_source || basemapKey);
        if (v.display_confidence) parts.push(v.display_confidence + ' confidence');
        if (!shortForm && shift >= 5) parts.push('shifted ' + Math.round(shift) + 'm from canonical point');
        if (!shortForm && v.display_notes) parts.push(v.display_notes);
        return shortForm
            ? 'map: ' + parts.slice(0, 2).join(' \u00b7 ')
            : 'Map placement: ' + parts.join(' \u00b7 ');
    }}
    if (basemapKey === 'Modern (CartoDB)') return shortForm ? '' : 'Map placement: canonical point on modern basemap.';
    const support = _matchesBasemapMapLayer(v.map_layer, basemapKey)
        ? 'no plan-specific geometry recorded yet'
        : 'nearest surveyed plan fallback';
    return shortForm
        ? 'map: canonical point on ' + basemapKey
        : 'Map placement: canonical point on ' + basemapKey + ' (' + support + ').';
}}

let playSpeed = 500;

// Evidence count per venue (for literary badge)
const evidenceCountByVenue = {{}};
EVIDENCE.forEach(p => {{
    if (p.venue_id) evidenceCountByVenue[p.venue_id] = (evidenceCountByVenue[p.venue_id] || 0) + 1;
}});

const state = {{ month: null, dow: null, band: null, literary: false,
                 selectedVenue: null, modality: null, tierView: false,
                 buildingType: null, contourMode: 'off', contourSense: 'smoke' }};

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
const layerControl = L.control.layers(baseLayers, {{}}, {{ collapsed: false }}).addTo(map);

// ── Auto-switching basemap ─────────────────────────────────────────────────
let _autoBasemap = true;
let _currentBaseKey = 'Modern (CartoDB)';

function updateBasemap(year) {{
    if (!_autoBasemap) return;
    const key = getBasemapKeyForYear(year);
    const targetLayer = baseLayers[key];
    Object.entries(baseLayers).forEach(([name, layer]) => {{
        if (name !== key && map.hasLayer(layer)) map.removeLayer(layer);
    }});
    if (key === _currentBaseKey && map.hasLayer(targetLayer)) return;
    if (!map.hasLayer(baseLayers[key])) baseLayers[key].addTo(map);
    _currentBaseKey = key;
}}

function bindBasemapManualOverride() {{
    const container = layerControl.getContainer();
    if (!container) return;
    const disableAuto = (evt) => {{
        if (!(evt.target instanceof HTMLElement)) return;
        if (!evt.target.classList.contains('leaflet-control-layers-selector')) return;
        _autoBasemap = false;
    }};
    container.addEventListener('click', disableAuto);
    container.addEventListener('change', disableAuto);
}}
bindBasemapManualOverride();

map.on('baselayerchange', (e) => {{
    _currentBaseKey = e.name;
}});

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
            const intensity = props[dominant + '_base'] !== undefined
                ? props[dominant + '_base']
                : props.smell_base || 0.3;
            const opacity = Math.min(0.10, intensity * 0.10);
            return {{
                fillColor: color,
                fillOpacity: opacity,
                stroke: false,
                interactive: false,
            }};
        }},
    }}).addTo(map);
    zoneFillLayer.bringToBack();
}}

function enableAutoBasemap() {{
    _autoBasemap = true;
    _currentBaseKey = '';  // force switch on next call
    const yr = parseInt(document.getElementById('year-slider').value);
    updateBasemap(yr);
}}

function _applyVenueDisplayGeometry(year) {{
    VENUES.forEach(v => {{
        const geom = _pickVenueGeometry(v, year);
        const nextLat = geom ? geom.lat : v.canonical_lat;
        const nextLon = geom ? geom.lon : v.canonical_lon;
        v.display_mode = geom ? 'epoch' : 'canonical';
        v.display_source = geom && geom.source_map ? geom.source_map : null;
        v.display_confidence = geom && geom.confidence ? geom.confidence : null;
        v.display_notes = geom && geom.notes ? geom.notes : null;

        const changed = v.lat !== nextLat || v.lon !== nextLon;
        v.lat = nextLat;
        v.lon = nextLon;
        if (!changed) return;

        if (markersByVenueId[v.id]) markersByVenueId[v.id].setLatLng([v.lat, v.lon]);
        if (smellRings[v.id]) smellRings[v.id].setLatLng([v.lat, v.lon]);
        if (noiseRings[v.id]) noiseRings[v.id].setLatLng([v.lat, v.lon]);
        if (_venueLabels[v.id]) _venueLabels[v.id].setLatLng([v.lat, v.lon]);
    }});
}}

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
    // Lord Mayor\u2019s Show: Guildhall \u2192 King St \u2192 Cheapside \u2192 Ludgate Hill \u2192 Fleet St \u2192 Temple Bar
    // Route varied by ward of new Mayor; this traces the core ceremonial spine.
    // Source: Lord Mayor\u2019s Show official history; Ian Visits.
    coords: [
      [51.5155, -0.0924],  // Guildhall
      [51.5147, -0.0926],  // King Street south \u2014 joining Cheapside
      [51.5147, -0.0955],  // Cheapside (Old Jewry junction)
      [51.5147, -0.0975],  // Cheapside (Foster Lane area)
      [51.5138, -0.0984],  // St Paul\u2019s Cathedral
      [51.5136, -0.1013],  // Ludgate Hill
      [51.5128, -0.1049],  // Ludgate Circus
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
    m.on('click', (e) => {{ L.DomEvent.stopPropagation(e); selectVenue(v.id); }});
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

const TIME_BANDS = ['Dawn', 'Morning', 'Midday', 'Afternoon', 'Evening', 'Night'];

function _dutyCycleAttenuation(fraction, floor) {{
    const clamped = Math.max(0, Math.min(1, fraction));
    if (clamped >= 1) return 1;
    return floor + (1 - floor) * clamped;
}}

function _eventMonthWindow(evt, year) {{
    if (evt.month_start === null) return null;
    let mStart = evt.month_start;
    if (evt.calendar_break && evt.month_start_ns && year >= evt.calendar_break) {{
        mStart = evt.month_start_ns;
    }}
    const mEnd = evt.month_end !== null ? evt.month_end : mStart;
    return [mStart, mEnd];
}}

function _eventActivationWeight(evt, year, month, dow, band) {{
    const ys = evt.year_start, ye = evt.year_end;
    if (ys !== null && year < ys) return 0;
    if (ye !== null && year > ye) return 0;

    let weight = 1;
    const monthWindow = _eventMonthWindow(evt, year);
    if (monthWindow) {{
        const [mStart, mEnd] = monthWindow;
        if (month !== null) {{
            if (month < mStart || month > mEnd) return 0;
        }} else {{
            weight *= _dutyCycleAttenuation((mEnd - mStart + 1) / 12, 0.30);
        }}
    }}

    if (evt.day_of_week) {{
        const days = evt.day_of_week.split('|');
        if (dow !== null) {{
            if (!days.includes(dow)) return 0;
        }} else {{
            weight *= _dutyCycleAttenuation(days.length / 7, 0.34);
        }}
    }}

    if (evt.time_bands) {{
        const bands = evt.time_bands.split('|');
        if (band !== null) {{
            if (!bands.includes(band)) return 0;
        }} else {{
            weight *= _dutyCycleAttenuation(bands.length / TIME_BANDS.length, 0.38);
        }}
    }}

    return weight;
}}

function _instanceActivationWeight(inst, month) {{
    if (month !== null) {{
        if (inst.month !== null && inst.month !== month) return 0;
        return 1;
    }}
    return inst.month !== null ? 0.32 : 1;
}}

function _eventMatchesBuildingType(evtId, buildingType) {{
    if (!buildingType) return true;
    return EVENT_VENUES.some(ev => {{
        if (ev.event_id !== evtId) return false;
        const venue = VENUES.find(v => v.id === ev.venue_id);
        return !!venue && venue.building_type === buildingType;
    }});
}}

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

        const activationWeight = _eventActivationWeight(evt, year, month, dow, band);
        if (activationWeight <= 0) return;

        loads.smell  = Math.min(1, loads.smell  + (evt.smell_load  || 0) * activationWeight);
        loads.noise  = Math.min(1, loads.noise  + (evt.noise_load  || 0) * activationWeight);
        loads.crowd  = Math.min(1, loads.crowd  + (evt.crowd_load  || 0) * activationWeight);
        loads.visual = Math.min(1, loads.visual + (evt.visual_load || 0) * activationWeight);
    }});

    // Event instances for specific year
    EVENT_INSTANCES.forEach(inst => {{
        if (inst.year !== year) return;
        const instWeight = _instanceActivationWeight(inst, month);
        if (instWeight <= 0) return;
        const linked = EVENT_VENUES.some(ev => ev.event_id === inst.event_id && ev.venue_id === venueId);
        if (!linked) return;
        const evt = EVENTS_BY_ID[inst.event_id];
        if (!evt) return;
        loads.smell  = Math.min(1, loads.smell  + (evt.smell_load  || 0) * instWeight);
        loads.noise  = Math.min(1, loads.noise  + (evt.noise_load  || 0) * instWeight);
        loads.crowd  = Math.min(1, loads.crowd  + (evt.crowd_load  || 0) * instWeight);
        loads.visual = Math.min(1, loads.visual + (evt.visual_load || 0) * instWeight);
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

    // Smoke load: SO2 index × enclosure modifier × zone industrial intensity
    const _smokDecade = Math.floor(year / 10) * 10;
    const _smokRow = SMOKE_DATA_ENV.find(s => s.decade_start === _smokDecade)
                  || SMOKE_DATA_ENV[SMOKE_DATA_ENV.length - 1];
    const _so2 = _smokRow ? _smokRow.so2_index : 0;
    const _zoneRaw = _venue ? getZoneForPoint(_venue.lat, _venue.lon) : null;
    const _zoneInterp = _zoneRaw ? interpolateZoneProps(_zoneRaw, year) : null;
    const _indIntensity = _zoneInterp ? (_zoneInterp.industrial_intensity || 0.3) : 0.3;
    loads.smoke = Math.min(1, _so2 * smokeMult * _indIntensity);

    loads.composite = (loads.smell + loads.noise + loads.crowd + loads.visual) / 4;
    return loads;
}}

function intensityColour(c, modality = null) {{
    if (c < 0.01) return '#aaaaaa';
    if (modality === 'crowd') {{
        if (c < 0.3) return '#d3a262';
        if (c < 0.6) return '#a86b3f';
        return '#7a4629';
    }}
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
    _applyVenueDisplayGeometry(year);
    updateEnvIndicators(year, month);
    _projectStreets(year);
    updateBasemap(year);
    updateZoneFills(year, month);

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
        // Hide venues outside their operational date range
        if (v.opened !== null && year < v.opened) {{
            const _m = markersByVenueId[v.id];
            if (_m) _m.setStyle({{ opacity: 0, fillOpacity: 0 }});
            return;
        }}
        if (v.closed !== null && year > v.closed) {{
            const _m = markersByVenueId[v.id];
            if (_m) _m.setStyle({{ opacity: 0, fillOpacity: 0 }});
            return;
        }}
        // Restore visibility for in-range venues
        {{
            const _m = markersByVenueId[v.id];
            if (_m) _m.setStyle({{ opacity: 1, fillOpacity: 0.7 }});
        }}
        const intensity = computeIntensity(v.id, year, month, dow, band);
        venueIntensityCache[v.id] = intensity;
        const marker = markersByVenueId[v.id];
        if (!marker) return;

        // If a sense filter is active, drive display from that modality alone
        const displayLoad = state.modality ? (intensity[state.modality] || 0) : intensity.composite;
        const isCrowdMode = state.modality === 'crowd';
        const col = intensityColour(displayLoad, state.modality);
        const r = displayLoad < 0.01
            ? (isCrowdMode ? 3.4 : 4)
            : (isCrowdMode
                ? 3.4 + Math.pow(displayLoad, 0.78) * 4.8
                : 4 + displayLoad * 14);

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
                fillOpacity: isCrowdMode ? 0.05 : 0.08,
                weight: 0.5,
                dashArray: dashArray,
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
                fillOpacity: isCrowdMode ? 0.60 : 0.78,
                weight: state.tierView ? 1.6 : (isCrowdMode ? 0.9 : 1),
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
        const placementTip = _venuePlacementSummary(v, year, true);
        if (placementTip) {{
            tip += `<br><span style="font-size:0.76em;opacity:0.50;font-style:italic">${{placementTip}}</span>`;
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
        const zoneBaseline = computeZoneBaseline(v.lat, v.lon, year, month);
        if (zoneBaseline && zoneBaseline.provenance.length > 1) {{
            tip += `<br><span style="font-size:0.78em;opacity:0.50;font-style:italic">~ ${{zoneBaseline.provenance.slice(1).join(' \u00b7 ')}}</span>`;
        }}
        marker.setTooltipContent(tip);

        // Spread rings
        if (smellRings[v.id]) smellRings[v.id].setRadius(intensity.smell * 400);
        if (noiseRings[v.id]) noiseRings[v.id].setRadius(intensity.noise * 300);

        if (!btDimmed && intensity.composite > 0.01) {{
            const evts = getActiveEvents(v.id, year, month, dow, band);
            evts.forEach(e => {{
                if (!activeEvents.find(x => x.evt.event_id === e.evt.event_id && x.venueId === v.id))
                    activeEvents.push({{...e, venueId: v.id, venueName: v.name}});
            }});
        }}
    }});

    // Show/hide procession route polylines
    Object.entries(routeLines).forEach(([evtId, line]) => {{
        if (isEventActive(evtId, year, month, dow, band) && _eventMatchesBuildingType(evtId, state.buildingType)) {{
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

    applyNightMode(band);
    updateMapLegend();
    if (state.contourMode !== 'off' && contourLayer && map.hasLayer(contourLayer)) {{
        contourLayer.redraw();
    }}
    if (_heatmapOn) {{ _doUpdateHeatmap(); _startPulseRings(); }}
}}

function getActiveEvents(venueId, year, month, dow, band) {{
    const results = [];
    EVENT_VENUES.forEach(ev => {{
        if (ev.venue_id !== venueId) return;
        const evt = EVENTS_BY_ID[ev.event_id];
        if (!evt) return;
        // Irregular/one_off events only appear via EVENT_INSTANCES (specific years)
        if (evt.recurrence === 'irregular' || evt.recurrence === 'one_off') return;
        const activationWeight = _eventActivationWeight(evt, year, month, dow, band);
        if (activationWeight <= 0) return;
        const inst = EVENT_INSTANCES.find(i => i.event_id === evt.event_id && i.year === year);
        results.push({{evt, inst: inst || null, weight: activationWeight}});
    }});
    EVENT_INSTANCES.forEach(inst => {{
        if (inst.year !== year) return;
        const instWeight = _instanceActivationWeight(inst, month);
        if (instWeight <= 0) return;
        if (!EVENT_VENUES.some(ev => ev.event_id === inst.event_id && ev.venue_id === venueId)) return;
        const evt = EVENTS_BY_ID[inst.event_id];
        if (!evt || results.find(r => r.evt.event_id === inst.event_id)) return;
        results.push({{evt, inst, weight: instWeight}});
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

    const bar = v => {{
        const filled = Math.round(Math.max(0, Math.min(1, v)) * 10);
        return '\u2588'.repeat(filled) + '\u2591'.repeat(10 - filled);
    }};
    const label = v => v > 0.7 ? 'high' : v > 0.45 ? 'moderate\u2013high' : v > 0.25 ? 'moderate' : v > 0.1 ? 'low\u2013moderate' : 'low';

    const modalities = [
        ['Smell',  baseline.smell,  '#b48a28'],
        ['Noise',  baseline.noise,  '#3c78c8'],
        ['Crowd',  baseline.crowd,  '#b43c3c'],
        ['Visual', baseline.visual, '#3ca050'],
    ];

    let html = `<div style="margin-bottom:10px;font-size:0.82em;opacity:0.65">\u2316 ${{lat.toFixed(4)}}, ${{lon.toFixed(4)}} &middot; ${{baseline.street_character}} streets</div>`;

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
        Zone inference \u2014 click a venue marker for documented evidence.
    </p>`;

    body.innerHTML = html;
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
    if (venue) {{
        html += `<div style="margin:0 0 10px;font-size:0.78em;opacity:0.58;font-style:italic">${{_venuePlacementSummary(venue, year, false)}}</div>`;
    }}
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
    return _eventActivationWeight(evt, year, month, dow, band) > 0;
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

function clearFilters() {{
    state.month = null; state.dow = null; state.band = null; state.modality = null; state.buildingType = null;
    state.selectedVenue = null;
    state.contourMode = 'off'; state.contourSense = 'smoke';
    document.querySelectorAll('.pill.active, .sense-pill.active, .btype-pill.active').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.contour-btn').forEach(b => b.classList.remove('active'));
    document.querySelector('.contour-btn[data-cmode="off"]')?.classList.add('active');
    document.querySelectorAll('.sense-btn').forEach(b => b.classList.remove('active'));
    document.querySelector('.sense-btn[data-sense="smoke"]')?.classList.add('active');
    document.getElementById('sense-row').style.display = 'none';
    if (contourLayer && map.hasLayer(contourLayer)) map.removeLayer(contourLayer);
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

// ── Street network + venue pixel positions ───────────────────────────────────
const venuePx = {{}};           // venue pixel positions, refreshed on move/resize

// Canyon field grid (used by heatmap for concentration modelling)
const FIELD_W = 80, FIELD_H = 60;
// Precomputed street direction field (rebuilt on pan/zoom)
const streetFieldDx  = new Float32Array(FIELD_W * FIELD_H);
const streetFieldDy  = new Float32Array(FIELD_W * FIELD_H);
const streetFieldMag = new Float32Array(FIELD_W * FIELD_H);  // 0=no street, 1=strong street
// Per-cell canyon H/W ratio (weighted avg of nearby segments). 1.2 = open; 3.0 = alley
const canyonFieldHw  = new Float32Array(FIELD_W * FIELD_H).fill(1.2);

function updateVenuePx() {{
    VENUES.forEach(v => {{
        const pt = map.latLngToContainerPoint([v.lat, v.lon]);
        venuePx[v.id] = {{ px: pt.x, py: pt.y }};
    }});
}}
updateVenuePx();

// Street segment pixel cache
let streetSegsPx = [];

// Highway-type width multiplier for channeling strength
function _hwayMult(t) {{
    if (t === 'primary' || t === 'trunk' || t === 'motorway') return 1.8;
    if (t === 'secondary') return 1.4;
    if (t === 'tertiary' || t === 'residential' || t === 'unclassified') return 1.0;
    return 0.6;  // footway, service, path, or unknown
}}

// Build per-cell street direction field using spatial bucketing (~10x speedup vs naive)
function _rebuildStreetField() {{
    if (!streetSegsPx.length) {{
        streetFieldDx.fill(0); streetFieldDy.fill(0); streetFieldMag.fill(0);
        canyonFieldHw.fill(1.2);
        return;
    }}
    const mapEl = document.getElementById('map');
    const W = mapEl.offsetWidth || 800, H = mapEl.offsetHeight || 600;
    const cW = W / FIELD_W, cH = H / FIELD_H;
    const SEARCH_PX = 40;
    // 8×6 spatial bucket grid
    const BW = 8, BH = 6;
    const bW = W / BW, bH = H / BH;
    const buckets = [];
    for (let i = 0; i < BW * BH; i++) buckets.push([]);
    streetSegsPx.forEach(seg => {{
        const minBx = Math.max(0, Math.floor(Math.min(seg.x0, seg.x1) / bW) - 1);
        const maxBx = Math.min(BW - 1, Math.floor(Math.max(seg.x0, seg.x1) / bW) + 1);
        const minBy = Math.max(0, Math.floor(Math.min(seg.y0, seg.y1) / bH) - 1);
        const maxBy = Math.min(BH - 1, Math.floor(Math.max(seg.y0, seg.y1) / bH) + 1);
        for (let by = minBy; by <= maxBy; by++)
            for (let bx = minBx; bx <= maxBx; bx++)
                buckets[by * BW + bx].push(seg);
    }});
    for (let gy = 0; gy < FIELD_H; gy++) {{
        for (let gx = 0; gx < FIELD_W; gx++) {{
            const cx = (gx + 0.5) * cW, cy = (gy + 0.5) * cH;
            let sdx = 0, sdy = 0, sumHw = 0, sumHwW = 0;
            const bx0 = Math.max(0, Math.floor(cx / bW) - 1);
            const bx1 = Math.min(BW - 1, Math.floor(cx / bW) + 1);
            const by0 = Math.max(0, Math.floor(cy / bH) - 1);
            const by1 = Math.min(BH - 1, Math.floor(cy / bH) + 1);
            const seen = new Set();
            for (let by = by0; by <= by1; by++) {{
                for (let bx = bx0; bx <= bx1; bx++) {{
                    for (const seg of buckets[by * BW + bx]) {{
                        if (seen.has(seg)) continue;
                        seen.add(seg);
                        const mx = (seg.x0 + seg.x1) * 0.5, my = (seg.y0 + seg.y1) * 0.5;
                        const ddx = mx - cx, ddy = my - cy;
                        const dist = Math.sqrt(ddx * ddx + ddy * ddy);
                        if (dist < 2 || dist > SEARCH_PX) continue;
                        const mult = _hwayMult(seg.t);
                        const w = mult / dist;
                        // Segment direction (unit vector)
                        const segDirX = (seg.x1 - seg.x0) / seg.len;
                        const segDirY = (seg.y1 - seg.y0) / seg.len;
                        sdx += segDirX * w;
                        sdy += segDirY * w;
                        // Canyon H/W weighted accumulation
                        const hw = seg.h || 1.2;
                        sumHw  += hw * w;
                        sumHwW += w;
                    }}
                }}
            }}
            const mag = Math.sqrt(sdx * sdx + sdy * sdy);
            const idx = gy * FIELD_W + gx;
            if (mag > 0.001) {{
                streetFieldDx[idx]  = sdx / mag;
                streetFieldDy[idx]  = sdy / mag;
                streetFieldMag[idx] = Math.min(1, mag * 0.08);
                canyonFieldHw[idx]  = sumHwW > 0.001 ? sumHw / sumHwW : 1.2;
            }} else {{
                streetFieldDx[idx] = 0; streetFieldDy[idx] = 0; streetFieldMag[idx] = 0;
                canyonFieldHw[idx] = 1.2;
            }}
        }}
    }}
}}

// Venue-anchor pass: stamp canyonFieldHw with documented H/W ratios around known venues.
// Blends with OHM-derived values — high-confidence historical data overrides highway-type estimates.
function _applyVenueCanyonAnchors() {{
    const mapEl = document.getElementById('map');
    const W = mapEl.offsetWidth || 800, H = mapEl.offsetHeight || 600;
    const ANCHOR_PX = 35;  // influence radius in pixels
    VENUES.forEach(v => {{
        if (typeof v.hw_ratio !== 'number') return;
        const vp = venuePx[v.id];
        if (!vp) return;
        const gxC = Math.min(FIELD_W - 1, Math.max(0, (vp.px / W) * FIELD_W) | 0);
        const gyC = Math.min(FIELD_H - 1, Math.max(0, (vp.py / H) * FIELD_H) | 0);
        const cellW = W / FIELD_W, cellH = H / FIELD_H;
        const gxR = Math.ceil(ANCHOR_PX / cellW) + 1;
        const gyR = Math.ceil(ANCHOR_PX / cellH) + 1;
        for (let gy = Math.max(0, gyC - gyR); gy <= Math.min(FIELD_H - 1, gyC + gyR); gy++) {{
            for (let gx = Math.max(0, gxC - gxR); gx <= Math.min(FIELD_W - 1, gxC + gxR); gx++) {{
                const cx = (gx + 0.5) * cellW, cy = (gy + 0.5) * cellH;
                const dist = Math.sqrt((cx - vp.px) ** 2 + (cy - vp.py) ** 2);
                if (dist > ANCHOR_PX) continue;
                // Linear blend: anchor weight 1.0 at centre → 0 at edge; mixed with existing value
                const t = dist / ANCHOR_PX;
                const w = 1 - t;
                const idx = gy * FIELD_W + gx;
                canyonFieldHw[idx] = canyonFieldHw[idx] * (1 - w) + v.hw_ratio * w;
            }}
        }}
    }});
}}

function _projectStreets(year) {{
    streetSegsPx = [];
    if (!STREET_NETWORK || !STREET_NETWORK.length) return;
    const yr = year || parseInt(document.getElementById('year-slider').value);
    STREET_NETWORK.forEach(entry => {{
        // New format: {{p:[[lat,lon],...], s:start_yr|null, e:end_yr|null, t:highway_type}}
        // Fallback: bare [[lat,lon],...] array for any old-format cached data
        const polyline = Array.isArray(entry) ? entry : entry.p;
        const s = Array.isArray(entry) ? null : entry.s;
        const e = Array.isArray(entry) ? null : entry.e;
        const t = Array.isArray(entry) ? '' : (entry.t || '');
        const h = Array.isArray(entry) ? 1.2 : (entry.h || 1.2);
        if (s !== null && s > yr) return;
        if (e !== null && e < yr) return;
        for (let i = 0; i < polyline.length - 1; i++) {{
            const a = map.latLngToContainerPoint([polyline[i][0],   polyline[i][1]]);
            const b = map.latLngToContainerPoint([polyline[i+1][0], polyline[i+1][1]]);
            const len = Math.sqrt((b.x-a.x)**2 + (b.y-a.y)**2);
            if (len < 2) continue;
            streetSegsPx.push({{ x0: a.x, y0: a.y, x1: b.x, y1: b.y, len, t, h, conn1: [], conn0: [] }});
        }}
    }});
    // Rebuild street direction field for the new projection
    _rebuildStreetField();
    // Overlay documented venue H/W ratios as high-confidence anchors
    _applyVenueCanyonAnchors();
}}

map.on('moveend zoomend', () => {{
    updateVenuePx();
    const yr = parseInt(document.getElementById('year-slider').value);
    _projectStreets(yr);
}});
if (STREET_NETWORK && STREET_NETWORK.length) {{
    const yr = parseInt(document.getElementById('year-slider').value);
    _projectStreets(yr);
}}

map.on('click', (e) => {{
    state.selectedVenue = null;
    const year  = parseInt(document.getElementById('year-slider').value);
    renderLocationPanel(e.latlng.lat, e.latlng.lng, year, state.month);
}});

// ── Heatmap overlay ────────────────────────────────────────────────────────────
const heatCanvas = document.getElementById('heatmap-canvas');
const heatCtx    = heatCanvas ? heatCanvas.getContext('2d') : null;
// Off-screen base canvas: stores the static intensity layer (re-rendered on data change)
// ── IDW Contour Surface ────────────────────────────────────────────────────────
const CONTOUR_RAMPS = {{
    atmosphere: [[245,235,220],[200,170,100],[140,95,40],[50,30,10]],
    smell:      [[245,235,220],[210,170,90],[180,110,50],[140,70,30]],
    noise:      [[230,240,250],[150,180,210],[80,120,170],[20,40,80]],
    crowd:      [[245,235,220],[220,160,140],[180,80,60],[130,20,20]],
    visual:     [[230,240,225],[160,185,140],[100,130,80],[40,70,25]],
    smoke:      [[245,235,220],[180,170,155],[120,105,85],[50,40,30]],
}};

function sampleRamp(ramp, t) {{
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

const ContourSurface = L.GridLayer.extend({{
    createTile: function(coords) {{
        const tile = document.createElement('canvas');
        const tileSize = this.getTileSize();
        tile.width = tileSize.x;
        tile.height = tileSize.y;
        const ctx = tile.getContext('2d');

        const zoom = coords.z;
        // Coarse grid — 32x32 is fast enough for smooth wash after bilinear upscale
        const gridW = 32;
        const gridH = 32;
        const cellW = tileSize.x / gridW;
        const cellH = tileSize.y / gridH;

        const mode = state.contourMode;
        if (mode === 'off') return tile;
        const rampKey = mode === 'atmosphere' ? 'atmosphere' : state.contourSense;
        const ramp = CONTOUR_RAMPS[rampKey] || CONTOUR_RAMPS.atmosphere;

        // Precompute venue pixel positions in tile space (avoids per-cell Haversine)
        const tileOriginX = coords.x * tileSize.x;
        const tileOriginY = coords.y * tileSize.y;
        // Metres per pixel at this zoom (approximate, using lat of London ~51.5°)
        const mPerPx = 156543.03 * Math.cos(51.5 * Math.PI / 180) / Math.pow(2, zoom);
        const venuesPx = [];
        VENUES.forEach(v => {{
            const cache = venueIntensityCache[v.id];
            if (!cache) return;
            const pt = map.project(L.latLng(v.lat, v.lon), zoom);
            const enc = v.enclosure || 'open';
            const cutoffM = enc === 'enclosed' ? 400 : enc === 'semi_open' ? 600 : 800;
            venuesPx.push({{
                px: pt.x - tileOriginX,
                py: pt.y - tileOriginY,
                lon: v.lon,
                cache: cache,
                cutoffPx: cutoffM / mPerPx,
                cutoffPx2: (cutoffM / mPerPx) ** 2,
            }});
        }});

        const _idwPass = (gridOut, modality, applyWindBias) => {{
            for (let gy = 0; gy < gridH; gy++) {{
                const py = (gy + 0.5) * cellH;
                for (let gx = 0; gx < gridW; gx++) {{
                    const px = (gx + 0.5) * cellW;
                    let wSum = 0, vSum = 0;
                    for (let vi = 0; vi < venuesPx.length; vi++) {{
                        const vp = venuesPx[vi];
                        const intensity = vp.cache[modality] || 0;
                        if (intensity <= 0.01) continue;
                        let dx = px - vp.px, dy = py - vp.py;
                        // Wind bias: pixel east of venue → dist compressed (reaches further)
                        if (applyWindBias) {{
                            if (dx > 0) dx *= 0.77;    // pixel is east of venue
                            else dx *= 1.43;            // pixel is west of venue
                        }}
                        const dist2 = dx * dx + dy * dy;
                        if (dist2 > vp.cutoffPx2) continue;
                        const distM = Math.sqrt(dist2) * mPerPx;
                        const w = 1.0 / Math.max(distM, 1) ** 2;
                        wSum += w;
                        vSum += w * intensity;
                    }}
                    gridOut[gy * gridW + gx] = wSum > 0 ? vSum / wSum : 0;
                }}
            }}
        }};

        const grid = new Float32Array(gridW * gridH);
        const maxAlpha = 0.65;

        if (mode === 'atmosphere') {{
            const smokeGrid = new Float32Array(gridW * gridH);
            const smellGrid = new Float32Array(gridW * gridH);
            _idwPass(smokeGrid, 'smoke', true);
            _idwPass(smellGrid, 'smell', false);
            for (let i = 0; i < grid.length; i++) {{
                grid[i] = 0.7 * smokeGrid[i] + 0.3 * smellGrid[i];
            }}
        }} else {{
            _idwPass(grid, state.contourSense, false);
        }}
        // Boost: IDW averages are inherently low (0.05–0.15); apply
        // sqrt gain so the surface is visible and contour thresholds fire
        for (let i = 0; i < grid.length; i++) {{
            grid[i] = Math.sqrt(Math.min(1, grid[i] * 3.0));
        }}

        const sigma = 1;  // light blur on 32x32 grid
        this._blurGrid(grid, gridW, gridH, sigma);

        // Bilinear interpolation for gradient wash
        const imgData = ctx.createImageData(tileSize.x, tileSize.y);
        for (let py2 = 0; py2 < tileSize.y; py2++) {{
            for (let px2 = 0; px2 < tileSize.x; px2++) {{
                const gxf = (px2 / tileSize.x) * gridW - 0.5;
                const gyf = (py2 / tileSize.y) * gridH - 0.5;
                const gx0 = Math.max(0, Math.floor(gxf));
                const gy0 = Math.max(0, Math.floor(gyf));
                const gx1 = Math.min(gridW - 1, gx0 + 1);
                const gy1 = Math.min(gridH - 1, gy0 + 1);
                const fx = Math.max(0, gxf - gx0), fy = Math.max(0, gyf - gy0);
                const val = (1-fx)*(1-fy) * grid[gy0*gridW+gx0]
                          +    fx *(1-fy) * grid[gy0*gridW+gx1]
                          + (1-fx)*   fy  * grid[gy1*gridW+gx0]
                          +    fx *   fy  * grid[gy1*gridW+gx1];
                const rgb = sampleRamp(ramp, val);
                const alpha = Math.round(val * maxAlpha * 255);
                const idx = (py2 * tileSize.x + px2) * 4;
                imgData.data[idx] = rgb[0];
                imgData.data[idx+1] = rgb[1];
                imgData.data[idx+2] = rgb[2];
                imgData.data[idx+3] = alpha;
            }}
        }}
        ctx.putImageData(imgData, 0, 0);

        this._drawContours(ctx, grid, gridW, gridH, cellW, cellH, ramp);
        return tile;
    }},

    _drawContours: function(ctx, grid, gridW, gridH, cellW, cellH, ramp) {{
        const thresholds = [
            {{ val: 0.25, dash: [4, 3], width: 0.8, alpha: 0.45 }},
            {{ val: 0.45, dash: [4, 3], width: 1.0, alpha: 0.55 }},
            {{ val: 0.65, dash: [],     width: 1.2, alpha: 0.70 }},
        ];
        thresholds.forEach(th => {{
            const rgb = sampleRamp(ramp, th.val);
            ctx.strokeStyle = `rgba(${{rgb[0]}},${{rgb[1]}},${{rgb[2]}},${{th.alpha}})`;
            ctx.lineWidth = th.width;
            ctx.setLineDash(th.dash);
            ctx.beginPath();
            for (let gy = 0; gy < gridH - 1; gy++) {{
                for (let gx = 0; gx < gridW - 1; gx++) {{
                    const tl = grid[gy * gridW + gx];
                    const tr = grid[gy * gridW + gx + 1];
                    const bl = grid[(gy+1) * gridW + gx];
                    const br = grid[(gy+1) * gridW + gx + 1];
                    const config = (tl >= th.val ? 8 : 0) | (tr >= th.val ? 4 : 0)
                                 | (br >= th.val ? 2 : 0) | (bl >= th.val ? 1 : 0);
                    if (config === 0 || config === 15) continue;
                    const lerp = (a, b) => a === b ? 0.5 : (th.val - a) / (b - a);
                    const cx0 = gx * cellW, cy0 = gy * cellH;
                    const cx1 = (gx+1) * cellW, cy1 = (gy+1) * cellH;
                    const top = [cx0 + lerp(tl, tr) * cellW, cy0];
                    const right = [cx1, cy0 + lerp(tr, br) * cellH];
                    const bottom = [cx0 + lerp(bl, br) * cellW, cy1];
                    const left = [cx0, cy0 + lerp(tl, bl) * cellH];
                    const segments = [];
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
                    segments.forEach(([a, b]) => {{ ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); }});
                }}
            }}
            ctx.stroke();
        }});
        ctx.setLineDash([]);
        this._drawContourLabels(ctx, grid, gridW, gridH, cellW, cellH, ramp);
    }},

    _drawContourLabels: function(ctx, grid, gridW, gridH, cellW, cellH, ramp) {{
        const thresholds = [0.25, 0.45, 0.65];
        ctx.font = 'italic 9px Georgia';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
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

    _blurGrid: function(grid, w, h, sigma) {{
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
}});

let contourLayer = null;
contourLayer = new ContourSurface({{ opacity: 1, zIndex: 400, className: 'contour-surface' }});

const _heatBase    = document.createElement('canvas');
const _heatBaseCtx = _heatBase.getContext('2d');
let _heatmapOn   = false;

function _resizeHeatmap() {{
    if (!heatCanvas) return;
    const mapEl = document.getElementById('map');
    const W = mapEl.offsetWidth, H = mapEl.offsetHeight;
    heatCanvas.width = W; heatCanvas.height = H;
    _heatBase.width  = W; _heatBase.height  = H;
}}
_resizeHeatmap();
window.addEventListener('resize', _resizeHeatmap);

const HEATMAP_RGB = {{
    smell:     [195, 130, 25],
    noise:     [60,  130, 220],
    crowd:     [168, 106, 58],
    visual:    [50,  175, 70],
    composite: [200, 105, 35],
}};

// Renders the static intensity layer to the off-screen base canvas
function _doUpdateHeatmap() {{
    if (!_heatBaseCtx || !_heatBase || !_heatmapOn) return;
    const W = _heatBase.width, H = _heatBase.height;
    if (!W || !H) return;
    const COLS = 60, ROWS = 45;
    const cellW = W / COLS, cellH = H / ROWS;
    const grid = new Float32Array(COLS * ROWS);
    const sense = state.modality || 'composite';
    const maxRadius = sense === 'noise' ? 135 : (sense === 'smell' ? 240 : 195);
    const [cr, cg, cb] = HEATMAP_RGB[sense] || HEATMAP_RGB.composite;

    VENUES.forEach(v => {{
        const loads = venueIntensityCache[v.id];
        if (!loads) return;
        const vp = venuePx[v.id];
        if (!vp) return;
        const intensity = sense === 'composite' ? loads.composite : (loads[sense] || 0);
        if (intensity < 0.015) return;
        const enc = v.enclosure || 'open';
        const encF = enc === 'enclosed' ? 0.35 : enc === 'semi_open' ? 0.65 : 1.0;
        const eff = intensity * encF;
        const colMin = Math.max(0, Math.floor((vp.px - maxRadius) / cellW));
        const colMax = Math.min(COLS - 1, Math.ceil( (vp.px + maxRadius) / cellW));
        const rowMin = Math.max(0, Math.floor((vp.py - maxRadius) / cellH));
        const rowMax = Math.min(ROWS - 1, Math.ceil( (vp.py + maxRadius) / cellH));
        const maxR2 = maxRadius * maxRadius;
        const applyCanyon = (sense === 'smoke' || sense === 'composite');
        for (let row = rowMin; row <= rowMax; row++) {{
            for (let col = colMin; col <= colMax; col++) {{
                const cx = (col + 0.5) * cellW, cy = (row + 0.5) * cellH;
                const dist2 = (cx - vp.px) ** 2 + (cy - vp.py) ** 2;
                if (dist2 > maxR2) continue;
                const t = dist2 / maxR2;
                // Canyon retention: narrow streets concentrate smoke/smell in heatmap
                let canyonMult = 1.0;
                if (applyCanyon) {{
                    const cgx = Math.min(FIELD_W-1, Math.max(0, (cx/W)*FIELD_W)|0);
                    const cgy = Math.min(FIELD_H-1, Math.max(0, (cy/H)*FIELD_H)|0);
                    const hw = canyonFieldHw[cgy * FIELD_W + cgx] || 1.2;
                    canyonMult = Math.min(1.8, hw / 1.2);
                }}
                grid[row * COLS + col] = Math.min(1, grid[row * COLS + col] + eff * Math.pow(1 - t, 1.4) * canyonMult);
            }}
        }}
    }});

    // 1-pass box blur
    const blurred = new Float32Array(COLS * ROWS);
    for (let row = 1; row < ROWS - 1; row++) {{
        for (let col = 1; col < COLS - 1; col++) {{
            let s = 0;
            for (let dr = -1; dr <= 1; dr++)
                for (let dc = -1; dc <= 1; dc++)
                    s += grid[(row + dr) * COLS + (col + dc)];
            blurred[row * COLS + col] = s / 9;
        }}
    }}

    _heatBaseCtx.clearRect(0, 0, W, H);
    for (let row = 0; row < ROWS; row++) {{
        for (let col = 0; col < COLS; col++) {{
            const val = blurred[row * COLS + col] || grid[row * COLS + col];
            if (val < 0.008) continue;
            const alpha = Math.min(0.70, val * 0.88);
            _heatBaseCtx.fillStyle = 'rgba(' + cr + ',' + cg + ',' + cb + ',' + alpha.toFixed(2) + ')';
            _heatBaseCtx.fillRect(
                Math.floor(col * cellW), Math.floor(row * cellH),
                Math.ceil(cellW) + 1, Math.ceil(cellH) + 1
            );
        }}
    }}
    // If no pulse loop running, blit to display canvas immediately
    if (!_pulseRaf) {{
        heatCtx.clearRect(0, 0, W, H);
        heatCtx.drawImage(_heatBase, 0, 0);
    }}
}}

// ── Heatmap pulse rings (RAF loop composites base + expanding rings) ──────────
let _pulseRaf = null;
const _pulseRings = [];

function _startPulseRings() {{
    if (_pulseRaf) cancelAnimationFrame(_pulseRaf);
    _pulseRings.length = 0;
    VENUES.forEach(v => {{
        const loads = venueIntensityCache[v.id];
        if (!loads || loads.composite < 0.35) return;
        const vp = venuePx[v.id];
        if (!vp) return;
        _pulseRings.push({{ vx: vp.px, vy: vp.py, t: Math.random(), intensity: loads.composite, vid: v.id }});
    }});
    function _pulseLoop() {{
        if (!_heatmapOn || !heatCtx || !heatCanvas) {{ _pulseRaf = null; return; }}
        const W = heatCanvas.width, H = heatCanvas.height;
        heatCtx.clearRect(0, 0, W, H);
        // Blit the static base layer (pre-computed in _doUpdateHeatmap)
        if (_heatBase.width === W && _heatBase.height === H)
            heatCtx.drawImage(_heatBase, 0, 0);
        // Overlay expanding rings
        const isNight = (state.band === 'night');
        const [cr, cg, cb] = HEATMAP_RGB[state.modality || 'composite'] || [200, 105, 35];
        _pulseRings.forEach(ring => {{
            ring.t += 0.007;
            if (ring.t > 1) {{
                ring.t = 0;
                const vp = venuePx[ring.vid];
                if (vp) {{ ring.vx = vp.px; ring.vy = vp.py; }}
            }}
            const loads = venueIntensityCache[ring.vid];
            if (!loads || loads.composite < 0.3) return;
            const r = ring.t * 55 + 6;
            const alpha = (1 - ring.t) * ring.intensity * (isNight ? 0.60 : 0.40);
            if (alpha < 0.01) return;
            heatCtx.strokeStyle = 'rgba(' + cr + ',' + cg + ',' + cb + ',' + alpha.toFixed(2) + ')';
            heatCtx.lineWidth = 2.5;
            heatCtx.beginPath();
            heatCtx.arc(ring.vx, ring.vy, r, 0, Math.PI * 2);
            heatCtx.stroke();
        }});
        _pulseRaf = requestAnimationFrame(_pulseLoop);
    }}
    _pulseRaf = requestAnimationFrame(_pulseLoop);
}}

function toggleHeatmap() {{
    _heatmapOn = !_heatmapOn;
    if (heatCanvas) heatCanvas.style.opacity = _heatmapOn ? '0.65' : '0';
    const btn = document.getElementById('heatmap-btn');
    if (btn) btn.classList.toggle('active', _heatmapOn);
    if (_heatmapOn) {{
        _doUpdateHeatmap();
        _startPulseRings();
    }} else {{
        if (_pulseRaf) {{ cancelAnimationFrame(_pulseRaf); _pulseRaf = null; }}
        if (heatCtx && heatCanvas) heatCtx.clearRect(0, 0, heatCanvas.width, heatCanvas.height);
    }}
}}

map.on('moveend zoomend', () => {{
    if (_heatmapOn) {{
        updateVenuePx();
        _doUpdateHeatmap();
        // refresh ring positions
        _pulseRings.forEach(ring => {{
            const vp = venuePx[ring.vid];
            if (vp) {{ ring.vx = vp.px; ring.vy = vp.py; }}
        }});
    }}
}});

// ── Night mode ─────────────────────────────────────────────────────────────────
function applyNightMode(band) {{
    const isNight = (band === 'night');
    const mapEl = document.getElementById('map');
    const ctrl  = document.getElementById('controls');
    const nightOverlay = document.getElementById('night-overlay');
    if (mapEl) mapEl.classList.toggle('night-mode', isNight);
    if (ctrl)  ctrl.classList.toggle('night-ctrl',  isNight);
    if (nightOverlay) nightOverlay.style.opacity = isNight ? '1' : '0';
    applyTimeTint(band);
}}

// ── Time-of-day atmospheric tint ───────────────────────────────────────────────
const TOD_TINTS = {{
    dawn:      {{ bg: 'radial-gradient(ellipse at 50% 110%, rgba(210,120,10,0.28) 0%, rgba(210,120,10,0) 60%)', op: '1' }},
    morning:   {{ bg: 'rgba(255,210,120,0.04)',                                                                  op: '1' }},
    midday:    {{ bg: 'rgba(200,220,255,0.02)',                                                                  op: '1' }},
    afternoon: {{ bg: 'rgba(255,180,60,0.06)',                                                                   op: '1' }},
    evening:   {{ bg: 'radial-gradient(ellipse at 50% 110%, rgba(200,70,0,0.30) 0%, rgba(180,50,0,0) 60%)',     op: '1' }},
    night:     {{ bg: 'rgba(0,0,0,0)',                                                                           op: '0' }},
}};
function applyTimeTint(band) {{
    const el = document.getElementById('tod-tint');
    if (!el) return;
    const t = TOD_TINTS[band] || {{ bg: 'rgba(0,0,0,0)', op: '0' }};
    el.style.background = t.bg;
    el.style.opacity    = t.op;
}}

// ── Venue name labels at zoom ≥ 14 ─────────────────────────────────────────────
const _venueLabels = {{}};
function _updateVenueLabels() {{
    const zoom = map.getZoom();
    if (zoom < 14) {{
        Object.values(_venueLabels).forEach(lyr => {{ if (map.hasLayer(lyr)) map.removeLayer(lyr); }});
        return;
    }}
    VENUES.forEach(v => {{
        if (typeof v.lat !== 'number' || typeof v.lon !== 'number' || !isFinite(v.lat) || !isFinite(v.lon)) return;
        if (!_venueLabels[v.id]) {{
            _venueLabels[v.id] = L.marker([v.lat, v.lon], {{
                icon: L.divIcon({{ className: 'venue-label', html: v.name, iconSize: null, iconAnchor: [0, 0] }}),
                interactive: false, keyboard: false, zIndexOffset: -100,
            }});
        }}
        if (!map.hasLayer(_venueLabels[v.id])) _venueLabels[v.id].addTo(map);
    }});
}}
map.on('zoomend', _updateVenueLabels);

// ── Map legend (DOM-safe, no innerHTML) ────────────────────────────────────────
function updateMapLegend() {{
    const el = document.getElementById('map-legend');
    if (!el) return;
    const senseNames  = {{ smell: 'Smell', noise: 'Noise', crowd: 'Crowd', visual: 'Visual' }};
    const activeLabel = state.modality ? (senseNames[state.modality] || 'Sensory') : 'Sensory intensity';
    const isNight = (state.band === 'night');
    el.style.background  = isNight ? 'rgba(10,18,32,0.92)' : 'rgba(255,255,255,0.93)';
    el.style.borderColor = isNight ? '#1a2a3a' : '#d8d4cc';
    el.style.color       = isNight ? '#8aabcc' : '#444';

    while (el.firstChild) el.removeChild(el.firstChild);

    const title = document.createElement('div');
    title.className = 'leg-title';
    title.textContent = activeLabel;
    el.appendChild(title);

    [
        {{ sz: 14, col: '#dc2626', lbl: 'High' }},
        {{ sz: 11, col: '#f97316', lbl: 'Moderate' }},
        {{ sz: 8,  col: '#f59e0b', lbl: 'Low' }},
        {{ sz: 6,  col: '#aaa',    lbl: 'Inactive' }},
    ].forEach(item => {{
        const row = document.createElement('div');
        row.className = 'leg-row';
        const dot = document.createElement('span');
        dot.className = 'leg-dot';
        dot.style.width = item.sz + 'px';
        dot.style.height = item.sz + 'px';
        dot.style.background = item.col;
        const lbl = document.createElement('span');
        lbl.textContent = item.lbl;
        row.appendChild(dot);
        row.appendChild(lbl);
        el.appendChild(row);
    }});

    if (state.tierView) {{
        const note = document.createElement('div');
        note.style.cssText = 'margin-top:4px;font-size:0.85em;opacity:0.6;font-style:italic';
        note.textContent = 'Tier view';
        el.appendChild(note);
    }}
}}

updateMap();
</script>
</body>
</html>
"""


def build(
    venues_path: Path = VENUES_PATH,
    db_path: Path = DB_PATH,
    out_path: Path = OUT_PATH,
    venue_geometries_path: Path = VENUE_GEOMETRIES_PATH,
    events_path: Path = EVENTS_PATH,
    event_venues_path: Path = EVENT_VENUES_PATH,
    event_instances_path: Path = EVENT_INSTANCES_PATH,
) -> None:
    data = load_data(
        venues_path,
        db_path,
        venue_geometries_path,
        events_path,
        event_venues_path,
        event_instances_path,
    )

    zones_path = Path(__file__).parent / "zones.json"
    zones_data = json.loads(zones_path.read_text(encoding="utf-8")) if zones_path.exists() else {"type": "FeatureCollection", "features": []}

    streets = fetch_ohm_streets()

    html = HTML_TEMPLATE.format(
        EVENTS_JSON          = json.dumps(data["events"],          ensure_ascii=False),
        EVENT_VENUES_JSON    = json.dumps(data["event_venues"],    ensure_ascii=False),
        EVENT_INSTANCES_JSON = json.dumps(data["event_instances"], ensure_ascii=False),
        VENUES_JSON          = json.dumps(data["venues"],          ensure_ascii=False),
        VENUE_GEOMETRIES_JSON = json.dumps(data["venue_geometries"], ensure_ascii=False),
        EVIDENCE_JSON        = json.dumps(data["evidence"],        ensure_ascii=False),
        CET_JSON             = json.dumps(data["cet"],             ensure_ascii=False),
        MORTALITY_JSON       = json.dumps(data["mortality"],       ensure_ascii=False),
        SMOKE_JSON           = json.dumps(data["smoke"],           ensure_ascii=False),
        STREET_NETWORK_JSON  = json.dumps(streets,                 ensure_ascii=False),
        ZONE_DATA_JSON       = json.dumps(zones_data,             ensure_ascii=False),
    )

    out_path.write_text(html, encoding="utf-8")
    n_venues = len(data["venues"])
    n_events = len(data["events"])
    n_ev     = len(data["evidence"])
    print(f"Sensory time map -> {out_path}")
    print(f"  {n_venues} venues  {n_events} event types  {n_ev} evidence passages")


if __name__ == "__main__":
    build()
