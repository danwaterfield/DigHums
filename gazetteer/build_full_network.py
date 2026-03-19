#!/usr/bin/env python3
"""
Build a self-contained HTML visualisation of the unified 18th-century
correspondence network.

Reads unified_correspondence_graph.csv and correspondent_locations.csv,
produces a D3 force-directed graph in full_network.html.

Usage:
    python3 gazetteer/build_full_network.py
    open gazetteer/full_network.html
"""

import csv
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
GRAPH_CSV = BASE / "unified_correspondence_graph.csv"
LOCATIONS_CSV = BASE / "correspondent_locations.csv"
PERSON_INFO = BASE / "person_info.json"
OUT_PATH = BASE / "full_network.html"
D3_CACHE = BASE / ".d3.v7.min.js"
D3_URL = "https://d3js.org/d3.v7.min.js"

# Source dataset colour palette
SOURCE_COLOURS = {
    "hemlow_catalogue": "#4a6fa5",   # slate blue — Burney catalogues
    "johnson_hill":     "#a07855",   # warm stone — Johnson
    "burke_1844":       "#5a8a7a",   # muted teal — Burke
    "walpole_1820":     "#7a7a8a",   # cool grey — Walpole
    "piozzi":           "#a07080",   # dusty rose — Piozzi
    "priestley":        "#6a5a8a",   # muted purple — Priestley
    "correspsearch":    "#8a7a5a",   # warm khaki — CorrespSearch
}

SOURCE_LABELS = {
    "hemlow_catalogue": "Burney catalogues",
    "johnson_hill":     "Johnson letters",
    "burke_1844":       "Burke correspondence",
    "walpole_1820":     "Walpole",
    "piozzi":           "Piozzi",
    "priestley":        "Priestley",
    "correspsearch":    "CorrespSearch",
}


def _get_d3_source() -> str:
    """Return D3 v7 minified JS, fetching and caching if needed."""
    if D3_CACHE.exists():
        return D3_CACHE.read_text(encoding="utf-8")
    import urllib.request
    print(f"Fetching D3 v7 from {D3_URL} ...")
    req = urllib.request.Request(D3_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        src = resp.read().decode("utf-8")
    D3_CACHE.write_text(src, encoding="utf-8")
    print(f"Cached -> {D3_CACHE}")
    return src


def load_graph() -> dict:
    """Load the unified graph CSV and build JSON-ready data structure."""
    edges = []
    node_weight = {}   # name -> total weight
    node_sources = {}  # name -> set of sources

    with open(GRAPH_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            a = row["person_a"].strip()
            b = row["person_b"].strip()
            w = int(row["weight"])
            y0_raw = row["year_min"].strip()
            y1_raw = row["year_max"].strip()
            y0 = int(y0_raw) if y0_raw else 0
            y1 = int(y1_raw) if y1_raw else 0
            src = row["sources"].strip()

            edges.append({
                "source": a,
                "target": b,
                "weight": w,
                "year_min": y0,
                "year_max": y1,
                "src": src,
            })

            for name in (a, b):
                node_weight[name] = node_weight.get(name, 0) + w
                if name not in node_sources:
                    node_sources[name] = set()
                node_sources[name].add(src)

    # Build nodes
    nodes = []
    for name in sorted(node_weight, key=lambda n: -node_weight[n]):
        srcs = sorted(node_sources[name])
        multi = len(srcs) > 1
        # Primary source = the one with most weight for this person
        src_weights = {}
        for e in edges:
            for side in ("source", "target"):
                if e[side] == name:
                    s = e["src"]
                    src_weights[s] = src_weights.get(s, 0) + e["weight"]
        primary = max(src_weights, key=src_weights.get) if src_weights else srcs[0]

        nodes.append({
            "id": name,
            "weight": node_weight[name],
            "sources": srcs,
            "primary": primary,
            "multi": multi,
        })

    # Load locations (pick the location with the widest date range per person)
    locations = {}
    with open(LOCATIONS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["name"].strip()
            lat = float(row["lat"])
            lon = float(row["lon"])
            # Just keep first entry (they're typically ordered by importance)
            if name not in locations:
                locations[name] = {"lat": lat, "lon": lon, "place": row.get("place_name", "")}

    # Attach location to nodes
    for n in nodes:
        loc = locations.get(n["id"])
        if loc:
            n["lat"] = loc["lat"]
            n["lon"] = loc["lon"]
            n["place"] = loc["place"]

    # Year range (exclude zero/missing and outliers beyond 1850)
    all_years = []
    for e in edges:
        if 1650 <= e["year_min"] <= 1850:
            all_years.append(e["year_min"])
        if 1650 <= e["year_max"] <= 1850:
            all_years.append(e["year_max"])
    year_min = min(all_years) if all_years else 1730
    year_max = max(all_years) if all_years else 1840

    return {
        "nodes": nodes,
        "edges": edges,
        "year_min": year_min,
        "year_max": year_max,
        "source_colours": SOURCE_COLOURS,
        "source_labels": SOURCE_LABELS,
    }


# ── HTML template ─────────────────────────────────────────────────
# Convention: ALL JS braces are doubled.  Only {D3_SOURCE},
# {GRAPH_JSON}, and {PERSON_INFO_JSON} use single braces (Python .replace() targets).

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>18th-Century Correspondence Network</title>
<style>
*,*::before,*::after {{ box-sizing:border-box; margin:0; padding:0; }}
html,body {{ height:100%; overflow:hidden; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #fafafa;
  color: #1a1a1a;
  display: flex;
  flex-direction: column;
}}

/* ── Header ─────────────────────────────────────── */
.header {{
  padding: 10px 24px;
  background: #fff;
  border-bottom: 1px solid #e0e0e0;
  display: flex;
  align-items: center;
  gap: 16px;
  flex-shrink: 0;
  z-index: 10;
  flex-wrap: wrap;
}}
.header h1 {{
  font-size: 16px;
  font-weight: 600;
  letter-spacing: -0.02em;
  color: #2d3436;
}}
.header .subtitle {{
  font-size: 12px;
  color: #888;
  font-weight: 400;
}}
.stats {{
  margin-left: auto;
  font-size: 11px;
  color: #999;
  display: flex;
  gap: 16px;
}}
.stats span {{ font-variant-numeric: tabular-nums; }}

/* ── Controls bar ──────────────────────────────── */
.controls {{
  padding: 8px 24px;
  background: #fff;
  border-bottom: 1px solid #e8e8e8;
  display: flex;
  align-items: center;
  gap: 20px;
  flex-shrink: 0;
  flex-wrap: wrap;
}}
.control-group {{
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 11px;
  color: #666;
}}
.control-group label {{
  white-space: nowrap;
  font-weight: 500;
  color: #555;
}}
.control-group input[type="range"] {{
  width: 120px;
  -webkit-appearance: none;
  appearance: none;
  height: 4px;
  background: #e0e0e0;
  border-radius: 2px;
  outline: none;
}}
.control-group input[type="range"]::-webkit-slider-thumb {{
  -webkit-appearance: none;
  width: 14px; height: 14px;
  border-radius: 50%;
  background: #2d3436;
  border: 2px solid #fff;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
  cursor: pointer;
}}
.control-group input[type="range"]::-moz-range-thumb {{
  width: 14px; height: 14px;
  border-radius: 50%;
  background: #2d3436;
  border: 2px solid #fff;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
  cursor: pointer;
}}
.val {{
  font-weight: 600;
  color: #2d3436;
  min-width: 20px;
  text-align: center;
}}

/* ── Search ────────────────────────────────────── */
.search-wrap {{
  position: relative;
}}
.search-wrap input {{
  font-size: 12px;
  padding: 5px 10px 5px 28px;
  border: 1px solid #d0d0d0;
  border-radius: 4px;
  width: 200px;
  outline: none;
  background: #fff;
}}
.search-wrap input:focus {{
  border-color: #2d3436;
}}
.search-icon {{
  position: absolute;
  left: 8px;
  top: 50%;
  transform: translateY(-50%);
  font-size: 13px;
  color: #999;
  pointer-events: none;
}}
.search-results {{
  position: absolute;
  top: 100%;
  left: 0;
  width: 260px;
  max-height: 200px;
  overflow-y: auto;
  background: #fff;
  border: 1px solid #d0d0d0;
  border-radius: 4px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  z-index: 100;
  display: none;
  font-size: 12px;
}}
.search-results .sr-item {{
  padding: 6px 10px;
  cursor: pointer;
  border-bottom: 1px solid #f0f0f0;
}}
.search-results .sr-item:hover {{
  background: #f5f5f5;
}}
.search-results .sr-item .sr-name {{
  font-weight: 500;
}}
.search-results .sr-item .sr-meta {{
  font-size: 10px;
  color: #999;
}}

/* ── Source checkboxes ─────────────────────────── */
.source-checks {{
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}}
.source-check {{
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
  cursor: pointer;
  white-space: nowrap;
}}
.source-check input {{
  margin: 0;
  cursor: pointer;
}}
.source-swatch {{
  width: 8px; height: 8px;
  border-radius: 2px;
  flex-shrink: 0;
}}

/* ── Reset button ─────────────────────────────── */
.reset-btn {{
  font-size: 11px;
  padding: 4px 14px;
  border: 1px solid #d0d0d0;
  border-radius: 12px;
  background: #fff;
  color: #555;
  cursor: pointer;
  white-space: nowrap;
  transition: background 0.15s, color 0.15s;
}}
.reset-btn:hover {{
  background: #2d3436;
  color: #fff;
  border-color: #2d3436;
}}

/* ── Bio in detail panel ─────────────────────── */
.dp-dates {{
  font-size: 12px;
  color: #999;
  margin-bottom: 2px;
}}
.dp-bio {{
  font-size: 13px;
  font-weight: 400;
  color: #555;
  line-height: 1.5;
  margin-top: 6px;
  margin-bottom: 4px;
}}
.dp-bio.dim {{
  color: #bbb;
  font-style: italic;
}}

/* ── Main layout ───────────────────────────────── */
.main {{
  flex: 1;
  display: flex;
  position: relative;
  overflow: hidden;
}}
.graph-area {{
  flex: 1;
  position: relative;
}}
svg {{ display: block; width: 100%; height: 100%; }}

/* ── Tooltip ───────────────────────────────────── */
.tooltip {{
  position: absolute;
  pointer-events: none;
  background: rgba(255,255,255,0.97);
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 8px 12px;
  font-size: 12px;
  line-height: 1.5;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  max-width: 320px;
  z-index: 20;
  display: none;
}}
.tooltip .tt-name {{ font-weight: 600; font-size: 13px; }}
.tooltip .tt-meta {{ font-size: 11px; color: #666; }}
.tooltip .tt-detail {{ margin-top: 4px; font-size: 11px; color: #555; }}

/* ── Legend ─────────────────────────────────────── */
.legend {{
  position: absolute;
  bottom: 12px;
  left: 12px;
  background: rgba(255,255,255,0.92);
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  padding: 8px 12px;
  font-size: 10px;
  line-height: 1.8;
  z-index: 5;
}}
.legend-item {{
  display: flex;
  align-items: center;
  gap: 6px;
}}
.legend-swatch {{
  width: 10px; height: 10px;
  border-radius: 2px;
  flex-shrink: 0;
}}
.legend-swatch.multi {{
  border: 2px solid #c8a84e;
}}

/* ── Detail panel ──────────────────────────────── */
.detail-panel {{
  width: 360px;
  background: #fff;
  border-left: 1px solid #e0e0e0;
  overflow-y: auto;
  flex-shrink: 0;
  transform: translateX(360px);
  transition: transform 0.25s ease;
  position: absolute;
  top: 0; right: 0; bottom: 0;
  z-index: 15;
}}
.detail-panel.open {{ transform: translateX(0); }}
.dp-header {{
  padding: 16px 20px;
  border-bottom: 1px solid #eee;
}}
.dp-close {{
  float: right;
  cursor: pointer;
  color: #999;
  font-size: 18px;
  line-height: 1;
  border: none;
  background: none;
}}
.dp-close:hover {{ color: #333; }}
.dp-name {{
  font-size: 18px;
  font-weight: 600;
  color: #1a1a1a;
  margin-bottom: 4px;
}}
.dp-sources {{
  font-size: 11px;
  color: #666;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 4px;
}}
.dp-source-chip {{
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 8px;
  border-radius: 10px;
  background: #f5f5f5;
  font-size: 10px;
}}
.dp-stats {{
  padding: 12px 20px;
  border-bottom: 1px solid #eee;
  font-size: 12px;
  color: #555;
  line-height: 1.8;
}}
.dp-stats strong {{ color: #1a1a1a; font-weight: 600; }}
.dp-correspondents {{
  padding: 12px 20px;
}}
.dp-correspondents h3 {{
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #999;
  margin-bottom: 8px;
}}
.dp-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 11px;
}}
.dp-table th {{
  text-align: left;
  font-weight: 500;
  color: #888;
  padding: 4px 6px;
  border-bottom: 1px solid #eee;
}}
.dp-table td {{
  padding: 4px 6px;
  border-bottom: 1px solid #f5f5f5;
  color: #444;
}}
.dp-table .clickable {{
  cursor: pointer;
  color: #4a6fa5;
}}
.dp-table .clickable:hover {{
  text-decoration: underline;
}}

/* ── Timeline ──────────────────────────────────── */
.timeline {{
  flex-shrink: 0;
  background: #fff;
  border-top: 1px solid #e0e0e0;
  padding: 10px 24px 14px;
}}
.timeline-label {{
  font-size: 12px;
  color: #555;
  margin-bottom: 6px;
  font-variant-numeric: tabular-nums;
}}
.timeline-label strong {{ color: #1a1a1a; }}
.slider-row {{
  display: flex;
  align-items: center;
  gap: 12px;
}}
.slider-row .yr {{
  font-size: 11px;
  font-weight: 600;
  color: #2d3436;
  min-width: 36px;
  text-align: center;
  font-variant-numeric: tabular-nums;
}}
.dual-slider {{
  flex: 1;
  position: relative;
  height: 28px;
}}
.dual-slider input[type="range"] {{
  -webkit-appearance: none;
  appearance: none;
  position: absolute;
  width: 100%;
  top: 0; left: 0;
  margin: 0;
  background: transparent;
  pointer-events: none;
  height: 28px;
}}
.dual-slider input[type="range"]::-webkit-slider-runnable-track {{
  height: 4px;
  background: #e0e0e0;
  border-radius: 2px;
}}
.dual-slider input[type="range"]::-webkit-slider-thumb {{
  -webkit-appearance: none;
  width: 16px; height: 16px;
  border-radius: 50%;
  background: #2d3436;
  border: 2px solid #fff;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
  margin-top: -6px;
  pointer-events: all;
  cursor: pointer;
}}
.dual-slider input[type="range"]::-moz-range-track {{
  height: 4px;
  background: #e0e0e0;
  border-radius: 2px;
  border: none;
}}
.dual-slider input[type="range"]::-moz-range-thumb {{
  width: 16px; height: 16px;
  border-radius: 50%;
  background: #2d3436;
  border: 2px solid #fff;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
  pointer-events: all;
  cursor: pointer;
}}

/* ── Narrow viewport ───────────────────────────── */
@media (max-width: 999px) {{
  .detail-panel {{
    width: 100%;
    transform: translateX(100%);
    box-shadow: -4px 0 16px rgba(0,0,0,0.1);
  }}
  .detail-panel.open {{ transform: translateX(0); }}
}}
</style>
</head>
<body>

<div class="header">
  <h1>18th-Century Correspondence Network</h1>
  <span class="subtitle">Unified multi-collection graph</span>
  <div class="stats">
    <span id="stat-nodes"></span>
    <span id="stat-edges"></span>
    <span id="stat-letters"></span>
  </div>
  <button class="reset-btn" id="reset-view" style="display:none;">Reset View</button>
</div>

<div class="controls">
  <div class="search-wrap">
    <span class="search-icon">&#x1F50D;</span>
    <input type="text" id="search-input" placeholder="Search names...">
    <div class="search-results" id="search-results"></div>
  </div>

  <div class="control-group">
    <label>Min. letters:</label>
    <input type="range" id="weight-slider" min="1" max="50" value="2">
    <span class="val" id="weight-val">2</span>
  </div>

  <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
    <div class="source-checks" id="source-checks"></div>
    <button class="phase-pill" id="select-all-src" style="font-size:9px;padding:2px 8px;">All</button>
    <button class="phase-pill" id="clear-all-src" style="font-size:9px;padding:2px 8px;">None</button>
  </div>
</div>

<div class="main">
  <div class="graph-area" id="graph-area">
    <svg id="network-svg"></svg>
    <div class="tooltip" id="tooltip"></div>
    <div class="legend" id="legend"></div>
  </div>
  <div class="detail-panel" id="detail-panel">
    <div class="dp-header">
      <button class="dp-close" id="dp-close">&times;</button>
      <div class="dp-name" id="dp-name"></div>
      <div class="dp-dates" id="dp-dates"></div>
      <div class="dp-bio" id="dp-bio"></div>
      <div class="dp-sources" id="dp-sources"></div>
    </div>
    <div class="dp-stats" id="dp-stats"></div>
    <div class="dp-correspondents">
      <h3>Correspondents</h3>
      <table class="dp-table">
        <thead><tr><th>Name</th><th>Letters</th><th>Dates</th><th>Source</th></tr></thead>
        <tbody id="dp-tbody"></tbody>
      </table>
    </div>
  </div>
</div>

<div class="timeline">
  <div class="timeline-label" id="tl-label"></div>
  <div class="slider-row">
    <span class="yr" id="yr-lo"></span>
    <div class="dual-slider">
      <input type="range" id="slider-lo">
      <input type="range" id="slider-hi">
    </div>
    <span class="yr" id="yr-hi"></span>
  </div>
</div>

<script>
{D3_SOURCE}
</script>
<script>
(function() {{
"use strict";

var DATA = {GRAPH_JSON};
var PERSON_INFO = {PERSON_INFO_JSON};

var SOURCE_COLOURS = DATA.source_colours;
var SOURCE_LABELS = DATA.source_labels;
var ALL_NODES = DATA.nodes;
var ALL_EDGES = DATA.edges;
var YEAR_MIN = DATA.year_min;
var YEAR_MAX = DATA.year_max;

/* ── State ───────────────────────────────────────── */
var rangeLo = YEAR_MIN, rangeHi = YEAR_MAX;
var minWeight = 2;
var activeSources = new Set(Object.keys(SOURCE_COLOURS));
var selectedNodeId = null;
var simulation = null;
var highlightedNodeId = null;

/* ── SVG setup ───────────────────────────────────── */
var svgEl = document.getElementById("network-svg");
var width = svgEl.clientWidth || 900;
var height = svgEl.clientHeight || 600;
var svg = d3.select("#network-svg");
var g = svg.append("g").attr("class", "graph-root");

var zoom = d3.zoom()
  .scaleExtent([0.1, 8])
  .on("zoom", function(event) {{ g.attr("transform", event.transform); }});
svg.call(zoom);
svg.on("click", function(event) {{
  if (event.target === svgEl) dismissDetail();
}});

var tooltip = document.getElementById("tooltip");

/* ── Build source checkboxes ─────────────────────── */
(function() {{
  var container = document.getElementById("source-checks");
  var keys = Object.keys(SOURCE_LABELS);
  keys.forEach(function(key) {{
    var lbl = document.createElement("label");
    lbl.className = "source-check";
    var cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = true;
    cb.dataset.src = key;
    cb.addEventListener("change", function() {{
      if (this.checked) activeSources.add(key);
      else activeSources.delete(key);
      rebuildGraph();
    }});
    var sw = document.createElement("span");
    sw.className = "source-swatch";
    sw.style.background = SOURCE_COLOURS[key];
    lbl.appendChild(cb);
    lbl.appendChild(sw);
    lbl.appendChild(document.createTextNode(" " + SOURCE_LABELS[key]));
    container.appendChild(lbl);
  }});
}})();

/* Select All / Clear All buttons */
document.getElementById("select-all-src").addEventListener("click", function() {{
  document.querySelectorAll("#source-checks input[type=checkbox]").forEach(function(cb) {{
    cb.checked = true;
    activeSources.add(cb.dataset.src);
  }});
  rebuildGraph();
}});
document.getElementById("clear-all-src").addEventListener("click", function() {{
  document.querySelectorAll("#source-checks input[type=checkbox]").forEach(function(cb) {{
    cb.checked = false;
    activeSources.delete(cb.dataset.src);
  }});
  rebuildGraph();
}});

/* ── Timeline setup ──────────────────────────────── */
var sliderLo = document.getElementById("slider-lo");
var sliderHi = document.getElementById("slider-hi");
sliderLo.min = YEAR_MIN; sliderLo.max = YEAR_MAX; sliderLo.value = YEAR_MIN;
sliderHi.min = YEAR_MIN; sliderHi.max = YEAR_MAX; sliderHi.value = YEAR_MAX;
document.getElementById("yr-lo").textContent = YEAR_MIN;
document.getElementById("yr-hi").textContent = YEAR_MAX;

sliderLo.addEventListener("input", function() {{
  rangeLo = +this.value;
  if (rangeLo > rangeHi) {{ rangeHi = rangeLo; sliderHi.value = rangeHi; }}
  document.getElementById("yr-lo").textContent = rangeLo;
  rebuildGraph();
}});
sliderHi.addEventListener("input", function() {{
  rangeHi = +this.value;
  if (rangeHi < rangeLo) {{ rangeLo = rangeHi; sliderLo.value = rangeLo; }}
  document.getElementById("yr-hi").textContent = rangeHi;
  rebuildGraph();
}});

/* ── Weight slider ───────────────────────────────── */
var weightSlider = document.getElementById("weight-slider");
var weightVal = document.getElementById("weight-val");
weightSlider.addEventListener("input", function() {{
  minWeight = +this.value;
  weightVal.textContent = minWeight;
  rebuildGraph();
}});

/* ── Search ──────────────────────────────────────── */
var searchInput = document.getElementById("search-input");
var searchResults = document.getElementById("search-results");

searchInput.addEventListener("input", function() {{
  var q = this.value.trim().toLowerCase();
  if (q.length < 2) {{
    searchResults.style.display = "none";
    if (highlightedNodeId) {{
      highlightedNodeId = null;
      updateHighlight();
    }}
    return;
  }}
  var matches = ALL_NODES.filter(function(n) {{
    return n.id.toLowerCase().indexOf(q) >= 0;
  }}).slice(0, 12);

  while (searchResults.firstChild) searchResults.removeChild(searchResults.firstChild);
  if (matches.length === 0) {{
    searchResults.style.display = "none";
    return;
  }}
  matches.forEach(function(n) {{
    var item = document.createElement("div");
    item.className = "sr-item";
    var nameSpan = document.createElement("span");
    nameSpan.className = "sr-name";
    nameSpan.textContent = n.id;
    var metaSpan = document.createElement("span");
    metaSpan.className = "sr-meta";
    metaSpan.textContent = "  " + n.weight + " letters \u00b7 " + n.sources.join(", ");
    item.appendChild(nameSpan);
    item.appendChild(metaSpan);
    item.addEventListener("click", function() {{
      searchResults.style.display = "none";
      searchInput.value = n.id;
      highlightAndZoom(n.id);
    }});
    searchResults.appendChild(item);
  }});
  searchResults.style.display = "block";
}});

searchInput.addEventListener("keydown", function(e) {{
  if (e.key === "Escape") {{
    searchResults.style.display = "none";
    searchInput.value = "";
    highlightedNodeId = null;
    updateHighlight();
  }}
}});

document.addEventListener("click", function(e) {{
  if (!e.target.closest(".search-wrap")) {{
    searchResults.style.display = "none";
  }}
}});

function highlightAndZoom(nodeId) {{
  focusOnNode(nodeId);
}}

function updateHighlight() {{
  if (!g) return;
  if (!highlightedNodeId) {{
    g.selectAll(".node-group").attr("opacity", 1);
    g.selectAll(".edge-line").attr("stroke-opacity", 0.15);
    return;
  }}
  /* Find connected node IDs */
  var connected = new Set();
  connected.add(highlightedNodeId);
  ALL_EDGES.forEach(function(e) {{
    if (e.source === highlightedNodeId || (e.source && e.source.id === highlightedNodeId))
      connected.add(typeof e.target === "object" ? e.target.id : e.target);
    if (e.target === highlightedNodeId || (e.target && e.target.id === highlightedNodeId))
      connected.add(typeof e.source === "object" ? e.source.id : e.source);
  }});

  g.selectAll(".node-group").attr("opacity", function(d) {{
    return connected.has(d.id) ? 1 : 0.08;
  }});
  g.selectAll(".edge-line").attr("stroke-opacity", function(d) {{
    var sid = typeof d.source === "object" ? d.source.id : d.source;
    var tid = typeof d.target === "object" ? d.target.id : d.target;
    return (sid === highlightedNodeId || tid === highlightedNodeId) ? 0.6 : 0.02;
  }});
}}

/* ── Node adjacency lookup ───────────────────────── */
function getNeighbours(nodeId, edges) {{
  var neighbours = [];
  edges.forEach(function(e) {{
    var sid = typeof e.source === "object" ? e.source.id : e.source;
    var tid = typeof e.target === "object" ? e.target.id : e.target;
    if (sid === nodeId) neighbours.push({{ id: tid, weight: e.weight, year_min: e.year_min, year_max: e.year_max, src: e.src }});
    if (tid === nodeId) neighbours.push({{ id: sid, weight: e.weight, year_min: e.year_min, year_max: e.year_max, src: e.src }});
  }});
  neighbours.sort(function(a, b) {{ return b.weight - a.weight; }});
  return neighbours;
}}

/* ── Build graph ─────────────────────────────────── */
function rebuildGraph() {{
  /* Filter edges by timeline, source, and weight */
  var filteredEdges = ALL_EDGES.filter(function(e) {{
    if (!activeSources.has(e.src)) return false;
    if (e.weight < minWeight) return false;
    /* Edge overlaps timeline window */
    if (e.year_max < rangeLo || e.year_min > rangeHi) return false;
    return true;
  }});

  /* Collect visible node IDs */
  var nodeIds = new Set();
  filteredEdges.forEach(function(e) {{
    nodeIds.add(e.source);
    nodeIds.add(e.target);
  }});

  var filteredNodes = ALL_NODES.filter(function(n) {{ return nodeIds.has(n.id); }});

  /* Deep clone for simulation */
  var simNodes = filteredNodes.map(function(n) {{
    return {{
      id: n.id,
      weight: n.weight,
      sources: n.sources,
      primary: n.primary,
      multi: n.multi,
      lat: n.lat,
      lon: n.lon,
      place: n.place
    }};
  }});
  var simEdges = filteredEdges.map(function(e) {{
    return {{
      source: e.source,
      target: e.target,
      weight: e.weight,
      year_min: e.year_min,
      year_max: e.year_max,
      src: e.src
    }};
  }});

  /* Update stats */
  var totalLetters = 0;
  filteredEdges.forEach(function(e) {{ totalLetters += e.weight; }});
  document.getElementById("stat-nodes").textContent = simNodes.length + " people";
  document.getElementById("stat-edges").textContent = simEdges.length + " connections";
  document.getElementById("stat-letters").textContent = totalLetters.toLocaleString() + " letters";

  /* Update timeline label */
  var tlLabel = document.getElementById("tl-label");
  while (tlLabel.firstChild) tlLabel.removeChild(tlLabel.firstChild);
  var strong = document.createElement("strong");
  strong.textContent = rangeLo + " \u2013 " + rangeHi;
  tlLabel.appendChild(strong);
  tlLabel.appendChild(document.createTextNode(" \u00b7 " + simNodes.length + " people, " + simEdges.length + " connections visible"));

  /* Rebuild legend */
  var legendEl = document.getElementById("legend");
  while (legendEl.firstChild) legendEl.removeChild(legendEl.firstChild);
  var presentSources = new Set();
  filteredEdges.forEach(function(e) {{ presentSources.add(e.src); }});
  Object.keys(SOURCE_LABELS).forEach(function(key) {{
    if (!presentSources.has(key)) return;
    var row = document.createElement("div");
    row.className = "legend-item";
    var swatch = document.createElement("div");
    swatch.className = "legend-swatch";
    swatch.style.background = SOURCE_COLOURS[key];
    var label = document.createElement("span");
    label.textContent = SOURCE_LABELS[key];
    row.appendChild(swatch);
    row.appendChild(label);
    legendEl.appendChild(row);
  }});
  /* Multi-source indicator */
  var hasMulti = simNodes.some(function(n) {{ return n.multi; }});
  if (hasMulti) {{
    var row2 = document.createElement("div");
    row2.className = "legend-item";
    var sw2 = document.createElement("div");
    sw2.className = "legend-swatch multi";
    sw2.style.background = "#2d3436";
    var lb2 = document.createElement("span");
    lb2.textContent = "Multi-source (bridge)";
    row2.appendChild(sw2);
    row2.appendChild(lb2);
    legendEl.appendChild(row2);
  }}

  /* Clear previous graph */
  g.selectAll("*").remove();
  if (simulation) simulation.stop();

  if (simNodes.length === 0) return;

  /* Scales */
  var maxWeight = d3.max(simNodes, function(n) {{ return n.weight; }}) || 1;
  var rScale = d3.scaleSqrt().domain([1, maxWeight]).range([3, 32]);
  var maxEdgeWeight = d3.max(simEdges, function(e) {{ return e.weight; }}) || 1;
  var wScale = d3.scaleSqrt().domain([1, maxEdgeWeight]).range([0.3, 5]);

  /* Links */
  var linkG = g.append("g").attr("class", "links");
  var link = linkG.selectAll("line")
    .data(simEdges)
    .join("line")
    .attr("class", "edge-line")
    .attr("stroke", function(d) {{ return SOURCE_COLOURS[d.src] || "#ccc"; }})
    .attr("stroke-opacity", 0.15)
    .attr("stroke-width", function(d) {{ return wScale(d.weight); }});

  /* Nodes */
  var nodeG = g.append("g").attr("class", "nodes");
  var node = nodeG.selectAll("g.node-group")
    .data(simNodes)
    .join("g")
    .attr("class", "node-group")
    .attr("cursor", "pointer");

  /* Gold ring for multi-source */
  node.filter(function(d) {{ return d.multi; }})
    .append("circle")
    .attr("class", "multi-ring")
    .attr("r", function(d) {{ return rScale(d.weight) + 3; }})
    .attr("fill", "none")
    .attr("stroke", "#c8a84e")
    .attr("stroke-width", 2);

  /* Main circle */
  node.append("circle")
    .attr("class", "node-circle")
    .attr("r", function(d) {{ return rScale(d.weight); }})
    .attr("fill", function(d) {{
      if (d.multi) return "#2d3436";
      return SOURCE_COLOURS[d.primary] || "#b0b0b0";
    }})
    .attr("stroke", "#fff")
    .attr("stroke-width", 1.5);

  /* Labels for nodes with weight > 50 */
  node.filter(function(d) {{ return d.weight > 50; }})
    .append("text")
    .text(function(d) {{
      var parts = d.id.split(" ");
      if (parts.length > 2) return parts[parts.length - 1];
      return d.id;
    }})
    .attr("dx", function(d) {{ return rScale(d.weight) + 4; }})
    .attr("dy", "0.35em")
    .attr("font-size", "10px")
    .attr("fill", "#555")
    .style("pointer-events", "none");

  /* Node interactions */
  node.on("mouseenter", function(event, d) {{
    /* Highlight neighbours */
    var connected = new Set();
    connected.add(d.id);
    simEdges.forEach(function(e) {{
      var sid = typeof e.source === "object" ? e.source.id : e.source;
      var tid = typeof e.target === "object" ? e.target.id : e.target;
      if (sid === d.id) connected.add(tid);
      if (tid === d.id) connected.add(sid);
    }});
    nodeG.selectAll(".node-group").attr("opacity", function(nd) {{
      return connected.has(nd.id) ? 1 : 0.1;
    }});
    linkG.selectAll("line").attr("stroke-opacity", function(ld) {{
      var sid = typeof ld.source === "object" ? ld.source.id : ld.source;
      var tid = typeof ld.target === "object" ? ld.target.id : ld.target;
      return (sid === d.id || tid === d.id) ? 0.7 : 0.02;
    }});

    /* Tooltip */
    while (tooltip.firstChild) tooltip.removeChild(tooltip.firstChild);
    var nameDiv = document.createElement("div");
    nameDiv.className = "tt-name";
    nameDiv.textContent = d.id;
    tooltip.appendChild(nameDiv);

    var metaDiv = document.createElement("div");
    metaDiv.className = "tt-meta";
    metaDiv.textContent = d.sources.map(function(s) {{ return SOURCE_LABELS[s] || s; }}).join(", ");
    tooltip.appendChild(metaDiv);

    var detDiv = document.createElement("div");
    detDiv.className = "tt-detail";
    var nbs = getNeighbours(d.id, simEdges);
    detDiv.textContent = d.weight + " total letters \u00b7 " + nbs.length + " correspondents";
    tooltip.appendChild(detDiv);

    if (d.place) {{
      var placeDiv = document.createElement("div");
      placeDiv.className = "tt-meta";
      placeDiv.textContent = d.place;
      tooltip.appendChild(placeDiv);
    }}

    tooltip.style.display = "block";
  }}).on("mousemove", function(event) {{
    var rect = svgEl.getBoundingClientRect();
    tooltip.style.left = (event.clientX - rect.left + 14) + "px";
    tooltip.style.top  = (event.clientY - rect.top  - 10) + "px";
  }}).on("mouseleave", function() {{
    tooltip.style.display = "none";
    if (focusedNodeId) {{
      /* Restore focus highlight (not hover) */
      updateHighlight();
    }} else if (!highlightedNodeId) {{
      nodeG.selectAll(".node-group").attr("opacity", 1);
      linkG.selectAll("line").attr("stroke-opacity", 0.15);
    }} else {{
      updateHighlight();
    }}
  }});

  /* Click -> detail panel */
  node.on("click", function(event, d) {{
    event.stopPropagation();
    showDetail(d);
  }});

  /* Force simulation */
  width = svgEl.clientWidth || 900;
  height = svgEl.clientHeight || 600;

  var nCount = simNodes.length;
  var chargeStrength = nCount > 500 ? -30 : nCount > 200 ? -60 : nCount > 50 ? -100 : -150;

  simulation = d3.forceSimulation(simNodes)
    .force("link", d3.forceLink(simEdges).id(function(d) {{ return d.id; }})
      .distance(function(d) {{ return 30 + 120 / Math.sqrt(d.weight || 1); }})
      .strength(function(d) {{ return 0.2 + 0.4 * Math.min(d.weight / (maxEdgeWeight || 1), 1); }})
    )
    .force("charge", d3.forceManyBody().strength(chargeStrength))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide().radius(function(d) {{
      return rScale(d.weight) + 2;
    }}))
    .on("tick", function() {{
      link
        .attr("x1", function(d) {{ return d.source.x; }})
        .attr("y1", function(d) {{ return d.source.y; }})
        .attr("x2", function(d) {{ return d.target.x; }})
        .attr("y2", function(d) {{ return d.target.y; }});
      node.attr("transform", function(d) {{ return "translate(" + d.x + "," + d.y + ")"; }});
    }});

  /* Drag */
  node.call(d3.drag()
    .on("start", function(event, d) {{
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x; d.fy = d.y;
    }})
    .on("drag", function(event, d) {{
      d.fx = event.x; d.fy = event.y;
    }})
    .on("end", function(event, d) {{
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null; d.fy = null;
    }})
  );
}}

/* ── Focus / reset ───────────────────────────────── */
var initialTransform = d3.zoomIdentity;
var focusedNodeId = null;

function focusOnNode(nodeId) {{
  focusedNodeId = nodeId;
  highlightedNodeId = nodeId;
  updateHighlight();
  document.getElementById("reset-view").style.display = "";

  /* Zoom to node */
  var simNode = null;
  if (simulation) {{
    simulation.nodes().forEach(function(n) {{
      if (n.id === nodeId) simNode = n;
    }});
  }}
  if (simNode && simNode.x != null) {{
    var scale = 1.5;
    var t = d3.zoomIdentity
      .translate(width / 2 - simNode.x * scale, height / 2 - simNode.y * scale)
      .scale(scale);
    svg.transition().duration(500).call(zoom.transform, t);
  }}
}}

function resetView() {{
  focusedNodeId = null;
  highlightedNodeId = null;
  selectedNodeId = null;
  updateHighlight();
  document.getElementById("detail-panel").classList.remove("open");
  document.getElementById("reset-view").style.display = "none";
  searchInput.value = "";
  svg.transition().duration(500).call(zoom.transform, initialTransform);
}}

document.getElementById("reset-view").addEventListener("click", resetView);

/* ── Detail panel ────────────────────────────────── */
function showDetail(d) {{
  selectedNodeId = d.id;
  var panel = document.getElementById("detail-panel");
  document.getElementById("dp-name").textContent = d.id;

  /* Biographical note */
  var info = PERSON_INFO[d.id];
  var datesEl = document.getElementById("dp-dates");
  var bioEl = document.getElementById("dp-bio");
  if (info) {{
    datesEl.textContent = info.dates || "";
    bioEl.textContent = info.bio || "";
    bioEl.className = "dp-bio";
  }} else {{
    datesEl.textContent = "";
    bioEl.textContent = "No biographical note available";
    bioEl.className = "dp-bio dim";
  }}

  /* Click-to-focus: zoom and highlight 1st degree network */
  focusOnNode(d.id);

  /* Source chips */
  var sourcesEl = document.getElementById("dp-sources");
  while (sourcesEl.firstChild) sourcesEl.removeChild(sourcesEl.firstChild);
  d.sources.forEach(function(s) {{
    var chip = document.createElement("span");
    chip.className = "dp-source-chip";
    var sw = document.createElement("span");
    sw.className = "source-swatch";
    sw.style.background = SOURCE_COLOURS[s] || "#ccc";
    chip.appendChild(sw);
    chip.appendChild(document.createTextNode(" " + (SOURCE_LABELS[s] || s)));
    sourcesEl.appendChild(chip);
  }});

  /* Stats */
  var statsEl = document.getElementById("dp-stats");
  while (statsEl.firstChild) statsEl.removeChild(statsEl.firstChild);

  var neighbours = getNeighbours(d.id, ALL_EDGES);

  var countLine = document.createElement("div");
  var b1 = document.createElement("strong");
  b1.textContent = d.weight;
  countLine.appendChild(b1);
  countLine.appendChild(document.createTextNode(" total letters across all collections"));
  statsEl.appendChild(countLine);

  var corrLine = document.createElement("div");
  var b2 = document.createElement("strong");
  b2.textContent = neighbours.length;
  corrLine.appendChild(b2);
  corrLine.appendChild(document.createTextNode(" correspondents"));
  statsEl.appendChild(corrLine);

  /* Date range */
  var allYears = [];
  neighbours.forEach(function(nb) {{
    allYears.push(nb.year_min);
    allYears.push(nb.year_max);
  }});
  if (allYears.length > 0) {{
    var dateLine = document.createElement("div");
    dateLine.textContent = d3.min(allYears) + " \u2013 " + d3.max(allYears);
    statsEl.appendChild(dateLine);
  }}

  if (d.place) {{
    var placeLine = document.createElement("div");
    placeLine.textContent = d.place;
    placeLine.style.color = "#888";
    placeLine.style.marginTop = "4px";
    statsEl.appendChild(placeLine);
  }}

  /* Correspondents table */
  var tbody = document.getElementById("dp-tbody");
  while (tbody.firstChild) tbody.removeChild(tbody.firstChild);
  neighbours.forEach(function(nb) {{
    var tr = document.createElement("tr");
    var tdName = document.createElement("td");
    tdName.className = "clickable";
    tdName.textContent = nb.id;
    tdName.addEventListener("click", function() {{
      /* Find and show detail for this node */
      var target = ALL_NODES.find(function(n) {{ return n.id === nb.id; }});
      if (target) showDetail(target);
    }});
    var tdW = document.createElement("td");
    tdW.textContent = nb.weight;
    var tdDates = document.createElement("td");
    tdDates.textContent = nb.year_min + "\u2013" + nb.year_max;
    tdDates.style.fontSize = "10px";
    var tdSrc = document.createElement("td");
    tdSrc.textContent = SOURCE_LABELS[nb.src] || nb.src;
    tdSrc.style.fontSize = "10px";
    tdSrc.style.color = SOURCE_COLOURS[nb.src] || "#999";
    tr.appendChild(tdName);
    tr.appendChild(tdW);
    tr.appendChild(tdDates);
    tr.appendChild(tdSrc);
    tbody.appendChild(tr);
  }});

  panel.classList.add("open");
}}

function dismissDetail() {{
  selectedNodeId = null;
  focusedNodeId = null;
  highlightedNodeId = null;
  updateHighlight();
  document.getElementById("detail-panel").classList.remove("open");
  document.getElementById("reset-view").style.display = "none";
}}

document.getElementById("dp-close").addEventListener("click", dismissDetail);

/* ── Resize ──────────────────────────────────────── */
window.addEventListener("resize", function() {{
  width = svgEl.clientWidth;
  height = svgEl.clientHeight;
  if (simulation) {{
    simulation.force("center", d3.forceCenter(width / 2, height / 2));
    simulation.alpha(0.3).restart();
  }}
}});

/* ── Init ────────────────────────────────────────── */
rebuildGraph();

}})();
</script>
</body>
</html>
"""


def build() -> None:
    """Build the unified correspondence network HTML."""
    data = load_graph()
    graph_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    # Load biographical notes
    if PERSON_INFO.exists():
        person_info = json.loads(PERSON_INFO.read_text(encoding="utf-8"))
    else:
        person_info = {}
    person_info_json = json.dumps(person_info, ensure_ascii=False, separators=(",", ":"))

    d3_src = _get_d3_source()

    # Un-double braces first, then insert D3 and data
    html = HTML_TEMPLATE.replace("{{", "{").replace("}}", "}")
    html = html.replace("{D3_SOURCE}", d3_src, 1)
    html = html.replace("{GRAPH_JSON}", graph_json, 1)
    html = html.replace("{PERSON_INFO_JSON}", person_info_json, 1)

    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"Full network -> {OUT_PATH}")
    print(f"  {len(data['nodes'])} nodes, {len(data['edges'])} edges")
    print(f"  Year range: {data['year_min']}-{data['year_max']}")
    print(f"  Sources: {', '.join(sorted(data['source_labels'].values()))}")


if __name__ == "__main__":
    build()
