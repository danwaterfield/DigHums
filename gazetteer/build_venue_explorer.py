#!/usr/bin/env python3
"""
Build the self-contained venue explorer HTML.

Reads venues.csv and sensory.db, writes venue_explorer.html.

Usage:
    python3 gazetteer/build_venue_explorer.py
    open gazetteer/venue_explorer.html
"""

import csv
import json
import re
import sqlite3
from pathlib import Path

VENUES_PATH = Path(__file__).parent / "venues.csv"
DB_PATH     = Path(__file__).parent / "sensory.db"
OUT_PATH    = Path(__file__).parent / "venue_explorer.html"


def fmt_author(s: str) -> str:
    """'FrancesBurney' -> 'Frances Burney'; 'burney' -> 'Burney'."""
    overrides = {"MGLewis": "M. G. Lewis"}
    if s in overrides:
        return overrides[s]
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    # All DB authors are single-word lowercase or CamelCase; no multi-word lowercase names exist.
    return spaced.capitalize() if " " not in spaced else spaced


def load_data(venues_path: Path, db_path: Path) -> list[dict]:
    """
    Return list of venue dicts, each with an 'evidence' array.
    All 73 venues are included; venues with no evidence have evidence=[].
    """
    with open(venues_path, newline="", encoding="utf-8") as f:
        venues = {
            row["id"]: {
                "id":           row["id"],
                "name":         row["name"],
                "lat":          float(row["lat"]),
                "lon":          float(row["lon"]),
                "enclosure":    row.get("enclosure", ""),
                "building_type": row.get("building_type", ""),
                "material":     row.get("material", ""),
                "capacity":     row.get("capacity", ""),
                "evidence":     [],
            }
            for row in csv.DictReader(f)
        }

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        for row in conn.execute("""
            SELECT venue_id, source_type, author, title, pub_year,
                   date_min, date_max, modality, text, context, valence, divergence
            FROM   sensory_evidence
            WHERE  venue_id IS NOT NULL
            ORDER  BY date_min
        """):
            vid = row["venue_id"]
            if vid in venues:
                venues[vid]["evidence"].append({
                    "source_type": row["source_type"],
                    "author":      fmt_author(row["author"] or ""),
                    "title":       row["title"] or "",
                    "pub_year":    row["pub_year"],
                    "date_min":    row["date_min"],
                    "date_max":    row["date_max"],
                    "modality":    row["modality"],
                    "text":        row["text"] or "",
                    "context":     row["context"] or "",
                    "valence":     row["valence"],
                    "divergence":  row["divergence"],
                })
    finally:
        conn.close()
    return list(venues.values())


def build(
    venues_path: Path = VENUES_PATH,
    db_path: Path     = DB_PATH,
    out_path: Path    = OUT_PATH,
) -> None:
    """Build venue_explorer.html."""
    venues  = load_data(venues_path, db_path)
    data_js = json.dumps(venues, ensure_ascii=False, separators=(",", ":"))

    # ── bookseller data ──────────────────────────────────────────────────
    booksellers_path = Path(__file__).parent / "booksellers.csv"
    bookseller_locs_path = Path(__file__).parent / "bookseller_locations.csv"

    bs_by_venue: dict[str, list[dict]] = {}
    if booksellers_path.exists() and bookseller_locs_path.exists():
        booksellers: dict[str, dict] = {}
        with open(booksellers_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                booksellers[row["bookseller_id"]] = row
        with open(bookseller_locs_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                vid = row["venue_id"]
                if not vid:
                    continue
                bs = booksellers.get(row["bookseller_id"], {})
                notes_raw = bs.get("notes", "")
                # Strip "Plomer (1668-1725); " prefix for display
                notes_display = re.sub(r'^Plomer \([^)]+\);\s*', '', notes_raw).strip('; ')
                entry = {
                    "name": bs.get("name", ""),
                    "sign": bs.get("sign", ""),
                    "type": bs.get("type", ""),
                    "date_min": int(row["date_min"]) if row["date_min"] else None,
                    "date_max": int(row["date_max"]) if row["date_max"] else None,
                    "address": row.get("address_detail", ""),
                    "notes": notes_display,
                }
                bs_by_venue.setdefault(vid, []).append(entry)

    bookseller_js = json.dumps(bs_by_venue, ensure_ascii=False, separators=(",", ":"))
    html = _render(data_js, bookseller_js)
    out_path.write_text(html, encoding="utf-8")
    geocoded = sum(1 for v in venues if v["evidence"])
    total_ev = sum(len(v["evidence"]) for v in venues)
    print(f"Venue explorer -> {out_path}")
    print(f"  {len(venues)} venues  {geocoded} with evidence  {total_ev} passages")


def _render(data_js: str, bookseller_js: str = "{}") -> str:
    return (HTML_TEMPLATE
            .replace("__VENUES_DATA__", data_js, 1)
            .replace("__BOOKSELLER_DATA__", bookseller_js, 1))


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Venue Explorer · 18c London &amp; Bath</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Inter', system-ui, sans-serif; background: #f4f1eb;
       display: flex; flex-direction: column; height: 100vh; overflow: hidden; }

/* ── top bar ── */
#topbar {
  flex-shrink: 0;
  background: rgba(255,255,255,0.97);
  border-bottom: 1px solid #ccc;
  box-shadow: 0 2px 8px rgba(0,0,0,0.10);
  padding: 8px 14px;
  display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  z-index: 1000;
}
#topbar .title { font-size: 14px; font-weight: 700; color: #1a1816; white-space: nowrap; }
.view-tab { background: #f4f1eb; border: 1px solid #d8d4cc; color: #5c5850; padding: 2px 10px; border-radius: 3px; font-size: 11px; font-weight: 500; text-decoration: none; display: inline-block; white-space: nowrap; }
.view-tab:hover { background: #ece8e0; color: #1a1816; }
.view-tab.active { background: #1e3c6e; border-color: #1e3c6e; color: #fff; font-weight: 600; cursor: default; }
.filter-group { display: flex; align-items: center; gap: 5px; flex-wrap: wrap; }
.filter-label { font-size: 11px; color: #9c9890; white-space: nowrap; font-weight: 500; }
.pill {
  font-size: 11px; padding: 3px 9px; border-radius: 3px; cursor: pointer;
  border: 1px solid #d8d4cc; background: #f4f1eb; color: #5c5850;
  transition: background 0.12s, color 0.12s; font-weight: 500;
}
.pill.active { background: #1e3c6e; color: #fff; border-color: #1e3c6e; }
.pill:hover:not(.active) { background: #ece8e0; color: #1a1816; }
.date-group { display: flex; align-items: center; gap: 6px; }
.date-group span { font-size: 11px; color: #555; min-width: 28px; text-align: center; }
.date-group input[type=range] { width: 90px; accent-color: #1e3c6e; cursor: pointer; }
.sep { color: #ccc; font-size: 14px; }

/* ── main area ── */
#main { flex: 1; display: flex; overflow: hidden; }
#map  { flex: 0 0 65%; }
#panel {
  flex: 0 0 35%;
  border-left: 1px solid #ddd;
  background: #faf9f7;
  display: flex; flex-direction: column;
  overflow: hidden;
}

/* ── panel header ── */
#panel-header {
  padding: 12px 14px 8px;
  border-bottom: 1px solid #e0dbd4;
  flex-shrink: 0;
}
#panel-title {
  font-size: 14px; font-weight: 700; color: #1a1816; margin-bottom: 3px;
}
#panel-stats { font-size: 11px; color: #888; margin-bottom: 8px; }
.panel-pills { display: flex; gap: 5px; flex-wrap: wrap; }

/* ── panel placeholder ── */
#panel-placeholder {
  flex: 1; display: flex; align-items: center; justify-content: center;
  padding: 24px; text-align: center; color: #aaa; font-style: italic;
  font-size: 13px; line-height: 1.6;
}

/* ── evidence list ── */
#evidence-list {
  flex: 1; overflow-y: auto; padding: 10px 12px;
}
.ev-card {
  background: white; border: 1px solid #e0dbd4; border-radius: 5px;
  padding: 10px 12px; margin-bottom: 8px;
  font-size: 12px; line-height: 1.5;
}
.ev-card-head {
  display: flex; align-items: baseline; gap: 7px; margin-bottom: 3px;
  flex-wrap: wrap;
}
.source-badge {
  border-radius: 3px; padding: 1px 6px; font-size: 10px;
  font-weight: bold; text-transform: uppercase; letter-spacing: 0.03em;
  flex-shrink: 0;
}
.src-fiction    { background: #dbe8f5; color: #1a5276; }
.src-diary      { background: #d5f0e0; color: #1a7a42; }
.src-topography { background: #f5e9d5; color: #7d4e1a; }
.src-poetry     { background: #ecdaf5; color: #5b1a7a; }
.src-letters    { background: #e8e8e8; color: #444; }
.src-legal      { background: #f5e0e0; color: #8b1a1a; }
.src-newspaper  { background: #f8ecd2; color: #8a5a11; }
.src-parish     { background: #d9f0ec; color: #0f615f; }
.src-institutional { background: #e4e8f6; color: #304c86; }
.ev-author { font-weight: 600; color: #1a1816; }
.ev-title  { color: #555; font-style: italic; }
.ev-date   { font-size: 10px; color: #999; margin-bottom: 5px; }
.ev-text   { color: #333; font-style: italic; line-height: 1.55; }
.ev-footer { display: flex; justify-content: flex-end; margin-top: 5px; }
.context-chip {
  background: #f0ede8; color: #666; border-radius: 3px;
  padding: 1px 6px; font-size: 10px; font-style: normal;
}
.valence-pip {
  width: 8px; height: 8px; border-radius: 50%;
  display: inline-block; flex-shrink: 0;
  margin-left: 4px; align-self: center;
}
.valence-pip.unpleasant { background: #c0392b; opacity: 0.55; }
.valence-pip.pleasant   { background: #9a6f2a; opacity: 0.55; }
.no-evidence {
  text-align: center; color: #aaa; font-style: italic;
  font-size: 12px; padding: 20px 0;
}
.ev-card.diverges {
  border-left: 3px solid #e67e22;
}
.diverge-badge {
  font-size: 9px; font-weight: bold; text-transform: uppercase;
  letter-spacing: 0.04em; padding: 1px 5px; border-radius: 2px;
  background: #fdebd0; color: #e67e22; border: 1px solid #e67e22;
  flex-shrink: 0; align-self: center;
}
</style>
</head>
<body>

<div id="topbar">
  <span class="title">Venue Explorer · 18c London &amp; Bath</span>
  <span style="display:flex;gap:4px;align-items:center;">
    <span style="font-size:11px;color:#9c9890;font-weight:500;margin-right:2px;">View</span>
    <a href="sensory_time_map.html" class="view-tab">Sensory Map</a>
    <span class="view-tab active">Evidence</span>
    <a href="narrative_map.html" class="view-tab">Narrative</a>
    <a href="comparison.html" class="view-tab">Comparison</a>
    <a href="sensory_timeline.html" class="view-tab">Timeline</a>
  </span>
  <span class="sep">|</span>
  <div class="filter-group">
    <span class="filter-label">Modality</span>
    <button class="pill active" data-f="modality" data-v="auditory">Auditory</button>
    <button class="pill active" data-f="modality" data-v="olfactory">Olfactory</button>
    <button class="pill active" data-f="modality" data-v="visual">Visual</button>
    <button class="pill active" data-f="modality" data-v="thermal">Thermal</button>
    <button class="pill active" data-f="modality" data-v="crowd">Crowd</button>
  </div>
  <span class="sep">|</span>
  <div class="filter-group">
    <span class="filter-label">Source</span>
    <select id="filter-source" style="font-size:11px;padding:2px 4px;border:1px solid #ccc;border-radius:4px;background:#fff;">
      <option value="">All sources</option>
      <option value="fiction">Fiction</option>
      <option value="diary">Diary</option>
      <option value="topography">Topography</option>
      <option value="poetry">Poetry</option>
      <option value="letters">Letters</option>
      <option value="legal">Legal</option>
      <option value="newspaper">Newspaper</option>
      <option value="parish">Parish</option>
      <option value="institutional">Institutional</option>
    </select>
  </div>
  <span class="sep">|</span>
  <div class="filter-group">
    <span class="filter-label">Venue type</span>
    <select id="filter-building" style="font-size:11px;padding:2px 4px;border:1px solid #ccc;border-radius:4px;background:#fff;">
      <option value="">All types</option>
      <option value="garden">Garden / Park</option>
      <option value="theatre">Theatre / Assembly</option>
      <option value="church">Church</option>
      <option value="street">Street / Square</option>
      <option value="market">Market</option>
      <option value="bookseller">Bookseller</option>
      <option value="prison">Prison / Court</option>
      <option value="menagerie">Menagerie</option>
    </select>
  </div>
  <span class="sep">|</span>
  <div class="date-group filter-group">
    <span class="filter-label">Date</span>
    <span id="lbl-from">1660</span>
    <input type="range" id="date-from" min="1660" max="1820" value="1660">
    <span>&#8211;</span>
    <input type="range" id="date-to"   min="1660" max="1820" value="1820">
    <span id="lbl-to">1820</span>
  </div>
  <a href="sensory_time_map.html" style="color:#1e3c6e;text-decoration:none;font-size:0.85em;margin-left:auto;font-weight:500;">Time map &#8594;</a>
</div>

<div id="main">
  <div id="map"></div>
  <div id="panel">
    <div id="panel-placeholder">
      Click a venue marker to explore its sensory evidence.
    </div>
    <div id="panel-header" style="display:none">
      <div id="panel-title"></div>
      <div id="panel-meta" style="display:flex;gap:5px;flex-wrap:wrap;margin:4px 0 5px"></div>
      <div id="panel-booksellers" style="margin:4px 0"></div>
      <div id="panel-stats"></div>
      <div class="panel-pills" id="panel-pills"></div>
    </div>
    <div id="evidence-list" style="display:none"></div>
  </div>
</div>

<script>
// ── embedded data ─────────────────────────────────────────────────────────
const VENUES = __VENUES_DATA__;
const BOOKSELLERS = __BOOKSELLER_DATA__;

// Building type colour scheme (shared with time map)
const BT_COLORS = {
    'garden':    '#4a7c4f', 'park':      '#4a7c4f',
    'theatre':   '#8b5e3c', 'assembly':  '#8b5e3c',
    'church':    '#6b5b8a',
    'street':    '#7a7a7a', 'square':    '#7a7a7a', 'district': '#7a7a7a',
    'market':    '#b8860b',
    'bookseller': '#8B4513',
    'prison':    '#8b0000', 'court':     '#8b0000', 'execution': '#8b0000',
};

function renderMetaChips(v) {
    const metaEl = document.getElementById('panel-meta');
    if (!metaEl) return;
    const parts = [v.enclosure, v.building_type, v.material, v.capacity].filter(Boolean);
    const color = BT_COLORS[v.building_type] || '#666';
    metaEl.innerHTML = parts.map(function(p) {
        return '<span style="font-size:10px;padding:2px 7px;border-radius:10px;border:1px solid '
            + color + ';color:' + color + ';background:rgba(0,0,0,0.04)">' + p + '</span>';
    }).join('');
}

function renderBooksellers(venueId) {
    var el = document.getElementById('panel-booksellers');
    if (!el) return;
    var occupants = BOOKSELLERS[venueId];
    if (!occupants || !occupants.length) { el.textContent = ''; return; }
    el.textContent = '';
    var hdr = document.createElement('div');
    hdr.style.cssText = 'margin-top:6px;font-size:11px;color:#666;font-weight:600';
    hdr.textContent = 'Book Trade';
    el.appendChild(hdr);
    occupants.forEach(function(b) {
        var row = document.createElement('div');
        row.style.cssText = 'font-size:12px;margin:2px 0';
        var nameSpan = document.createElement('strong');
        nameSpan.textContent = b.name;
        row.appendChild(nameSpan);
        if (b.sign) {
            var signSpan = document.createElement('span');
            signSpan.textContent = ' (' + b.sign + ')';
            row.appendChild(signSpan);
        }
        var dateSpan = document.createElement('span');
        dateSpan.style.color = '#888';
        dateSpan.textContent = ' ' + (b.date_min || '?') + '\u2013' + (b.date_max || '?');
        row.appendChild(dateSpan);
        if (b.type) {
            var typeSpan = document.createElement('span');
            typeSpan.style.cssText = 'color:#8B4513;font-size:10px;margin-left:4px';
            typeSpan.textContent = b.type.replace(/\\|/g, ', ');
            row.appendChild(typeSpan);
        }
        el.appendChild(row);
        if (b.address) {
            var addrDiv = document.createElement('div');
            addrDiv.style.cssText = 'font-size:10px;color:#999;margin-left:8px';
            addrDiv.textContent = b.address;
            el.appendChild(addrDiv);
        }
        if (b.notes) {
            var notesDiv = document.createElement('div');
            notesDiv.style.cssText = 'font-size:10px;color:#777;margin-left:8px;font-style:italic';
            notesDiv.textContent = b.notes;
            el.appendChild(notesDiv);
        }
    });
}

// ── Leaflet map ───────────────────────────────────────────────────────────
const map = L.map('map', { zoomControl: true }).setView([51.51, -0.13], 13);

const baseLayers = {
  'Modern (CartoDB)': L.tileLayer(
    'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
    { attribution: '&copy; OpenStreetMap contributors &copy; CARTO', maxZoom: 19 }
  ).addTo(map),
  'Rocque 1746': L.tileLayer(
    'https://www.dhi.ac.uk/san/llptiles/molarocque/{z}/{x}/{y}.png',
    { attribution: 'Map tiles &copy; Museum of London Archaeology / DHI, based on John Rocque 1746',
      minZoom: 13, maxZoom: 15, maxNativeZoom: 15, opacity: 0.9 }
  ),
  'Horwood 1792\u201399': L.tileLayer(
    'https://www.romanticlondon.org/horwoodplan/{z}/{x}/{y}.png',
    { attribution: 'Map tiles &copy; Romantic London project, based on Richard Horwood 1792\u201399',
      minZoom: 11, maxZoom: 17, maxNativeZoom: 16, opacity: 0.9 }
  ),
};
L.control.layers(baseLayers, {}, { collapsed: false }).addTo(map);

// ── state ─────────────────────────────────────────────────────────────────
const state = {
  modalities:      new Set(['auditory','olfactory','visual','thermal','crowd']),
  sourceFilter:    '',
  buildingFilter:  '',
  dateFrom:        1660,
  dateTo:          1820,
  selectedId:      null,
  panelModalities: new Set(['auditory','olfactory','visual','thermal','crowd']),
};

// ── helpers ───────────────────────────────────────────────────────────────
function matchesGlobal(ev) {
  return state.modalities.has(ev.modality)
      && (!state.sourceFilter || ev.source_type === state.sourceFilter)
      && (ev.date_max === null || ev.date_max >= state.dateFrom)
      && (ev.date_min === null || ev.date_min <= state.dateTo);
}

function markerRadius(total) {
  return 6 + Math.log1p(total) * 3;
}

// ── markers ───────────────────────────────────────────────────────────────
const markers = {};

const BT_GROUPS = {
  'garden': 'garden', 'park': 'garden',
  'theatre': 'theatre', 'assembly': 'theatre',
  'church': 'church',
  'street': 'street', 'square': 'street', 'district': 'street',
  'market': 'market',
  'bookseller': 'bookseller',
  'prison': 'prison', 'court': 'prison', 'execution': 'prison',
  'menagerie': 'menagerie',
};

function renderMarkers() {
  VENUES.forEach(function(v) {
    var btGroup = BT_GROUPS[v.building_type] || v.building_type;
    var hidden = state.buildingFilter && btGroup !== state.buildingFilter;
    if (hidden) {
      if (markers[v.id]) markers[v.id].setStyle({ opacity: 0, fillOpacity: 0 });
      return;
    }
    const count  = v.evidence.filter(matchesGlobal).length;
    const total  = v.evidence.length;
    const active = count > 0;
    const sel    = state.selectedId === v.id;
    const r      = markerRadius(total || 1);

    const btColor = BT_COLORS[v.building_type] || '#8b6914';
    const opts = {
      radius:      r,
      color:       sel ? '#2c3e50' : btColor,
      fillColor:   btColor,
      fillOpacity: sel ? 0.85 : (active ? 0.50 : 0.20),
      weight:      sel ? 2.5 : 1.5,
    };

    if (markers[v.id]) {
      markers[v.id].setStyle(opts);
      markers[v.id].setRadius(r);
    } else {
      const m = L.circleMarker([v.lat, v.lon], opts).addTo(map);
      m.bindTooltip(v.name + (total ? ' (' + total + ')' : ''), { sticky: true });
      m.on('click', function() { openPanel(v.id); });
      markers[v.id] = m;
    }
  });
}

// ── side panel ────────────────────────────────────────────────────────────
const SOURCE_CLASSES = {
  fiction: 'src-fiction', diary: 'src-diary',
  topography: 'src-topography', poetry: 'src-poetry',
  letters: 'src-letters', legal: 'src-legal',
  newspaper: 'src-newspaper', parish: 'src-parish',
  institutional: 'src-institutional',
};

function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function renderCard(ev) {
  const cls     = SOURCE_CLASSES[ev.source_type] || 'src-letters';
  const text    = ev.text.length > 380 ? ev.text.slice(0, 380) + '\u2026' : ev.text;
  const dateStr = ev.pub_year
    ? (ev.date_min && ev.date_min !== ev.pub_year
        ? ev.pub_year + ' \u00b7 set ' + ev.date_min
          + (ev.date_max != null ? '\u2013' + ev.date_max : '')
        : String(ev.pub_year))
    : (ev.date_min
        ? 'c.\u00a0' + ev.date_min
          + (ev.date_max != null ? '\u2013' + ev.date_max : '')
        : '');
  const divergesCls = ev.divergence === 'diverges' ? ' diverges' : '';
  return '<div class="ev-card' + divergesCls + '">'
    + '<div class="ev-card-head">'
    + '<span class="source-badge ' + cls + '">' + ev.source_type + '</span>'
    + '<span class="ev-author">' + esc(ev.author) + '</span>'
    + '<span class="ev-title">' + esc(ev.title) + '</span>'
    + (ev.valence === 'unpleasant' ? '<span class="valence-pip unpleasant" title="unpleasant"></span>'
       : ev.valence === 'pleasant'  ? '<span class="valence-pip pleasant" title="pleasant"></span>'
       : '')
    + (ev.divergence === 'diverges' ? '<span class="diverge-badge" title="Passage diverges from expected sensory profile for this event">diverges</span>' : '')
    + '</div>'
    + '<div class="ev-date">' + dateStr + '</div>'
    + '<div class="ev-text">\u201c' + esc(text) + '\u201d</div>'
    + '<div class="ev-footer"><span class="context-chip">' + esc(ev.context) + '</span></div>'
    + '</div>';
}

function renderPanel() {
  const v = VENUES.find(function(x) { return x.id === state.selectedId; });
  if (!v) return;

  const globalFiltered = v.evidence.filter(matchesGlobal);
  const filtered = globalFiltered.filter(function(ev) {
    return state.panelModalities.has(ev.modality);
  });

  // modality breakdown for header
  const breakdown = {};
  globalFiltered.forEach(function(ev) {
    breakdown[ev.modality] = (breakdown[ev.modality] || 0) + 1;
  });
  const bStr = Object.entries(breakdown)
    .sort(function(a,b) { return b[1]-a[1]; })
    .map(function(e) { return e[1] + '\u00a0' + e[0]; })
    .join(' \u00b7 ') || 'no matches';

  document.getElementById('panel-title').textContent = v.name;
  renderMetaChips(v);
  renderBooksellers(v.id);
  document.getElementById('panel-stats').textContent =
    globalFiltered.length + ' passage' + (globalFiltered.length !== 1 ? 's' : '') + ' \u00b7 ' + bStr;

  // local modality pills
  const pillsEl = document.getElementById('panel-pills');
  pillsEl.innerHTML = ['auditory','olfactory','visual','thermal','crowd'].map(function(m) {
    const active = state.panelModalities.has(m) ? ' active' : '';
    return '<button class="pill' + active + '" data-panel-mod="' + m + '">'
      + m.charAt(0).toUpperCase() + m.slice(1) + '</button>';
  }).join('');

  // evidence cards
  const listEl = document.getElementById('evidence-list');
  listEl.innerHTML = filtered.length
    ? filtered.map(renderCard).join('')
    : '<div class="no-evidence">No evidence matches current filters.</div>';

  document.getElementById('panel-placeholder').style.display = 'none';
  document.getElementById('panel-header').style.display      = 'block';
  document.getElementById('evidence-list').style.display     = 'block';
}

function openPanel(venueId) {
  state.selectedId      = venueId;
  state.panelModalities = new Set(['auditory','olfactory','visual','thermal','crowd']);
  renderPanel();
  renderMarkers();
}

// ── filter wiring ─────────────────────────────────────────────────────────
document.querySelectorAll('.pill[data-f]').forEach(function(btn) {
  btn.addEventListener('click', function() {
    const f   = btn.dataset.f;
    const v   = btn.dataset.v;
    const set = state.modalities;
    if (set.has(v)) { set.delete(v); btn.classList.remove('active'); }
    else            { set.add(v);    btn.classList.add('active');    }
    renderMarkers();
    if (state.selectedId) renderPanel();
  });
});

document.getElementById('filter-source').addEventListener('change', function() {
  state.sourceFilter = this.value;
  renderMarkers();
  if (state.selectedId) renderPanel();
});

document.getElementById('filter-building').addEventListener('change', function() {
  state.buildingFilter = this.value;
  renderMarkers();
  if (state.selectedId) renderPanel();
});

document.getElementById('panel-pills').addEventListener('click', function(e) {
  const btn = e.target.closest('[data-panel-mod]');
  if (!btn) return;
  const m = btn.dataset.panelMod;
  if (state.panelModalities.has(m)) state.panelModalities.delete(m);
  else                               state.panelModalities.add(m);
  renderPanel();
});

document.getElementById('date-from').addEventListener('input', function() {
  const val      = parseInt(this.value);
  const toSlider = document.getElementById('date-to');
  if (val > parseInt(toSlider.value)) {
    toSlider.value = val;
    state.dateTo = val;
    document.getElementById('lbl-to').textContent = val;
  }
  state.dateFrom = val;
  document.getElementById('lbl-from').textContent = val;
  renderMarkers();
  if (state.selectedId) renderPanel();
});

document.getElementById('date-to').addEventListener('input', function() {
  const val        = parseInt(this.value);
  const fromSlider = document.getElementById('date-from');
  if (val < parseInt(fromSlider.value)) {
    fromSlider.value = val;
    state.dateFrom = val;
    document.getElementById('lbl-from').textContent = val;
  }
  state.dateTo = val;
  document.getElementById('lbl-to').textContent = val;
  renderMarkers();
  if (state.selectedId) renderPanel();
});

// ── initial render ────────────────────────────────────────────────────────
renderMarkers();
</script>
</body>
</html>"""


if __name__ == "__main__":
    build()
