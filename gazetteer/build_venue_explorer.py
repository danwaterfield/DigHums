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
                "id":       row["id"],
                "name":     row["name"],
                "lat":      float(row["lat"]),
                "lon":      float(row["lon"]),
                "evidence": [],
            }
            for row in csv.DictReader(f)
        }

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        for row in conn.execute("""
            SELECT venue_id, source_type, author, title, pub_year,
                   date_min, date_max, modality, text, context
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
    html    = _render(data_js)
    out_path.write_text(html, encoding="utf-8")
    geocoded = sum(1 for v in venues if v["evidence"])
    total_ev = sum(len(v["evidence"]) for v in venues)
    print(f"Venue explorer -> {out_path}")
    print(f"  {len(venues)} venues  {geocoded} with evidence  {total_ev} passages")


def _render(data_js: str) -> str:
    return HTML_TEMPLATE.replace("__VENUES_DATA__", data_js, 1)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sensory Map · 18c London &amp; Bath</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: Georgia, 'Times New Roman', serif; background: #f5f0eb;
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
#topbar .title { font-size: 15px; font-weight: bold; color: #2c3e50; white-space: nowrap; }
#topbar .nav-link { font-size: 12px; color: #666; text-decoration: none; white-space: nowrap; }
#topbar .nav-link:hover { color: #2c3e50; text-decoration: underline; }
.filter-group { display: flex; align-items: center; gap: 5px; flex-wrap: wrap; }
.filter-label { font-size: 11px; color: #888; white-space: nowrap; }
.pill {
  font-size: 11px; padding: 3px 9px; border-radius: 12px; cursor: pointer;
  border: 1px solid #bbb; background: white; color: #555;
  transition: background 0.12s, color 0.12s;
}
.pill.active { background: #2c3e50; color: white; border-color: #2c3e50; }
.pill:hover:not(.active) { background: #eee; }
.date-group { display: flex; align-items: center; gap: 6px; }
.date-group span { font-size: 11px; color: #555; min-width: 28px; text-align: center; }
.date-group input[type=range] { width: 90px; accent-color: #2c3e50; cursor: pointer; }
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
  font-size: 15px; font-weight: bold; color: #2c3e50; margin-bottom: 3px;
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
.ev-author { font-weight: bold; color: #2c3e50; }
.ev-title  { color: #555; font-style: italic; }
.ev-date   { font-size: 10px; color: #999; margin-bottom: 5px; }
.ev-text   { color: #333; font-style: italic; line-height: 1.55; }
.ev-footer { display: flex; justify-content: flex-end; margin-top: 5px; }
.context-chip {
  background: #f0ede8; color: #666; border-radius: 3px;
  padding: 1px 6px; font-size: 10px; font-style: normal;
}
.no-evidence {
  text-align: center; color: #aaa; font-style: italic;
  font-size: 12px; padding: 20px 0;
}
</style>
</head>
<body>

<div id="topbar">
  <span class="title">Sensory Map · 18c London &amp; Bath</span>
  <a href="narrative_map.html" class="nav-link">&#8592; Narrative Map</a>
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
    <button class="pill active" data-f="source" data-v="fiction">Fiction</button>
    <button class="pill active" data-f="source" data-v="diary">Diary</button>
    <button class="pill active" data-f="source" data-v="topography">Topography</button>
    <button class="pill active" data-f="source" data-v="poetry">Poetry</button>
    <button class="pill active" data-f="source" data-v="letters">Letters</button>
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
</div>

<div id="main">
  <div id="map"></div>
  <div id="panel">
    <div id="panel-placeholder">
      Click a venue marker to explore its sensory evidence.
    </div>
    <div id="panel-header" style="display:none">
      <div id="panel-title"></div>
      <div id="panel-stats"></div>
      <div class="panel-pills" id="panel-pills"></div>
    </div>
    <div id="evidence-list" style="display:none"></div>
  </div>
</div>

<script>
// ── embedded data ─────────────────────────────────────────────────────────
const VENUES = __VENUES_DATA__;

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
  sources:         new Set(['fiction','diary','topography','poetry','letters']),
  dateFrom:        1660,
  dateTo:          1820,
  selectedId:      null,
  panelModalities: new Set(['auditory','olfactory','visual','thermal','crowd']),
};

// ── helpers ───────────────────────────────────────────────────────────────
function matchesGlobal(ev) {
  return state.modalities.has(ev.modality)
      && state.sources.has(ev.source_type)
      && (ev.date_max === null || ev.date_max >= state.dateFrom)
      && (ev.date_min === null || ev.date_min <= state.dateTo);
}

function markerRadius(total) {
  return 6 + Math.log1p(total) * 3;
}

// ── markers ───────────────────────────────────────────────────────────────
const markers = {};

function renderMarkers() {
  VENUES.forEach(function(v) {
    const count  = v.evidence.filter(matchesGlobal).length;
    const total  = v.evidence.length;
    const active = count > 0;
    const sel    = state.selectedId === v.id;
    const r      = markerRadius(total || 1);

    const opts = {
      radius:      r,
      color:       sel ? '#2c3e50' : (active ? '#8b6914' : '#999'),
      fillColor:   active ? '#c9a84c' : '#ccc',
      fillOpacity: sel ? 0.95 : (active ? 0.70 : 0.35),
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
  topography: 'src-topography', poetry: 'src-poetry', letters: 'src-letters',
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
  return '<div class="ev-card">'
    + '<div class="ev-card-head">'
    + '<span class="source-badge ' + cls + '">' + ev.source_type + '</span>'
    + '<span class="ev-author">' + esc(ev.author) + '</span>'
    + '<span class="ev-title">' + esc(ev.title) + '</span>'
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
    const set = f === 'modality' ? state.modalities : state.sources;
    if (set.has(v)) { set.delete(v); btn.classList.remove('active'); }
    else            { set.add(v);    btn.classList.add('active');    }
    renderMarkers();
    if (state.selectedId) renderPanel();
  });
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
