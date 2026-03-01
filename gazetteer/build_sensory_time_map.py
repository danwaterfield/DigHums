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
from pathlib import Path

VENUES_PATH = Path(__file__).parent / "venues.csv"
DB_PATH     = Path(__file__).parent / "sensory.db"
OUT_PATH    = Path(__file__).parent / "sensory_time_map.html"


def load_data(venues_path: Path, db_path: Path) -> dict:
    with open(venues_path, newline="", encoding="utf-8") as f:
        venues = [
            {"id": r["id"], "name": r["name"],
             "lat": float(r["lat"]), "lon": float(r["lon"])}
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
    finally:
        conn.close()
    return {
        "venues": venues,
        "events": events,
        "event_venues": event_venues,
        "event_instances": event_instances,
        "evidence": evidence,
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
#panel-header {{ background: #2c2c2c; color: #e8e0d0; padding: 10px 14px; font-size: 0.85em; letter-spacing: 0.08em; flex-shrink: 0; }}
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
</style>
</head>
<body>
<div id="controls">
  <div id="header-row">
    <span class="title">SENSORY TIME MAP</span>
    <div id="year-control">
      <button class="step-btn" onclick="stepYear(-10)">&#8592;</button>
      <input type="range" id="year-slider" min="1660" max="1820" value="1750" oninput="updateMap()">
      <span id="year-display">1750</span>
      <button class="step-btn" onclick="stepYear(10)">&#8594;</button>
    </div>
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
</div>
<div id="main">
  <div id="map"></div>
  <div id="panel">
    <div id="panel-header">ACTIVE EVENTS</div>
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

const EVENTS_BY_ID = Object.fromEntries(EVENTS.map(e => [e.event_id, e]));
const state = {{ month: null, dow: null, band: null, literary: false, selectedVenue: null }};

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

const markersByVenueId = {{}};
VENUES.forEach(v => {{
    const m = L.circleMarker([v.lat, v.lon], {{
        radius: 4, fillColor: '#aaa', color: '#888',
        fillOpacity: 0.4, weight: 1
    }}).addTo(map);
    m.bindTooltip(v.name, {{permanent: false, direction: 'top'}});
    m.on('click', () => selectVenue(v.id));
    markersByVenueId[v.id] = m;
}});

function computeIntensity(venueId, year, month, dow, band) {{
    const loads = {{smell: 0, noise: 0, crowd: 0, visual: 0}};

    EVENT_VENUES.forEach(ev => {{
        if (ev.venue_id !== venueId) return;
        const evt = EVENTS_BY_ID[ev.event_id];
        if (!evt) return;

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

function updateMap() {{
    const year  = parseInt(document.getElementById('year-slider').value);
    const month = state.month;
    const dow   = state.dow;
    const band  = state.band;
    document.getElementById('year-display').textContent = year;

    let activeEvents = [];

    VENUES.forEach(v => {{
        const intensity = computeIntensity(v.id, year, month, dow, band);
        const marker = markersByVenueId[v.id];
        if (!marker) return;
        const r = intensity.composite < 0.01 ? 4 : 4 + intensity.crowd * 14;
        const col = intensityColour(intensity.composite);
        marker.setRadius(r);
        marker.setStyle({{
            fillColor: col, color: col,
            fillOpacity: intensity.composite < 0.01 ? 0.25 : 0.75,
            weight: 1
        }});
        marker._intensity = intensity;

        if (intensity.composite > 0.01) {{
            const evts = getActiveEvents(v.id, year, month, dow, band);
            evts.forEach(e => {{ if (!activeEvents.find(x => x.evt.event_id === e.evt.event_id && x.venueId === v.id)) activeEvents.push({{...e, venueId: v.id, venueName: v.name}}); }});
        }}
    }});

    if (state.selectedVenue) {{
        renderVenuePanel(state.selectedVenue, year, month, dow, band);
    }} else {{
        renderGlobalPanel(activeEvents);
    }}
}}

function getActiveEvents(venueId, year, month, dow, band) {{
    const results = [];
    EVENT_VENUES.forEach(ev => {{
        if (ev.venue_id !== venueId) return;
        const evt = EVENTS_BY_ID[ev.event_id];
        if (!evt) return;
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
    document.getElementById('panel-header').textContent = venue ? venue.name.toUpperCase() : 'VENUE';

    const evts = getActiveEvents(venueId, year, month, dow, band);
    let html = evts.length
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

updateMap();
</script>
</body>
</html>
"""


def build(venues_path: Path = VENUES_PATH, db_path: Path = DB_PATH,
          out_path: Path = OUT_PATH) -> None:
    data = load_data(venues_path, db_path)

    html = HTML_TEMPLATE.format(
        EVENTS_JSON          = json.dumps(data["events"],          ensure_ascii=False),
        EVENT_VENUES_JSON    = json.dumps(data["event_venues"],    ensure_ascii=False),
        EVENT_INSTANCES_JSON = json.dumps(data["event_instances"], ensure_ascii=False),
        VENUES_JSON          = json.dumps(data["venues"],          ensure_ascii=False),
        EVIDENCE_JSON        = json.dumps(data["evidence"],        ensure_ascii=False),
    )

    out_path.write_text(html, encoding="utf-8")
    n_venues = len(data["venues"])
    n_events = len(data["events"])
    n_ev     = len(data["evidence"])
    print(f"Sensory time map -> {out_path}")
    print(f"  {n_venues} venues  {n_events} event types  {n_ev} evidence passages")


if __name__ == "__main__":
    build()
