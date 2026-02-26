#!/usr/bin/env python3
"""
Build an interactive narrative path map for individual novels.

Reads narrative_mentions.json and writes narrative_map.html.

For each text the map shows:
  - Venue markers sized by mention count up to the current timeline position
  - A path line connecting venues in first-encounter order (blue → red gradient)
  - A context snippet from the current event
  - A play button that animates the slider through the novel

Usage:
    python3 gazetteer/validate_venues.py --narrative   # generate data first
    python3 gazetteer/build_narrative_map.py
    open gazetteer/narrative_map.html
"""

import json
import re
from pathlib import Path


ROCQUE_URL  = "https://www.dhi.ac.uk/san/llptiles/molarocque/{z}/{x}/{y}.png"
ROCQUE_ATTR = "Map tiles © Museum of London Archaeology / DHI, based on John Rocque 1746"
HORWOOD_URL  = "https://www.romanticlondon.org/horwoodplan/{z}/{x}/{y}.png"
HORWOOD_ATTR = "Map tiles © Romantic London project, based on Richard Horwood 1792–99"

MIN_EVENTS = 5   # texts with fewer events are excluded from the selector


def fmt_author(s: str) -> str:
    """'FrancesBurney' → 'Frances Burney'."""
    overrides = {"MGLewis": "M. G. Lewis", "FrancesSheridan": "Frances Sheridan"}
    if s in overrides:
        return overrides[s]
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", s)


def build(data_path: Path, out_path: Path) -> None:
    with open(data_path, encoding="utf-8") as f:
        all_texts = json.load(f)

    texts = [t for t in all_texts if t["event_count"] >= MIN_EVENTS]
    texts.sort(key=lambda t: t["pub_year"] or 0)

    # Add display author name before embedding
    for t in texts:
        t["display_author"] = fmt_author(t["author"])

    data_js = json.dumps(texts, ensure_ascii=False, separators=(",", ":"))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Novel Path Map · 18c Fiction</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: Georgia, 'Times New Roman', serif; background: #f5f5f0; }}

  /* ── controls bar ── */
  #controls {{
    position: fixed; top: 0; left: 0; right: 0; z-index: 1000;
    background: rgba(255,255,255,0.96);
    border-bottom: 1px solid #ccc;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    padding: 10px 16px;
    display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
  }}
  #controls .title {{
    font-size: 15px; font-weight: bold; white-space: nowrap; color: #2c3e50;
  }}
  #text-select {{
    font-family: Georgia, serif; font-size: 13px;
    padding: 4px 8px; border: 1px solid #bbb; border-radius: 4px;
    background: white; cursor: pointer; min-width: 280px;
  }}
  #play-btn {{
    font-size: 18px; background: none; border: 1px solid #bbb;
    border-radius: 4px; padding: 2px 10px; cursor: pointer;
    line-height: 1.4; color: #2c3e50;
  }}
  #play-btn:hover {{ background: #eee; }}
  .slider-wrap {{ flex: 1; min-width: 160px; }}
  #timeline {{
    width: 100%; accent-color: #2c3e50;
    cursor: pointer;
  }}
  #pos-label {{
    font-size: 12px; color: #555; white-space: nowrap; min-width: 110px;
  }}
  #stats {{
    font-size: 12px; color: #888; white-space: nowrap;
  }}

  /* ── map ── */
  #map {{
    position: fixed; top: 56px; left: 0; right: 0;
    bottom: 72px;
  }}

  /* ── context panel ── */
  #context-panel {{
    position: fixed; bottom: 0; left: 0; right: 0; z-index: 1000;
    background: rgba(255,255,255,0.96);
    border-top: 1px solid #ccc;
    padding: 10px 16px;
    font-size: 12px; color: #333;
    height: 72px; overflow: hidden;
    display: flex; align-items: center; gap: 10px;
  }}
  #ctx-badge {{
    background: #2c3e50; color: white;
    border-radius: 3px; padding: 2px 7px;
    font-size: 11px; white-space: nowrap; flex-shrink: 0;
  }}
  #ctx-venue {{
    font-weight: bold; white-space: nowrap; flex-shrink: 0; color: #2c3e50;
  }}
  #ctx-text {{
    color: #555; font-style: italic;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }}

  /* ── gradient legend ── */
  #grad-legend {{
    position: fixed; bottom: 82px; right: 14px; z-index: 1000;
    background: rgba(255,255,255,0.92); border: 1px solid #ccc;
    border-radius: 5px; padding: 8px 12px; font-size: 11px;
    box-shadow: 1px 1px 5px rgba(0,0,0,0.12);
  }}
  #grad-bar {{
    width: 120px; height: 10px; border-radius: 3px; margin: 4px 0;
    background: linear-gradient(to right, hsl(240,70%,50%), hsl(120,60%,40%), hsl(30,80%,50%));
  }}
  .grad-labels {{ display: flex; justify-content: space-between; color: #888; }}
</style>
</head>
<body>

<div id="controls">
  <span class="title">Novel Path Map</span>
  <select id="text-select"></select>
  <button id="play-btn" title="Play / pause">&#9654;</button>
  <div class="slider-wrap">
    <input type="range" id="timeline" min="0" max="1000" value="1000">
  </div>
  <span id="pos-label">100% through text</span>
  <span id="stats">—</span>
</div>

<div id="map"></div>

<div id="context-panel">
  <span id="ctx-badge">—</span>
  <span id="ctx-venue">—</span>
  <span id="ctx-text">Select a text to begin.</span>
</div>

<div id="grad-legend">
  <div style="font-weight:bold;margin-bottom:2px">Narrative position</div>
  <div id="grad-bar"></div>
  <div class="grad-labels"><span>Opening</span><span>End</span></div>
  <div style="margin-top:5px;color:#888">
    ● size = mention count<br>
    ─ path = first encounter order
  </div>
</div>

<script>
// ── embedded data ──────────────────────────────────────────────────────────
const TEXTS = {data_js};

const ROCQUE  = "{ROCQUE_URL}";
const HORWOOD = "{HORWOOD_URL}";

// ── map setup ──────────────────────────────────────────────────────────────
const map = L.map('map', {{ zoomControl: true }}).setView([51.51, -0.13], 13);

const baseLayers = {{
  'Modern (CartoDB)': L.tileLayer(
    'https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',
    {{ attribution: '© OpenStreetMap contributors © CARTO', maxZoom: 19 }}
  ).addTo(map),
  'Rocque 1746': L.tileLayer(ROCQUE, {{
    attribution: '{ROCQUE_ATTR}',
    minZoom: 13, maxZoom: 15, maxNativeZoom: 15, opacity: 0.9,
  }}),
  'Horwood 1792–99': L.tileLayer(HORWOOD, {{
    attribution: '{HORWOOD_ATTR}',
    minZoom: 11, maxZoom: 17, maxNativeZoom: 16, opacity: 0.9,
  }}),
}};
L.control.layers(baseLayers, {{}}, {{ collapsed: false }}).addTo(map);

// ── colour gradient ────────────────────────────────────────────────────────
function posToColor(pos) {{
  // HSL: 240° (blue) at pos=0 → 30° (orange) at pos=1
  const hue = Math.round(240 - pos * 210);
  const sat = 70, lit = pos < 0.5 ? 45 : 42;
  return `hsl(${{hue}},${{sat}}%,${{lit}}%)`;
}}

// ── state ──────────────────────────────────────────────────────────────────
let layers     = [];   // all active Leaflet layers for cleanup
let polylines  = [];
let playing    = false;
let playTimer  = null;

function clearLayers() {{
  layers.forEach(l => map.removeLayer(l));
  layers = [];
}}

// ── populate selector ─────────────────────────────────────────────────────
const sel = document.getElementById('text-select');
TEXTS.forEach((t, i) => {{
  const opt = document.createElement('option');
  opt.value = i;
  opt.textContent = `${{t.display_author}} — ${{t.title}} (${{t.pub_year || '?'}})`;
  sel.appendChild(opt);
}});

// ── main render ────────────────────────────────────────────────────────────
function render() {{
  clearLayers();

  const tData  = TEXTS[parseInt(sel.value)];
  const sliderVal = parseInt(document.getElementById('timeline').value);
  const posThreshold = sliderVal / 1000;

  // events up to current position
  const active = tData.events.filter(e => e.pos <= posThreshold);
  if (active.length === 0) {{
    document.getElementById('pos-label').textContent =
      `${{Math.round(posThreshold * 100)}}% through text`;
    document.getElementById('stats').textContent = '—';
    document.getElementById('ctx-text').textContent = 'No locations yet.';
    return;
  }}

  // ── aggregate mentions per venue ─────────────────────────────────────────
  const venueCounts = {{}};  // venue_id → count
  const venueFirst  = {{}};  // venue_id → first event object
  const firstEncounterOrder = [];

  for (const ev of active) {{
    venueCounts[ev.venue_id] = (venueCounts[ev.venue_id] || 0) + 1;
    if (!venueFirst[ev.venue_id]) {{
      venueFirst[ev.venue_id] = ev;
      firstEncounterOrder.push(ev);
    }}
  }}

  // ── draw path through first-encounter order ──────────────────────────────
  for (let i = 0; i < firstEncounterOrder.length - 1; i++) {{
    const a = firstEncounterOrder[i];
    const b = firstEncounterOrder[i + 1];
    // skip trivially short segments (same or adjacent coord)
    const dist = Math.hypot(a.lat - b.lat, a.lon - b.lon);
    const color = posToColor(a.pos);
    const weight = dist < 0.001 ? 1 : 2.5;
    const dash = dist > 0.08 ? '6 4' : null;  // dashed for long jumps (London↔Bath)
    const pl = L.polyline([[a.lat, a.lon], [b.lat, b.lon]], {{
      color, weight, opacity: 0.75, dashArray: dash,
    }}).addTo(map);
    layers.push(pl);
  }}

  // ── draw venue markers ───────────────────────────────────────────────────
  const lastEvent = active[active.length - 1];

  for (const [vid, count] of Object.entries(venueCounts)) {{
    const ev = venueFirst[vid];
    const isLatest = (vid === lastEvent.venue_id);
    const color = posToColor(ev.pos);
    const radius = Math.max(5, Math.sqrt(count) * 3.5);

    const marker = L.circleMarker([ev.lat, ev.lon], {{
      radius:      isLatest ? radius + 4 : radius,
      color:       isLatest ? '#2c3e50' : color,
      fillColor:   color,
      fillOpacity: isLatest ? 0.95 : 0.65,
      weight:      isLatest ? 2.5 : 1.5,
    }}).addTo(map);
    layers.push(marker);

    // Build popup with all mentions up to here
    const eventsHere = active.filter(e => e.venue_id === vid);
    const rows = eventsHere.slice(-4).map(e =>
      `<tr>
         <td style="color:#888;padding:1px 6px">${{Math.round(e.pos*100)}}%</td>
         <td style="padding:1px 6px;font-style:italic">${{e.context}}</td>
       </tr>`
    ).join('');
    const extra = eventsHere.length > 4
      ? `<tr><td colspan="2" style="color:#888;padding:1px 6px">
           + ${{eventsHere.length - 4}} earlier mention(s)
         </td></tr>` : '';

    marker.bindPopup(`
      <div style="font-family:Georgia,serif;max-width:340px">
        <div style="font-size:13px;font-weight:bold;margin-bottom:4px">
          ${{ev.venue_name}}
        </div>
        <div style="font-size:11px;color:#888;margin-bottom:6px">
          ${{count}} mention${{count > 1 ? 's' : ''}} up to ${{Math.round(posThreshold*100)}}%
        </div>
        <table style="font-size:11px;border-collapse:collapse;width:100%">
          ${{rows}}${{extra}}
        </table>
      </div>`, {{ maxWidth: 360 }});
  }}

  // ── update info panel ─────────────────────────────────────────────────────
  const pct = Math.round(posThreshold * 100);
  document.getElementById('pos-label').textContent = `${{pct}}% through text`;
  document.getElementById('stats').textContent =
    `${{firstEncounterOrder.length}} venue${{firstEncounterOrder.length !== 1 ? 's' : ''}}, ` +
    `${{active.length}} mention${{active.length !== 1 ? 's' : ''}}`;

  document.getElementById('ctx-badge').textContent =
    Math.round(lastEvent.pos * 100) + '%';
  document.getElementById('ctx-venue').textContent = lastEvent.venue_name;
  document.getElementById('ctx-text').textContent  = lastEvent.context;
}}

// ── text selection ─────────────────────────────────────────────────────────
function onTextChange() {{
  stopPlay();
  document.getElementById('timeline').value = 1000;
  const t = TEXTS[parseInt(sel.value)];
  // fit map to the text's events
  const coords = t.events.map(e => [e.lat, e.lon]);
  if (coords.length > 0) map.fitBounds(L.latLngBounds(coords).pad(0.15));
  render();
}}
sel.addEventListener('change', onTextChange);
document.getElementById('timeline').addEventListener('input', render);

// ── play / pause ───────────────────────────────────────────────────────────
const playBtn = document.getElementById('play-btn');

function stopPlay() {{
  playing = false;
  clearInterval(playTimer);
  playBtn.textContent = '▶';
}}

playBtn.addEventListener('click', () => {{
  if (playing) {{
    stopPlay();
  }} else {{
    const slider = document.getElementById('timeline');
    // rewind if at end
    if (parseInt(slider.value) >= 1000) slider.value = 0;
    playing = true;
    playBtn.textContent = '⏸';
    playTimer = setInterval(() => {{
      const s = document.getElementById('timeline');
      const v = parseInt(s.value) + 5;
      if (v >= 1000) {{
        s.value = 1000;
        render();
        stopPlay();
      }} else {{
        s.value = v;
        render();
      }}
    }}, 60);
  }}
}});

// ── initial render ─────────────────────────────────────────────────────────
onTextChange();
</script>
</body>
</html>
"""

    out_path.write_text(html, encoding="utf-8")
    print(f"Narrative map → {out_path}")
    print(f"  {len(texts)} texts  ({sum(t['event_count'] for t in texts)} events total)")


if __name__ == "__main__":
    here = Path(__file__).parent
    build(here / "narrative_mentions.json", here / "narrative_map.html")
