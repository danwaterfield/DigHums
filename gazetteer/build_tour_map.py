#!/usr/bin/env python3
"""
Build a self-contained HTML visualisation of Charles Burney's two European
tours (1770 Italy, 1772 Germany/Low Countries) with animated routes and
co-presence events.

Usage:
    python3 gazetteer/build_tour_map.py
    open gazetteer/tour_map.html
"""

import csv
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOUR_1770_PATH = ROOT / "gazetteer" / "cb_tour_1770_verified.csv"
TOUR_1772_PATH = ROOT / "gazetteer" / "cb_tour_1772_verified.csv"
COPRESENCE_PATH = ROOT / "gazetteer" / "copresence_events.csv"
OUT_PATH = Path(__file__).resolve().parent / "tour_map.html"


def _load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _date_to_iso(s: str) -> str:
    """Return ISO date string, or empty string if unparseable."""
    s = s.strip()
    if not s:
        return ""
    try:
        date.fromisoformat(s)
        return s
    except ValueError:
        return ""


def _parse_tour(rows: list[dict], tour_id: str) -> list[dict]:
    stops = []
    for r in rows:
        city = r.get("city", "").strip()
        if not city:
            continue
        arrive = _date_to_iso(r.get("date_arrive", ""))
        depart = _date_to_iso(r.get("date_depart", ""))
        stops.append({
            "tour": tour_id,
            "city": city,
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "arrive": arrive,
            "depart": depart,
            "notes": r.get("notes", "").strip(),
        })
    return stops


def _parse_copresence(rows: list[dict]) -> list[dict]:
    events = []
    for r in rows:
        loc = r.get("location", "").strip()
        if not loc:
            continue
        events.append({
            "location": loc,
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "date_start": _date_to_iso(r.get("date_start", "")),
            "date_end": _date_to_iso(r.get("date_end", "")),
            "persons": [p.strip() for p in r.get("persons_present", "").split("|") if p.strip()],
            "encounters": r.get("verified_encounters", "").strip(),
            "source": r.get("source", "").strip(),
            "notes": r.get("notes", "").strip(),
        })
    return events


def build() -> None:
    tour_1770 = _parse_tour(_load_csv(TOUR_1770_PATH), "1770")
    tour_1772 = _parse_tour(_load_csv(TOUR_1772_PATH), "1772")
    copresence = _parse_copresence(_load_csv(COPRESENCE_PATH))

    data = {
        "tour_1770": tour_1770,
        "tour_1772": tour_1772,
        "copresence": copresence,
    }
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    # Un-double braces, then insert data
    html = HTML_TEMPLATE.replace("{{", "{").replace("}}", "}")
    html = html.replace("{TOUR_DATA_JSON}", data_json, 1)

    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"Tour map -> {OUT_PATH}")
    print(f"  1770 stops: {len(tour_1770)}")
    print(f"  1772 stops: {len(tour_1772)}")
    print(f"  Co-presence events: {len(copresence)}")


# ── HTML Template ─────────────────────────────────────────────────────
# All JS braces are doubled.  {TOUR_DATA_JSON} is the only placeholder.

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Charles Burney — European Tours 1770 &amp; 1772</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
*,*::before,*::after {{ box-sizing:border-box; }}
body {{ margin:0; font-family:'Segoe UI',system-ui,-apple-system,sans-serif; background:#1a1a2e; color:#e0e0e0; }}
#map {{ position:absolute; top:0; left:0; right:0; bottom:0; z-index:0; }}

/* ── Control panel ────────────────────────────────────────── */
#controls {{
  position:absolute; top:12px; left:60px; z-index:1000;
  background:rgba(26,26,46,0.92); border-radius:10px;
  padding:14px 18px; min-width:340px; max-width:420px;
  box-shadow:0 4px 20px rgba(0,0,0,0.4); border:1px solid rgba(255,255,255,0.08);
}}
#controls h1 {{
  margin:0 0 10px; font-size:15px; font-weight:600; color:#d4c5a9;
  letter-spacing:0.3px;
}}
.tour-pills {{ display:flex; gap:6px; margin-bottom:12px; }}
.tour-pills button {{
  padding:5px 14px; border:1px solid rgba(255,255,255,0.15); border-radius:16px;
  background:transparent; color:#bbb; font-size:12px; cursor:pointer;
  transition:all 0.2s;
}}
.tour-pills button.active {{ color:#fff; border-color:currentColor; }}
.tour-pills button[data-tour="1770"].active {{ background:rgba(74,111,165,0.3); color:#7ba3d4; border-color:#7ba3d4; }}
.tour-pills button[data-tour="1772"].active {{ background:rgba(160,120,85,0.3); color:#c9a87c; border-color:#c9a87c; }}
.tour-pills button[data-tour="both"].active {{ background:rgba(180,160,130,0.2); color:#d4c5a9; border-color:#d4c5a9; }}

/* Timeline */
.timeline-row {{ display:flex; align-items:center; gap:8px; margin-bottom:8px; }}
#playBtn {{
  width:30px; height:30px; border:none; border-radius:50%;
  background:rgba(212,197,169,0.15); color:#d4c5a9; font-size:14px;
  cursor:pointer; display:flex; align-items:center; justify-content:center;
  transition:background 0.2s;
}}
#playBtn:hover {{ background:rgba(212,197,169,0.3); }}
#timeline {{
  flex:1; -webkit-appearance:none; appearance:none; height:4px;
  background:rgba(255,255,255,0.12); border-radius:2px; outline:none;
  cursor:pointer;
}}
#timeline::-webkit-slider-thumb {{
  -webkit-appearance:none; width:14px; height:14px; border-radius:50%;
  background:#d4c5a9; cursor:pointer;
}}
#timeline::-moz-range-thumb {{
  width:14px; height:14px; border-radius:50%; background:#d4c5a9;
  border:none; cursor:pointer;
}}
#dateLabel {{ font-size:11px; color:#999; min-width:80px; text-align:right; }}

/* Speed */
.speed-row {{ display:flex; align-items:center; gap:6px; margin-bottom:10px; }}
.speed-row label {{ font-size:11px; color:#888; }}
.speed-row button {{
  padding:3px 10px; border:1px solid rgba(255,255,255,0.1); border-radius:12px;
  background:transparent; color:#999; font-size:11px; cursor:pointer; transition:all 0.2s;
}}
.speed-row button.active {{ color:#d4c5a9; border-color:#d4c5a9; background:rgba(212,197,169,0.12); }}

/* Current stop info */
#stopInfo {{
  font-size:12px; color:#aaa; line-height:1.5; min-height:20px;
  border-top:1px solid rgba(255,255,255,0.06); padding-top:8px;
}}
#stopInfo .city-name {{ color:#d4c5a9; font-weight:600; font-size:13px; }}
#stopInfo .persons {{ color:#8bafdb; }}

/* ── Legend ────────────────────────────────────────────────── */
#legend {{
  position:absolute; bottom:28px; left:12px; z-index:1000;
  background:rgba(26,26,46,0.88); border-radius:8px;
  padding:10px 14px; font-size:11px;
  border:1px solid rgba(255,255,255,0.06);
}}
#legend h3 {{ margin:0 0 6px; font-size:11px; color:#999; font-weight:500; text-transform:uppercase; letter-spacing:0.5px; }}
.legend-item {{ display:flex; align-items:center; gap:8px; margin-bottom:4px; }}
.legend-swatch {{ width:24px; height:3px; border-radius:2px; }}
.legend-dot {{ width:10px; height:10px; border-radius:50%; border:2px solid; background:transparent; }}
.legend-pulse {{ width:12px; height:12px; border-radius:50%; position:relative; }}
.legend-pulse::after {{
  content:''; position:absolute; inset:-3px; border-radius:50%;
  border:1.5px solid #e6c656; opacity:0.5;
}}

/* Leaflet overrides */
.leaflet-popup-content-wrapper {{
  background:rgba(26,26,46,0.95); color:#ddd; border-radius:8px;
  border:1px solid rgba(255,255,255,0.1);
}}
.leaflet-popup-tip {{ background:rgba(26,26,46,0.95); }}
.leaflet-popup-content {{ font-size:12px; line-height:1.5; }}
.leaflet-popup-content .popup-city {{ font-size:14px; font-weight:600; color:#d4c5a9; margin-bottom:4px; }}
.leaflet-popup-content .popup-dates {{ color:#999; margin-bottom:6px; }}
.leaflet-popup-content .popup-persons {{ color:#8bafdb; }}
.leaflet-popup-content .popup-encounters {{ color:#aaa; font-style:italic; margin-top:4px; }}
.leaflet-popup-content .popup-notes {{ color:#888; margin-top:4px; font-size:11px; }}
.leaflet-popup-close-button {{ color:#888 !important; }}

/* Pulsing animation for co-presence markers */
@keyframes pulse-ring {{
  0% {{ transform:scale(1); opacity:0.6; }}
  100% {{ transform:scale(2.2); opacity:0; }}
}}
</style>
</head>
<body>
<div id="map"></div>

<div id="controls">
  <div style="font-size:11px;margin-bottom:8px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
    <a href="../index.html" style="color:#9a6f2a;text-decoration:none;">&#8592; Projects</a>
    <span style="color:#ccc;">|</span>
    <a href="full_network.html" style="color:#666;text-decoration:none;">Full Network</a>
    <a href="correspondent_network.html" style="color:#666;text-decoration:none;">Burney Family</a>
    <span style="color:#1a1816;font-weight:600;">Tours</span>
  </div>
  <h1>Charles Burney &mdash; European Tours</h1>
  <p style="font-size:12px;color:#999;margin:4px 0 10px;line-height:1.5;">Two research journeys for the <em>History of Music</em>: France &amp; Italy (Jun&ndash;Dec 1770) and Germany, Austria &amp; the Low Countries (Jul&ndash;Nov 1772). Gold markers show verified encounters with named individuals. Click any stop for details.</p>
  <div class="tour-pills">
    <button data-tour="1770" class="active">1770 Italy</button>
    <button data-tour="1772" class="active">1772 Germany</button>
    <button data-tour="both" class="active">Both</button>
  </div>
  <div class="timeline-row">
    <button id="playBtn" title="Play/Pause">&#9654;</button>
    <input type="range" id="timeline" min="0" max="1000" value="0">
    <span id="dateLabel">Jun 1770</span>
  </div>
  <div class="speed-row">
    <label>Speed:</label>
    <button data-speed="0.3">Slow</button>
    <button data-speed="1" class="active">Medium</button>
    <button data-speed="3">Fast</button>
  </div>
  <div id="stopInfo"></div>
</div>

<div id="legend">
  <h3>Legend</h3>
  <div class="legend-item"><div class="legend-swatch" style="background:#4a6fa5"></div> 1770 Italy Tour</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#a07855"></div> 1772 Germany Tour</div>
  <div class="legend-item"><div class="legend-dot" style="border-color:#4a6fa5"></div> Tour stop</div>
  <div class="legend-item">
    <div class="legend-pulse" style="background:rgba(230,198,86,0.25); border-radius:50%; border:2px solid #e6c656;"></div>
    Co-presence event
  </div>
  <div style="margin-top:4px; color:#777; font-size:10px;">Marker size = days at stop</div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
(function() {{
  "use strict";

  // ── Data ──────────────────────────────────────────────────
  var DATA = {TOUR_DATA_JSON};

  // ── Date helpers ──────────────────────────────────────────
  function parseDate(s) {{
    if (!s) return null;
    var p = s.split("-");
    return new Date(+p[0], +p[1]-1, +p[2]);
  }}
  function daysBetween(a, b) {{
    if (!a || !b) return 1;
    var d = Math.round((b - a) / 86400000);
    return d < 1 ? 1 : d;
  }}
  function formatDate(d) {{
    if (!d) return "";
    var months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
    return d.getDate() + " " + months[d.getMonth()] + " " + d.getFullYear();
  }}
  function dateToFrac(d) {{
    if (!d) return 0;
    return (d - TIMELINE_START) / (TIMELINE_END - TIMELINE_START);
  }}

  // ── Compute timeline bounds ───────────────────────────────
  var TIMELINE_START = new Date(1770, 5, 1);   // 1 Jun 1770
  var TIMELINE_END   = new Date(1772, 11, 1);  // 1 Dec 1772
  var TIMELINE_SPAN  = TIMELINE_END - TIMELINE_START;

  // ── Prepare stops with parsed dates and duration ──────────
  var allStops = DATA.tour_1770.concat(DATA.tour_1772);
  allStops.forEach(function(s) {{
    s._arrive = parseDate(s.arrive);
    s._depart = parseDate(s.depart);
    s._days = daysBetween(s._arrive, s._depart);
  }});

  // Build co-presence lookup by approximate lat/lon
  var copresenceByKey = {{}};
  DATA.copresence.forEach(function(c) {{
    c._start = parseDate(c.date_start);
    c._end   = parseDate(c.date_end);
    var key = c.lat.toFixed(2) + "," + c.lon.toFixed(2);
    if (!copresenceByKey[key]) copresenceByKey[key] = [];
    copresenceByKey[key].push(c);
  }});

  function getCopresence(stop) {{
    var key = stop.lat.toFixed(2) + "," + stop.lon.toFixed(2);
    return copresenceByKey[key] || [];
  }}

  // ── Escape HTML for safe rendering ────────────────────────
  function esc(s) {{
    var el = document.createElement("span");
    el.textContent = s;
    return el.innerHTML;
  }}

  // ── Map ───────────────────────────────────────────────────
  var map = L.map("map", {{
    zoomControl: false,
    attributionControl: true
  }}).setView([47, 10], 5);

  L.control.zoom({{ position: "bottomright" }}).addTo(map);

  L.tileLayer("https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}@2x.png", {{
    attribution: '&copy; <a href="https://carto.com">CARTO</a> &copy; <a href="https://osm.org/copyright">OSM</a>',
    maxZoom: 18
  }}).addTo(map);

  // ── Tour colours ──────────────────────────────────────────
  var COLORS = {{
    "1770": "#4a6fa5",
    "1772": "#a07855"
  }};

  // ── Build route polylines ─────────────────────────────────
  function buildPolyline(stops, color) {{
    var coords = stops.map(function(s) {{ return [s.lat, s.lon]; }});
    return L.polyline(coords, {{
      color: color,
      weight: 2.5,
      opacity: 0.7,
      dashArray: "6 4",
      smoothFactor: 1.5
    }});
  }}

  var line1770 = buildPolyline(DATA.tour_1770, COLORS["1770"]);
  var line1772 = buildPolyline(DATA.tour_1772, COLORS["1772"]);

  // ── Build stop markers ────────────────────────────────────
  function buildPopupContent(stop) {{
    var cp = getCopresence(stop);
    var html = '<div class="popup-city">' + esc(stop.city) + '</div>';
    html += '<div class="popup-dates">';
    if (stop._arrive) html += formatDate(stop._arrive);
    if (stop._depart && stop._depart.getTime() !== stop._arrive.getTime())
      html += ' &ndash; ' + formatDate(stop._depart);
    html += ' (' + stop._days + (stop._days === 1 ? ' day' : ' days') + ')</div>';

    if (cp.length > 0) {{
      cp.forEach(function(c) {{
        var others = c.persons.filter(function(p) {{ return p !== "Charles Burney"; }});
        if (others.length) {{
          html += '<div class="popup-persons"><strong>People met:</strong> ' + esc(others.join(", ")) + '</div>';
        }}
        if (c.encounters) {{
          html += '<div class="popup-encounters">' + esc(c.encounters.replace(/\|/g, "; ")) + '</div>';
        }}
      }});
    }}
    if (stop.notes) {{
      html += '<div class="popup-notes">' + esc(stop.notes) + '</div>';
    }}
    return html;
  }}

  function markerRadius(days) {{
    return Math.max(4, Math.min(14, 3 + Math.sqrt(days) * 2.5));
  }}

  var markers1770 = [];
  var markers1772 = [];

  function buildMarkers(stops, color, arr) {{
    stops.forEach(function(s) {{
      var cp = getCopresence(s);
      var hasCopresence = cp.length > 0;
      var r = markerRadius(s._days);

      var marker = L.circleMarker([s.lat, s.lon], {{
        radius: r,
        fillColor: hasCopresence ? "#e6c656" : color,
        fillOpacity: hasCopresence ? 0.35 : 0.25,
        color: hasCopresence ? "#e6c656" : color,
        weight: hasCopresence ? 2.5 : 1.8,
        opacity: 0.9
      }});
      marker.bindPopup(buildPopupContent(s), {{ maxWidth: 320 }});
      marker._stopData = s;
      marker._hasCopresence = hasCopresence;
      arr.push(marker);

      // Add pulse ring for co-presence
      if (hasCopresence) {{
        var pulse = L.circleMarker([s.lat, s.lon], {{
          radius: r + 6,
          fillColor: "transparent",
          fillOpacity: 0,
          color: "#e6c656",
          weight: 1.5,
          opacity: 0.3
        }});
        marker._pulseRing = pulse;
        arr.push(pulse);
      }}
    }});
  }}

  buildMarkers(DATA.tour_1770, COLORS["1770"], markers1770);
  buildMarkers(DATA.tour_1772, COLORS["1772"], markers1772);

  // ── Layer groups ──────────────────────────────────────────
  var group1770 = L.layerGroup([line1770].concat(markers1770));
  var group1772 = L.layerGroup([line1772].concat(markers1772));
  group1770.addTo(map);
  group1772.addTo(map);

  // ── Moving dot (animation marker) ─────────────────────────
  var movingDot = L.circleMarker([0, 0], {{
    radius: 6,
    fillColor: "#fff",
    fillOpacity: 0.9,
    color: "#fff",
    weight: 2,
    opacity: 1
  }});

  // ── State ─────────────────────────────────────────────────
  var state = {{
    playing: false,
    speed: 1,
    currentFrac: 0,
    visibleTours: {{ "1770": true, "1772": true }},
    animFrame: null,
    lastTime: 0,
    activePopup: null,
    popupTimeout: null
  }};

  // ── Build sorted event list for animation ─────────────────
  var animEvents = [];
  function addAnimEvents(stops, tourId) {{
    stops.forEach(function(s) {{
      if (s._arrive) {{
        animEvents.push({{
          time: s._arrive,
          frac: dateToFrac(s._arrive),
          lat: s.lat,
          lon: s.lon,
          stop: s,
          tour: tourId
        }});
      }}
    }});
  }}
  addAnimEvents(DATA.tour_1770, "1770");
  addAnimEvents(DATA.tour_1772, "1772");
  animEvents.sort(function(a, b) {{ return a.frac - b.frac; }});

  // ── Interpolate position along a tour at a given fraction ──
  function interpolatePosition(stops, frac) {{
    if (stops.length === 0) return null;
    var dated = stops.filter(function(s) {{ return s._arrive; }});
    if (dated.length === 0) return null;

    var fracs = dated.map(function(s) {{ return dateToFrac(s._arrive); }});

    if (frac <= fracs[0]) return {{ lat: dated[0].lat, lon: dated[0].lon, stop: dated[0] }};
    if (frac >= fracs[fracs.length - 1]) {{
      var last = dated[dated.length - 1];
      return {{ lat: last.lat, lon: last.lon, stop: last }};
    }}

    for (var i = 0; i < fracs.length - 1; i++) {{
      if (frac >= fracs[i] && frac <= fracs[i + 1]) {{
        var t = (frac - fracs[i]) / (fracs[i + 1] - fracs[i]);
        return {{
          lat: dated[i].lat + (dated[i + 1].lat - dated[i].lat) * t,
          lon: dated[i].lon + (dated[i + 1].lon - dated[i].lon) * t,
          stop: t < 0.5 ? dated[i] : dated[i + 1]
        }};
      }}
    }}
    return {{ lat: dated[0].lat, lon: dated[0].lon, stop: dated[0] }};
  }}

  // ── Find current stop at given fraction ───────────────────
  function findNearestStop(frac) {{
    var best = null;
    var bestDist = Infinity;
    animEvents.forEach(function(e) {{
      if (!state.visibleTours[e.tour]) return;
      var d = Math.abs(e.frac - frac);
      if (d < bestDist) {{ bestDist = d; best = e; }}
    }});
    return best;
  }}

  // ── Update display for a given timeline fraction ──────────
  function updateTimeline(frac) {{
    frac = Math.max(0, Math.min(1, frac));
    state.currentFrac = frac;

    document.getElementById("timeline").value = Math.round(frac * 1000);

    var currentDate = new Date(TIMELINE_START.getTime() + frac * TIMELINE_SPAN);
    document.getElementById("dateLabel").textContent = formatDate(currentDate);

    // Position moving dot
    var pos1770 = state.visibleTours["1770"] ? interpolatePosition(DATA.tour_1770, frac) : null;
    var pos1772 = state.visibleTours["1772"] ? interpolatePosition(DATA.tour_1772, frac) : null;

    // Determine which tour is active at this point in time
    var gap1770Start = dateToFrac(new Date(1770, 5, 1));
    var gap1770End = dateToFrac(new Date(1771, 0, 1));
    var gap1772Start = dateToFrac(new Date(1772, 6, 1));
    var gap1772End = dateToFrac(new Date(1772, 11, 1));

    var show1770dot = state.visibleTours["1770"] && frac >= gap1770Start && frac <= gap1770End;
    var show1772dot = state.visibleTours["1772"] && frac >= gap1772Start && frac <= gap1772End;

    if (show1770dot && pos1770) {{
      movingDot.setLatLng([pos1770.lat, pos1770.lon]);
      movingDot.setStyle({{ fillColor: COLORS["1770"], color: COLORS["1770"] }});
      if (!map.hasLayer(movingDot)) movingDot.addTo(map);
    }} else if (show1772dot && pos1772) {{
      movingDot.setLatLng([pos1772.lat, pos1772.lon]);
      movingDot.setStyle({{ fillColor: COLORS["1772"], color: COLORS["1772"] }});
      if (!map.hasLayer(movingDot)) movingDot.addTo(map);
    }} else {{
      if (map.hasLayer(movingDot)) map.removeLayer(movingDot);
    }}

    // Show stop info
    var nearest = findNearestStop(frac);
    updateStopInfo(nearest, frac);

    // Reveal markers up to current time during animation
    if (state.playing) {{
      revealMarkers(markers1770, frac);
      revealMarkers(markers1772, frac);
    }}
  }}

  function revealMarkers(markerArr, frac) {{
    markerArr.forEach(function(m) {{
      if (!m._stopData) return;
      var stopFrac = m._stopData._arrive ? dateToFrac(m._stopData._arrive) : 0;
      var shouldShow = stopFrac <= frac + 0.002;
      if (shouldShow) {{
        m.setStyle({{ fillOpacity: m._hasCopresence ? 0.35 : 0.25, opacity: 0.9 }});
      }} else {{
        m.setStyle({{ fillOpacity: 0.06, opacity: 0.15 }});
      }}
    }});
  }}

  function resetMarkerOpacity() {{
    markers1770.concat(markers1772).forEach(function(m) {{
      if (!m._stopData) return;
      m.setStyle({{
        fillOpacity: m._hasCopresence ? 0.35 : 0.25,
        opacity: 0.9
      }});
    }});
  }}

  function updateStopInfo(event, frac) {{
    var el = document.getElementById("stopInfo");
    if (!event) {{ el.textContent = ""; return; }}

    if (Math.abs(event.frac - frac) > 0.015) {{ el.textContent = ""; return; }}

    var s = event.stop;
    var cp = getCopresence(s);

    // Build DOM nodes instead of setting innerHTML
    el.textContent = "";
    var citySpan = document.createElement("span");
    citySpan.className = "city-name";
    citySpan.textContent = s.city;
    el.appendChild(citySpan);

    var dateText = " \u2014 " + formatDate(s._arrive);
    if (s._days > 1) dateText += " (" + s._days + " days)";
    el.appendChild(document.createTextNode(dateText));

    if (cp.length > 0) {{
      cp.forEach(function(c) {{
        var others = c.persons.filter(function(p) {{ return p !== "Charles Burney"; }});
        if (others.length) {{
          el.appendChild(document.createElement("br"));
          var persSpan = document.createElement("span");
          persSpan.className = "persons";
          persSpan.textContent = "Met: " + others.join(", ");
          el.appendChild(persSpan);
        }}
      }});
    }}

    // Auto-show popup during animation
    if (state.playing && Math.abs(event.frac - frac) < 0.003) {{
      showStopPopup(event);
    }}
  }}

  function showStopPopup(event) {{
    var allMarkers = markers1770.concat(markers1772);
    for (var i = 0; i < allMarkers.length; i++) {{
      var m = allMarkers[i];
      if (m._stopData && m._stopData === event.stop) {{
        if (state.activePopup !== m) {{
          if (state.popupTimeout) clearTimeout(state.popupTimeout);
          m.openPopup();
          state.activePopup = m;
          state.popupTimeout = setTimeout(function() {{
            m.closePopup();
            state.activePopup = null;
          }}, 3000);
        }}
        break;
      }}
    }}
  }}

  // ── Animation loop ────────────────────────────────────────
  function animate(timestamp) {{
    if (!state.playing) return;
    if (!state.lastTime) state.lastTime = timestamp;

    var elapsed = timestamp - state.lastTime;
    state.lastTime = timestamp;

    // Speed: at speed=1, traverse full timeline in ~60 seconds
    var increment = (elapsed / 60000) * state.speed;
    state.currentFrac += increment;

    if (state.currentFrac >= 1) {{
      state.currentFrac = 1;
      stopAnimation();
      updateTimeline(1);
      resetMarkerOpacity();
      return;
    }}

    updateTimeline(state.currentFrac);
    state.animFrame = requestAnimationFrame(animate);
  }}

  function startAnimation() {{
    state.playing = true;
    state.lastTime = 0;
    document.getElementById("playBtn").textContent = "\u23F8";
    // Dim all markers at start
    if (state.currentFrac < 0.01) {{
      markers1770.concat(markers1772).forEach(function(m) {{
        if (m._stopData) m.setStyle({{ fillOpacity: 0.06, opacity: 0.15 }});
      }});
    }}
    state.animFrame = requestAnimationFrame(animate);
  }}

  function stopAnimation() {{
    state.playing = false;
    document.getElementById("playBtn").textContent = "\u25B6";
    if (state.animFrame) cancelAnimationFrame(state.animFrame);
    state.animFrame = null;
    resetMarkerOpacity();
  }}

  // ── UI Bindings ───────────────────────────────────────────
  document.getElementById("playBtn").addEventListener("click", function() {{
    if (state.playing) {{
      stopAnimation();
    }} else {{
      if (state.currentFrac >= 0.99) state.currentFrac = 0;
      startAnimation();
    }}
  }});

  document.getElementById("timeline").addEventListener("input", function() {{
    var frac = +this.value / 1000;
    if (state.playing) stopAnimation();
    state.currentFrac = frac;
    updateTimeline(frac);
    resetMarkerOpacity();
  }});

  // Speed buttons
  document.querySelectorAll(".speed-row button").forEach(function(btn) {{
    btn.addEventListener("click", function() {{
      document.querySelectorAll(".speed-row button").forEach(function(b) {{ b.classList.remove("active"); }});
      this.classList.add("active");
      state.speed = parseFloat(this.getAttribute("data-speed"));
    }});
  }});

  // Tour toggle pills
  document.querySelectorAll(".tour-pills button").forEach(function(btn) {{
    btn.addEventListener("click", function() {{
      var tour = this.getAttribute("data-tour");
      if (tour === "both") {{
        var allActive = state.visibleTours["1770"] && state.visibleTours["1772"];
        state.visibleTours["1770"] = !allActive;
        state.visibleTours["1772"] = !allActive;
      }} else {{
        state.visibleTours[tour] = !state.visibleTours[tour];
      }}

      document.querySelector('[data-tour="1770"]').classList.toggle("active", state.visibleTours["1770"]);
      document.querySelector('[data-tour="1772"]').classList.toggle("active", state.visibleTours["1772"]);
      document.querySelector('[data-tour="both"]').classList.toggle("active",
        state.visibleTours["1770"] && state.visibleTours["1772"]);

      if (state.visibleTours["1770"]) {{ if (!map.hasLayer(group1770)) group1770.addTo(map); }}
      else {{ if (map.hasLayer(group1770)) map.removeLayer(group1770); }}

      if (state.visibleTours["1772"]) {{ if (!map.hasLayer(group1772)) group1772.addTo(map); }}
      else {{ if (map.hasLayer(group1772)) map.removeLayer(group1772); }}
    }});
  }});

  // ── Pulsing co-presence animation ─────────────────────────
  var pulsePhase = 0;
  setInterval(function() {{
    pulsePhase = (pulsePhase + 1) % 60;
    var scale = 1 + 0.4 * Math.sin(pulsePhase * Math.PI / 30);
    markers1770.concat(markers1772).forEach(function(m) {{
      if (m._pulseRing && map.hasLayer(m._pulseRing)) {{
        var baseR = markerRadius(m._stopData._days) + 6;
        m._pulseRing.setRadius(baseR * scale);
        m._pulseRing.setStyle({{ opacity: 0.15 + 0.15 * Math.sin(pulsePhase * Math.PI / 30) }});
      }}
    }});
  }}, 50);

  // ── Initial state ─────────────────────────────────────────
  updateTimeline(0);
  resetMarkerOpacity();

}})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    build()
