#!/usr/bin/env python3
"""
Build an interactive HTML map of 18c venue mentions across the fiction corpus.

Reads gazetteer/mentions.json and writes gazetteer/map.html.

Marker radius scales with sqrt(total_mentions).
Colour encodes venue type.
Popups show per-text mention counts sorted chronologically.
Anachronistic mentions are flagged in the popup.

Historical base tiles:
    Rocque 1746  — NLS tile service (see NLS_TILES comment below)
    Horwood 1799 — NLS tile service
    Modern       — CartoDB Positron (always reliable fallback)

Usage:
    python3 gazetteer/build_map.py
    open gazetteer/map.html
"""

import json
import math
from pathlib import Path

import folium
from folium import plugins


# ---------------------------------------------------------------------------
# Historical tile layer URLs
#
# Rocque 1746 — served by the MOLA / Locating London's Past project (DHI)
#   https://www.locatinglondon.org/  (min z=13, max z=15)
#
# Horwood 1792–99 — served by the Romantic London project
#   https://www.romanticlondon.org/explore-horwoods-plan/  (min z=11, max z=16)
#
# Both are tested live as of 2026-02.
# ---------------------------------------------------------------------------
NLS_ROCQUE = (
    "https://www.dhi.ac.uk/san/llptiles/molarocque/{z}/{x}/{y}.png",
    "Rocque's London 1746 (MOLA / Locating London's Past)",
    "Map tiles © Museum of London Archaeology / DHI, based on John Rocque 1746",
)
NLS_HORWOOD = (
    "https://www.romanticlondon.org/horwoodplan/{z}/{x}/{y}.png",
    "Horwood's London 1792–99 (Romantic London)",
    "Map tiles © Romantic London project, based on Richard Horwood 1792–99",
)

# ---------------------------------------------------------------------------
# Venue type → display colour
# ---------------------------------------------------------------------------
VENUE_TYPE = {
    "LON001": "garden",    # Vauxhall
    "LON002": "garden",    # Ranelagh
    "LON003": "garden",    # Marylebone Gardens
    "BAT007": "garden",    # Sydney Gardens
    "LON006": "theatre",   # Drury Lane
    "LON007": "theatre",   # Covent Garden
    "LON008": "theatre",   # King's Theatre
    "LON004": "assembly",  # The Pantheon
    "LON005": "assembly",  # Almack's
    "BAT001": "assembly",  # Lower Assembly Rooms
    "BAT002": "assembly",  # Upper Assembly Rooms
    "BAT003": "assembly",  # Pump Room
    "LON018": "club",      # White's
}
COLOUR = {
    "garden":   "#27ae60",   # green
    "theatre":  "#e74c3c",   # red
    "assembly": "#2980b9",   # blue
    "club":     "#8e44ad",   # purple
    "street":   "#d35400",   # burnt orange  (default)
}
TYPE_LABEL = {
    "garden":  "Pleasure garden",
    "theatre": "Theatre / opera house",
    "assembly":"Assembly room / social venue",
    "club":    "Gentlemen's club",
    "street":  "Street, square or park",
}


def venue_colour(venue_id: str) -> str:
    return COLOUR[VENUE_TYPE.get(venue_id, "street")]


def venue_type_label(venue_id: str) -> str:
    return TYPE_LABEL[VENUE_TYPE.get(venue_id, "street")]


# ---------------------------------------------------------------------------
# Anachronism lookup (mirrors validate_venues.py logic, read from JSON)
# ---------------------------------------------------------------------------
def is_anachronistic(mention: dict, opened: int | None, closed: int | None) -> bool:
    yr = mention.get("text_year")
    if yr is None:
        return False
    if opened and yr < opened:
        return True
    if closed and yr > closed:
        return True
    return False


# ---------------------------------------------------------------------------
# Popup HTML
# ---------------------------------------------------------------------------
def make_popup(venue: dict) -> str:
    name    = venue["name"]
    city    = venue["city"]
    opened  = venue["opened"]
    closed  = venue["closed"]
    total   = venue["total_mentions"]
    vtype   = venue_type_label(venue["venue_id"])
    colour  = venue_colour(venue["venue_id"])

    date_str = ""
    if opened:
        date_str = f"Open {opened}"
        if closed:
            date_str += f"–{closed}"
        else:
            date_str += "–"
    elif closed:
        date_str = f"Closed {closed}"

    n_texts = len(venue["mentions"])

    rows = []
    for m in venue["mentions"]:
        yr   = m["text_year"] or m["pub_year"] or "?"
        auth = m["author"].replace("Frances", "F.").replace("Samuel", "S.") \
                          .replace("Henry", "H.").replace("Tobias", "T.") \
                          .replace("Jane", "J.").replace("Ann", "A.") \
                          .replace("Charlotte", "C.").replace("Maria", "M.") \
                          .replace("Horace", "Ho.").replace("Clara", "Cl.") \
                          .replace("William", "W.").replace("Eliza", "El.")
        flag = ""
        if is_anachronistic(m, opened, closed):
            flag = " ⚠"
        count_cell = f'<td style="text-align:right;padding:1px 6px">{m["count"]}{flag}</td>'
        rows.append(
            f'<tr>'
            f'<td style="padding:1px 6px;color:#666">{yr}</td>'
            f'<td style="padding:1px 6px">{auth}</td>'
            f'<td style="padding:1px 6px;font-style:italic">{m["title"]}</td>'
            f'{count_cell}'
            f'</tr>'
        )

    table = (
        '<table style="font-size:11px;border-collapse:collapse;width:100%">'
        '<thead><tr style="background:#f0f0f0">'
        '<th style="padding:2px 6px;text-align:left">Year</th>'
        '<th style="padding:2px 6px;text-align:left">Author</th>'
        '<th style="padding:2px 6px;text-align:left">Title</th>'
        '<th style="padding:2px 6px;text-align:right">n</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>'
    )

    html = f"""
<div style="font-family:Georgia,serif;max-width:360px;padding:4px">
  <div style="border-left:4px solid {colour};padding-left:8px;margin-bottom:6px">
    <div style="font-size:14px;font-weight:bold;margin-bottom:2px">{name}</div>
    <div style="font-size:11px;color:#555">{city} · {vtype}</div>
    {"" if not date_str else f'<div style="font-size:11px;color:#888">{date_str}</div>'}
  </div>
  <div style="font-size:12px;margin-bottom:6px">
    <b>{total}</b> mention{"s" if total != 1 else ""} across
    <b>{n_texts}</b> text{"s" if n_texts != 1 else ""}
  </div>
  {table}
  {"" if not any(is_anachronistic(m, opened, closed) for m in venue["mentions"])
      else '<div style="font-size:10px;color:#c0392b;margin-top:4px">⚠ anachronistic mention</div>'}
</div>
"""
    return html


# ---------------------------------------------------------------------------
# Legend HTML (injected as a floating div)
# ---------------------------------------------------------------------------
LEGEND_HTML = """
<div style="
    position: fixed; bottom: 30px; left: 30px; z-index: 1000;
    background: rgba(255,255,255,0.92); border: 1px solid #ccc;
    border-radius: 6px; padding: 10px 14px;
    font-family: Georgia, serif; font-size: 12px; line-height: 1.8;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.15)">
  <div style="font-weight:bold;margin-bottom:4px">Venue type</div>
  <div><span style="color:#27ae60;font-size:16px">●</span> Pleasure garden</div>
  <div><span style="color:#e74c3c;font-size:16px">●</span> Theatre / opera house</div>
  <div><span style="color:#2980b9;font-size:16px">●</span> Assembly room</div>
  <div><span style="color:#8e44ad;font-size:16px">●</span> Gentlemen's club</div>
  <div><span style="color:#d35400;font-size:16px">●</span> Street, square or park</div>
  <hr style="margin:6px 0;border:none;border-top:1px solid #ddd">
  <div style="color:#888;font-size:10px">Radius ∝ √(mention count)</div>
  <div style="color:#888;font-size:10px">Click a marker to see breakdown</div>
</div>
"""


# ---------------------------------------------------------------------------
# Title / info panel HTML
# ---------------------------------------------------------------------------
TITLE_HTML = """
<div style="
    position: fixed; top: 16px; left: 60px; z-index: 1000;
    background: rgba(255,255,255,0.92); border: 1px solid #ccc;
    border-radius: 6px; padding: 8px 14px;
    font-family: Georgia, serif;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.15)">
  <div style="font-size:15px;font-weight:bold">
    Urban Space in 18c Fiction
  </div>
  <div style="font-size:11px;color:#555;margin-top:2px">
    Venue mentions · 33 texts · 1719–1817
  </div>
</div>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build(data_path: Path, out_path: Path) -> None:
    with open(data_path, encoding="utf-8") as f:
        venues = json.load(f)

    # Map centred on London, zoom shows both London and Bath
    m = folium.Map(
        location=[51.51, -0.13],
        zoom_start=12,
        tiles=None,          # add tile layers manually for full control
    )

    # --- Base tile layers ---
    folium.TileLayer(
        tiles="CartoDB positron",
        name="Modern (CartoDB Positron)",
        control=True,
        overlay=False,
    ).add_to(m)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        control=True,
        overlay=False,
    ).add_to(m)

    folium.TileLayer(
        tiles=NLS_ROCQUE[0],
        name=NLS_ROCQUE[1],
        attr=NLS_ROCQUE[2],
        control=True,
        overlay=False,
        opacity=0.9,
        min_zoom=13,
        max_zoom=15,
        max_native_zoom=15,
    ).add_to(m)

    folium.TileLayer(
        tiles=NLS_HORWOOD[0],
        name=NLS_HORWOOD[1],
        attr=NLS_HORWOOD[2],
        control=True,
        overlay=False,
        opacity=0.9,
        min_zoom=11,
        max_zoom=17,
        max_native_zoom=16,
    ).add_to(m)

    # --- Layer groups by venue type ---
    groups = {
        "garden":   folium.FeatureGroup(name="Pleasure gardens", show=True),
        "theatre":  folium.FeatureGroup(name="Theatres &amp; opera houses", show=True),
        "assembly": folium.FeatureGroup(name="Assembly rooms", show=True),
        "club":     folium.FeatureGroup(name="Gentlemen's clubs", show=True),
        "street":   folium.FeatureGroup(name="Streets, squares &amp; parks", show=True),
    }

    for venue in venues:
        vtype  = VENUE_TYPE.get(venue["venue_id"], "street")
        colour = COLOUR[vtype]
        radius = max(6, math.sqrt(venue["total_mentions"]) * 2.5)

        # Scale popup height with number of texts to avoid clipping
        popup_height = min(400, max(220, 80 + len(venue["mentions"]) * 22))

        popup_html = make_popup(venue)
        popup = folium.Popup(
            folium.IFrame(popup_html, width=400, height=popup_height),
            max_width=420,
        )

        tooltip = folium.Tooltip(
            f"<b>{venue['name']}</b><br>{venue['total_mentions']} mention"
            f"{'s' if venue['total_mentions'] != 1 else ''}",
            style="font-family:Georgia,serif;font-size:12px"
        )

        marker = folium.CircleMarker(
            location=[venue["lat"], venue["lon"]],
            radius=radius,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.65,
            weight=1.5,
            popup=popup,
            tooltip=tooltip,
        )

        groups[vtype].add_child(marker)

    for g in groups.values():
        g.add_to(m)

    # --- Minimap ---
    plugins.MiniMap(toggle_display=True, tile_layer="CartoDB positron").add_to(m)

    # --- Layer control ---
    folium.LayerControl(collapsed=False).add_to(m)

    # --- Floating HTML panels ---
    m.get_root().html.add_child(folium.Element(LEGEND_HTML))
    m.get_root().html.add_child(folium.Element(TITLE_HTML))

    m.save(str(out_path))
    print(f"Map written → {out_path}")
    print(f"  {len(venues)} venues plotted")
    total_mentions = sum(v["total_mentions"] for v in venues)
    print(f"  {total_mentions} total mentions across corpus")


if __name__ == "__main__":
    here = Path(__file__).parent
    build(here / "mentions.json", here / "map.html")
