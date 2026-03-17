#!/usr/bin/env python3
"""
Build the self-contained sensory timeline HTML.

Reads venues.csv and sensory.db, writes sensory_timeline.html.

Usage:
    python3 gazetteer/build_sensory_timeline.py
    open gazetteer/sensory_timeline.html
"""

import csv
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

VENUES_PATH = Path(__file__).parent / "venues.csv"
DB_PATH     = Path(__file__).parent / "sensory.db"
OUT_PATH    = Path(__file__).parent / "sensory_timeline.html"

MIN_PASSAGES = 20
MIN_SOURCES  = 3


# ── helpers ────────────────────────────────────────────────────────────────


def fmt_author(s: str) -> str:
    """'FrancesBurney' -> 'Frances Burney'; 'burney' -> 'Burney'."""
    overrides = {"MGLewis": "M. G. Lewis"}
    if s in overrides:
        return overrides[s]
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    return spaced.capitalize() if " " not in spaced else spaced


def assign_decade(date_min: int) -> int:
    """Return the decade for a given year: assign_decade(1725) == 1720."""
    return (date_min // 10) * 10


def venue_qualifies(
    passages: list[dict],
    min_passages: int = MIN_PASSAGES,
    min_sources: int  = MIN_SOURCES,
) -> bool:
    """Return True if the passage list meets the evidence threshold."""
    if len(passages) < min_passages:
        return False
    distinct_sources = {p["source_id"] for p in passages}
    return len(distinct_sources) >= min_sources


def select_highlight(passages: list[dict]) -> dict:
    """
    Choose the best representative passage from a list.

    Preference order:
      1. Fiction over non-fiction (when length is within 20% of the longest).
      2. Longest text otherwise.
    """
    if not passages:
        raise ValueError("passages must be non-empty")
    longest_len = max(len(p["text"]) for p in passages)
    threshold   = longest_len * 0.8

    # Candidates whose text length >= threshold
    candidates = [p for p in passages if len(p["text"]) >= threshold]

    # Prefer fiction among candidates
    fiction = [p for p in candidates if p.get("source_type") == "fiction"]
    if fiction:
        return max(fiction, key=lambda p: len(p["text"]))
    return max(candidates, key=lambda p: len(p["text"]))


# ── data loading ───────────────────────────────────────────────────────────


def load_venues(venues_path: Path) -> dict[str, dict]:
    """Load LON-prefix venues from CSV, keyed by id."""
    with open(venues_path, newline="", encoding="utf-8") as f:
        return {
            row["id"]: {
                "id":            row["id"],
                "name":          row["name"],
                "lat":           float(row["lat"]),
                "lon":           float(row["lon"]),
                "enclosure":     row.get("enclosure", ""),
                "building_type": row.get("building_type", ""),
                "material":      row.get("material", ""),
                "capacity":      row.get("capacity", ""),
            }
            for row in csv.DictReader(f)
            if row["id"].startswith("LON")
        }


def load_evidence(db_path: Path) -> dict[str, list[dict]]:
    """
    Query sensory_evidence for LON venues.

    Returns dict keyed by venue_id, each value a list of passage dicts.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    by_venue: dict[str, list[dict]] = defaultdict(list)
    try:
        for row in conn.execute("""
            SELECT venue_id, source_id, source_type, author, title,
                   pub_year, date_min, date_max, modality, valence, text
            FROM   sensory_evidence
            WHERE  venue_id LIKE 'LON%'
              AND  date_min IS NOT NULL
            ORDER  BY date_min
        """):
            by_venue[row["venue_id"]].append({
                "source_id":   row["source_id"],
                "source_type": row["source_type"],
                "author":      fmt_author(row["author"] or ""),
                "title":       row["title"] or "",
                "pub_year":    row["pub_year"],
                "date_min":    row["date_min"],
                "date_max":    row["date_max"],
                "modality":    row["modality"],
                "valence":     row["valence"],
                "text":        row["text"] or "",
            })
    finally:
        conn.close()
    return dict(by_venue)


# ── computation ────────────────────────────────────────────────────────────


def compute_decade_summaries(
    evidence_by_venue: dict[str, list[dict]],
) -> dict[str, dict[int, dict]]:
    """
    For each qualifying venue, compute per-decade summary dicts.

    Returns:
        {venue_id: {decade: {"passage_count": int, "source_count": int,
                             "modality_breakdown": {modality: int}}}}
    """
    result: dict[str, dict[int, dict]] = {}
    for venue_id, passages in evidence_by_venue.items():
        if not venue_qualifies(passages):
            continue
        decades: dict[int, dict] = {}
        for p in passages:
            decade = assign_decade(p["date_min"])
            if decade not in decades:
                decades[decade] = {
                    "passage_count": 0,
                    "source_ids":    set(),
                    "modality_breakdown": defaultdict(int),
                }
            decades[decade]["passage_count"] += 1
            decades[decade]["source_ids"].add(p["source_id"])
            decades[decade]["modality_breakdown"][p["modality"]] += 1

        # Serialise sets and defaultdicts
        serialised: dict[int, dict] = {}
        for decade, summary in decades.items():
            serialised[decade] = {
                "passage_count":      summary["passage_count"],
                "source_count":       len(summary["source_ids"]),
                "modality_breakdown": dict(summary["modality_breakdown"]),
            }
        result[venue_id] = serialised
    return result


def compute_highlights(
    evidence_by_venue: dict[str, list[dict]],
) -> dict[str, dict[int, dict]]:
    """
    For each qualifying venue and decade, select one highlight passage.

    Returns:
        {venue_id: {decade: passage_dict}}
    """
    result: dict[str, dict[int, dict]] = {}
    for venue_id, passages in evidence_by_venue.items():
        if not venue_qualifies(passages):
            continue
        by_decade: dict[int, list[dict]] = defaultdict(list)
        for p in passages:
            by_decade[assign_decade(p["date_min"])].append(p)
        result[venue_id] = {
            decade: select_highlight(ps)
            for decade, ps in by_decade.items()
        }
    return result


# ── build ──────────────────────────────────────────────────────────────────


def build(
    venues_path: Path = VENUES_PATH,
    db_path:     Path = DB_PATH,
    out_path:    Path = OUT_PATH,
) -> None:
    """Build sensory_timeline.html."""
    venues_all        = load_venues(venues_path)
    evidence_by_venue = load_evidence(db_path)

    # Filter venues to qualifying ones only
    decade_summaries = compute_decade_summaries(evidence_by_venue)
    highlights       = compute_highlights(evidence_by_venue)
    qualifying_ids   = set(decade_summaries.keys())

    venues_out   = [v for vid, v in venues_all.items() if vid in qualifying_ids]
    evidence_out = {vid: ps for vid, ps in evidence_by_venue.items() if vid in qualifying_ids}

    html = HTML_TEMPLATE.format(
        VENUES_JSON           = json.dumps(venues_out, ensure_ascii=False, separators=(",", ":")),
        EVIDENCE_JSON         = json.dumps(evidence_out, ensure_ascii=False, separators=(",", ":")),
        DECADE_SUMMARIES_JSON = json.dumps(
            {vid: {str(k): v for k, v in ds.items()} for vid, ds in decade_summaries.items()},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        HIGHLIGHTS_JSON       = json.dumps(
            {vid: {str(k): v for k, v in hl.items()} for vid, hl in highlights.items()},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )
    out_path.write_text(html, encoding="utf-8")

    total_ev = sum(len(ps) for ps in evidence_out.values())
    print(f"Sensory timeline -> {out_path}")
    print(f"  {len(venues_out)} qualifying venues  {total_ev} passages")


# ── template ───────────────────────────────────────────────────────────────
# Note: literal JS braces are doubled ({{ }}) so str.format() works.

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sensory Timeline &middot; 18c London</title>
<style>
/* ── reset ── */
* {{ box-sizing: border-box; margin: 0; padding: 0; }}

/* ── page layout ── */
body {{ font-family: 'Inter', system-ui, sans-serif; background: #fff; color: #111; }}
.page {{ display: flex; height: 100vh; overflow: hidden; }}

/* ── decade sidebar ── */
.decade-nav {{
  width: 64px;
  background: #111;
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
  overflow-y: auto;
}}
.decade-nav h3 {{
  font-size: 9px;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: #666;
  padding: 10px 0 6px 8px;
}}
.decade-btn {{
  display: block;
  width: 100%;
  padding: 7px 8px;
  background: none;
  border: none;
  border-left: 3px solid transparent;
  color: #aaa;
  font-size: 11px;
  text-align: left;
  cursor: pointer;
}}
.decade-btn.active {{
  border-left: 3px solid #3b82f6;
  color: #fff;
}}
.decade-btn:not(.has-data) {{ color: #444; }}

/* ── main column ── */
.main {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; }}

/* ── top bar ── */
.top-bar {{
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 8px 16px;
  border-bottom: 1px solid #ddd;
  flex-shrink: 0;
  flex-wrap: wrap;
}}
.view-nav {{ display: flex; gap: 12px; align-items: center; font-size: 12px; }}
.view-nav a {{ color: #555; text-decoration: none; }}
.view-nav a:hover {{ color: #111; }}
.view-nav .active-view {{ color: #111; font-weight: 600; }}
.venue-selector {{
  font-size: 12px;
  padding: 4px 6px;
  border: 1px solid #ccc;
  background: #fff;
  color: #111;
}}
.mode-toggle {{ display: flex; gap: 0; margin-left: auto; }}
.mode-btn {{
  padding: 4px 10px;
  font-size: 12px;
  border: 1px solid #ccc;
  background: #fff;
  color: #555;
  cursor: pointer;
}}
.mode-btn.active {{ background: #111; color: #fff; border-color: #111; }}
.mode-btn:first-child {{ border-right: none; }}

/* ── timeline strip ── */
.timeline-strip {{
  display: flex;
  height: 72px;
  border-bottom: 1px solid #ddd;
  flex-shrink: 0;
  overflow-x: auto;
}}
.timeline-decade {{
  flex: 1;
  min-width: 48px;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  justify-content: flex-end;
  padding: 4px 4px 6px 4px;
  border-right: 1px solid #eee;
  cursor: pointer;
}}
.timeline-decade:last-child {{ border-right: none; }}
.timeline-decade.active {{ background: #f0f6ff; }}
.timeline-decade .label {{
  font-size: 9px;
  color: #999;
  margin-bottom: 4px;
  letter-spacing: 0.3px;
}}
.sense-bars {{ display: flex; gap: 2px; align-items: flex-end; height: 36px; width: 100%; }}
.sense-bar {{
  width: 5px;
  min-height: 1px;
  flex-shrink: 0;
}}
.sense-bar.smell   {{ background: #f59e0b; }}
.sense-bar.noise   {{ background: #3b82f6; }}
.sense-bar.crowd   {{ background: #ef4444; }}
.sense-bar.visual  {{ background: #10b981; }}
.sense-bar.thermal {{ background: #8b5cf6; }}

/* ── sense legend ── */
.sense-legend {{
  display: flex;
  gap: 14px;
  padding: 4px 12px;
  font-size: 10px;
  color: #555;
  border-bottom: 1px solid #eee;
  flex-shrink: 0;
  align-items: center;
}}
.legend-dot {{
  display: inline-block;
  width: 8px;
  height: 8px;
  margin-right: 4px;
  vertical-align: middle;
}}

/* ── content area ── */
.content-area {{ flex: 1; overflow-y: auto; padding: 0 0 80px 0; }}
.content-inner {{ max-width: 720px; margin: 0 auto; padding: 16px; }}

/* ── decade sections ── */
.decade-section {{ margin-bottom: 32px; }}
.decade-heading {{
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  color: #555;
  padding: 8px 0 6px 0;
  border-bottom: 1px solid #ddd;
  margin-bottom: 12px;
}}

/* ── cards ── */
.card {{
  padding: 12px 14px;
  margin-bottom: 10px;
  border-top: 1px solid #eee;
}}
.card-fiction    {{ border-left: 3px solid #333; padding-left: 11px; }}
.card-nonfiction {{ border-left: 1px solid #ccc; padding-left: 13px; }}
.card.dimmed {{ opacity: 0.25; }}
.card-header {{ display: flex; gap: 8px; align-items: baseline; margin-bottom: 4px; }}
.source-label {{
  font-size: 10px;
  font-variant: small-caps;
  letter-spacing: 0.5px;
  color: #666;
}}
.card-author {{ font-size: 11px; color: #333; }}
.card-title  {{ font-size: 11px; color: #555; font-style: italic; }}
.card-date   {{ font-size: 10px; color: #888; margin-left: auto; white-space: nowrap; }}
.modality-tag {{
  font-size: 9px;
  color: #999;
  margin-bottom: 4px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}}
.card-text {{
  font-family: Georgia, serif;
  font-style: italic;
  line-height: 1.75;
  font-size: 15px;
  color: #222;
}}
.card-source {{ font-size: 10px; color: #888; margin-top: 6px; }}

/* ── compare row ── */
.compare-row {{ display: flex; gap: 12px; margin-bottom: 10px; }}
.compare-row .card {{ flex: 1; margin-bottom: 0; }}

/* ── story bar ── */
.story-bar {{
  position: fixed;
  bottom: 0;
  left: 64px;
  right: 0;
  background: #111;
  color: #eee;
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 8px 20px;
  font-size: 12px;
  z-index: 100;
}}
.story-pos {{ font-weight: 700; min-width: 60px; }}
.story-hints {{ color: #888; font-size: 11px; }}
.story-hints kbd {{
  background: #333;
  color: #eee;
  padding: 1px 5px;
  font-size: 10px;
  font-family: monospace;
}}
</style>
</head>
<body>

<div class="page">

  <!-- decade sidebar -->
  <nav class="decade-nav">
    <h3>Decade</h3>
    <button class="decade-btn" data-decade="1660">1660</button>
    <button class="decade-btn" data-decade="1670">1670</button>
    <button class="decade-btn" data-decade="1680">1680</button>
    <button class="decade-btn" data-decade="1690">1690</button>
    <button class="decade-btn" data-decade="1700">1700</button>
    <button class="decade-btn" data-decade="1710">1710</button>
    <button class="decade-btn" data-decade="1720">1720</button>
    <button class="decade-btn" data-decade="1730">1730</button>
    <button class="decade-btn" data-decade="1740">1740</button>
    <button class="decade-btn" data-decade="1750">1750</button>
    <button class="decade-btn" data-decade="1760">1760</button>
    <button class="decade-btn" data-decade="1770">1770</button>
    <button class="decade-btn" data-decade="1780">1780</button>
    <button class="decade-btn" data-decade="1790">1790</button>
    <button class="decade-btn" data-decade="1800">1800</button>
    <button class="decade-btn" data-decade="1810">1810</button>
  </nav>

  <!-- main -->
  <div class="main">

    <!-- top bar -->
    <div class="top-bar">
      <div class="view-nav">
        <a href="sensory_time_map.html">Sensory Map</a>
        <a href="venue_explorer.html">Evidence</a>
        <a href="comparison.html">Comparison</a>
        <span class="active-view">Timeline</span>
      </div>
      <select class="venue-selector"></select>
      <div class="mode-toggle">
        <button class="mode-btn" data-mode="story">Story</button>
        <button class="mode-btn active" data-mode="explore">Explore</button>
      </div>
    </div>

    <!-- timeline strip -->
    <div class="timeline-strip">
      <div class="timeline-decade" data-decade="1660"><span class="label">1660</span><div class="sense-bars"><div class="sense-bar smell"></div><div class="sense-bar noise"></div><div class="sense-bar crowd"></div><div class="sense-bar visual"></div><div class="sense-bar thermal"></div></div></div>
      <div class="timeline-decade" data-decade="1670"><span class="label">1670</span><div class="sense-bars"><div class="sense-bar smell"></div><div class="sense-bar noise"></div><div class="sense-bar crowd"></div><div class="sense-bar visual"></div><div class="sense-bar thermal"></div></div></div>
      <div class="timeline-decade" data-decade="1680"><span class="label">1680</span><div class="sense-bars"><div class="sense-bar smell"></div><div class="sense-bar noise"></div><div class="sense-bar crowd"></div><div class="sense-bar visual"></div><div class="sense-bar thermal"></div></div></div>
      <div class="timeline-decade" data-decade="1690"><span class="label">1690</span><div class="sense-bars"><div class="sense-bar smell"></div><div class="sense-bar noise"></div><div class="sense-bar crowd"></div><div class="sense-bar visual"></div><div class="sense-bar thermal"></div></div></div>
      <div class="timeline-decade" data-decade="1700"><span class="label">1700</span><div class="sense-bars"><div class="sense-bar smell"></div><div class="sense-bar noise"></div><div class="sense-bar crowd"></div><div class="sense-bar visual"></div><div class="sense-bar thermal"></div></div></div>
      <div class="timeline-decade" data-decade="1710"><span class="label">1710</span><div class="sense-bars"><div class="sense-bar smell"></div><div class="sense-bar noise"></div><div class="sense-bar crowd"></div><div class="sense-bar visual"></div><div class="sense-bar thermal"></div></div></div>
      <div class="timeline-decade" data-decade="1720"><span class="label">1720</span><div class="sense-bars"><div class="sense-bar smell"></div><div class="sense-bar noise"></div><div class="sense-bar crowd"></div><div class="sense-bar visual"></div><div class="sense-bar thermal"></div></div></div>
      <div class="timeline-decade" data-decade="1730"><span class="label">1730</span><div class="sense-bars"><div class="sense-bar smell"></div><div class="sense-bar noise"></div><div class="sense-bar crowd"></div><div class="sense-bar visual"></div><div class="sense-bar thermal"></div></div></div>
      <div class="timeline-decade" data-decade="1740"><span class="label">1740</span><div class="sense-bars"><div class="sense-bar smell"></div><div class="sense-bar noise"></div><div class="sense-bar crowd"></div><div class="sense-bar visual"></div><div class="sense-bar thermal"></div></div></div>
      <div class="timeline-decade" data-decade="1750"><span class="label">1750</span><div class="sense-bars"><div class="sense-bar smell"></div><div class="sense-bar noise"></div><div class="sense-bar crowd"></div><div class="sense-bar visual"></div><div class="sense-bar thermal"></div></div></div>
      <div class="timeline-decade" data-decade="1760"><span class="label">1760</span><div class="sense-bars"><div class="sense-bar smell"></div><div class="sense-bar noise"></div><div class="sense-bar crowd"></div><div class="sense-bar visual"></div><div class="sense-bar thermal"></div></div></div>
      <div class="timeline-decade" data-decade="1770"><span class="label">1770</span><div class="sense-bars"><div class="sense-bar smell"></div><div class="sense-bar noise"></div><div class="sense-bar crowd"></div><div class="sense-bar visual"></div><div class="sense-bar thermal"></div></div></div>
      <div class="timeline-decade" data-decade="1780"><span class="label">1780</span><div class="sense-bars"><div class="sense-bar smell"></div><div class="sense-bar noise"></div><div class="sense-bar crowd"></div><div class="sense-bar visual"></div><div class="sense-bar thermal"></div></div></div>
      <div class="timeline-decade" data-decade="1790"><span class="label">1790</span><div class="sense-bars"><div class="sense-bar smell"></div><div class="sense-bar noise"></div><div class="sense-bar crowd"></div><div class="sense-bar visual"></div><div class="sense-bar thermal"></div></div></div>
      <div class="timeline-decade" data-decade="1800"><span class="label">1800</span><div class="sense-bars"><div class="sense-bar smell"></div><div class="sense-bar noise"></div><div class="sense-bar crowd"></div><div class="sense-bar visual"></div><div class="sense-bar thermal"></div></div></div>
      <div class="timeline-decade" data-decade="1810"><span class="label">1810</span><div class="sense-bars"><div class="sense-bar smell"></div><div class="sense-bar noise"></div><div class="sense-bar crowd"></div><div class="sense-bar visual"></div><div class="sense-bar thermal"></div></div></div>
    </div>

    <!-- sense legend -->
    <div class="sense-legend">
      <span><span class="legend-dot" style="background:#f59e0b"></span>Smell</span>
      <span><span class="legend-dot" style="background:#3b82f6"></span>Noise</span>
      <span><span class="legend-dot" style="background:#ef4444"></span>Crowd</span>
      <span><span class="legend-dot" style="background:#10b981"></span>Visual</span>
      <span><span class="legend-dot" style="background:#8b5cf6"></span>Thermal</span>
    </div>

    <!-- content area -->
    <div class="content-area" id="content-area">
      <div class="content-inner" id="content-inner"></div>
    </div>

  </div><!-- .main -->

</div><!-- .page -->

<!-- story bar (shown in story mode) -->
<div class="story-bar" id="story-bar" style="display:none">
  <span class="story-pos" id="story-pos">1 / 1</span>
  <span class="story-hints">
    <kbd>Space</kbd> / <kbd>&#8594;</kbd> next &nbsp;
    <kbd>&#8592;</kbd> prev &nbsp;
    <kbd>Esc</kbd> exit
  </span>
</div>

<script>
// ── embedded data ───────────────────────────────────────────────────────────
const VENUES           = {VENUES_JSON};
const EVIDENCE         = {EVIDENCE_JSON};
const DECADE_SUMMARIES = {DECADE_SUMMARIES_JSON};
const HIGHLIGHTS       = {HIGHLIGHTS_JSON};

// ── state ───────────────────────────────────────────────────────────────────
const state = {{ venueId: VENUES.length ? VENUES[0].id : null, mode: 'explore', storyIndex: 0 }};

// ── label maps ──────────────────────────────────────────────────────────────
const SOURCE_LABELS = {{
  fiction:       'Fiction',
  topography:    'Topography',
  diary:         'Diary',
  letters:       'Letters',
  legal:         'Legal',
  institutional: 'Institutional',
  poetry:        'Poetry',
}};

const MOD_LABELS = {{
  olfactory: 'smell',
  auditory:  'noise',
  crowd:     'crowd',
  visual:    'visual',
  thermal:   'thermal',
}};

const DECADES = [1660,1670,1680,1690,1700,1710,1720,1730,1740,1750,1760,1770,1780,1790,1800,1810];

// ── helpers ─────────────────────────────────────────────────────────────────
function esc(s) {{
  return String(s || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}}

// ── venue selector ──────────────────────────────────────────────────────────
function populateVenueSelector() {{
  var sel = document.querySelector('.venue-selector');
  VENUES.forEach(function(v) {{
    var opt = document.createElement('option');
    opt.value = v.id;
    opt.textContent = v.name;
    if (v.id === state.venueId) opt.selected = true;
    sel.appendChild(opt);
  }});
  sel.addEventListener('change', function() {{
    state.venueId = this.value;
    state.storyIndex = 0;
    render();
  }});
}}

// ── strip rendering ─────────────────────────────────────────────────────────
function renderStrip() {{
  var summaries = state.venueId ? (DECADE_SUMMARIES[state.venueId] || {{}}) : {{}};

  // Compute peak counts per modality across all decades for normalisation
  var modalities = ['olfactory', 'auditory', 'crowd', 'visual', 'thermal'];
  var peaks = {{}};
  modalities.forEach(function(m) {{ peaks[m] = 0; }});
  DECADES.forEach(function(d) {{
    var s = summaries[String(d)];
    if (!s) return;
    var mb = s.modality_breakdown || {{}};
    modalities.forEach(function(m) {{
      if ((mb[m] || 0) > peaks[m]) peaks[m] = mb[m] || 0;
    }});
  }});

  var barClasses = ['smell', 'noise', 'crowd', 'visual', 'thermal'];

  DECADES.forEach(function(d) {{
    var stripCell = document.querySelector('.timeline-strip .timeline-decade[data-decade="' + d + '"]');
    if (!stripCell) return;
    var s = summaries[String(d)];
    var mb = s ? (s.modality_breakdown || {{}}) : {{}};
    var rawMods = ['olfactory', 'auditory', 'crowd', 'visual', 'thermal'];
    var bars = stripCell.querySelectorAll('.sense-bar');
    rawMods.forEach(function(m, i) {{
      var peak = peaks[m] || 1;
      var val  = mb[m] || 0;
      var pct  = Math.round((val / peak) * 36);
      bars[i].style.height = (pct || 0) + 'px';
    }});
    stripCell.classList.toggle('active', String(d) === String(state.activeDecade || ''));
  }});
}}

// ── sidebar rendering ───────────────────────────────────────────────────────
function renderSidebar() {{
  var summaries = state.venueId ? (DECADE_SUMMARIES[state.venueId] || {{}}) : {{}};
  DECADES.forEach(function(d) {{
    var btn = document.querySelector('.decade-nav .decade-btn[data-decade="' + d + '"]');
    if (!btn) return;
    var hasData = !!(summaries[String(d)]);
    btn.classList.toggle('has-data', hasData);
    btn.classList.toggle('active', String(d) === String(state.activeDecade || ''));
  }});
}}

// ── card rendering ──────────────────────────────────────────────────────────
function renderCard(ev, idx, hlSet) {{
  var isFiction   = ev.source_type === 'fiction';
  var cardClass   = isFiction ? 'card card-fiction' : 'card card-nonfiction';
  var isDimmed    = state.mode === 'story' && !hlSet[idx];
  if (isDimmed) cardClass += ' dimmed';

  var sourceLabel = SOURCE_LABELS[ev.source_type] || (ev.source_type || '').toUpperCase();
  var modLabel    = MOD_LABELS[ev.modality] || ev.modality || '';

  // Date string
  var evDecade    = Math.floor((ev.date_min || 0) / 10) * 10;
  var pubDecade   = ev.pub_year ? Math.floor(ev.pub_year / 10) * 10 : evDecade;
  var dateStr     = ev.pub_year ? String(ev.pub_year) : '';
  if (ev.pub_year && pubDecade !== evDecade) {{
    dateStr += ' (describing ' + evDecade + 's)';
  }}

  var text = ev.text || '';
  var snippet = text.length > 500 ? text.slice(0, 500) + '\u2026' : text;

  return '<div class="' + esc(cardClass) + '" data-idx="' + idx + '">' +
    '<div class="card-header">' +
      '<span class="source-label">' + esc(sourceLabel) + '</span>' +
      '<span class="card-author">' + esc(ev.author) + '</span>' +
      '<span class="card-title">' + esc(ev.title) + '</span>' +
      (dateStr ? '<span class="card-date">' + esc(dateStr) + '</span>' : '') +
    '</div>' +
    '<div class="modality-tag">' + esc(modLabel) + '</div>' +
    '<div class="card-text">\u201c' + esc(snippet) + '\u201d</div>' +
    '<div class="card-source">' + esc(ev.title) + (ev.pub_year ? ' (' + ev.pub_year + ')' : '') + '</div>' +
  '</div>';
}}

// ── content rendering ───────────────────────────────────────────────────────
function renderContent() {{
  var inner  = document.getElementById('content-inner');
  if (!inner) return;
  var evidence  = state.venueId ? (EVIDENCE[state.venueId] || []) : [];
  if (!evidence.length) {{
    inner.innerHTML = '<p style="color:#888;padding:20px 0;font-size:13px;">No evidence for this venue.</p>';
    return;
  }}

  // Group by decade
  var byDecade = {{}};
  evidence.forEach(function(ev, idx) {{
    var d = Math.floor((ev.date_min || 0) / 10) * 10;
    if (!byDecade[d]) byDecade[d] = [];
    byDecade[d].push({{ ev: ev, idx: idx }});
  }});

  // Precompute highlight set once for story mode dimming
  var hlSet = {{}};
  if (state.mode === 'story') {{
    getStoryHighlights().forEach(function(i) {{ hlSet[i] = true; }});
  }}

  var html = '';
  var sortedDecades = Object.keys(byDecade).map(Number).sort(function(a,b) {{ return a-b; }});

  sortedDecades.forEach(function(d) {{
    var items = byDecade[d];
    var passageCount = items.length;
    var sourceIds = {{}};
    items.forEach(function(it) {{ sourceIds[it.ev.source_id] = true; }});
    var sourceCount = Object.keys(sourceIds).length;

    html += '<div class="decade-section" data-decade="' + d + '" id="decade-' + d + '">';
    html += '<div class="decade-heading">' +
      esc(d + 's') + ' \u2014 ' + passageCount + ' passage' + (passageCount !== 1 ? 's' : '') +
      ' from ' + sourceCount + ' source' + (sourceCount !== 1 ? 's' : '') +
    '</div>';

    // Comparison row: longest fiction + longest non-fiction side by side
    var fictionItems    = items.filter(function(it) {{ return it.ev.source_type === 'fiction'; }});
    var nonfictionItems = items.filter(function(it) {{ return it.ev.source_type !== 'fiction'; }});

    if (fictionItems.length && nonfictionItems.length) {{
      var bestFic    = fictionItems.reduce(function(a, b) {{ return (b.ev.text||'').length > (a.ev.text||'').length ? b : a; }});
      var bestNonFic = nonfictionItems.reduce(function(a, b) {{ return (b.ev.text||'').length > (a.ev.text||'').length ? b : a; }});
      html += '<div class="compare-row">' +
        renderCard(bestFic.ev, bestFic.idx, hlSet) +
        renderCard(bestNonFic.ev, bestNonFic.idx, hlSet) +
      '</div>';
      // Remaining items (those not used in compare row)
      var compareIdxs = {{}};
      compareIdxs[bestFic.idx] = true;
      compareIdxs[bestNonFic.idx] = true;
      items.forEach(function(it) {{
        if (!compareIdxs[it.idx]) html += renderCard(it.ev, it.idx, hlSet);
      }});
    }} else {{
      items.forEach(function(it) {{ html += renderCard(it.ev, it.idx, hlSet); }});
    }}

    html += '</div>';
  }});

  inner.innerHTML = html;
}}

// ── scroll spy ──────────────────────────────────────────────────────────────
function setupScrollSpy() {{
  var sections = document.querySelectorAll('.decade-section');
  if (!sections.length) return;
  var observer = new IntersectionObserver(function(entries) {{
    entries.forEach(function(entry) {{
      if (entry.isIntersecting) {{
        var d = entry.target.getAttribute('data-decade');
        state.activeDecade = d;
        // Update sidebar
        document.querySelectorAll('.decade-btn').forEach(function(btn) {{
          btn.classList.toggle('active', btn.getAttribute('data-decade') === d);
        }});
        // Update strip
        document.querySelectorAll('.timeline-decade').forEach(function(cell) {{
          cell.classList.toggle('active', cell.getAttribute('data-decade') === d);
        }});
      }}
    }});
  }}, {{ root: document.getElementById('content-area'), threshold: 0.2 }});
  sections.forEach(function(s) {{ observer.observe(s); }});
}}

// ── decade click navigation ─────────────────────────────────────────────────
function setupDecadeClicks() {{
  document.querySelectorAll('.decade-btn, .timeline-decade').forEach(function(el) {{
    el.addEventListener('click', function() {{
      var d = this.getAttribute('data-decade');
      var target = document.getElementById('decade-' + d);
      if (target) target.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
    }});
  }});
}}

// ── story mode ──────────────────────────────────────────────────────────────
function getStoryHighlights() {{
  var highlights = state.venueId ? (HIGHLIGHTS[state.venueId] || {{}}) : {{}};
  var evidence   = state.venueId ? (EVIDENCE[state.venueId] || []) : [];
  var result = [];
  DECADES.forEach(function(d) {{
    var hl = highlights[String(d)];
    if (!hl) return;
    for (var i = 0; i < evidence.length; i++) {{
      if (evidence[i].text === hl.text && evidence[i].source_id === hl.source_id) {{
        result.push(i);
        break;
      }}
    }}
  }});
  return result;
}}

function enterStory() {{
  state.mode       = 'story';
  state.storyIndex = 0;
  document.querySelectorAll('.mode-btn').forEach(function(btn) {{
    btn.classList.toggle('active', btn.getAttribute('data-mode') === 'story');
  }});
  document.getElementById('story-bar').style.display = 'flex';
  render();
  scrollToHighlight();
}}

function exitStory() {{
  state.mode = 'explore';
  document.querySelectorAll('.mode-btn').forEach(function(btn) {{
    btn.classList.toggle('active', btn.getAttribute('data-mode') === 'explore');
  }});
  document.getElementById('story-bar').style.display = 'none';
  render();
}}

function scrollToHighlight() {{
  var hlIndices = getStoryHighlights();
  var idx = hlIndices[state.storyIndex];
  if (idx === undefined) return;
  var card = document.querySelector('.card[data-idx="' + idx + '"]');
  if (card) card.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
  updateStoryBar();
}}

function updateStoryBar() {{
  var hlIndices = getStoryHighlights();
  var pos = document.getElementById('story-pos');
  if (pos) pos.textContent = (state.storyIndex + 1) + ' / ' + hlIndices.length;
}}

function storyAdvance(dir) {{
  var hlIndices = getStoryHighlights();
  var next = state.storyIndex + dir;
  if (next < 0 || next >= hlIndices.length) return;
  state.storyIndex = next;
  render();
  scrollToHighlight();
}}

// ── keyboard ────────────────────────────────────────────────────────────────
document.addEventListener('keydown', function(e) {{
  if (state.mode !== 'story') return;
  if (e.key === 'ArrowRight' || e.key === ' ') {{ e.preventDefault(); storyAdvance(1); }}
  else if (e.key === 'ArrowLeft')               {{ e.preventDefault(); storyAdvance(-1); }}
  else if (e.key === 'Escape')                  {{ exitStory(); }}
}});

// ── mode toggle ─────────────────────────────────────────────────────────────
document.querySelectorAll('.mode-btn').forEach(function(btn) {{
  btn.addEventListener('click', function() {{
    var m = this.getAttribute('data-mode');
    if (m === 'story')   enterStory();
    else                 exitStory();
  }});
}});

// ── render ──────────────────────────────────────────────────────────────────
function render() {{
  renderStrip();
  renderSidebar();
  renderContent();
  setupScrollSpy();
  if (state.mode === 'story') scrollToHighlight();
}}

// ── init ─────────────────────────────────────────────────────────────────────
populateVenueSelector();
setupDecadeClicks();
render();
</script>
</body>
</html>"""


if __name__ == "__main__":
    build()
