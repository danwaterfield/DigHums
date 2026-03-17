#!/usr/bin/env python3
"""
Build the self-contained venue comparison HTML.

Reads venues.csv and sensory.db, writes comparison.html.

For each venue the page shows two columns:
  Left  — Fiction passages (novels)
  Right — Non-fiction passages (diary, topography, letters, legal,
           poetry, newspaper, parish, institutional)

Each column displays:
  - Modality breakdown bar
  - Valence distribution (pleasant / neutral / unpleasant)
  - Scrollable passage list

A shared venue selector drives both columns.

Usage:
    python3 gazetteer/build_comparison.py
    open gazetteer/comparison.html
"""

import csv
import json
import re
import sqlite3
from pathlib import Path

VENUES_PATH = Path(__file__).parent / "venues.csv"
DB_PATH     = Path(__file__).parent / "sensory.db"
OUT_PATH    = Path(__file__).parent / "comparison.html"

FICTION_TYPES    = {"fiction"}
NONFICTION_TYPES = {
    "diary", "topography", "letters", "legal", "poetry",
    "newspaper", "parish", "institutional",
}


def fmt_author(s: str) -> str:
    overrides = {"MGLewis": "M. G. Lewis"}
    if s in overrides:
        return overrides[s]
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    return spaced.capitalize() if " " not in spaced else spaced


def load_data(venues_path: Path, db_path: Path) -> list[dict]:
    """
    Return list of venue dicts with 'fiction' and 'nonfiction' evidence arrays.
    Venues with no evidence of either type are included (with empty arrays).
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
                "fiction":      [],
                "nonfiction":   [],
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
            if vid not in venues:
                continue
            ev = {
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
            }
            st = row["source_type"]
            if st in FICTION_TYPES:
                venues[vid]["fiction"].append(ev)
            elif st in NONFICTION_TYPES:
                venues[vid]["nonfiction"].append(ev)
    finally:
        conn.close()
    return list(venues.values())


def build(
    venues_path: Path = VENUES_PATH,
    db_path: Path     = DB_PATH,
    out_path: Path    = OUT_PATH,
) -> None:
    venues  = load_data(venues_path, db_path)
    data_js = json.dumps(venues, ensure_ascii=False, separators=(",", ":"))
    html    = _render(data_js)
    out_path.write_text(html, encoding="utf-8")
    with_data = sum(
        1 for v in venues if v["fiction"] or v["nonfiction"]
    )
    print(f"Comparison view -> {out_path}")
    print(f"  {len(venues)} venues  {with_data} with evidence")


def _render(data_js: str) -> str:
    return HTML_TEMPLATE.replace("__VENUES_DATA__", data_js, 1)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Fiction vs Non-Fiction Comparison · 18c London &amp; Bath</title>
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
  z-index: 100;
}
#topbar .title { font-size: 15px; font-weight: bold; color: #2c3e50; white-space: nowrap; }
.view-tab { background: #f0ede6; border: 1px solid #ccc; color: #555; padding: 2px 10px; border-radius: 3px; font-size: 11px; font-weight: 500; text-decoration: none; display: inline-block; white-space: nowrap; }
.view-tab:hover { background: #e4e0d8; color: #2c3e50; }
.view-tab.active { background: #2c3e50; border-color: #2c3e50; color: #fff; font-weight: 600; cursor: default; }
#venue-select {
  font-family: Georgia, serif; font-size: 13px;
  padding: 4px 8px; border: 1px solid #bbb; border-radius: 4px;
  background: white; cursor: pointer; min-width: 260px; max-width: 380px;
}
#topbar .subtitle { font-size: 11px; color: #888; margin-left: auto; }

/* ── two-column layout ── */
#columns {
  flex: 1; display: flex; overflow: hidden;
}
.col {
  flex: 1; display: flex; flex-direction: column;
  overflow: hidden; border-right: 1px solid #ddd;
  background: #faf9f7;
}
.col:last-child { border-right: none; }

/* ── column header ── */
.col-header {
  flex-shrink: 0;
  padding: 10px 14px 8px;
  border-bottom: 1px solid #e0dbd4;
}
.col-label {
  font-size: 13px; font-weight: bold; color: #2c3e50; margin-bottom: 4px;
  display: flex; align-items: center; gap: 8px;
}
.col-label .badge {
  font-size: 11px; padding: 1px 7px; border-radius: 10px;
  font-weight: normal;
}
.fiction-badge    { background: #dbe8f5; color: #1a5276; }
.nonfiction-badge { background: #d5f0e0; color: #1a7a42; }

/* ── modality bar ── */
.modality-bar {
  display: flex; height: 10px; border-radius: 3px;
  overflow: hidden; margin-bottom: 4px; background: #e8e4de;
}
.mod-seg { transition: width 0.3s; }
.mod-auditory  { background: #5b9bd5; }
.mod-olfactory { background: #c97c3b; }
.mod-visual    { background: #6aab6e; }
.mod-thermal   { background: #d45f5f; }
.mod-crowd     { background: #9b7dcc; }

/* ── valence bar ── */
.valence-row {
  display: flex; gap: 4px; align-items: center; font-size: 10px; color: #888;
  margin-bottom: 3px;
}
.val-dot {
  width: 8px; height: 8px; border-radius: 50%; display: inline-block;
}
.val-pleasant   { background: #9a6f2a; }
.val-neutral    { background: #bbb; }
.val-unpleasant { background: #c0392b; }
.col-stats { font-size: 11px; color: #888; }

/* ── evidence list ── */
.ev-list {
  flex: 1; overflow-y: auto; padding: 10px 12px;
}
.ev-card {
  background: white; border: 1px solid #e0dbd4; border-radius: 5px;
  padding: 9px 11px; margin-bottom: 7px;
  font-size: 12px; line-height: 1.5;
}
.ev-card.diverges { border-left: 3px solid #e67e22; }
.ev-card-head {
  display: flex; align-items: baseline; gap: 6px; margin-bottom: 2px;
  flex-wrap: wrap;
}
.source-badge {
  border-radius: 3px; padding: 1px 5px; font-size: 10px;
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
.ev-author { font-weight: bold; color: #2c3e50; }
.ev-title  { color: #555; font-style: italic; }
.ev-date   { font-size: 10px; color: #999; margin-bottom: 4px; }
.ev-text   { color: #333; font-style: italic; line-height: 1.55; }
.ev-footer { display: flex; justify-content: flex-end; margin-top: 4px; }
.context-chip {
  background: #f0ede8; color: #666; border-radius: 3px;
  padding: 1px 5px; font-size: 10px; font-style: normal;
}
.valence-pip {
  width: 8px; height: 8px; border-radius: 50%;
  display: inline-block; flex-shrink: 0; margin-left: 2px; align-self: center;
}
.valence-pip.unpleasant { background: #c0392b; opacity: 0.55; }
.valence-pip.pleasant   { background: #9a6f2a; opacity: 0.55; }
.diverge-badge {
  font-size: 9px; font-weight: bold; text-transform: uppercase;
  letter-spacing: 0.04em; padding: 1px 4px; border-radius: 2px;
  background: #fdebd0; color: #e67e22; border: 1px solid #e67e22;
  flex-shrink: 0; align-self: center;
}
.no-evidence {
  text-align: center; color: #aaa; font-style: italic;
  font-size: 12px; padding: 24px 0;
}
</style>
</head>
<body>

<div id="topbar">
  <span class="title">Fiction vs Non-Fiction · 18c Sensory Evidence</span>
  <span style="display:flex;gap:4px;align-items:center;">
    <span style="font-size:11px;color:#888;font-weight:500;margin-right:2px;">View</span>
    <a href="sensory_time_map.html" class="view-tab">Sensory Map</a>
    <a href="venue_explorer.html" class="view-tab">Evidence</a>
    <a href="narrative_map.html" class="view-tab">Narrative</a>
    <span class="view-tab active">Comparison</span>
    <a href="sensory_timeline.html" class="view-tab">Timeline</a>
  </span>
  <select id="venue-select"></select>
  <span class="subtitle" id="top-stats"></span>
  <span id="top-meta" style="display:flex;gap:4px;flex-wrap:wrap;align-items:center"></span>
</div>

<div id="columns">
  <div class="col" id="col-fiction">
    <div class="col-header">
      <div class="col-label">
        Fiction
        <span class="badge fiction-badge" id="fic-count">0 passages</span>
      </div>
      <div class="modality-bar" id="fic-mod-bar"></div>
      <div class="valence-row" id="fic-val-row"></div>
    </div>
    <div class="ev-list" id="fic-list"></div>
  </div>
  <div class="col" id="col-nonfiction">
    <div class="col-header">
      <div class="col-label">
        Non-Fiction
        <span class="badge nonfiction-badge" id="nf-count">0 passages</span>
      </div>
      <div class="modality-bar" id="nf-mod-bar"></div>
      <div class="valence-row" id="nf-val-row"></div>
    </div>
    <div class="ev-list" id="nf-list"></div>
  </div>
</div>

<script>
const VENUES = __VENUES_DATA__;

const BT_COLORS = {
  'garden':    '#4a7c4f', 'park':      '#4a7c4f',
  'theatre':   '#8b5e3c', 'assembly':  '#8b5e3c',
  'church':    '#6b5b8a',
  'street':    '#7a7a7a', 'square':    '#7a7a7a', 'district': '#7a7a7a',
  'market':    '#b8860b',
  'prison':    '#8b0000', 'court':     '#8b0000', 'execution': '#8b0000',
};

function renderMetaChips(v) {
  const el = document.getElementById('top-meta');
  if (!el) return;
  const parts = [v.enclosure, v.building_type, v.material, v.capacity].filter(Boolean);
  const color = BT_COLORS[v.building_type] || '#666';
  el.innerHTML = parts.map(function(p) {
    return '<span style="font-size:10px;padding:2px 7px;border-radius:10px;border:1px solid '
        + color + ';color:' + color + ';background:rgba(0,0,0,0.04)">' + p + '</span>';
  }).join('');
}

const SOURCE_CLASSES = {
  fiction: 'src-fiction', diary: 'src-diary',
  topography: 'src-topography', poetry: 'src-poetry',
  letters: 'src-letters', legal: 'src-legal',
  newspaper: 'src-newspaper', parish: 'src-parish',
  institutional: 'src-institutional',
};
const MOD_COLORS = {
  auditory: '#5b9bd5', olfactory: '#c97c3b', visual: '#6aab6e',
  thermal: '#d45f5f', crowd: '#9b7dcc',
};

function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── populate venue selector ────────────────────────────────────────────────
const sel = document.getElementById('venue-select');
VENUES.forEach(function(v, i) {
  const total = v.fiction.length + v.nonfiction.length;
  if (total === 0) return;
  const opt = document.createElement('option');
  opt.value = i;
  opt.textContent = v.name + ' (' + total + ')';
  sel.appendChild(opt);
});

// ── helpers ────────────────────────────────────────────────────────────────
function modBreakdown(evs) {
  const counts = {};
  evs.forEach(function(e) { counts[e.modality] = (counts[e.modality] || 0) + 1; });
  return counts;
}

function valBreakdown(evs) {
  const c = { pleasant: 0, neutral: 0, unpleasant: 0 };
  evs.forEach(function(e) {
    const v = e.valence || 'neutral';
    if (c[v] !== undefined) c[v]++;
    else c.neutral++;
  });
  return c;
}

function renderModBar(containerId, evs) {
  const el = document.getElementById(containerId);
  const counts = modBreakdown(evs);
  const total  = evs.length || 1;
  el.innerHTML = Object.entries(counts)
    .sort(function(a, b) { return b[1] - a[1]; })
    .map(function(e) {
      const pct = (e[1] / total * 100).toFixed(1);
      const col = MOD_COLORS[e[0]] || '#aaa';
      return '<div class="mod-seg mod-' + e[0] + '" style="width:' + pct + '%;background:' + col + '"'
           + ' title="' + e[0] + ': ' + e[1] + '"></div>';
    }).join('');
}

function renderValRow(containerId, evs) {
  const el = document.getElementById(containerId);
  const c  = valBreakdown(evs);
  const total = evs.length || 1;
  el.innerHTML =
    '<span class="val-dot val-pleasant"></span>'
    + '<span>' + c.pleasant + ' pleasant (' + Math.round(c.pleasant/total*100) + '%) &nbsp;</span>'
    + '<span class="val-dot val-neutral"></span>'
    + '<span>' + c.neutral  + ' neutral (' + Math.round(c.neutral/total*100)  + '%) &nbsp;</span>'
    + '<span class="val-dot val-unpleasant"></span>'
    + '<span>' + c.unpleasant + ' unpleasant (' + Math.round(c.unpleasant/total*100) + '%)</span>';
}

function renderCard(ev) {
  const cls     = SOURCE_CLASSES[ev.source_type] || 'src-letters';
  const text    = ev.text.length > 360 ? ev.text.slice(0, 360) + '\u2026' : ev.text;
  const dateStr = ev.pub_year
    ? (ev.date_min && ev.date_min !== ev.pub_year
        ? ev.pub_year + ' \u00b7 set ' + ev.date_min
          + (ev.date_max != null ? '\u2013' + ev.date_max : '')
        : String(ev.pub_year))
    : (ev.date_min
        ? 'c.\u00a0' + ev.date_min + (ev.date_max != null ? '\u2013' + ev.date_max : '')
        : '');
  const divCls  = ev.divergence === 'diverges' ? ' diverges' : '';
  return '<div class="ev-card' + divCls + '">'
    + '<div class="ev-card-head">'
    + '<span class="source-badge ' + cls + '">' + ev.source_type + '</span>'
    + '<span class="ev-author">' + esc(ev.author) + '</span>'
    + '<span class="ev-title">' + esc(ev.title) + '</span>'
    + (ev.valence === 'unpleasant' ? '<span class="valence-pip unpleasant" title="unpleasant"></span>'
       : ev.valence === 'pleasant'  ? '<span class="valence-pip pleasant" title="pleasant"></span>'
       : '')
    + (ev.divergence === 'diverges'
        ? '<span class="diverge-badge" title="Diverges from expected sensory profile">diverges</span>'
        : '')
    + '</div>'
    + '<div class="ev-date">' + dateStr + '</div>'
    + '<div class="ev-text">\u201c' + esc(text) + '\u201d</div>'
    + '<div class="ev-footer"><span class="context-chip">' + esc(ev.context) + '</span></div>'
    + '</div>';
}

function renderList(listId, evs) {
  const el = document.getElementById(listId);
  el.innerHTML = evs.length
    ? evs.map(renderCard).join('')
    : '<div class="no-evidence">No evidence for this venue.</div>';
}

// ── main render ────────────────────────────────────────────────────────────
function render() {
  const idx = parseInt(sel.value);
  if (isNaN(idx)) return;
  const v = VENUES[idx];

  const fic = v.fiction;
  const nf  = v.nonfiction;

  document.getElementById('fic-count').textContent =
    fic.length + ' passage' + (fic.length !== 1 ? 's' : '');
  document.getElementById('nf-count').textContent =
    nf.length  + ' passage' + (nf.length  !== 1 ? 's' : '');

  document.getElementById('top-stats').textContent =
    v.name + ' \u00b7 ' + (fic.length + nf.length) + ' total passages';
  renderMetaChips(v);

  renderModBar('fic-mod-bar', fic);
  renderModBar('nf-mod-bar',  nf);
  renderValRow('fic-val-row', fic);
  renderValRow('nf-val-row',  nf);
  renderList('fic-list', fic);
  renderList('nf-list',  nf);
}

sel.addEventListener('change', render);
if (sel.options.length > 0) render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    build()
