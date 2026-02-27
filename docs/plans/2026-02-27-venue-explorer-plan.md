# Venue Explorer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `venue_explorer.html` — a self-contained Leaflet map that makes the Phase 1 `sensory.db` data explorable via venue markers, a side evidence panel, and modality/source/date filters.

**Architecture:** Python build script (`build_venue_explorer.py`) queries `sensory.db` and `venues.csv`, serialises all 74 venues + their evidence arrays as an embedded JS constant, and writes a single HTML file. All filtering happens client-side. Same pattern as `build_narrative_map.py`.

**Tech Stack:** Python 3, SQLite, Leaflet 1.9.4 (CDN), vanilla JS, no extra libraries.

---

## Task 1: Build script — data loading

**Files:**
- Create: `gazetteer/build_venue_explorer.py`
- Create: `gazetteer/tests/test_build_venue_explorer.py`

### Step 1: Write the failing tests

```python
# gazetteer/tests/test_build_venue_explorer.py
import csv
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from build_venue_explorer import load_data

VENUES_PATH = Path(__file__).parent.parent / "venues.csv"
DB_PATH     = Path(__file__).parent.parent / "sensory.db"


def test_all_venues_present():
    """All 74 venues from venues.csv appear in the output."""
    with open(VENUES_PATH, newline="") as f:
        expected_ids = {row["id"] for row in csv.DictReader(f)}
    venues = load_data(VENUES_PATH, DB_PATH)
    actual_ids = {v["id"] for v in venues}
    assert actual_ids == expected_ids


def test_evidence_counts_match_db():
    """Total evidence count matches sensory_evidence WHERE venue_id IS NOT NULL."""
    conn = sqlite3.connect(DB_PATH)
    db_count = conn.execute(
        "SELECT COUNT(*) FROM sensory_evidence WHERE venue_id IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    venues = load_data(VENUES_PATH, DB_PATH)
    total = sum(len(v["evidence"]) for v in venues)
    assert total == db_count


def test_evidence_sorted_by_date_min():
    """Evidence within each venue is sorted by date_min ascending."""
    venues = load_data(VENUES_PATH, DB_PATH)
    for v in venues:
        dates = [e["date_min"] for e in v["evidence"] if e["date_min"]]
        assert dates == sorted(dates), f"Unsorted evidence at {v['id']}"


def test_venues_have_required_fields():
    """Each venue dict has id, name, lat, lon, evidence list."""
    venues = load_data(VENUES_PATH, DB_PATH)
    for v in venues:
        for field in ("id", "name", "lat", "lon", "evidence"):
            assert field in v, f"Missing field '{field}' in venue {v.get('id')}"
        assert isinstance(v["evidence"], list)
```

### Step 2: Run tests to confirm they fail

```bash
/opt/homebrew/bin/python3.10 -m pytest gazetteer/tests/test_build_venue_explorer.py -v
```
Expected: `ERROR` (ImportError — `build_venue_explorer` doesn't exist yet)

### Step 3: Implement `load_data()`

Create `gazetteer/build_venue_explorer.py`:

```python
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

ROCQUE_URL   = "https://www.dhi.ac.uk/san/llptiles/molarocque/{z}/{x}/{y}.png"
ROCQUE_ATTR  = "Map tiles © Museum of London Archaeology / DHI, based on John Rocque 1746"
HORWOOD_URL  = "https://www.romanticlondon.org/horwoodplan/{z}/{x}/{y}.png"
HORWOOD_ATTR = "Map tiles © Romantic London project, based on Richard Horwood 1792–99"


def fmt_author(s: str) -> str:
    """'FrancesBurney' → 'Frances Burney'; 'burney' → 'Burney'."""
    overrides = {"MGLewis": "M. G. Lewis"}
    if s in overrides:
        return overrides[s]
    # CamelCase: insert spaces before capitals
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    # lowercase single-word surname: just capitalise
    return spaced.capitalize() if " " not in spaced else spaced


def load_data(venues_path: Path, db_path: Path) -> list[dict]:
    """
    Return list of venue dicts, each with an 'evidence' array.
    All 74 venues are included; venues with no evidence have evidence=[].
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
    print(f"Venue explorer → {out_path}")
    print(f"  {len(venues)} venues  {geocoded} with evidence  {total_ev} passages")


def _render(data_js: str) -> str:
    """Return the complete HTML string."""
    return HTML_TEMPLATE.replace("__VENUES_DATA__", data_js)


# HTML_TEMPLATE is defined in Task 2 below.
HTML_TEMPLATE = "<html><body>TODO</body></html>"


if __name__ == "__main__":
    build()
```

### Step 4: Run tests — all 4 should pass

```bash
/opt/homebrew/bin/python3.10 -m pytest gazetteer/tests/test_build_venue_explorer.py -v
```
Expected: `4 passed`

### Step 5: Commit

```bash
git add gazetteer/build_venue_explorer.py gazetteer/tests/test_build_venue_explorer.py
git commit -m "feat: add venue explorer build script (data loading)"
```

---

## Task 2: Complete HTML template

**Files:**
- Modify: `gazetteer/build_venue_explorer.py` — replace `HTML_TEMPLATE` stub with full template

### Step 1: Write the failing test

Add to `gazetteer/tests/test_build_venue_explorer.py`:

```python
import subprocess
import tempfile


def test_build_generates_valid_html():
    """Running build() produces an HTML file containing all expected landmarks."""
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        out = Path(f.name)
    from build_venue_explorer import build
    build(VENUES_PATH, DB_PATH, out)
    html = out.read_text(encoding="utf-8")
    assert "const VENUES" in html, "Missing VENUES constant"
    assert "Vauxhall" in html,     "Missing Vauxhall venue"
    assert "Ranelagh" in html,     "Missing Ranelagh venue"
    assert "leaflet" in html.lower(), "Missing Leaflet"
    assert len(html) > 20_000,    "HTML suspiciously short"
    out.unlink()
```

### Step 2: Run to confirm it fails

```bash
/opt/homebrew/bin/python3.10 -m pytest gazetteer/tests/test_build_venue_explorer.py::test_build_generates_valid_html -v
```
Expected: FAIL (`"Missing VENUES constant"` — current template is just a stub)

### Step 3: Replace `HTML_TEMPLATE` in `build_venue_explorer.py`

Replace the last three lines of `build_venue_explorer.py` (the `HTML_TEMPLATE` stub and the `if __name__` block) with the full template below, then re-add the `if __name__` block at the end.

```python
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sensory Map · 18c London & Bath</title>
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
  <a href="narrative_map.html" class="nav-link">← Narrative Map</a>
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
    <span>–</span>
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
    { attribution: '© OpenStreetMap contributors © CARTO', maxZoom: 19 }
  ).addTo(map),
  'Rocque 1746': L.tileLayer(
    'https://www.dhi.ac.uk/san/llptiles/molarocque/{z}/{x}/{y}.png',
    { attribution: 'Map tiles © Museum of London Archaeology / DHI, based on John Rocque 1746',
      minZoom: 13, maxZoom: 15, maxNativeZoom: 15, opacity: 0.9 }
  ),
  'Horwood 1792–99': L.tileLayer(
    'https://www.romanticlondon.org/horwoodplan/{z}/{x}/{y}.png',
    { attribution: 'Map tiles © Romantic London project, based on Richard Horwood 1792–99',
      minZoom: 11, maxZoom: 17, maxNativeZoom: 16, opacity: 0.9 }
  ),
};
L.control.layers(baseLayers, {}, { collapsed: false }).addTo(map);

// ── state ─────────────────────────────────────────────────────────────────
const state = {
  modalities:     new Set(['auditory','olfactory','visual','thermal','crowd']),
  sources:        new Set(['fiction','diary','topography','poetry','letters']),
  dateFrom:       1660,
  dateTo:         1820,
  selectedId:     null,
  panelModalities: new Set(['auditory','olfactory','visual','thermal','crowd']),
};

// ── marker layer ──────────────────────────────────────────────────────────
const markers = {};

function matchesGlobal(ev) {
  return state.modalities.has(ev.modality)
      && state.sources.has(ev.source_type)
      && (ev.date_max === null || ev.date_max >= state.dateFrom)
      && (ev.date_min === null || ev.date_min <= state.dateTo);
}

function activeCount(venue) {
  return venue.evidence.filter(matchesGlobal).length;
}

function markerRadius(total) {
  return 6 + Math.log1p(total) * 3;
}

function renderMarkers() {
  VENUES.forEach(v => {
    const count   = activeCount(v);
    const total   = v.evidence.length;
    const active  = count > 0;
    const sel     = state.selectedId === v.id;
    const radius  = markerRadius(total || 1);

    const opts = {
      radius,
      color:       sel    ? '#2c3e50' : (active ? '#8b6914' : '#999'),
      fillColor:   active ? '#c9a84c' : '#ccc',
      fillOpacity: sel    ? 0.95      : (active ? 0.70 : 0.35),
      weight:      sel    ? 2.5       : 1.5,
    };

    if (markers[v.id]) {
      markers[v.id].setStyle(opts);
      markers[v.id].setRadius(radius);
    } else {
      const m = L.circleMarker([v.lat, v.lon], opts).addTo(map);
      m.bindTooltip(v.name + (total ? ` (${total})` : ''), { sticky: true });
      m.on('click', () => openPanel(v.id));
      markers[v.id] = m;
    }
  });
}

// ── side panel ────────────────────────────────────────────────────────────
const SOURCE_CLASSES = {
  fiction: 'src-fiction', diary: 'src-diary',
  topography: 'src-topography', poetry: 'src-poetry', letters: 'src-letters',
};

function renderCard(ev) {
  const cls   = SOURCE_CLASSES[ev.source_type] || 'src-letters';
  const text  = ev.text.length > 380 ? ev.text.slice(0, 380) + '…' : ev.text;
  const dateStr = ev.pub_year
    ? (ev.date_min && ev.date_min !== ev.pub_year
        ? `${ev.pub_year} · set ${ev.date_min}–${ev.date_max}`
        : String(ev.pub_year))
    : (ev.date_min ? `c. ${ev.date_min}–${ev.date_max}` : '');
  return `<div class="ev-card">
    <div class="ev-card-head">
      <span class="source-badge ${cls}">${ev.source_type}</span>
      <span class="ev-author">${ev.author}</span>
      <span class="ev-title">${ev.title}</span>
    </div>
    <div class="ev-date">${dateStr}</div>
    <div class="ev-text">&ldquo;${text}&rdquo;</div>
    <div class="ev-footer"><span class="context-chip">${ev.context}</span></div>
  </div>`;
}

function renderPanel() {
  const v = VENUES.find(x => x.id === state.selectedId);
  if (!v) return;

  // apply global + panel modality filters
  const filtered = v.evidence.filter(ev =>
    matchesGlobal(ev) && state.panelModalities.has(ev.modality)
  );

  // modality breakdown for header
  const breakdown = {};
  v.evidence.filter(matchesGlobal).forEach(ev => {
    breakdown[ev.modality] = (breakdown[ev.modality] || 0) + 1;
  });
  const bStr = Object.entries(breakdown)
    .sort((a,b) => b[1]-a[1])
    .map(([m,n]) => `${n} ${m}`).join(' · ') || 'no matches';

  document.getElementById('panel-title').textContent = v.name;
  document.getElementById('panel-stats').textContent =
    `${filtered.length} passage${filtered.length !== 1 ? 's' : ''} · ${bStr}`;

  // local modality pills
  const pillsEl = document.getElementById('panel-pills');
  pillsEl.innerHTML = ['auditory','olfactory','visual','thermal','crowd'].map(m =>
    `<button class="pill ${state.panelModalities.has(m) ? 'active' : ''}"
       data-panel-mod="${m}">${m.charAt(0).toUpperCase()+m.slice(1)}</button>`
  ).join('');

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
  state.selectedId = venueId;
  state.panelModalities = new Set(['auditory','olfactory','visual','thermal','crowd']);
  renderPanel();
  renderMarkers();
}

// ── filter wiring ─────────────────────────────────────────────────────────
// Global pill buttons (modality + source)
document.querySelectorAll('.pill[data-f]').forEach(btn => {
  btn.addEventListener('click', () => {
    const f   = btn.dataset.f;
    const v   = btn.dataset.v;
    const set = f === 'modality' ? state.modalities : state.sources;
    if (set.has(v)) { set.delete(v); btn.classList.remove('active'); }
    else            { set.add(v);    btn.classList.add('active');    }
    renderMarkers();
    if (state.selectedId) renderPanel();
  });
});

// Local (panel) modality pills — delegated
document.getElementById('panel-pills').addEventListener('click', e => {
  const btn = e.target.closest('[data-panel-mod]');
  if (!btn) return;
  const m = btn.dataset.panelMod;
  if (state.panelModalities.has(m)) state.panelModalities.delete(m);
  else                               state.panelModalities.add(m);
  renderPanel();
});

// Date range
document.getElementById('date-from').addEventListener('input', function() {
  const val = parseInt(this.value);
  const toSlider = document.getElementById('date-to');
  if (val > parseInt(toSlider.value)) { toSlider.value = val; state.dateTo = val; }
  state.dateFrom = val;
  document.getElementById('lbl-from').textContent = val;
  renderMarkers();
  if (state.selectedId) renderPanel();
});
document.getElementById('date-to').addEventListener('input', function() {
  const val = parseInt(this.value);
  const fromSlider = document.getElementById('date-from');
  if (val < parseInt(fromSlider.value)) { fromSlider.value = val; state.dateFrom = val; }
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
```

### Step 4: Run the new test

```bash
/opt/homebrew/bin/python3.10 -m pytest gazetteer/tests/test_build_venue_explorer.py -v
```
Expected: `5 passed`

### Step 5: Build the HTML and open it

```bash
python3 gazetteer/build_venue_explorer.py
open gazetteer/venue_explorer.html
```

Verify visually:
- 74 circle markers visible on map (London cluster + Bath cluster)
- Hovering shows venue name tooltip
- Clicking Vauxhall Spring Gardens opens side panel with evidence cards
- Evidence cards show source type badge (coloured), author, title, date, passage, context chip
- Modality pills in top bar toggle markers grey when deactivated
- Date sliders update "From"/"To" labels and affect active marker count

### Step 6: Commit

```bash
git add gazetteer/build_venue_explorer.py gazetteer/venue_explorer.html \
        gazetteer/tests/test_build_venue_explorer.py
git commit -m "feat: add venue explorer — Leaflet map of sensory evidence"
```

---

## Task 3: Smoke test + README update

**Files:**
- Modify: `README.md`

### Step 1: Run the full test suite

```bash
/opt/homebrew/bin/python3.10 -m pytest gazetteer/tests/ -v
```
Expected: all 19 tests pass (14 existing + 5 new)

### Step 2: Update README

In `README.md`, find the "Urban Gazetteer & Interactive Maps" section and add the venue explorer after the narrative map entry:

```markdown
### Venue Explorer (`gazetteer/venue_explorer.html`)

Interactive map of the Phase 1 sensory evidence store. Click any venue to browse
assembled passages filtered by modality (auditory, olfactory, visual, thermal, crowd),
source type (fiction, diary, topography, poetry, letters), and date range.

Built from `sensory.db` (8,099 deduplicated passages across 37 sources, 565 geocoded).

To regenerate after updating `sensory.db`:
```bash
python3 gazetteer/build_venue_explorer.py
```
```

### Step 3: Commit

```bash
git add README.md
git commit -m "docs: add venue explorer to README"
```

### Step 4: Push

```bash
git push
```
