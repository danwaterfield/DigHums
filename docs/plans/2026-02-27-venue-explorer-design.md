# Venue Explorer — Design Document
## 2026-02-27

---

## Goal

A self-contained Leaflet HTML file that makes the Phase 1 sensory evidence store (`sensory.db`) explorable. Click any venue on the map, browse the assembled passages filtered by modality, source type, and date range.

---

## Architecture

**Static HTML, embedded data** — same pattern as `narrative_map.html`.

A Python build script (`build_venue_explorer.py`) queries `sensory.db`, joins against `venues.csv`, and bakes all data into `venue_explorer.html` as a JS constant. No server required; file opens directly in a browser.

```
build_venue_explorer.py
  └─ reads venues.csv          (74 venues — full gazetteer)
  └─ reads sensory.db          (565 geocoded passages)
  └─ writes venue_explorer.html
```

Running: `python3 gazetteer/build_venue_explorer.py`

---

## Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ [title + ← narrative map]  [modality pills]  [source pills]     │
│                             ──────── date slider ────────       │  ← top bar
├──────────────────────────────────────┬──────────────────────────┤
│                                      │  VENUE NAME              │
│           LEAFLET MAP                │  89 passages · 4 modal.  │
│                                      │  ─────────────────────── │
│   ●  ●      ●                        │  [auditory] [visual] …   │
│        ●●  ●   ●                     │  ─────────────────────── │
│                                      │  ┌──────────────────────┐│
│                                      │  │ fiction · Burney     ││
│                                      │  │ 1778 · set 1770–1778 ││
│                                      │  │ "The din of the mob…"││
│                                      │  └──────────────────────┘│
└──────────────────────────────────────┴──────────────────────────┘
```

- Map: ~65% width; side panel always visible (no flyout required)
- Side panel: "select a venue to explore" when nothing selected
- Top bar: title | modality pills | source type pills | dual date slider (1660–1820)

---

## Venue Markers

- `L.circleMarker`, radius = `6 + Math.log1p(count) * 3` (log-scaled by evidence count)
- Default colour: muted gold `#c9a84c` with dark border — readable on Rocque tiles
- Hovered: brightens, tooltip shows name + count
- Selected: filled darker, stays highlighted
- Venues with no evidence matching current filters: fade to `#aaa` (not hidden — full gazetteer always visible)
- Tile layers: Rocque 1746, Horwood 1792–99, OpenStreetMap — layer switcher top-right

---

## Side Panel

**Header:** venue name, total matching passage count, modality breakdown (e.g. `4 auditory · 2 olfactory · 1 visual`), updated live as filters change.

**Within-panel modality pills:** further narrow evidence without affecting global map state.

**Evidence cards** (sorted by `date_min` ascending):

```
┌──────────────────────────────────────────┐
│ [fiction]  Frances Burney · Evelina      │
│ 1778 · set 1770–1778                     │
│ ─────────────────────────────────────── │
│ "The din of the carriages was            │
│  insupportable, and the crowd so         │
│  thick we could scarce move…"            │
│                                    [din] │
└──────────────────────────────────────────┘
```

- Source type badge, colour-coded: fiction=blue, diary=green, topography=brown, poetry=purple, letters=grey
- Setting date range shown separately from pub year
- Triggering term shown as chip (bottom-right)

---

## Filters

**Global (top bar):**
- Modality pills: Auditory | Olfactory | Visual | Thermal | Crowd (toggle, multi-select; all active by default)
- Source type pills: Fiction | Diary | Topography | Poetry | Letters (toggle, multi-select; all active by default)
- Date slider: dual-handle range (1660–1820), two overlapping `<input type="range">` styled in CSS — no extra library

Global filters affect map marker colour (active vs grey). Clicking any marker opens its panel regardless of filter state.

**Local (within panel):**
- Same modality pills repeated inside the panel for quick within-venue narrowing

---

## Data Structure

```js
const VENUES = [
  {
    id: "LON001",
    name: "Vauxhall Spring Gardens",
    lat: 51.4863,
    lon: -0.1228,
    evidence: [
      {
        source_type: "fiction",
        author: "burney",
        title: "Evelina",
        pub_year: 1778,
        date_min: 1770,
        date_max: 1778,
        modality: "auditory",
        text: "The din of the carriages…",
        context: "din"
      },
      …
    ]
  },
  …
];
```

Venues with no evidence have `evidence: []`.

---

## Files

| File | Purpose |
|------|---------|
| `gazetteer/build_venue_explorer.py` | Build script: queries DB, writes HTML |
| `gazetteer/venue_explorer.html` | Generated output — self-contained |
| `gazetteer/tests/test_build_venue_explorer.py` | Tests: all venues present, counts match DB, HTML renders |

---

## Out of Scope (Phase 5+)

- Fiction vs. non-fiction two-column compare view
- Divergence highlighting
- Environmental layer toggles (CET, Bills of Mortality, smoke)
- OSRM street routing integration
