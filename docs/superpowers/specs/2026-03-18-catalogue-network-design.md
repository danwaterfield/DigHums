# Catalogue-Powered Correspondent Network (Phase 1)

## Summary

Replace the current 243-entry OUP-only correspondent network with the full
Hemlow catalogue data: 6,057 Frances entries + 991 CB Sr entries + 908 CB Jr
entries. Three switchable network views (Frances / Charles Sr / Charles Jr)
in a single HTML file with shared UI infrastructure. Professional aesthetic
matching the existing `correspondent_network.html`.

## Data Sources

| Source | File | Entries | Period |
|--------|------|---------|--------|
| Frances Burney d'Arblay | `catalogue_frances_darblay.csv` | 6,057 | 1768–1839 |
| Charles Burney Mus Doc | `catalogue_cb_sr.csv` | 991 | 1749–1814 |
| Charles Burney DD | `catalogue_cb_jr.csv` | 908 | 1767–1817 |

Each CSV has columns: `date, direction, correspondent, repository, first_line`.

The existing OUP-parsed data (243 selections, 28 correspondents) is retained
as a high-confidence subset within the Frances view.

## Data Pipeline

### 1. Name normalisation (`gazetteer/burney_names.py`)

A shared module containing:

- `CANONICAL_NAMES`: dict mapping every variant to a canonical form.
  Seeded from the existing `NAME_ALIASES` + the catalogue abbreviation key
  (SBP → Susanna Burney Phillips, HLTP → Hester Thrale Piozzi, etc.)
- `COMMUNITIES`: dict mapping canonical names to emotional communities
  (Family, Literary, Court, Publishers, Intimate circle, French circle,
  Musical circle, Scholarly/Church, Royal)
- `normalise(name)` function
- `community(name)` function

Parsing artefacts in the CSV ([?], NYPL(B), "to", BM(Bar)) are filtered
out during loading, not in the names module.

### 2. Date parsing

Extract year from the catalogue date field. Handle:
- `[1784]`, `1 Jan 1784`, `[post 1782]`, `[c 1790]`, `[1786-91]`
- For ranges like `[1786-91]`, take the midpoint year
- For `[?]` or unparseable dates, exclude from timeline but include in
  network totals

### 3. Build script (`gazetteer/build_correspondent_network.py`)

Extended to:
1. Load all three CSVs
2. Normalise names via `burney_names.py`
3. Filter parsing artefacts
4. Compute per-correspondent: letter count, direction (to/from counts),
   date range, letters per year
5. Build three separate network data structures (one per family member)
6. Embed all three as JSON in the HTML

Output structure per network:
```json
{
  "subject": "Frances Burney d'Arblay",
  "nodes": [...],
  "edges": [...],
  "letters": [...],
  "phases": [...]
}
```

Each node includes:
- `id`, `community`, `count`, `to_count`, `from_count`
- `year_min`, `year_max`
- `repositories`: list of repository codes where letters are held

Each edge includes:
- `source`, `target`, `weight`, `to_weight`, `from_weight`

### 4. Phase definitions

Frances phases: same as existing (Apprentice Years through Widowhood).

Charles Sr phases:
- 1749–1760: Lynn & Early Career
- 1760–1770: London Establishment
- 1770–1776: Continental Tours & History of Music
- 1776–1789: Streatham & Literary Fame
- 1789–1800: Late Career
- 1800–1814: Final Years

Charles Jr phases:
- 1767–1786: Early Life & Cambridge
- 1786–1800: Schoolmaster & Greek Scholar
- 1800–1817: Late Career & DD

## Visualisation

### Network switcher

Three tabs in the header: **Frances** | **Charles Sr** | **Charles Jr**.
Clicking switches the entire graph — different nodes, edges, phases,
timeline range. Smooth transition: shared correspondents (people in
multiple networks) hold position, others fade in/out.

A small indicator shows entry count and correspondent count for each tab.

### Force graph

Same as current design but scaled for larger networks:
- Frances: ~500 correspondents (top 100 shown by default, expand on demand)
- Charles Sr: ~130 correspondents
- Charles Jr: ~100 correspondents

For Frances, a **threshold slider** controls how many correspondents are
visible (minimum letter count to show). Default: 5 letters. Drag to 1 to
see everyone, or up to 20+ to focus on the core network.

Node sizing, colouring, drag, zoom, tooltip, detail panel all work as
in v1.

### Directional edges

Edges now show direction via arrow markers:
- Arrow thickness indicates volume in that direction
- Hover on an edge shows "42 to, 38 from" in a tooltip
- Significant asymmetry (>3:1 ratio) gets a subtle colour warning

### Timeline slider

Same dual-handle design. Range adapts to the active network:
- Frances: 1768–1839
- Charles Sr: 1749–1814
- Charles Jr: 1767–1817

Phase preset pills update to match the active network's phases.

### Detail panel

Extended to show:
- **Direction breakdown**: "42 sent, 38 received" with a visual bar
- **Repository distribution**: small list of where letters are held
  (NYPL(B): 34, BM(Bar): 12, Osb: 8)
- **Cross-network presence**: if this correspondent also appears in another
  family member's network, show a link ("Also in Charles Sr: 40 letters")
- **Letter list**: scrollable table with date, direction, repository

### Shared correspondents indicator

When viewing Frances's network, correspondents who also appear in
Charles Sr or Jr's networks get a small secondary ring or badge.
This makes the cross-network connections visible without switching tabs.

### UI / Aesthetic

Same as v1: white/light-grey background, system sans-serif, muted palette.
New community colours for the expanded categories:
- Family: #4a6fa5 (slate blue)
- Literary: #a07855 (warm stone)
- Court: #5a8a7a (muted teal)
- Publishers: #7a7a8a (cool grey)
- Intimate circle: #a07080 (dusty rose)
- French circle: #6a5a8a (muted purple)
- Musical circle: #8a6a50 (warm brown)
- Scholarly/Church: #5a7a6a (sage)
- Royal: #8a7a5a (gold-grey)

## Build Pattern

- `gazetteer/burney_names.py` — shared name dictionary (new file)
- `gazetteer/build_correspondent_network.py` — extended build script
- `gazetteer/correspondent_network.html` — extended output
- `gazetteer/tests/test_burney_names.py` — name dictionary tests (new)
- `gazetteer/tests/test_build_correspondent_network.py` — extended tests

## Test Coverage

- Name normalisation: all catalogue abbreviations resolve correctly
- Parsing artefact filtering: [?], NYPL(B), "to" etc. excluded
- Date parsing: handles all variant formats in the CSV
- Three networks built with correct entry counts
- Cross-network correspondents identified correctly
- Direction counts: to + from = total for each edge
- HTML output: contains all three network datasets
- Phase presets update when switching networks
- Threshold slider reduces visible nodes correctly

## Out of Scope (Phase 2)

- Body-text mention extraction
- Editorial absence overlay (ghost nodes, sparklines, asymmetry analysis)
- Auction catalogue integration
- Manuscript provenance tracking
