# Sensory Timeline — Design Spec

## Purpose

A venue-focused temporal narrative view showing how the sensory experience of
London changed across the long eighteenth century (1660–1820). Serves two
audiences:

1. **Lecture presentation** — a curated "story mode" stepping through
   highlights for undergraduates, demonstrating how places like Vauxhall or
   Smithfield transformed over decades.
2. **Student/researcher exploration** — an interactive mode where any
   well-documented venue can be browsed freely, with all evidence passages
   visible and chronologically ordered.

The primary pedagogical goals are:

- **Temporal change as the spine** — how a single venue's sensory profile
  evolved decade by decade.
- **Intensity and spatial contrast** — how shockingly intense London was, and
  how different places had completely different characters.
- **Fiction/non-fiction comparison** — placing literary sources alongside
  diaries, travel accounts, and legal records as a teaching moment about
  evidence and genre.

Deployed as a static HTML page on GitHub Pages, built by a Python script from
`sensory.db`, following the same pattern as the existing four views.

## Data Model

### Venue selection

All London venues (filtered by `venue_id LIKE 'LON%'`) meeting the threshold:
20+ evidence passages and 3+ distinct sources. Currently ~38 venues qualify,
including Vauxhall (108 passages, 21 sources), Smithfield (64, 52), Newgate
(134, 45), the Strand (108, 26), and Drury Lane (73, 19). Rural venues
(`RUR001`, `RUR002`) are excluded by the LON prefix filter — they represent
Continental Gothic and rural settings outside the project's urban scope.

### Evidence passages

Each passage carries:

| Field | Source |
|-------|--------|
| `source_id` | FK to `sources` table |
| `author` | e.g. `burney`, `CarlMoritz` |
| `title` | e.g. `Evelina`, `Travels in England in 1782` |
| `source_type` | `fiction`, `topography`, `diary`, `letters`, `legal`, `institutional`, `poetry` |
| `date_min`, `date_max` | Integer years |
| `modality` | `auditory`, `visual`, `crowd`, `olfactory`, `thermal` |
| `valence` | `pleasant`, `unpleasant`, `neutral`, `mixed`, `positive`, `negative` |
| `text` | The passage text |

### Decade assignment

A passage is assigned to the decade containing `date_min`, floored:
`decade = floor(date_min / 10) * 10`. A passage with `date_min=1766` goes into
the 1760s. Passages spanning multiple decades appear once, in the earliest.

### Decade summaries

For each venue + decade, a summary object:

```json
{
  "decade": 1760,
  "passage_count": 22,
  "source_count": 3,
  "modalities": {
    "auditory": 8,
    "visual": 6,
    "crowd": 5,
    "olfactory": 2,
    "thermal": 1
  }
}
```

These drive the timeline strip bar widths. Bar width is proportional to passage
count per modality, normalised against the venue's peak decade.

### JSON payloads baked into HTML

- `{VENUES_JSON}` — venue metadata (id, name, lat, lon, building_type,
  enclosure, passage/source counts).
- `{EVIDENCE_JSON}` — all passages for qualifying venues, sorted by venue then
  date_min. Each passage includes: source_id, author, title, source_type,
  pub_year, date_min, date_max, modality, valence, text. The `context` and
  `divergence` fields from the database are omitted — they serve the venue
  explorer's different purpose.
- `{DECADE_SUMMARIES_JSON}` — per-venue per-decade summary objects.

## UI Structure

### Layout

Three zones, no border-radius anywhere. Typography-led, minimal chrome.

```
┌──────┬──────────────────────────────────────────┐
│      │  Venue selector          [Story][Explore]│
│ 1660 ├──────────────────────────────────────────┤
│ 1670 │  Timeline strip: sensory bands by decade │
│ 1680 │  Smell ████  Noise ██████  etc.          │
│ 1690 ├──────────────────────────────────────────┤
│ 1700 │                                          │
│ 1710 │  1740s — 16 passages from 4 sources      │
│ 1720 │  ┌──────────────────────────────────┐    │
│ 1730 │  │ FICTION  Richardson, Clarissa     │    │
│▐1740 │  │ sound · crowd                    │    │
│ 1750 │  │ "At all public diversions, she   │    │
│ 1760 │  │  was the leader..."              │    │
│ 1770 │  └──────────────────────────────────┘    │
│ 1780 │                                          │
│ 1790 │  1760s — 22 passages from 3 sources      │
│ 1800 │  ┌──────────────────────────────────┐    │
│ 1810 │  │ FICTION  Smollett, Humphry C.     │    │
│      │  │ "Give it noise, confusion..."    │    │
│      │  └──────────────────────────────────┘    │
└──────┴──────────────────────────────────────────┘
```

### Decade sidebar (left, ~64px)

Vertical list of decades. Each shows a small intensity indicator. Clicking
scrolls the content area. Active decade updates via scroll-spy
(IntersectionObserver on decade header elements). Decades with no passages
for the selected venue show a dimmed label and no intensity bar; clicking
them is a no-op. The content area only renders decade sections that have
passages.

### Timeline strip (top, ~72px)

Horizontal bar chart: one column per decade, five stacked bars for the five
modalities: smell (amber), noise (blue), crowd (red), visual (green), thermal
(purple). These are the only uses of colour in the UI. Thermal data is sparse
but present for some venues (e.g. Smollett on the cold night air at Vauxhall).
A small legend row sits below the strip. Clicking a column scrolls to that
decade.

### Content area

Scrollable evidence passages grouped by decade.

**Decade headers**: uppercase, letterspaced, small — e.g.
`1760s — 22 passages from 3 sources`.

**Evidence cards**: square-edged, white background, thin border. Left border
encodes source type: a heavier weight or slightly darker shade for fiction vs
non-fiction (not coloured pills). Inside each card:

- Source type label (plain text, small caps: FICTION / TRAVEL / DIARY / LEGAL)
- Author and title
- Date — show `pub_year` as the primary date; if `date_min` differs (fiction
  describing an earlier period), show both: "1771 (describing 1760s)"
- Modality tags (small, grey, understated)
- Passage text in serif italic (Georgia)
- Source attribution line

**Comparison rows**: where fiction and non-fiction passages exist in the same
decade for the same venue, they appear side-by-side in a two-column layout.
"Non-fiction" means any source_type other than `fiction` (i.e. topography,
diary, letters, legal, institutional, poetry). If multiple fiction or multiple
non-fiction passages exist, show the single best of each (longest text) in the
comparison row; remaining passages appear as regular cards below. Comparison
rows appear in both Story and Explore modes.

### Top bar

- Venue selector (dropdown, all qualifying venues)
- Story / Explore toggle (segmented control, minimal styling)

## Story Mode

Auto-curated, no manual editorial pass needed.

**Highlight selection algorithm**: for each venue, for each decade that has
passages, select one highlight passage using:

1. Prefer longer text (more vivid, more context) — this is the primary
   discriminator since confidence values are nearly all 1.0
2. Prefer passages where the same decade also has a passage of a different
   `source_type` (enables the fiction/non-fiction comparison moment)
3. Tiebreak: prefer `fiction` over other source types (more compelling to read)

**Behaviour**:

- Space or right-arrow advances to the next highlight
- Left-arrow goes back
- Esc exits to explore mode
- Non-highlight cards are dimmed (low opacity) but remain visible for context
- Decade sidebar and timeline strip update in sync
- A small fixed bar at the bottom shows position: "3 / 11" and keyboard hints

## Build Script

**File**: `gazetteer/build_sensory_timeline.py`

**Pattern**: same as the other three builders.

1. Connect to `sensory.db`
2. Query venues meeting threshold (20+ passages, 3+ sources, venue_id LIKE
   'LON%')
3. Query all evidence passages for those venues
4. Compute decade summaries
5. Run highlight selection algorithm for story mode
6. Render `HTML_TEMPLATE` with `str.format()` — doubled braces for JS, single
   braces for Python placeholders
7. Write `gazetteer/sensory_timeline.html`

**Template conventions**: all literal JS braces doubled (`{{`, `}}`). Python
placeholders: `{VENUES_JSON}`, `{EVIDENCE_JSON}`, `{DECADE_SUMMARIES_JSON}`,
`{HIGHLIGHTS_JSON}`.

## Navigation Integration

The existing view nav bar gains a fifth entry: "Timeline". This link is added
to:

- `build_sensory_time_map.py` (sensory_time_map.html)
- `build_venue_explorer.py` (venue_explorer.html)
- `build_comparison.py` (comparison.html)
- The narrative map if it exists

The timeline page links back to the other four views in the same nav bar.
"Timeline" is inserted as the last entry, after "Comparison".

## Visual Design Principles

- **No border-radius** — square edges on all cards, badges, controls
- **Monochrome base** — black, white, greys. Colour only for sensory band
  encoding in the timeline strip (amber, blue, red, green)
- **Source-type distinction via typography** — weight/caps differences, not
  coloured pills. Thin left border shade difference (darker for fiction,
  lighter for non-fiction)
- **Modality tags** — small, grey, understated
- **Quotation text** — Georgia serif italic, generous line height (1.75),
  dominant in the visual hierarchy
- **Minimal chrome** — no shadows, no gradients, no rounded corners. The
  content is the interface.

## Test Suite

**File**: `gazetteer/tests/test_build_sensory_timeline.py`

Uses the same fixture pattern: subprocess builds the HTML, tests assert against
the output string.

Tests to include:

- Venue threshold filtering (20+ passages, 3+ sources)
- Decade assignment correctness
- All qualifying venues present in JSON
- Evidence passages present and grouped by decade
- Story mode highlights present
- Source-type badges in HTML
- Modality tags in HTML
- Decade sidebar buttons present
- Timeline strip sensory bars present
- Sense legend present
- Navigation links to other views present
- Story/Explore mode toggle present

## Scope

### In scope

- New build script (`build_sensory_timeline.py`) + output
  (`sensory_timeline.html`)
- ~38 London venues meeting the evidence threshold
- All evidence passages for those venues, decade-grouped
- Timeline strip with five sensory bands
- Decade sidebar with scroll-spy
- Story mode (auto-curated) and Explore mode (free scroll)
- Fiction/non-fiction visual distinction and side-by-side comparison
- Navigation link added to existing views
- Test suite

### Not in scope

- Vector base map / figure-ground (separate future project)
- Bath venues (no data in database yet)
- Manual editorial curation of story mode
- Export / PDF / image output
- Contour surface fixes (separate task)
- Map integration within the timeline view
