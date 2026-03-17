# Sensory Timeline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a venue-focused temporal narrative view (`sensory_timeline.html`) showing how London's sensory experience changed decade by decade, with story mode for lectures and explore mode for self-guided browsing.

**Architecture:** Python build script queries `sensory.db` for qualifying venues (20+ passages, 3+ sources, LON prefix), computes decade summaries and story-mode highlights, renders a self-contained HTML page via `str.format()` with doubled-brace JS templates. The HTML has three zones: decade sidebar, timeline strip, and scrollable evidence cards.

**Tech Stack:** Python 3 (sqlite3, json, csv, pathlib), vanilla JS, CSS. No external dependencies beyond what the page loads (Inter font via Google Fonts).

**Spec:** `docs/superpowers/specs/2026-03-17-sensory-timeline-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `gazetteer/build_sensory_timeline.py` | Build script: queries DB, computes summaries/highlights, renders HTML template |
| `gazetteer/sensory_timeline.html` | Generated output (checked in) |
| `gazetteer/tests/test_build_sensory_timeline.py` | Test suite using subprocess fixture |
| `gazetteer/build_sensory_time_map.py` | Modified: add Timeline nav link |
| `gazetteer/build_venue_explorer.py` | Modified: add Timeline nav link |
| `gazetteer/build_comparison.py` | Modified: add Timeline nav link |

---

## Chunk 1: Build script data layer and test foundation

### Task 1: Create build script with data queries and test fixture

**Files:**
- Create: `gazetteer/build_sensory_timeline.py`
- Create: `gazetteer/tests/test_build_sensory_timeline.py`

- [ ] **Step 1: Write the test fixture and basic tests**

Create `gazetteer/tests/test_build_sensory_timeline.py`:

```python
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

REPO_ROOT = Path(__file__).parent.parent.parent
HTML_PATH = REPO_ROOT / "gazetteer" / "sensory_timeline.html"


@pytest.fixture(scope="module")
def html():
    """Build the timeline and return HTML content."""
    result = subprocess.run(
        [sys.executable, "gazetteer/build_sensory_timeline.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return HTML_PATH.read_text(encoding="utf-8")


def test_html_generated(html):
    assert "<html" in html
    assert "Sensory Timeline" in html


def test_venues_json_present(html):
    """Qualifying venues must be baked into the page."""
    assert "VENUES" in html
    # Vauxhall has 108 passages, 21 sources - must qualify
    assert "Vauxhall" in html


def test_evidence_json_present(html):
    """Evidence passages must be baked into the page."""
    assert "EVIDENCE" in html


def test_decade_summaries_present(html):
    """Decade summary data must be baked into the page."""
    assert "DECADE_SUMMARIES" in html


def test_highlights_present(html):
    """Story mode highlights must be baked into the page."""
    assert "HIGHLIGHTS" in html


def test_only_london_venues(html):
    """Rural venues (RUR prefix) must be excluded."""
    assert "RUR001" not in html
    assert "RUR002" not in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest gazetteer/tests/test_build_sensory_timeline.py -v`
Expected: FAIL (build script does not exist)

- [ ] **Step 3: Create the build script with data loading**

Create `gazetteer/build_sensory_timeline.py`. The script must:

1. Read venue metadata from `venues.csv`, filtering to LON-prefix venues
2. Query `sensory_evidence` from `sensory.db` for those venues (fields: venue_id, source_id, source_type, author, title, pub_year, date_min, date_max, modality, valence, text)
3. Compute decade assignment: `decade = (date_min // 10) * 10`
4. Filter to qualifying venues (20+ passages, 3+ distinct source_ids)
5. Compute decade summaries (passage_count, source_count, modality breakdown per venue per decade)
6. Run highlight selection (per venue per decade: prefer longest text, prefer fiction when cross-genre is available)
7. Render `HTML_TEMPLATE` via `str.format()` with `{VENUES_JSON}`, `{EVIDENCE_JSON}`, `{DECADE_SUMMARIES_JSON}`, `{HIGHLIGHTS_JSON}` placeholders
8. Write to `gazetteer/sensory_timeline.html`

Use `fmt_author()` from `build_venue_explorer.py` (copy the function — it handles CamelCase splitting and overrides). The initial `HTML_TEMPLATE` is a minimal placeholder that just injects the four JSON payloads into `<script>` tags.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest gazetteer/tests/test_build_sensory_timeline.py -v`
Expected: all PASS

- [ ] **Step 5: Run full gazetteer test suite**

Run: `pytest gazetteer/tests/ -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add gazetteer/build_sensory_timeline.py gazetteer/tests/test_build_sensory_timeline.py gazetteer/sensory_timeline.html
git commit -m "feat: sensory timeline build script with data layer and tests"
```

---

### Task 2: Add tests for decade assignment, threshold, and highlights

**Files:**
- Modify: `gazetteer/tests/test_build_sensory_timeline.py`

- [ ] **Step 1: Add data-layer unit tests**

Append to `gazetteer/tests/test_build_sensory_timeline.py`:

```python
from build_sensory_timeline import (
    compute_decade_summaries,
    find_qualifying_venues,
    select_highlights,
)


def test_decade_assignment():
    """date_min=1766 should map to decade 1760."""
    ev = [{"venue_id": "LON001", "source_id": "s1", "source_type": "fiction",
           "modality": "auditory", "text": "x" * 50, "decade": 1760,
           "author": "A", "title": "T", "pub_year": 1771,
           "date_min": 1766, "date_max": 1771, "valence": "neutral"}]
    summaries = compute_decade_summaries(ev)
    assert 1760 in [s["decade"] for s in summaries["LON001"]]


def test_threshold_filtering():
    """Venues below 20 passages or 3 sources must be excluded."""
    evidence = [
        {"venue_id": "LON999", "source_id": f"s{i % 2}", "source_type": "fiction",
         "modality": "auditory", "text": "x", "decade": 1760,
         "author": "A", "title": "T", "pub_year": 1771,
         "date_min": 1766, "date_max": 1771, "valence": "neutral"}
        for i in range(25)  # 25 passages but only 2 sources
    ]
    assert "LON999" not in find_qualifying_venues(evidence)


def test_highlight_prefers_longer():
    """Highlight selection should prefer longer passages."""
    evidence = [
        {"venue_id": "LON001", "source_id": "s1", "source_type": "fiction",
         "modality": "auditory", "text": "short", "decade": 1760,
         "author": "A", "title": "T", "pub_year": 1771,
         "date_min": 1766, "date_max": 1771, "valence": "neutral"},
        {"venue_id": "LON001", "source_id": "s1", "source_type": "fiction",
         "modality": "auditory", "text": "a much longer passage with more detail",
         "decade": 1760,
         "author": "A", "title": "T", "pub_year": 1771,
         "date_min": 1766, "date_max": 1771, "valence": "neutral"},
    ]
    highlights = select_highlights(evidence)
    assert highlights["LON001"][0]["index"] == 1  # the longer one
```

- [ ] **Step 2: Run tests**

Run: `pytest gazetteer/tests/test_build_sensory_timeline.py -v`
Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add gazetteer/tests/test_build_sensory_timeline.py
git commit -m "test: add unit tests for decade assignment, threshold, highlights"
```

---

## Chunk 2: HTML template — CSS and structure

### Task 3: Add the full CSS and HTML skeleton to the template

**Files:**
- Modify: `gazetteer/build_sensory_timeline.py`
- Modify: `gazetteer/tests/test_build_sensory_timeline.py`

- [ ] **Step 1: Write tests for UI structure**

Append to `gazetteer/tests/test_build_sensory_timeline.py`:

```python
def test_decade_sidebar_buttons(html):
    """Decade sidebar must have buttons for each decade."""
    assert 'data-decade="1660"' in html
    assert 'data-decade="1750"' in html
    assert 'data-decade="1810"' in html


def test_timeline_strip_present(html):
    """Timeline strip with sensory bars must exist."""
    assert "timeline-strip" in html


def test_sense_legend(html):
    """Sense legend must label all five modalities."""
    for label in ["Smell", "Noise", "Crowd", "Visual", "Thermal"]:
        assert label in html


def test_venue_selector(html):
    """Venue dropdown must be present."""
    assert "<select" in html
    assert "venue-selector" in html


def test_story_explore_toggle(html):
    """Story/Explore mode toggle must be present."""
    assert "Story" in html
    assert "Explore" in html


def test_view_nav_links(html):
    """Navigation links to other views must be present."""
    assert "sensory_time_map.html" in html
    assert "venue_explorer.html" in html
    assert "comparison.html" in html


def test_no_border_radius(html):
    """No border-radius in the CSS (design principle)."""
    import re
    matches = re.findall(r'border-radius:\s*[^0;][^;]*;', html)
    real_radii = [m for m in matches if not re.match(r'border-radius:\s*0', m)]
    assert len(real_radii) == 0, f"Found border-radius: {real_radii}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest gazetteer/tests/test_build_sensory_timeline.py::test_decade_sidebar_buttons -v`
Expected: FAIL

- [ ] **Step 3: Replace the placeholder HTML_TEMPLATE with the full CSS and structure**

Replace the `HTML_TEMPLATE` string in `gazetteer/build_sensory_timeline.py` with the complete template. This is a large block. Key requirements:

**CSS rules:**
- No `border-radius` anywhere
- Monochrome: black, white, greys only
- Colour only for the five sensory bar encodings: smell `#f59e0b`, noise `#3b82f6`, crowd `#ef4444`, visual `#10b981`, thermal `#8b5cf6`
- `.card-fiction` left border: `3px solid #333`; `.card-nonfiction` left border: `1px solid #ccc`
- Source labels: small-caps, `font-size: 10px`, `letter-spacing: 0.5px`, `color: #666`
- Modality tags: `font-size: 9px`, `color: #999`
- `.card-text`: `font-family: Georgia, serif; font-style: italic; line-height: 1.75; font-size: 15px`
- `.dimmed`: `opacity: 0.25` (for story mode non-highlights)
- `.compare-row`: `display: flex; gap: 12px` (side-by-side fiction/non-fiction)
- `.story-bar`: fixed bottom, `background: #111`, `color: #eee`
- `.decade-nav`: `width: 64px`, `background: #111`, vertical flex
- `.decade-btn.active`: left border accent `#3b82f6`
- `.decade-btn:not(.has-data)`: `color: #444`, no click action
- `.timeline-decade.active`: `background: #f0f6ff`
- Page uses `display: flex; height: 100vh` for the three-zone layout

**HTML structure:**

```
div.page (flex row)
  nav.decade-nav
    h3 "Decade"
    button.decade-btn[data-decade="1660"] x 16 (1660 through 1810)
  div.main (flex column)
    div.top-bar
      span.view-nav (links to other views)
      select.venue-selector (populated by JS)
      div.mode-toggle
        button.mode-btn[data-mode="story"] "Story"
        button.mode-btn.active[data-mode="explore"] "Explore"
    div.timeline-strip (flex row)
      div.timeline-decade[data-decade] x 16
        span.label (decade label)
        div.sense-bars
          div.sense-bar.smell
          div.sense-bar.noise
          div.sense-bar.crowd
          div.sense-bar.visual
          div.sense-bar.thermal
    div.sense-legend
      5 x span with coloured dot + label
    div.content-area
      div.content-inner (max-width: 720px, margin: 0 auto)
        (populated by JS)
  div.story-bar (initially hidden)
    span.story-pos
    kbd hints
```

**Script block** injects the four JSON payloads and declares:

```javascript
const VENUES = {VENUES_JSON};
const EVIDENCE = {EVIDENCE_JSON};
const DECADE_SUMMARIES = {DECADE_SUMMARIES_JSON};
const HIGHLIGHTS = {HIGHLIGHTS_JSON};

const state = {{ venueId: VENUES.length ? VENUES[0].id : null, mode: 'explore', storyIndex: 0 }};
```

Note: all JS content uses `textContent` for user-facing text or builds DOM strings from pre-sanitised JSON data (author names, titles, passage text are all from the controlled `sensory.db` database, not user input). The `esc()` helper function must HTML-encode `&`, `<`, `>`, `"` in all rendered text values.

- [ ] **Step 4: Rebuild and run tests**

Run: `python3 gazetteer/build_sensory_timeline.py && pytest gazetteer/tests/test_build_sensory_timeline.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add gazetteer/build_sensory_timeline.py gazetteer/sensory_timeline.html gazetteer/tests/test_build_sensory_timeline.py
git commit -m "feat: sensory timeline HTML template with CSS and layout"
```

---

## Chunk 3: JavaScript — rendering, interaction, story mode

### Task 4: Implement venue rendering and decade scroll

**Files:**
- Modify: `gazetteer/build_sensory_timeline.py` (JS in HTML_TEMPLATE)

- [ ] **Step 1: Add tests for rendered evidence cards**

Append to `gazetteer/tests/test_build_sensory_timeline.py`:

```python
def test_source_type_labels(html):
    """Source type labels must appear as small-caps text."""
    assert "source_type" in html
    assert "FICTION" in html or "fiction" in html.lower()


def test_modality_tags_in_template(html):
    """Modality tag rendering must be present."""
    assert "modality" in html


def test_georgia_font(html):
    """Passage text must use Georgia serif."""
    assert "Georgia" in html
```

- [ ] **Step 2: Implement the JS rendering functions**

Add to the `<script>` block in the HTML template. The JS must implement:

1. **`esc(s)`** — HTML-encode `&`, `<`, `>`, `"` in string values
2. **`populateVenueSelector()`** — fill the `<select>` from `VENUES`, wire change handler to update `state.venueId` and call `render()`
3. **`renderStrip()`** — for each `.timeline-decade` column, look up the venue's decade summary, set each `.sense-bar` width as `(count / peak * 100)%` where peak is the max passage_count across all decades for that venue
4. **`renderSidebar()`** — toggle `.has-data` class on `.decade-btn` elements based on whether the venue has data for that decade
5. **`renderCard(ev, idx)`** — render an evidence card with: source label (small-caps), author + title, date (show pub_year, and if floor(pub_year/10)*10 differs from decade, append "describing {decade}s"), modality tag, passage text (Georgia italic, truncated at 500 chars), source attribution. Apply `.card-fiction` or `.card-nonfiction` class. Apply `.dimmed` if in story mode and not a highlight.
6. **`renderContent()`** — group evidence by decade, render decade headers with passage/source counts. For each decade with both fiction and non-fiction, render a `.compare-row` with the longest fiction and longest non-fiction passage side-by-side, then render remaining cards below.
7. **`setupScrollSpy()`** — IntersectionObserver on `.decade-section` elements, updating active state on sidebar and strip
8. **`setupDecadeClicks()`** — click handlers on `.decade-btn` and `.timeline-decade` to scroll to the matching `.decade-section`

SOURCE_LABELS map: fiction->FICTION, topography->TOPOGRAPHY, diary->DIARY, letters->LETTERS, legal->LEGAL, institutional->INSTITUTIONAL, poetry->POETRY.

MOD_LABELS map: olfactory->smell, auditory->noise, crowd->crowd, visual->visual, thermal->thermal.

- [ ] **Step 3: Rebuild and run tests**

Run: `python3 gazetteer/build_sensory_timeline.py && pytest gazetteer/tests/test_build_sensory_timeline.py -v`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add gazetteer/build_sensory_timeline.py gazetteer/sensory_timeline.html gazetteer/tests/test_build_sensory_timeline.py
git commit -m "feat: timeline JS rendering - cards, strip, sidebar, scroll-spy"
```

---

### Task 5: Implement story mode

**Files:**
- Modify: `gazetteer/build_sensory_timeline.py` (JS in HTML_TEMPLATE)
- Modify: `gazetteer/tests/test_build_sensory_timeline.py`

- [ ] **Step 1: Add story mode tests**

Append to `gazetteer/tests/test_build_sensory_timeline.py`:

```python
def test_story_mode_keyboard(html):
    """Story mode must handle Space, ArrowRight, ArrowLeft, Escape."""
    assert "ArrowRight" in html
    assert "ArrowLeft" in html
    assert "Escape" in html


def test_story_bar_present(html):
    """Story mode position bar must exist."""
    assert "story-bar" in html
```

- [ ] **Step 2: Add story mode JS**

Implement in the `<script>` block:

1. **`enterStory()`** — set `state.mode = 'story'`, `state.storyIndex = 0`, update toggle button active states, show `.story-bar`, re-render, scroll to first highlight
2. **`exitStory()`** — set `state.mode = 'explore'`, update buttons, hide `.story-bar`, re-render
3. **`scrollToHighlight()`** — find the card matching the current highlight index, `scrollIntoView({ behavior: 'smooth', block: 'center' })`, update position display
4. **`updateStoryBar()`** — set `.story-pos` text to `"{current} / {total}"`
5. **`storyAdvance(dir)`** — increment/decrement `state.storyIndex` within bounds, re-render and scroll
6. **Keyboard handler** — `keydown` listener: Space/ArrowRight advance, ArrowLeft goes back, Escape exits story mode. Only active when `state.mode === 'story'`.
7. **Mode toggle click handlers** — wire `.mode-btn` buttons to `enterStory()` / `exitStory()`

- [ ] **Step 3: Rebuild and run tests**

Run: `python3 gazetteer/build_sensory_timeline.py && pytest gazetteer/tests/test_build_sensory_timeline.py -v`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add gazetteer/build_sensory_timeline.py gazetteer/sensory_timeline.html gazetteer/tests/test_build_sensory_timeline.py
git commit -m "feat: story mode with keyboard navigation and position bar"
```

---

### Task 6: Add the main render function and initialisation

**Files:**
- Modify: `gazetteer/build_sensory_timeline.py` (JS in HTML_TEMPLATE)

- [ ] **Step 1: Add the main render() and init**

Add to the `<script>` block:

1. **`render()`** — calls `renderStrip()`, `renderSidebar()`, `renderContent()`, `setupScrollSpy()`, and if in story mode, `scrollToHighlight()`
2. **Init sequence** — call `populateVenueSelector()`, `setupDecadeClicks()`, `render()`

- [ ] **Step 2: Rebuild and open in browser**

Run: `python3 gazetteer/build_sensory_timeline.py`
Open `gazetteer/sensory_timeline.html` in browser. Verify:
- Venue selector populated and working
- Timeline strip shows sensory bars
- Evidence cards render with decade grouping
- Decade sidebar scroll-spy works
- Story/Explore toggle works
- Keyboard navigation in story mode works

- [ ] **Step 3: Commit**

```bash
git add gazetteer/build_sensory_timeline.py gazetteer/sensory_timeline.html
git commit -m "feat: timeline init and render loop"
```

---

## Chunk 4: Navigation integration and final validation

### Task 7: Add Timeline link to existing views

**Files:**
- Modify: `gazetteer/build_sensory_time_map.py`
- Modify: `gazetteer/build_venue_explorer.py`
- Modify: `gazetteer/build_comparison.py`

- [ ] **Step 1: Add test for nav link in existing views**

Append to `gazetteer/tests/test_build_sensory_timeline.py`:

```python
def test_sensory_map_has_timeline_link():
    """Sensory time map must link to timeline."""
    path = REPO_ROOT / "gazetteer" / "sensory_time_map.html"
    if path.exists():
        content = path.read_text(encoding="utf-8")
        assert "sensory_timeline.html" in content


def test_venue_explorer_has_timeline_link():
    """Venue explorer must link to timeline."""
    path = REPO_ROOT / "gazetteer" / "venue_explorer.html"
    if path.exists():
        content = path.read_text(encoding="utf-8")
        assert "sensory_timeline.html" in content
```

- [ ] **Step 2: Add Timeline link to sensory time map**

In `gazetteer/build_sensory_time_map.py`, find the view nav section (around line 526-529) and add after the Comparison link:

```html
    <a href="sensory_timeline.html" class="view-tab">Timeline</a>
```

- [ ] **Step 3: Add Timeline link to venue explorer**

In `gazetteer/build_venue_explorer.py`, find the view nav section (around line 238-241) and add after the Comparison link:

```html
    <a href="sensory_timeline.html" class="view-tab">Timeline</a>
```

- [ ] **Step 4: Add Timeline link to comparison view**

In `gazetteer/build_comparison.py`, find the view nav section and add the same link.

- [ ] **Step 5: Rebuild all views**

Run:
```bash
python3 gazetteer/build_sensory_time_map.py
python3 gazetteer/build_venue_explorer.py
python3 gazetteer/build_comparison.py
python3 gazetteer/build_sensory_timeline.py
```

- [ ] **Step 6: Run full test suite**

Run: `pytest gazetteer/tests/ -v`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add gazetteer/build_sensory_time_map.py gazetteer/build_venue_explorer.py gazetteer/build_comparison.py gazetteer/sensory_time_map.html gazetteer/venue_explorer.html gazetteer/comparison.html gazetteer/tests/test_build_sensory_timeline.py
git commit -m "feat: add Timeline nav link to all views"
```

---

### Task 8: Final validation and browser testing

**Files:**
- No new files

- [ ] **Step 1: Run full test suite**

Run: `pytest gazetteer/tests/ -v`
Expected: all pass

- [ ] **Step 2: Browser validation checklist**

Open `gazetteer/sensory_timeline.html` in browser and verify:

- [ ] Page loads without JS errors
- [ ] Venue selector shows ~38 venues, alphabetically sorted
- [ ] Selecting a venue updates the timeline strip, sidebar, and content
- [ ] Timeline strip bars reflect actual passage counts per modality
- [ ] Decade sidebar highlights active decade on scroll
- [ ] Clicking a decade in sidebar or strip scrolls content
- [ ] Evidence cards show: source label, author, title, date, modality tag, passage text
- [ ] Fiction cards have darker left border than non-fiction
- [ ] Comparison rows appear when fiction + non-fiction exist in same decade
- [ ] Date display shows "1771 (describing 1760s)" when pub_year differs from decade
- [ ] Story mode: clicking Story activates, Space advances, Esc exits
- [ ] Story mode: non-highlight cards are dimmed
- [ ] Story mode: position bar shows "3 / 11" etc
- [ ] Nav links: Sensory Map, Evidence, Narrative, Comparison all present and working
- [ ] No border-radius anywhere in the UI
- [ ] Passage text is Georgia serif italic

- [ ] **Step 3: Commit generated HTML**

```bash
git add gazetteer/sensory_timeline.html
git commit -m "build: regenerate all HTML views with timeline nav links"
```
