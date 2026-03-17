# Correspondent Network Analysis and Visualisation

## Summary

A self-contained interactive HTML visualisation of Frances Burney's correspondence
network, built from the structured headers in the OUP *Journals and Letters*
(2011, ed. Sabor & Troide). Force-directed graph with a dual-handle timeline
slider. Professional analyst-presentation aesthetic.

## Data Source

The 243 numbered selections in `nonfiction/FrancesBurney/JournalsAndLetters.txt`.
Each selection has a structured header of the form:

```
N. [From] {Letter|Journal Letter|Verse Letter|Letters} to CORRESPONDENT DATE
```

Journals addressed to "Nobody" (private entries) are excluded from the network
but counted in the timeline for context density.

## Data Pipeline

`gazetteer/build_correspondent_network.py`:

1. Read the OUP text file.
2. Parse headers with a regex capturing: selection number, entry type
   (journal / letter / journal letter / verse letter), correspondent name,
   and date string.
3. Parse dates to year (and month where available). Handle ranges
   (e.g. "August–September 1773") by taking the start.
4. Normalise correspondent names via a lookup dict:
   - "Susanna Burney" / "Susanna Phillips" → "Susanna Burney Phillips"
   - "Dr Burney" / "Dr Charles Burney" → "Dr Charles Burney"
   - "Hester Lynch Thrale" / "Hester Lynch Piozzi" → "Hester Thrale Piozzi"
   - "Charlotte Cambridge" / "Charlotte Broome" → "Charlotte Broome"
   - etc.
5. Assign each correspondent to an emotional community (lookup dict):
   - **Family**: Dr Charles Burney, Susanna Burney Phillips, Esther Burney,
     Charlotte Broome, Charlotte Barrett, James Burney, Charles Burney (brother),
     Charles Parr Burney, Maria Rishton, Alexandre d'Arblay, Alexander d'Arblay
   - **Literary / Bluestocking**: Samuel Crisp, Samuel Johnson, Hester Thrale Piozzi,
     Georgiana Waddington
   - **Court**: Queen Charlotte, Princess Elizabeth, Margaret Planta,
     William Lowndes
   - **Publishers**: Thomas Lowndes, Longman Hurst Rees Orme and Brown,
     Messrs Longman and Company
   - **Intimate circle (post-court)**: Frederica Locke, Amelia Angerstein,
     Viscountess Keith, William Wilberforce
6. Assign each selection to a life-phase based on date, matching the OUP
   chapter divisions:
   - 1768–1777: Apprentice Years
   - 1778–1781: Evelina & Streatham
   - 1782–1786: Cecilia & Prelude to Court
   - 1786–1791: Court Years
   - 1791–1792: London & Western Tour
   - 1793–1795: Courtship & Marriage
   - 1796–1802: Camilla & Camilla Cottage
   - 1802–1812: France
   - 1812–1814: Interlude / The Wanderer
   - 1814–1815: Waterloo
   - 1815–1818: Final Years with d'Arblay
   - 1818–1839: Widowhood
7. Output: JSON blob embedded in the HTML template via Python string
   substitution (doubled braces for JS, single for placeholders — per
   project convention).

## Visualisation

### Force-directed graph (D3.js v7, via CDN)

- **Burney** is a fixed central node (larger, distinct styling).
- **Correspondent nodes**: size proportional to total letter count.
  Colour by emotional community. Muted professional palette:
  - Family: slate blue (#4a6fa5)
  - Literary / Bluestocking: warm stone (#a07855)
  - Court: muted teal (#5a8a7a)
  - Publishers: cool grey (#7a7a8a)
  - Intimate circle: dusty rose (#a07080)
- **Edges**: thickness proportional to letter count. Same colour as
  correspondent node, at reduced opacity.
- **Interaction**:
  - Hover node → tooltip: name, community, letter count, date range.
  - Click node → detail panel slides in from right: full name, community,
    letter count, date range, list of individual letter dates with
    entry type and life-phase.
  - Nodes are draggable; simulation settles naturally.
  - Click background to dismiss detail panel.

### Timeline slider

- Dual-handle range slider below the graph.
- Range: 1768–1839.
- Dragging adjusts the visible window:
  - Nodes outside the range fade to 10% opacity and lose force influence.
  - Edges outside the range fade similarly.
  - Node sizes recompute based on filtered letter count.
  - A label shows the active range and letter count.
- **Preset buttons** for each life-phase (matching OUP chapters) for
  quick navigation. Clicking a preset sets the slider handles and
  triggers the filter.

### Detail panel

Right-side panel, collapses when nothing selected. Shows:
- Correspondent name
- Emotional community (with colour chip)
- Total letters / letters in current range
- Date range of correspondence
- Table of individual letters: date, entry type (letter / journal letter /
  verse letter), life-phase

### UI / Aesthetic

- White / light-grey background (#fafafa).
- System sans-serif font stack.
- No decorative borders, shadows, or gratuitous chrome.
- Graph area ~70% viewport height.
- Timeline slider below graph, ~80px.
- Detail panel 320px right sidebar, slides in on selection.
- Muted, desaturated palette throughout. No saturated primaries.
- Responsive but optimised for desktop / presentation display (1200px+).
- Legend: small, unobtrusive, bottom-left — community colour key.

## Build Pattern

- Script: `gazetteer/build_correspondent_network.py`
- Output: `gazetteer/correspondent_network.html`
- Tests: `gazetteer/tests/test_build_correspondent_network.py`
- Self-contained single HTML file, all JS/CSS inline, D3 from CDN.
- Template uses doubled braces for JS, single braces for Python
  placeholders (`{NETWORK_JSON}`).

## Test Coverage

- Header parsing: verify all 243 selections are parsed, correct count
  of letters vs journals.
- Name normalisation: verify merged identities (Susanna Burney = Susanna
  Phillips, etc.).
- Community assignment: verify every correspondent has a community.
- Life-phase assignment: verify date → phase mapping for edge cases.
- JSON output structure: verify nodes and edges have required fields.
- HTML output: verify the file is produced and contains expected
  placeholders filled.

## Out of Scope (for now)

- Body-text mention extraction (the "C" layer — future enhancement).
- Gutenberg *Diary and Letters* volumes.
- Cross-correspondent edges (who knew whom independently of Burney).
