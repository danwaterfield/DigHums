# Correspondent Network Analysis and Visualisation

## Summary

A self-contained interactive HTML visualisation of Frances Burney's correspondence
network, built from the structured headers in the OUP *Journals and Letters*
(2011, ed. Sabor & Troide). Force-directed graph with a dual-handle timeline
slider. Professional analyst-presentation aesthetic.

## Data Source

The numbered selections in `nonfiction/FrancesBurney/JournalsAndLetters.txt`.
Headers are parsed from lines matching `^\d+\. `. The actual count should be
determined at parse time rather than hardcoded.

### Header Variants

The source contains several header patterns beyond simple letters:

| Pattern | Example | Handling |
|---------|---------|----------|
| Letter to X | `44. From Letter to Susanna Burney 5 July 1778` | Standard: extract correspondent + date |
| Journal Letter to X | `46. From Journal Letter to Susanna Burney 23 August 1778` | Standard: extract correspondent + date |
| Verse Letter to X | `9. Verse Letter to Dr Charles Burney 23 June 1769` | Standard: extract correspondent + date |
| Letters (plural) to X | `36. Letters to Thomas Lowndes 25 and 26 December 1776` | Count as 1 selection, extract correspondent |
| Multi-recipient | `79. From Journal Letter to Susanna Burney and Charlotte Ann Burney June 1781` | Create one edge per recipient |
| Pure journal (no "to") | `1. Journal 27 March 1768` | Exclude from network; count in timeline density |
| Location-prefixed journal | `209. Waterloo Journal 27 April and 13 May 1815` | Exclude from network (no correspondent) |
| "Journal for" | `214. Journal for 22 July 1815` | Exclude from network |
| Compound entry | `26. From Letter to Samuel Crisp 2 March 1775 and Journal 1775` | Extract the letter component only |

### Date Qualifiers

Dates may include qualifiers that are stripped to extract the year:

- `c. 16 February 1779` → 1779
- `post 12 October 1779` → 1779
- `pre-15 December 1814` → 1814
- `between 12 and 28 April 1814` → 1814
- `late June 1779` → 1779
- `mid or late May 1802` → 1802
- `August–September 1773` → 1773 (take start)

Month is extracted where unambiguously available; otherwise year only.

## Data Pipeline

`gazetteer/build_correspondent_network.py`:

1. Read the OUP text file (path relative to repo root:
   `nonfiction/FrancesBurney/JournalsAndLetters.txt`).
2. Parse headers with a regex. Entries with no "to" clause (pure journals,
   location-prefixed journals, "Journal for") are excluded from the network
   but counted for timeline context density.
3. Multi-recipient letters ("to X and Y") create one edge per recipient.
   Split on " and " in the correspondent field, but only where the result
   yields known names — handle "William and Frederica Locke" as two
   recipients by checking against the normalisation dict.
4. Parse dates: strip qualifiers (c., post, pre-, late, early, mid,
   between...and), extract year and month where available.
5. Normalise correspondent names via a lookup dict:
   - "Susanna Burney" / "Susanna Phillips" → "Susanna Burney Phillips"
   - "Dr Burney" / "Dr Charles Burney" → "Dr Charles Burney"
   - "Hester Lynch Thrale" / "Hester Lynch Piozzi" → "Hester Thrale Piozzi"
   - "Charlotte Cambridge" / "Charlotte Broome" → "Charlotte Broome"
   - "Charlotte Ann Burney" → "Charlotte Ann Burney" (distinct from Charlotte Broome — different sister)
   - "Alexandre d'Arblay" (husband) vs "Alexander d'Arblay" (son) — kept
     distinct, these are two different people
   - "Longman, Hurst, Rees, Orme and Brown" / "Messrs Longman and Company"
     → "Longman & Co"
6. Assign each correspondent to an emotional community (lookup dict):
   - **Family**: Dr Charles Burney, Susanna Burney Phillips, Esther Burney,
     Charlotte Broome, Charlotte Ann Burney, Charlotte Barrett, James Burney,
     Charles Burney (brother), Charles Parr Burney, Maria Rishton,
     Alexandre d'Arblay, Alexander d'Arblay
   - **Literary / Bluestocking**: Samuel Crisp, Samuel Johnson,
     Hester Thrale Piozzi, Georgiana Waddington
   - **Court**: Queen Charlotte, Princess Elizabeth, Margaret Planta,
     William Lowndes
   - **Publishers**: Thomas Lowndes, Longman & Co
   - **Intimate circle (post-court)**: Frederica Locke, Amelia Angerstein,
     William Locke, Viscountess Keith, William Wilberforce
   - Any unlisted correspondent is assigned "Unknown" and flagged in
     stderr during build so the dict can be updated.
7. Assign each selection to a life-phase based on date, matching the OUP
   chapter divisions. Boundary years are assigned to the later phase
   (e.g. 1786 → Court Years, not Cecilia):
   - 1768–1777: Apprentice Years
   - 1778–1781: Evelina & Streatham
   - 1782–1785: Cecilia & Prelude to Court
   - 1786–1791: Court Years
   - 1791–1792: London & Western Tour (1791 → this phase, not Court)
   - 1793–1795: Courtship & Marriage
   - 1796–1802: Camilla & Camilla Cottage
   - 1803–1812: France (1802 entries assigned here if month >= July,
     else to previous phase — the move happened mid-1802)
   - 1812–1814: Interlude / The Wanderer
   - 1814–1815: Waterloo
   - 1815–1818: Final Years with d'Arblay
   - 1819–1839: Widowhood
8. Output: JSON blob embedded in the HTML template via Python string
   substitution (doubled braces for JS, single for placeholders — per
   project convention).

## Visualisation

### Force-directed graph (D3.js v7, inlined)

D3 v7 minified (~280KB) is inlined in the HTML to maintain the project's
self-contained convention. No CDN dependency.

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
  On narrow viewports (<1000px), the detail panel overlays rather than
  sitting beside the graph.
- Legend: small, unobtrusive, bottom-left — community colour key.

## Build Pattern

- Script: `gazetteer/build_correspondent_network.py`
- Output: `gazetteer/correspondent_network.html`
- Tests: `gazetteer/tests/test_build_correspondent_network.py`
- Self-contained single HTML file, all JS/CSS inline, D3 inlined.
- Template uses doubled braces for JS, single braces for Python
  placeholders (`{NETWORK_JSON}`).

## Test Coverage

- Header parsing: verify the parsed selection count matches actual headers
  in source. Correct count of letters vs journals.
- Multi-recipient parsing: verify "Susanna Burney and Charlotte Ann Burney"
  produces two edges.
- Approximate date handling: verify "c.", "post", "pre-", "between", "late"
  qualifiers are stripped and year extracted correctly.
- Location-prefixed journals: verify "Waterloo Journal" and "Ilfracombe
  Journal" are excluded from the network.
- Compound entries: verify "Letter to X ... and Journal ..." extracts the
  letter component only.
- Name normalisation: verify merged identities (Susanna Burney = Susanna
  Phillips, etc.) and that Alexandre/Alexander d'Arblay remain distinct.
- Community assignment: verify every correspondent has a community.
  Verify unlisted correspondents are flagged.
- Life-phase assignment: verify date → phase mapping for boundary years.
- Pure journal exclusion: verify journal entries without correspondents
  are excluded from the network but present in timeline density data.
- JSON output structure: verify nodes and edges have required fields.
- HTML output: verify the file is produced and contains expected
  placeholders filled.

## Out of Scope (for now)

- Body-text mention extraction (the "C" layer — future enhancement).
- Gutenberg *Diary and Letters* volumes.
- Cross-correspondent edges (who knew whom independently of Burney).
- Export to SVG/PNG (future enhancement for presentations).
