# Old Bailey Integration — Design Document
## 2026-02-28

---

## Goal

Extend the sensory evidence store (`sensory.db`) with legal records from the Old Bailey Proceedings API, expanding the gazetteer to cover plebeian and street-level urban spaces, and adding a `valence` field to the evidence schema to support future environmental overlay visualisation.

---

## Motivation

The adjacent question driving this project is: *what would a person standing in a given place have actually heard, smelt, and seen on a given day?* — especially a person who had grown up in the countryside. The literary corpus answers this partly, but from a literary register. Old Bailey nuisance prosecutions answer it differently: a slaughterhouse operator is dragged to court specifically because the smell crossed a threshold. That is quantified sensory overwhelm, not narrative convention.

Legal records also cover spaces that fiction ignores: prisons, markets, execution grounds, rookeries. These are the spaces where sensory intensity was highest and literary representation thinnest.

---

## Architecture

```
extract_old_bailey.py
  └─ reads venues.csv          (expanded gazetteer — new venues added)
  └─ queries Old Bailey API    (paginated, venue-anchored)
  └─ caches responses          (gazetteer/sources/legal/<venue_id>_<alias>_p<n>.json)
  └─ applies tag_modalities()  (existing lexicon-based extractor)
  └─ writes to sensory.db      (INSERT OR IGNORE)

build_venue_explorer.py        (rerun — no structural changes)
  └─ reads updated sensory.db
  └─ writes venue_explorer.html
```

Running: `python3 gazetteer/extract_old_bailey.py`

---

## Section 1: New Venues

~25 venues added to `venues.csv` in five categories, all with London coordinates and aliases drawn from period nomenclature.

### Prisons
| ID | Name | Notes |
|----|------|-------|
| LON074 | Newgate Prison | Primary criminal prison; adjacent to Old Bailey courthouse |
| LON075 | Fleet Prison | Debtors' prison; Fleet Street area |
| LON076 | Marshalsea Prison | Debtors' prison; Southwark |
| LON077 | Bridewell Prison | House of correction; Blackfriars |
| LON078 | King's Bench Prison | Southwark; debtors and misdemeanours |
| LON079 | Tothill Fields Bridewell | Westminster house of correction |

### Courts
| ID | Name | Notes |
|----|------|-------|
| LON080 | Old Bailey Courthouse | Sessions House; central criminal court |
| LON081 | Bow Street Magistrates Court | Fielding's court; founded 1739 |

### Markets
| ID | Name | Notes |
|----|------|-------|
| LON082 | Smithfield Market | Livestock market; notorious smell and noise |
| LON083 | Billingsgate Fish Market | Lower Thames Street; fish smell |
| LON084 | Leadenhall Market | Poultry and general produce |
| LON085 | Covent Garden Market | Fruit, vegetables, flowers |

### Rookeries and Street Areas
| ID | Name | Notes |
|----|------|-------|
| LON086 | St Giles / Seven Dials | Dense slum; Irish population |
| LON087 | Whitechapel | East End; crowded and industrial |
| LON088 | Spitalfields | Huguenot weavers; silk industry |
| LON089 | Wapping | Riverside; sailors and dockhands |
| LON090 | Cheapside | Major commercial thoroughfare |
| LON091 | Holborn | Mixed; lawyers' quarter merging with slums |
| LON092 | Fleet Street | Print trade, taverns |
| LON093 | Southwark / Borough | South bank; tanneries, brewing |

### Execution Sites
| ID | Name | Notes |
|----|------|-------|
| LON094 | Tyburn Gallows | Public executions; Oxford Street area |
| LON095 | Newgate Gallows | Post-1783 executions outside Newgate |

---

## Section 2: `extract_old_bailey.py` Architecture

### API
Endpoint: `https://www.dhi.ac.uk/api/data/oldbailey_record?text=<alias>&_limit=10&_offset=<n>`

Returns JSON. Each record contains:
- `reference`: trial reference (e.g. `t17650116-1`)
- `date`: integer `YYYYMMDD`
- `text`: full trial transcript

### Query loop
For each venue, iterate its aliases. For each alias, paginate until response count < page size.

### Caching
Responses cached as JSON at `gazetteer/sources/legal/<venue_id>_<alias_slug>_p<n>.json`. Re-runs skip the network; delete cache to re-fetch.

### Date filtering
Only trials dated 1660–1820 are inserted. Date parsed from the integer `YYYYMMDD` field.

### Sensory extraction
Trial text passed through existing `tag_modalities()` from `extract_sensory.py`. No new NLP required.

### Venue assignment
Since we searched by alias, `venue_id` is known directly — `geocode_passage` is bypassed entirely.

### `source_id`
`old_bailey_<reference>` (e.g. `old_bailey_t17650116-1`). The UNIQUE constraint on `(source_id, char_offset, modality)` handles deduplication on re-runs.

### Rate limiting
0.3s sleep between API requests (skipped for cache hits).

---

## Section 3: Schema Update — `valence` Field

**Motivation:** The nullschool-style environmental overlay (future Phase 4/5) needs a quantified affective register per passage. Storing it now avoids a costly re-extraction pass later.

**New column:** `valence TEXT` — one of `pleasant`, `neutral`, `unpleasant`. Nullable; defaults to `NULL`.

**Tagging logic:**
- Old Bailey passages: default `unpleasant` (prosecution implies threshold crossed)
- Fiction/diary/topography: inferred from a small valence lexicon
  - Unpleasant: *stench, reek, fetid, din, hubbub, squalor, filth, effluvia, noisome, press, jostle*
  - Pleasant: *fragrant, sweet, music, harmony, elegant, charming, gay, pleasant*
  - Neutral: everything else

**Migration:** `ALTER TABLE sensory_evidence ADD COLUMN valence TEXT` — backward-compatible.

---

## Section 4: Venue Explorer Updates

Three changes to `build_venue_explorer.py` / the generated `venue_explorer.html`:

1. **"Legal" source-type filter pill** — added to the top-bar source row alongside Fiction, Diary, Topography, Poetry, Letters. Toggle behaviour identical; active by default.

2. **Legal badge colour** — crimson: background `#f5e0e0`, text `#8b1a1a`. Visually distinct and tonally appropriate.

3. **Valence indicator on evidence cards** — subtle pip in the card's top-right corner:
   - `unpleasant`: muted red dot
   - `pleasant`: muted gold dot
   - `neutral` / `NULL`: no pip

4. **Rebuild** — `python3 gazetteer/build_venue_explorer.py` regenerates `venue_explorer.html` after DB is populated. No structural changes to the build script.

---

## Files

| File | Action |
|------|--------|
| `gazetteer/venues.csv` | Add ~25 new venues |
| `gazetteer/sensory_db.py` | Add `valence TEXT` column; migration script |
| `gazetteer/extract_old_bailey.py` | New file — API querying, caching, extraction |
| `gazetteer/extract_sensory.py` | Add valence tagging to `tag_modalities()` |
| `gazetteer/build_venue_explorer.py` | Legal pill + badge colour + valence pip |
| `gazetteer/sources/legal/` | New directory — cached API responses |
| `gazetteer/tests/test_extract_old_bailey.py` | New test file |

---

## Out of Scope (Phase 4/5)

- Nullschool-style wind/smell diffusion overlay (requires environmental rasters)
- CET temperature integration
- Bills of Mortality parish layer
- Smoke burden estimate from coal consumption
- "Rural baseline" contrast view (Austen/Radcliffe countryside passages as sensory counter-register)

---

## Longer-Term Vision

The experiential framing — *what would a newcomer from the countryside have experienced standing at Smithfield on a July afternoon?* — requires layering:

1. Textual evidence (what sources say about sensory qualities) ← Phase 1 + Phase 2
2. Environmental conditions (temperature, wind direction, humidity) ← Phase 4
3. Smell/sound diffusion from known point sources (slaughterhouses, tanneries, coal fires, coffeehouses) ← Phase 4
4. A composite "sensory intensity" score per venue per time period ← Phase 5 UI

The valence field added in Phase 2 seeds step 4. No further schema changes should be needed for the overlay.
