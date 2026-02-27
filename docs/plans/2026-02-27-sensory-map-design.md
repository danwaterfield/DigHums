# Sensory-Historical Map of 18th-Century London and Bath
## Design Document — 2026-02-27

---

## Vision

A multi-source, venue-anchored interface for exploring the sensory experience of 18th-century urban space. Every venue in the gazetteer accumulates evidence from independent, heterogeneous sources — fiction, diary, legal record, periodical, topographical survey, environmental data — each filterable without any one being treated as foundational or causal. The discrepancies between sources are where the scholarship lives.

Two primary modes:

1. **Venue explorer** — "What was Vauxhall Gardens like on a busy evening in 1765?" Select a venue, set a time window, filter by sensory modality and source type. Browse assembled evidence; toggle environmental layers.

2. **Narrative path mode** — the existing narrative map, enriched with actual street routing (replacing straight lines) and sensory annotations drawn from all sources along the route.

Both modes share the same underlying data. Neither is primary.

---

## Core Principle

No data layer is treated as foundational or causal. Economic geography, geology, climate, and literary representation are co-equal evidence streams. The interface enables correlation and divergence analysis without pre-committing to a hierarchy. Where the Bills of Mortality record fever clustering in Southwark and the novels ignore it, that divergence is surfaced as an observation, not explained away.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Leaflet Interface                  │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │ Venue Explorer│  │Narrative Path│  │ Layer Ctrl│  │
│  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘  │
└─────────┼────────────────┼────────────────┼─────────┘
          │                │                │
┌─────────▼────────────────▼────────────────▼─────────┐
│                  Evidence Store                      │
│  venue_id │ date_range │ modality │ source_type      │
│  author   │ text       │ coords   │ confidence       │
└─────────────────────────┬───────────────────────────┘
                          │
        ┌─────────────────┼──────────────────┐
        │                 │                  │
┌───────▼──────┐  ┌───────▼──────┐  ┌───────▼──────┐
│ Textual      │  │ Sensory NLP  │  │ Environmental │
│ Ingest       │  │ Pipeline     │  │ Rasters       │
│ Pipeline     │  │              │  │               │
└──────────────┘  └──────────────┘  └──────────────┘
```

---

## Data Schema

### `sensory_evidence` table (SQLite or JSON)

```
venue_id        str       — links to venues.csv (e.g. LON005)
venue_name      str       — denormalised for convenience
lat, lon        float
source_type     enum      — fiction | diary | periodical | legal |
                            topography | poetry | environmental
author          str       — CamelCase, matches corpus convention
title           str
pub_year        int       — year of publication
date_min        int       — earliest plausible date of described scene
date_max        int       — latest plausible date of described scene
modality        enum      — auditory | olfactory | visual | tactile |
                            thermal | crowd | economic | unclassified
text            str       — extracted passage (up to ~500 chars)
context         str       — surrounding sentence for readability
char_offset     int       — position in source text (for narrative use)
pos             float     — 0–1 narrative position (fiction only)
confidence      float     — extraction confidence (0–1)
notes           str       — curatorial notes
```

The `date_min`/`date_max` fields acknowledge that a passage published in 1778 may describe events set in 1750 — particularly relevant for Burney's diary (written contemporaneously) vs. her fiction (retrospectively shaped).

### `environmental_layers` (GeoJSON/GeoTIFF per layer)

```
layer_id        str       — e.g. "cet_temperature_1750", "bills_fever_1750s"
layer_type      enum      — climate | geology | mortality | economic |
                            smoke | morphological
date_min        int
date_max        int
description     str
source          str
spatial_res     str       — e.g. "parish", "grid_500m", "point"
```

---

## Data Sources

### Textual (to ingest)

| Source | Type | Period | Location | Notes |
|--------|------|--------|----------|-------|
| Novel corpus (existing) | fiction | 1719–1817 | corpus/ | already parsed |
| Spectator, Tatler (existing) | periodical | 1709–14 | non-fiction corpus | already ingested |
| Defoe, *Tour Through Great Britain* | topography | 1724–26 | Gutenberg | foundational survey |
| Gay, *Trivia* | poetry | 1716 | Gutenberg | sensory poem about London streets |
| Evelyn, *Fumifugium* | environmental | 1661 | Gutenberg | first anti-pollution tract; pre-period |
| Pennant, *Of London* | topography | 1790 | Gutenberg | late-century survey |
| Burney, *Diary and Letters* | diary | 1768–1839 | Gutenberg (multi-vol) | same author as fiction; direct comparison |
| Boswell, *London Journal* | diary | 1762–63 | Gutenberg | granular sensory detail |
| Walpole letters | letters | 1732–1797 | Gutenberg (partial) | venue-specific observations |
| Pepys diary | diary | 1660–69 | Gutenberg | pre-period; invaluable sensory baseline |
| Johnson, *Rambler* + *Idler* | periodical | 1750–60 | Gutenberg | urban observation essays |
| Anstey, *New Bath Guide* | poetry | 1766 | Gutenberg | Bath-specific, satirical, sensory |
| Wood, *Essay on Bath* | topography | 1749 | Archive.org | written by Bath's architect |
| Old Bailey Proceedings | legal | 1674–1820 | oldbaileyonline.org API | plebeian/nocturnal; extraordinary detail |

### Environmental (non-textual)

| Source | Type | Period | Access |
|--------|------|--------|--------|
| Central England Temperature series | climate | 1659–present | Met Office (free download) |
| Bills of Mortality | mortality/epidemiological | 1603–1840 | London Lives / digitized |
| British Geological Survey | geology | static | BGS (free WMS) |
| Coal consumption records | smoke proxy | 1700s–1800s | House of Commons papers |
| Parish rate books | economic | by decade | British History Online (partial) |
| Rocque 1746 / Horwood 1792–99 | morphological | 1746 / 1799 | already in map (tiles) |

---

## Sensory NLP Pipeline

Extraction proceeds in two passes:

**Pass 1 — Lexicon-based**
Curated vocabulary lists per modality, drawing on period-specific terms:
- *Auditory*: din, clatter, rattling, cries, huzza, hubbub, discord, tolling, rumble
- *Olfactory*: stench, effluvia, perfume, reek, vapour, odour, fetid, fragrant, smoke
- *Visual*: glare, gloom, throng, dazzling, murky, smoky, illuminated, narrow, lofty
- *Thermal*: sultry, damp, raw, fog, mist, frost, close (air)
- *Crowd/density*: press, mob, jostle, thronged, crammed, deserted, sparsely

**Pass 2 — Embedding-based**
Use the existing sentence-transformer infrastructure (all-MiniLM-L6-v2 already installed) to find passages semantically similar to seed sentences per modality, catching oblique or euphemistic sensory language that the lexicon misses.

Geocoding: passages are assigned to venues either by co-occurrence with a known venue alias (within ±500 words) or, for topographical texts, by explicit street/district mention matched against venues.csv.

---

## Interface Design

### Venue Explorer

- Click any venue marker → side panel opens
- Panel tabs: **All evidence** | **Auditory** | **Olfactory** | **Visual** | **Environmental**
- Each tab shows filterable cards: source type badge, author, date range, excerpt
- Time range slider at top filters all cards simultaneously
- "Compare sources" button: two-column view of fiction vs. non-fiction for the same venue and period
- Divergence highlight: passages rated as diverging from environmental baseline flagged with a subtle indicator (not prescriptive — just surfaces the gap for the scholar to interpret)

### Environmental Layers (map toggles)

- CET temperature anomaly (decade averages, choropleth)
- Bills of Mortality fever index (parish-level choropleth)
- Smoke burden estimate (point-source diffusion from known coal use, prevailing westerlies)
- Geological substrate (Thames alluvium, gravel terraces, clay)
- Economic tier (rate book assessments, simplified to 5 tiers)
- Street morphology (Rocque-derived street width proxy — already have the tiles)

Each layer is independently toggleable; none is set as default-on. No layer is described as "causing" anything in the UI copy.

### Narrative Path Mode (extension of existing)

- Replace straight-line connections with OSM-routed paths (using OSRM or Leaflet Routing Machine), displayed on the Rocque/Horwood tile layer
- Along each route segment, surface sensory evidence cards from all sources describing that street or area
- Environmental layer values sampled along the route (e.g. "this stretch passes through a high-smoke-burden zone in the 1770s estimate")
- Toggle: "show fiction only" / "show all sources" / "show divergences"

---

## Build Phases

### Phase 1 — Evidence schema and initial ingestion
- Define SQLite schema (or JSON equivalent) for `sensory_evidence`
- Ingest Gutenberg sources: Defoe Tour, Gay Trivia, Burney Diary vols 1–2, Boswell London Journal, Evelyn Fumifugium, Pennant, Anstey
- Write `ingest_sources.py` following the existing corpus/ conventions
- Build lexicon-based extractor (Pass 1) — fast, auditable, no model needed
- Populate evidence store; verify with manual spot-checks against 3–4 key venues (Vauxhall, Ranelagh, King's Theatre, Pulteney Street)

### Phase 2 — Old Bailey integration
- Query Old Bailey API for venue-adjacent proceedings (by street/parish)
- Parse and ingest into evidence store with `source_type = legal`
- Note: Old Bailey XML is well-structured; extraction is more parsing than NLP

### Phase 3 — Embedding-based extraction (Pass 2)
- Extend existing `embedding_baseline.py` infrastructure for sensory passage retrieval
- Run against full corpus + new textual sources
- Merge with Pass 1 results; deduplicate

### Phase 4 — Environmental layers
- Download and process CET temperature series
- Source Bills of Mortality parish data
- Build smoke burden estimate (simple diffusion model from coal consumption)
- Package as GeoJSON layers for Leaflet

### Phase 5 — Interface
- Venue explorer panel (extends existing Folium/Leaflet map)
- Environmental layer toggles
- Route enrichment (OSRM integration for narrative path mode)
- Time range filter

---

## What This Enables Analytically

- **Authorial selectivity**: does Burney's fictional Vauxhall match her diary Vauxhall? Where she diverges, why — social propriety, genre convention, narrative function?
- **Genre and sensory register**: Smollett amplifies filth and noise; Austen suppresses it entirely. The environmental baseline lets you quantify the gap rather than just assert it.
- **The westward drift**: Mayfair sits upwind and uphill of the City. The smoke burden and mortality data encode this; the fiction may or may not. That spatial-social correspondence (or divergence) is mappable.
- **Bath as therapeutic counter-space**: Bramble's body in *Humphry Clinker* as barometer. The CET data and Bath's morphology (wide streets, limestone, uphill site) provide the environmental argument.
- **Temporal change**: venues change over time. Vauxhall in 1740 (high social status) is different from Vauxhall in 1790 (middling, declining). The evidence store's date fields let you track that.

---

## Open Questions

- **OSRM vs. Leaflet Routing Machine** for street routing — OSRM requires a local server; LRM queries OSRM's public API (rate-limited). For a small number of routes, LRM public API is probably fine.
- **Period accuracy of OSM routing**: central London's street network is largely stable since the 18th century but Regent Street (1820s) and Victoria Embankment (1860s) are post-period. Routes through those areas should be flagged or avoided.
- **Bills of Mortality digitisation quality**: some decades are well-digitized (London Lives), others require manual transcription. Phase 4 scope may need trimming.
- **Vectorisation / semantic comparison work** (separate but related): the embedding-based author/work comparison discussed in parallel should share the sentence-transformer infrastructure built in Phase 3.
