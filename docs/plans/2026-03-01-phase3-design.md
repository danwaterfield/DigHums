# Phase 3: Temporal Events Layer — Design Document
## 2026-03-01

---

## Goal

Add a time-parameterised sensory layer to the project: a curated dataset of recurring events (weekly markets, annual fairs, executions, civic processions, seasonal venue operations) that modulate the sensory environment of each venue across the day, week, and year. Events are modelled as structured data with per-modality intensity loads; specific attested occurrences are encoded as instances tied to sources. A new interactive map — `sensory_time_map.html` — lets users set a year, month, day-of-week, and time-of-day band, and see which venues were sensorially active and why. A literary comparison layer overlays the existing textual evidence for contrast.

---

## Motivation

The current evidence store (Phases 1–2) answers the question *what do sources say about sensory experience at this venue?* Phase 3 answers the complementary question: *what was institutionally happening at this venue at a given moment, regardless of whether a novelist or diarist happened to record it?*

The gap between the two registers is analytically valuable. Smithfield on a Friday morning is predicted by the institutional record to be one of the loudest, most pungent places in London. When Burney and Smollett describe it, they confirm this — but from a literary register that inflects the sensory with social anxiety, class disgust, and narrative purpose. The time map makes that gap visible.

Events also cover spaces the literary corpus ignores. No novelist describes the smell of Billingsgate at dawn. The execution crowd at Tyburn appears only obliquely in fiction. The institutional layer reaches where literary evidence does not.

This work lays the data foundation for Phase 4/5 (CET temperature integration, point-source diffusion modelling), which will add environmental columns to the same event rows rather than redesigning the schema.

---

## Key Secondary Sources

The curated events dataset is grounded in the following scholarship:

- Peter Linebaugh, *The London Hanged: Crime and Civil Society in the Eighteenth Century* (Cambridge UP, 1992; Verso paperback 2006). ISBN 9780521457583. Standard account of Tyburn executions and their social function.
- Vic Gatrell, *The Hanging Tree: Execution and the English People, 1770–1868* (Oxford UP, 1994). ISBN 9780192853325. Covers the Tyburn–Newgate transition (1783) and execution crowd culture.
- Roy Porter, *London: A Social History* (Hamish Hamilton, 1994; Penguin 1996). ISBN 9780140105933. Wide-ranging survey covering markets, pleasure gardens, street life.
- Jerry White, *London in the Eighteenth Century: A Great and Monstrous Thing* (Bodley Head, 2012; Vintage 2013). ISBN 9781847921802. Most comprehensive single-volume treatment of 18th-century London trade, markets, and urban experience.
- David Coke & Alan Borg, *Vauxhall Gardens: A History* (Yale UP for Paul Mellon Centre, 2011). ISBN 9780300173826. Definitive monograph; season dates, admission prices, programme.
- Malcolm Thick, *The Neat House Gardens: Early Market Gardening Around London* (Prospect Books, 1998). ISBN 9780907325789. Supply hinterland for Covent Garden and London markets.
- William Andrews, *Famous Frosts and Frost Fairs in Great Britain* (1887). Free via Project Gutenberg. Primary Victorian compilation; exact dates for all major Frost Fairs.
- Alain Corbin, *Village Bells: Sound and Meaning in the Nineteenth-Century French Countryside* (Columbia UP, 1998). Methodological touchstone for bells and acoustic territorial identity; cited in Burney Notes research.
- Daniel Defoe, *A Tour Through the Whole Island of Great Britain* (1724–26). Primary source for Smithfield scale: "without question, the greatest [market] in the world."
- Nicholas Rogers, *Crowds, Culture and Politics in Georgian Britain* (Oxford UP, 1998). Crowd behaviour, riot ritual, and the sensory character of public assembly.
- Holger Hoock, *Empires of the Imagination: Politics, War, and the Arts in the British World, 1750–1850* (Profile Books, 2010). Documents Handel Commemoration (1784) and state ceremonial.
- Museum of London, *Sensory Smithfield* research report (2017). Available at sensorysmithfield.com. Direct sensory history study of the market.
- Capital Punishment UK, *Overview of Executions at Tyburn 1735–1783*. Online database; confirms weekday patterns (predominantly Monday).
- Old Bailey Online, *The Journey from Newgate to Tyburn*. Contextual essay; procession route and timing.

---

## Section 1: Data Model

### New files

**`gazetteer/events.csv`** — recurring event types.

| Field | Type | Notes |
|---|---|---|
| `event_id` | TEXT | e.g. `EVT001` |
| `name` | TEXT | e.g. "Smithfield Weekly Market" |
| `category` | TEXT | `weekly_market`, `annual_fair`, `execution`, `civic_procession`, `seasonal_operation`, `cultural_event`, `frost_fair` |
| `month_start` | INT 1–12 | NULL = year-round |
| `month_end` | INT 1–12 | NULL = year-round |
| `day_of_week` | TEXT | `Mon`, `Mon,Fri`, NULL = any day |
| `time_bands` | TEXT | pipe-separated subset of `dawn,morning,midday,afternoon,evening,night` |
| `year_start` | INT | first year active within 1660–1820 |
| `year_end` | INT | last year active within 1660–1820 |
| `recurrence` | TEXT | `weekly`, `annual`, `irregular`, `one_off` |
| `smell_load` | REAL 0–1 | curated estimate |
| `noise_load` | REAL 0–1 | curated estimate |
| `crowd_load` | REAL 0–1 | curated estimate |
| `visual_load` | REAL 0–1 | curated estimate |
| `calendar_break` | INT | year of OS→NS calendar shift affecting this event (1752 for most); NULL if inapplicable |
| `month_start_ns` | INT | post-1752 month_start if different |
| `notes` | TEXT | historiographical caveats; contested interpretations |
| `sources` | TEXT | semicolon-separated citation strings |

**`gazetteer/event_venues.csv`** — join table linking events to venues (many-to-many).

| Field | Type |
|---|---|
| `event_id` | TEXT |
| `venue_id` | TEXT |

**`gazetteer/event_instances.csv`** — attested specific occurrences.

| Field | Type | Notes |
|---|---|---|
| `instance_id` | TEXT | e.g. `INS001` |
| `event_id` | TEXT | FK to events.csv |
| `year` | INT | |
| `month` | INT | |
| `day` | INT | NULL if only month known |
| `source_id` | TEXT | FK to `sources` table, or free-text external ref |
| `notes` | TEXT | what the source says |

### New DB tables (added to `sensory_db.py`)

```sql
CREATE TABLE IF NOT EXISTS events (
    event_id      TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    category      TEXT NOT NULL,
    month_start   INTEGER,
    month_end     INTEGER,
    day_of_week   TEXT,
    time_bands    TEXT NOT NULL,
    year_start    INTEGER,
    year_end      INTEGER,
    recurrence    TEXT NOT NULL,
    smell_load    REAL DEFAULT 0,
    noise_load    REAL DEFAULT 0,
    crowd_load    REAL DEFAULT 0,
    visual_load   REAL DEFAULT 0,
    calendar_break INTEGER,
    month_start_ns INTEGER,
    notes         TEXT,
    sources       TEXT
);

CREATE TABLE IF NOT EXISTS event_venues (
    event_id   TEXT REFERENCES events(event_id),
    venue_id   TEXT,
    PRIMARY KEY (event_id, venue_id)
);

CREATE TABLE IF NOT EXISTS event_instances (
    instance_id  TEXT PRIMARY KEY,
    event_id     TEXT REFERENCES events(event_id),
    year         INTEGER,
    month        INTEGER,
    day          INTEGER,
    source_id    TEXT,
    notes        TEXT
);
```

### Migration on `sensory_evidence`

`ALTER TABLE sensory_evidence ADD COLUMN event_id TEXT` — nullable FK. Populated only for the small curated set of passages that demonstrably describe event-day conditions. The existing migration guard in `init_db()` handles this.

---

## Section 2: Curated Events Dataset

### Time bands

Six bands, replacing an hourly slider:

| Band | Hours | Rationale |
|---|---|---|
| Dawn | 04:00–07:00 | Wholesale markets open; drovers arrive |
| Morning | 07:00–12:00 | Market and execution peak |
| Midday | 12:00–15:00 | Market winding down; crowds dispersing |
| Afternoon | 15:00–18:00 | Street life; transition period |
| Evening | 18:00–22:00 | Pleasure gardens open; theatre |
| Night | 22:00–04:00 | Late entertainment; drovers arriving for Monday market |

### Weekly markets

| event_id | name | Venues | Days | Bands | smell / noise / crowd | Sources |
|---|---|---|---|---|---|---|
| EVT001 | Smithfield Weekly Market | LON082 | Mon, Fri | night, dawn, morning | 1.0 / 0.9 / 0.9 | Defoe, *Tour* (1724–26); White; MoL Sensory Smithfield report |
| EVT002 | Billingsgate Fish Market | LON083 | Mon–Sat | dawn, morning | 1.0 / 0.8 / 0.7 | 1698 Act of Parliament; White; City of London official history |
| EVT003 | Leadenhall Market | LON084 | Mon–Fri | dawn, morning, midday | 0.7 / 0.6 / 0.6 | *Old and New London* (BHO); Porter |
| EVT004 | Covent Garden Market | LON085 | Tue, Thu, Sat | night, dawn, morning | 0.5 / 0.7 / 0.8 | *Old and New London* (BHO); Thick, *Neat House Gardens* |

Note on Smithfield: Defoe (1724–26) records it as "without question, the greatest [market] in the world." Drovers with livestock arrived Sunday night before the Monday market, making the Night band active from the Sunday. White, *London in the 18th Century*, describes the immense quantities of animal waste deposited in surrounding streets. Smell load set to maximum (1.0).

### Annual fairs

Note on calendar reform: the Calendar (New Style) Act 1750 (effective September 1752) dropped 11 days. Fairs were legally exempted from the new calendar to preserve their seasonal position, so many shifted their calendar date. Bartholomew Fair moved from 24 August (OS) to 3 September (NS). This is encoded via `calendar_break=1752`, `month_start_ns=9`.

| event_id | name | Venues | Period (OS/NS) | Duration | smell / noise / crowd | Active | Sources |
|---|---|---|---|---|---|---|---|
| EVT010 | Bartholomew Fair | LON082 | Aug 24 (pre-1752) / Sep 3 (post-1752) | ~4 days officially; ~14 in practice | 0.9 / 1.0 / 1.0 | 1660–1855 | *Old and New London* (BHO vol.2); History Today; Wikipedia |
| EVT011 | Southwark Fair | LON093 | Sep 7 charter; expanding to ~2 weeks | ~2 weeks | 0.8 / 0.9 / 0.9 | 1660–1762 | Georgian Cities (Sorbonne); Past Tense; Hogarth (1733) |
| EVT012 | Greenwich Fair | off-map | Easter (3 days) + Whitsun | 3 days each | 0.3 / 0.8 / 1.0 | 1660–1857 | Friends of Greenwich Park |
| EVT013 | May Fair | LON area | First 2 weeks of May | 2 weeks | 0.7 / 0.9 / 0.9 | 1660–1708 | Porter; White |

### Frost Fairs (irregular — encoded as `event_instances` only, not recurring)

| instance_id | event_id | Year | Dates | Duration | Notes | Source |
|---|---|---|---|---|---|---|
| INS001 | EVT020 | 1684 | Jan 5 – Feb 6 | ~6 weeks | Booths, printing presses, horse racing on ice; ice 11 inches thick | Andrews, *Famous Frosts*; History Today |
| INS002 | EVT020 | 1716 | Dec 1715 – Feb 1716 | several weeks | "Great cook's shop" on ice | Andrews |
| INS003 | EVT020 | 1740 | Dec 25 1739 – Feb 17 1740 | ~7 weeks | Printed broadsides survive dated Jan 14 and Feb 5 1740 | Andrews |
| INS004 | EVT020 | 1789 | Feb 1789 | brief | Melting ice carried away ship at Rotherhithe | Andrews |
| INS005 | EVT020 | 1814 | Jan 30 – Feb 5 | ~6 days | Elephant led across ice near Blackfriars; last Frost Fair | Andrews; Regency History |

### Executions

| event_id | name | Venues | Schedule | Bands | noise / crowd / visual | Active | Sources |
|---|---|---|---|---|---|---|---|
| EVT030 | Tyburn Hanging Day | LON094 | ~8x/year, almost always Mon | morning | 0.4 / 1.0 / 1.0 | 1660–Nov 3 1783 | Linebaugh; Gatrell; Capital Punishment UK |
| EVT031 | Newgate Gallows Execution | LON095 | more frequent post-1783, Mon | morning | 0.3 / 0.9 / 0.9 | 1783–1820 | Gatrell; Linebaugh |
| EVT032 | Tyburn Procession (route) | LON074, LON086, LON090 | same days as EVT030 | morning | — / 0.8 / 0.8 | 1660–1783 | Old Bailey Online: "Journey from Newgate to Tyburn" |

Notes: Linebaugh (*The London Hanged*) documents ~8 hanging days per year. Capital Punishment UK's decade-by-decade tables confirm Monday as near-invariable. Crowds of 10,000–50,000 documented for celebrity executions. The three-mile procession from Newgate to Tyburn via Holborn, St Giles, and Oxford Street (the "Tyburn Road") took up to three hours; crowd loads are assigned to the entire procession route (EVT032).

### Seasonal venue operations

These set a baseline: is a venue sensorially active at all in a given period?

| event_id | name | Venues | Season | Bands | smell / noise / crowd | Active | Sources |
|---|---|---|---|---|---|---|---|
| EVT040 | Vauxhall Gardens Open Season | LON001 | Late May – Sep (Mon/Wed/Fri evenings; daily earlier) | evening, night | 0.4 / 0.9 / 0.8 | 1660–1859 | Coke & Borg, *Vauxhall Gardens: A History* |
| EVT041 | Ranelagh Gardens Open Season | LON002 | Easter – Aug (Mon/Wed/Fri) | evening, night | 0.3 / 0.8 / 0.7 | 1742–1803 | Wikipedia; Digitens |
| EVT042 | Marylebone Gardens Open Season | LON003 | Spring – Autumn | evening | 0.3 / 0.7 / 0.6 | 1660–1778 | Porter; White |

Notes on Vauxhall: 1 shilling admission from 1732 (Tyers's reopening); closed Sundays from c.1760s. Season opening date described in *Microcosm of London* (1808–10) as "about the latter end of May." Smell load is lower than markets — pleasure gardens were designed to suppress urban odour through water features, scented walks, and distance from the City.

### Civic processions and annual ceremonies

| event_id | name | Venues | Date | Bands | crowd / visual | Notes | Sources |
|---|---|---|---|---|---|---|---|
| EVT050 | Lord Mayor's Show | LON048, LON080 | Oct 29 (pre-1752) / Nov 9 (post-1752) | morning, midday | 0.9 / 1.0 | Route varied annually through ward of new Mayor | Lord Mayor's Show official site; Wikipedia |
| EVT051 | Gunpowder Plot (Guy Fawkes) | city-wide | Nov 5 | evening, night | 0.7 / 0.8 | Bell-ringing, bonfires; contested (Corbin on bells as territorial sound); Tory/Whig inflection | Corbin, *Village Bells*; Rogers, *Crowds* |

### One-off attested instances

| instance_id | event_id | Date | Venue | Notes | Source |
|---|---|---|---|---|---|
| INS010 | EVT060 | May 26, 1784 | Westminster Abbey | Handel Commemoration Concert; "greatest musical event of the century"; unprecedented news coverage | Hoock, *Empires of the Imagination*; Burney research notes |
| INS011 | EVT060 | 1784 | LON004 (Pantheon) | Second concert in series; Evelina's narrator described Pantheon as "more the appearance of a chapel than a place of diversion" | Hoock; Burney, *Evelina* |
| INS012 | EVT061 | Jun 2–9, 1780 | City-wide | Gordon Riots; crowds rang bells "upon arrival"; forced illumination of houses; chapels attacked | Rogers, *Crowds, Culture and Politics in Georgian Britain*; Burney research notes |

---

## Section 3: UI — Sensory Time Map

### File

`gazetteer/sensory_time_map.html` — generated by `gazetteer/build_sensory_time_map.py`. Self-contained static HTML with embedded JSON; deploys to GitHub Pages. Cross-links to `venue_explorer.html`.

### Layout

```
┌──────────────────────────────────────────────────────────────┐
│  SENSORY TIME MAP   [1660 ────●────────── 1820]  [← step →] │
│  [Jan][Feb][Mar][Apr][May][Jun][Jul][Aug][Sep][Oct][Nov][Dec] │
│  [Mon][Tue][Wed][Thu][Fri][Sat][Sun]                         │
│  [Dawn][Morning][Midday][Afternoon][Evening][Night]          │
│                                    [☰ Literary layer  OFF→] │
├────────────────────────────────┬─────────────────────────────┤
│                                │  ACTIVE EVENTS              │
│       LEAFLET MAP              │  ─────────────────────────  │
│                                │  Smithfield Weekly Market   │
│  ◉ large red = high intensity  │  smell ████████████ 1.0     │
│  ○ small grey = inactive       │  noise ███████████░ 0.9     │
│                                │  crowd ███████████░ 0.9     │
│  [hover → tooltip]             │  Mon–Fri · Dawn · Morning   │
│  [click → panel narrows]       │  ─────────────────────────  │
│                                │  [other active events...]   │
└────────────────────────────────┴─────────────────────────────┘
```

### Map layer

Each venue renders as a Leaflet `circleMarker`. Radius encodes `crowd_load` (range 4px–18px). Fill colour encodes composite intensity on a grey → amber → red scale. Venues with no active events at the selected time are shown as small grey dots — present, but quiet. Hovering shows a tooltip: venue name + list of active event names. Clicking narrows the right panel to that venue.

### Time controls

- **Year slider** (1660–1820): draggable; step buttons (← →) move by decade.
- **Month pills** (Jan–Dec): single-select; clicking a month activates it.
- **Day-of-week pills** (Mon–Sun): single-select.
- **Time-band pills** (Dawn / Morning / Midday / Afternoon / Evening / Night): single-select.
- Selecting "any" (no pill active) for day or band means "show maximum intensity across all possibilities" — useful for a quick overview.

### Right panel

**Default — "Active events" across all venues** at the selected time. Each entry: event name, venue name, per-modality intensity bars, brief description, sources cited. Collapsed by default; click to expand.

**On venue click** — narrows to that venue. Shows:
1. Event cards (from events table matching the time query)
2. Event instance cards (from event_instances matching the year ± tolerance)
3. If literary layer is on: evidence passage cards from `sensory_evidence` (filtered by `venue_id`; sorted by proximity of `pub_year` to selected year)

### Literary comparison layer

Toggle button. When active, adds a second set of Leaflet markers to the map: small coloured dots at each venue (coloured by dominant modality — same palette as `venue_explorer.html`). The contrast between the institutional prediction (event layer) and the textual witness (evidence layer) is the analytical product. Gap = what writers chose not to notice, or noticed differently.

### Intensity computation

Runs entirely in JavaScript at interaction time. The HTML embeds:
- `EVENTS` JSON (~40 rows)
- `VENUES` JSON (~95 rows with coordinates)
- `EVENT_VENUES` JSON (join table)
- `EVENT_INSTANCES` JSON
- `EVIDENCE` JSON (passages with venue_id, for the literary layer)

`computeIntensity(venueId, year, month, dow, band)` iterates matching events, sums loads, clamps to 1.0. Instant response; no pre-computation or server required.

**Calendar reform handling:** Events with `calendar_break=1752` use `month_start`/`month_end` for `year < 1752` and `month_start_ns`/`month_end` for `year >= 1752`. The JS function applies this transparently.

---

## Section 4: Architecture

### New files

| File | Purpose |
|---|---|
| `gazetteer/events.csv` | ~40 recurring event types |
| `gazetteer/event_venues.csv` | Event–venue join table |
| `gazetteer/event_instances.csv` | Attested specific occurrences |
| `gazetteer/extract_events.py` | Loads CSVs into sensory.db; idempotent |
| `gazetteer/build_sensory_time_map.py` | Generates sensory_time_map.html |
| `gazetteer/sensory_time_map.html` | Generated output (not committed; built on deploy) |
| `gazetteer/tests/test_extract_events.py` | Schema + load tests |
| `gazetteer/tests/test_build_sensory_time_map.py` | HTML generation tests |

### Modified files

| File | Change |
|---|---|
| `gazetteer/sensory_db.py` | Add `events`, `event_venues`, `event_instances` DDL; migration adds `event_id TEXT` to `sensory_evidence` |
| `gazetteer/venue_explorer.html` | Add "Time map →" link in top bar |

### Build pipeline

```
events.csv + event_venues.csv + event_instances.csv
    └─ extract_events.py ──────────► sensory.db

sensory.db
    ├─ build_venue_explorer.py ────► venue_explorer.html      (existing)
    └─ build_sensory_time_map.py ──► sensory_time_map.html    (new)
```

---

## Out of Scope (Phase 4/5)

- CET temperature records and seasonal temperature overlay
- Wind direction / smell diffusion modelling from point sources
- Composite "sensory intensity" score incorporating environmental data
- Bills of Mortality integration (death clustering as sensory/demographic signal)
- Smoke burden estimates from coal consumption data
- Rural baseline contrast view

The `events` schema is designed for Phase 4/5 extensibility: add `temp_modifier REAL` and `wind_sensitivity TEXT` columns to `events` without restructuring the table. The intensity computation function in JS is written to accept additional modifiers.

---

## Historiographical Note

Sensory experience in 18th-century London was ideologically contested, not merely ambient. Alain Corbin (*Village Bells*) established that acoustic events — bell-ringing, market cries, crowd noise — shaped community identity and territorial belonging differently for different social groups. Nicholas Rogers (*Crowds, Culture and Politics*) shows that crowd events followed ritual scripts legible to participants but opaque to later observers. The `notes` field on each event row encodes these contested meanings where relevant: Gunpowder Plot bonfires meant different things to Tories and Dissenters; Tyburn hangings were read as civic theatre, moral lesson, and carnivalesque inversion simultaneously.

The intensity scores are curated estimates encoding historiographical judgement, not measured data. They should be read as hypotheses about relative sensory load, not empirical claims. The `sources` field on each row makes the evidentiary basis transparent.
