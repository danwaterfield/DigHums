# Narrative Pace Analysis Tool — Design Spec

## Purpose

Computationally measure the ratio of "filler" to "event" across the 54-novel corpus (1719–1817), tracking how the novel form's commitment to everyday regularity intensifies over the century. Grounded in Moretti's "Serious Century" thesis, Auerbach's *Mimesis*, and Ghosh's *The Great Derangement* argument that the realist novel's formal ecology screens out environmental catastrophe.

A secondary axis tracks free indirect discourse (FID) as the formal mechanism that normalises both internal and external experience — collapsing the boundary between character interiority and narrator authority, producing the regularised world-model that Ghosh identifies as the novel's ecological blind spot.

## Corpus

54 texts from `burney-attribution/data/metadata_v2.csv`, spanning 11 authors, 6 genres, 1719–1817. Multi-volume works (Cecilia 3 vols, Clarissa 9 vols, The Wanderer 5 vols, The Old Manor House 4 vols, Joseph Andrews 2 vols, The Italian 2 vols) are concatenated into single works with volume boundary positions recorded.

## Architecture

Two scripts, following existing gazetteer conventions:

### `gazetteer/analyse_narrative_pace.py`

Analysis engine. Reads corpus texts, classifies sentences, outputs structured data.

**Inputs:**
- `burney-attribution/data/metadata_v2.csv` — canonical text list
- Corpus `.txt` files via file paths in metadata
- `gazetteer/sensory.db` — for Ecology view environmental evidence counts

**Outputs:**
- `gazetteer/narrative_pace.db` — SQLite database with sentence-level scores (scholarly drill-down)
- `gazetteer/narrative_pace_data.json` — 200-point resampled arcs + summaries (visualisation)

**Processing pipeline:**
1. Strip Project Gutenberg boilerplate from each text
2. Concatenate multi-volume works in volume order
3. Segment into sentences (rule-based: period/question/exclamation + whitespace, with abbreviation handling)
4. Classify each sentence (see Classification Engine below)
5. Record sentence-level scores to SQLite
6. Smooth with Savitzky-Golay filter (window = ~5% of sentence count per novel)
7. Resample to 200 evenly-spaced points
8. Compute whole-novel summary proportions
9. Query sensory.db for environmental evidence counts by decade
10. Write JSON

### `gazetteer/build_narrative_pace.py`

Visualisation builder. Reads JSON, writes self-contained HTML.

**Input:** `gazetteer/narrative_pace_data.json`
**Output:** `gazetteer/narrative_pace.html`

Follows the `HTML_TEMPLATE` + `__PLACEHOLDER__` replacement pattern used by `build_comparison.py` and `build_venue_explorer.py`.

## Classification Engine

Lexicon-based, operating at sentence level. Dialogue is detected first and excluded from other classifiers. The remaining five classifiers each produce a 0–1 score per sentence, normalised to sum to 1.0.

### Category Definitions and Signals

**1. Dialogue**
- Primary: text enclosed in quotation marks (straight or curly), or following em-dash speech conventions
- Secondary: speech verb proximity. Full period-verified list (ranked by corpus frequency): "said", "cried" (= exclaimed, not wept), "answered", "returned" (= replied; archaic but 4th most common in corpus), "replied", "continued" (= went on speaking), "added", "repeated", "interrupted", "declared", "exclaimed", "demanded" (= asked/inquired in 18c, not modern "insisted"), "pursued" (= continued speaking; period-specific, now archaic), "resumed" (= began speaking again), "entreated", "whispered", "observed" (= remarked; weight as speech verb near quotation marks), "inquired", "rejoined" (= replied to a reply), "ejaculated" (= exclaimed suddenly; no sexual connotation in 18c)
- Binary: a sentence is either dialogue or not. If dialogue, the other five categories are not scored.

**2. Singulative narration** — unique events told once
- Temporal rupture adverbs: "suddenly", "at once", "at that moment", "immediately", "instantly", "in a moment", "without warning", "presently" (= immediately/soon in 18c, NOT "currently"), "directly" (= immediately in 18c, NOT directionally), "on a sudden", "in a trice", "ere long"
- Simple past tense action verbs (high-agency: "seized", "ran", "threw", "struck", "fled", "entered", "discovered")
- Unique temporal markers: "that evening", "the next morning", "on Tuesday" (specific, not habitual)
- Exclamatory constructions (without speech marks)
- New proper noun introductions
- Period-specific archaic temporals: "ere" (= before), "betimes" (= early), "by and by" (= soon)

**3. Iterative narration** — habitual/repeated events
- Habitual "would" (not conditional): "she would often", "he would every morning"
- "used to", "was accustomed to", "was wont to" (valid but low frequency — 17 instances vs 735 for "used to")
- Frequency adverbs: "every", "often", "always", "never", "usually", "generally", "commonly", "frequently", "daily", "nightly", "each morning"
- Generic plurals in subject position ("the ladies would...", "visitors were always...")
- Imperfect/habitual constructions
- "ever and anon" (= from time to time; archaic but period-specific)

**4. Description** — static scene-setting
- Copular verb + adjective chains ("the room was large and handsomely furnished")
- Spatial prepositions: "above", "beneath", "beside", "beyond", "within", "around", "between"
- Colour, material, size, and architectural terms
- Absence of temporal progression (no temporal adverbs or conjunctions)
- Present-tense set-pieces embedded in past-tense narration

**5. Free indirect discourse** — character consciousness rendered as narration
- Third person + backshifted tense + evaluative language
- Exclamatory syntax without speech marks: "How delightful!", "What a noble prospect!"
- Interrogative syntax without speech marks: "Was she to endure this?"
- Deictic shift: "here", "now", "this", "tomorrow", "yesterday" appearing in past-tense narration (proximal deictics in distal tense context)
- Evaluative adjectives/adverbs expressing character judgment (period-verified, ranked by corpus frequency): "cruel", "dreadful", "charming" (stronger than modern — closer to "enchanting"), "agreeable"/"disagreeable" (the period's standard social judgment pair), "amiable" (= worthy of love, not merely "friendly"), "wretched", "shocking", "odious", "delightful", "barbarous", "horrid" (used as colloquial intensifier), "insufferable", "divine" (exclamatory hyperbole), "unaccountable" (= inexplicable; period-specific bewilderment marker), "insupportable" (= unbearable; now archaic), "prodigious" (used as intensifying adverb = "extremely"), "monstrous" (likewise used as adverb = "extremely")
- Epistemic hedges in narration: "indeed" (5,345 corpus instances — most common hedge by far), "perhaps", "surely" (stronger assertive force in 18c — signals character's emotional conviction), "certainly", "truly", "no doubt", "doubtless", "I dare say", "it seemed"

**6. Commentary** — narrator's own voice
- First-person-plural: "we", "our", "us" (narrator + reader)
- Direct reader address: "reader", "the reader will..."
- Generalising nouns: "mankind", "the world", "a woman", "every man"
- Present-tense maxims and generalisations
- Moral/philosophical vocabulary (period-verified): "honour" (gendered: male = public reputation; female = chastity), "virtue", "duty", "tenderness" (key sensibility term, = emotional responsiveness), "gratitude" (morally loaded; could shade into romantic obligation), "esteem" (= serious moral approval, rational counterpart to love), "prudence" (ambivalent: practical self-interest, especially in marriage), "delicacy" (gendered: refined moral/sexual sensitivity), "sensibility" (= capacity for refined emotional response, NOT "good sense"), "propriety", "fortitude" (feminine ideal of endurance), "benevolence" (Shaftesburian natural virtue), "compassion", "modesty", "discretion", "condescension" (POSITIVE in 18c = gracious lowering of a superior; NOT modern pejorative)
- Subjunctive constructions

### Period False Friends — Words Excluded From Markers

The following common 18c words have shifted meaning substantially and must NOT be used with modern senses as classification markers:

| Word | Corpus count | 18c meaning | Modern meaning | Risk if misused |
|------|-------------|-------------|----------------|-----------------|
| sensible | ~520 | "aware of, capable of feeling" | "practical, rational" | Misclassifies sensibility as rationality |
| nice | ~171 | "precise, fastidious, delicate" | "pleasant, kind" | Inverts valence |
| awful | ~128 | "awe-inspiring, sublime" | "terrible, bad" | Inverts valence |
| romantic | ~209 | "fanciful, unrealistic" (often pejorative) | "relating to love" | Misclassifies irony as sincerity |
| enthusiasm | ~93 | "religious fanaticism" (pejorative) | "eager interest" (positive) | Inverts valence |
| indifferent | ~317 | "impartial, of no consequence" | "uncaring" | False negative affect |
| impertinent | ~211 | "irrelevant, not pertaining" | "rude" | Misclassifies register |
| nervous | ~15 | "sinewy, vigorous" | "anxious" | Inverts meaning entirely |
| liberal | ~88 | "generous, free-giving" | "politically left" | Anachronistic classification |

### Scoring Details

- Each classifier produces a raw score based on marker density (marker hits / sentence length in words). Multi-word markers (e.g. "at that moment") count as one hit regardless of their word length.
- Raw scores are normalised to sum to 1.0 across the five non-dialogue categories.
- For ambiguous sentences with no strong markers, scores will tend toward uniform distribution. This is acceptable — the smoothing step absorbs this noise.
- The lexicon is versioned (`lexicon_version` field in JSON output) to allow iterative refinement.

## Smoothing

- **Method:** Savitzky-Golay filter (polynomial order 3).
- **Window size:** ~5% of total sentences per novel, always odd, minimum 11.
- **Applied independently** to each of the six category signals.
- **Resampling:** after smoothing, interpolate to 200 evenly-spaced points along normalised narrative position (0.0–1.0).
- The HTML exposes a smoothing-level slider that adjusts the window size parameter, recomputing client-side from the 200-point data. This is an approximation (re-smoothing already-smoothed data) but sufficient for interactive exploration. The SQLite retains raw sentence-level data for rigorous re-analysis.

## Data Structures

### SQLite schema (`narrative_pace.db`)

```sql
CREATE TABLE novels (
    id TEXT PRIMARY KEY,       -- e.g. "burney_evelina_1778"
    author TEXT,
    title TEXT,
    year INTEGER,
    genre TEXT,
    word_count INTEGER,
    sentence_count INTEGER,
    volume_boundaries TEXT     -- JSON array of normalised positions
);

CREATE TABLE sentences (
    novel_id TEXT REFERENCES novels(id),
    sentence_index INTEGER,
    position REAL,             -- normalised 0.0–1.0
    text TEXT,
    is_dialogue INTEGER,
    singulative REAL,
    iterative REAL,
    description REAL,
    fid REAL,
    commentary REAL,
    dominant_category TEXT,
    PRIMARY KEY (novel_id, sentence_index)
);

CREATE TABLE environmental (
    decade INTEGER PRIMARY KEY,    -- e.g. 1720, 1730, ...
    smoke_count INTEGER,
    smell_count INTEGER,
    pollution_count INTEGER,
    total_evidence INTEGER
);
```

### JSON structure (`narrative_pace_data.json`)

```json
{
  "novels": [
    {
      "id": "burney_evelina_1778",
      "author": "burney",
      "title": "Evelina",
      "year": 1778,
      "genre": "epistolary",
      "word_count": 123456,
      "sentence_count": 5432,
      "volume_boundaries": [],
      "arc": {
        "positions": [0.0, 0.005, ...],
        "dialogue": [0.23, 0.18, ...],
        "singulative": [0.31, 0.28, ...],
        "iterative": [0.12, 0.15, ...],
        "description": [0.19, 0.22, ...],
        "fid": [0.08, 0.10, ...],
        "commentary": [0.07, 0.07, ...]
      },
      "summary": {
        "dialogue": 0.24,
        "singulative": 0.29,
        "iterative": 0.14,
        "description": 0.18,
        "fid": 0.09,
        "commentary": 0.06
      }
    }
  ],
  "environmental": [
    {"decade": 1660, "smoke": 12, "smell": 8, "pollution": 5, "total": 25},
    ...
  ],
  "lexicon_version": "1.0",
  "generated": "2026-04-05T..."
}
```

## Visualisation

Single-page HTML with four tab views. Canvas/SVG charts rendered client-side from the embedded JSON data. No external dependencies (self-contained HTML, consistent with all other gazetteer tools).

### Tab 1: "Century" (default landing view)

The Moretti/Ghosh chronological argument.

- **Top:** Scatter plot. X-axis = publication year (1719–1817). Y-axis = filler proportion (iterative + description + FID, as percentage). Each dot = one novel, coloured by genre. Hover shows title/author/year. Click navigates to Arcs view. Regression trend line overlaid.
- **Bottom:** Stacked horizontal bars, one per novel, ordered chronologically. Each bar shows the six-category breakdown. Lets the viewer see not just how much filler but what kind — does FID rise while iterative stays flat? Does dialogue decline?
- **Genre colour key:** domestic (green), gothic (purple), picaresque (red), epistolary (blue), amatory (orange), satirical (grey).

### Tab 2: "Arcs"

Individual novel narrative shapes.

- **Novel selector:** dropdown listing all novels by author — title (year).
- **Main chart:** Stacked area chart. X-axis = narrative position (0.0–1.0). Y-axis = proportion of each category. Six colour-coded layers.
- **Volume boundaries:** dashed vertical lines at volume break positions, labelled "Vol 1 | Vol 2" etc.
- **Controls:**
  - Smoothing slider (adjusts window size)
  - Blended / Dominant toggle (continuous scores vs winner-takes-all)
  - Show volume boundaries checkbox
  - Overlay selector (add 1–2 additional novels for comparison, rendered as semi-transparent layers)
- **Legend:** colour key with whole-novel percentages for the selected work.

### Tab 3: "Grid"

Small multiples, all novels.

- Grid of miniature stacked-area sparkline cards, ordered chronologically (left-to-right, top-to-bottom).
- Each card shows: year, title, author, and a tiny version of the stacked area chart.
- Click any card to navigate to its arc in Tab 2.
- Filter pills above the grid: by genre, by author.
- The visual question: does the colour balance shift as you scan left to right across the century?

### Tab 4: "Ecology"

The Ghosh cross-reference — what the novels screen out.

- **Dual panel:**
  - Left: novels' filler + FID proportion over time (same scatter as Tab 1 but focused on filler + FID only).
  - Right: environmental evidence density from `sensory.db` — smoke, smell, and pollution evidence counts by decade (1660–1820).
- **Argument callout:** text box framing the interpretive claim: as the left panel rises (regularisation), the right panel also rises (environmental reality), and the divergence between formal representation and lived experience widens.
- **Evidence table below:** specific environmental phenomena from non-fiction sources (Evelyn's *Fumifugium*, Grosley, Moritz, Gay's *Trivia*) alongside what the contemporary novels were doing at the same date. Drawn from existing sensory.db evidence passages tagged with environmental modalities.

## Dependencies

Python standard library plus:
- `scipy.signal.savgol_filter` — for Savitzky-Golay smoothing
- No other external dependencies. Sentence segmentation and classification are rule-based using stdlib `re`.

## File Inventory

| File | Type | Description |
|------|------|-------------|
| `gazetteer/analyse_narrative_pace.py` | Script | Analysis engine |
| `gazetteer/build_narrative_pace.py` | Script | HTML builder |
| `gazetteer/narrative_pace.db` | Data | Sentence-level SQLite (generated) |
| `gazetteer/narrative_pace_data.json` | Data | 200-point resampled JSON (generated) |
| `gazetteer/narrative_pace.html` | Output | Self-contained visualisation (generated) |

## Testing

- `gazetteer/tests/test_analyse_narrative_pace.py` — unit tests for each classifier (known sentences with expected dominant category), boilerplate stripping, volume concatenation, score normalisation.
- `gazetteer/tests/test_build_narrative_pace.py` — HTML fixture test (subprocess build, check data injection and tab structure).

## Out of Scope

- LLM-based classification or validation (future enhancement)
- Sentence-level drill-down in the HTML (data available in SQLite for separate tooling)
- Cross-corpus comparison with non-18c texts
- Automated arc clustering / typology (could be added later using the JSON output)
