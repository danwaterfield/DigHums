# DigHums

Digital humanities workspace for eighteenth-century fiction, authorship
attribution, and historical geography.

This repository is no longer just a plain-text corpus. It now contains:

1. A literary corpus at the repository root.
2. A machine-learning attribution project in `burney-attribution/`.
3. A historical-geography and sensory-evidence project in `gazetteer/`.

## Current Repository State

- Root literary holdings: 16 author directories, 56 `.txt` files, and one
  `.docx` duplicate (`FrancesBurney/CeciliaVol1.docx`).
- Supplemental material: `nonfiction/FrancesBurney/` contains three diary and
  letters volumes not currently wired into the modeling pipeline.
- Canonical modeling corpus: `burney-attribution/data/metadata_v2.csv`
  defines 14 authors, 34 works, and 53 source files.
- Gazetteer inputs: `gazetteer/venues.csv` currently lists 99 venues.
- Checked-in gazetteer database: `gazetteer/sensory.db` currently contains
  11,831 evidence rows across 458 source IDs, with 20 recurring events and
  8 dated event instances.

## Canonical Vs. Supplemental Data

The repo contains more text files than the current modeling pipeline uses.
When accuracy matters, use these rules:

- Treat the root author directories as the raw literary holdings.
- Treat `burney-attribution/data/metadata_v2.csv` as the source of truth for
  which fiction texts are currently wired into the attribution and gazetteer
  fiction pipelines.
- Treat `burney-attribution/data/metadata.csv`,
  `burney-attribution/scripts/train_bert.py`, and
  `burney-attribution/results/test_results.txt` plus
  `burney-attribution/results/anonymous_attribution_test.json` as legacy v1
  artifacts from the earlier 7-author experiment.
- Treat `FrancesSheridan/`, `HenryMackenzie/`,
  `SamuelRichardson/SirCharlesGrandison_Vol4.txt`, and
  `nonfiction/FrancesBurney/` as supplemental holdings until they are ingested
  into `metadata_v2.csv` or another explicit dataset.

## Repository Structure

```text
DigHums/
├── [Author Directories]/          # Raw literary texts at corpus root
│   ├── AnnRadcliffe/
│   ├── FrancesBurney/
│   ├── JaneAusten/
│   └── ... (12 more)
├── nonfiction/                    # Supplemental non-fiction holdings
│   └── FrancesBurney/
├── burney-attribution/            # Authorship attribution project
│   ├── data/
│   ├── scripts/
│   ├── results/
│   ├── models/
│   └── README.md
├── gazetteer/                     # Historical geography + sensory maps
│   ├── sources/
│   ├── tests/
│   ├── venues.csv
│   ├── sensory.db
│   └── *.html
├── docs/plans/                    # Design and implementation plans
├── CORPUS_CATALOG.md              # Corpus inventory and status notes
└── CLAUDE.md                      # Repository guide for coding agents
```

## Projects

### 1. Root Corpus

The root of the repository holds the source texts: mostly Project Gutenberg
novels, organized by author. Multi-volume works remain split by volume.

The corpus now has two practical layers:

- A broader holdings layer at the repository root.
- A narrower, structured layer in `burney-attribution/data/metadata_v2.csv`
  that powers the current code.

See `CORPUS_CATALOG.md` for a breakdown of modeled and supplemental texts.

### 2. Burney Attribution

`burney-attribution/` contains the authorship-attribution pipeline: corpus
ingest, preprocessing, metadata, chunking, model training, and evaluation.

Important evaluation note:

- Legacy v1 results: the original 7-author experiment used overlapping chunk
  splits within the same works and achieved 99.9% chunk-level accuracy
  (`results/test_results.txt`), plus near-perfect anonymous attribution on the
  checked-in v1 model (`results/anonymous_attribution_test.json`).
- Current v2 results: `scripts/train_bert_v2.py` switches to work-level splits
  and non-overlapping chunks. The checked-in work-level holdout result is
  61.4% chunk accuracy on the subset of authors with multiple works
  (`results/bert_v2_holdout.json`).

In other words: the repository preserves the striking early result, but the
more conservative v2 evaluation is the one to cite when discussing
generalization across works.

### 3. Gazetteer

`gazetteer/` is a separate but related project for mapping urban space,
sensory evidence, and event intensity across London and Bath.

The pipeline is:

1. Curate venues and sources in CSV.
2. Extract passages into `sensory.db`.
3. Attach venues, events, valence, and related metadata.
4. Render self-contained HTML outputs such as:
   - `map.html`
   - `narrative_map.html`
   - `venue_explorer.html`
   - `sensory_time_map.html`
   - `comparison.html`
   - `gordon_riots.html`

The checked-in HTML files and SQLite database are useful deliverables, but the
authoritative structured inputs are the Python scripts plus the CSV catalogs.

## Quick Start

### Browse the modeled corpus

```bash
python3 - <<'PY'
import csv
from collections import defaultdict

works = defaultdict(set)
with open("burney-attribution/data/metadata_v2.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        works[row["author"]].add(row["title"])

for author in sorted(works):
    print(author, len(works[author]))
PY
```

### Run the current attribution pipeline

```bash
cd burney-attribution
python3 scripts/ingest_corpus.py
python3 scripts/preprocess.py
python3 scripts/train_bert_v2.py
```

### Rebuild the gazetteer outputs

```bash
python3 gazetteer/extract_sensory.py --write
python3 gazetteer/extract_fiction.py --write
python3 gazetteer/extract_events.py --write
python3 gazetteer/build_venue_explorer.py
python3 gazetteer/build_sensory_time_map.py
```

## Key Files

- `CORPUS_CATALOG.md`: corpus inventory, modeled texts, and supplemental texts.
- `CLAUDE.md`: practical guidance for coding agents working in this repo.
- `burney-attribution/README.md`: current and legacy attribution workflows.
- `burney-attribution/ROADMAP.md`: research roadmap for the attribution work.
- `gazetteer/venues.csv`: curated venue gazetteer.
- `gazetteer/sources_catalog.csv`: non-fiction source catalog.
- `gazetteer/correspondence_enrichment_acquisition_matrix.csv`: staged
  external data matrix for correspondence, person and address enrichment, and
  economic or social context.
- `gazetteer/correspondence_enrichment/README.md`: optional workspace for
  authority IDs, relationships, and address assertions that can enrich the
  correspondence graph without becoming a build dependency.
- `gazetteer/events.csv`: recurring event definitions for the time map.
- `docs/plans/2026-03-23-correspondence-enrichment-acquisition-plan.md`:
  rights-aware acquisition plan for the correspondence enrichment layer.

## Caveats

- Counts in older markdown files may reflect the earlier 7-author or 13-author
  expansion stages rather than the current checked-in repository.
- Some generated artifacts and local research outputs are committed alongside
  source files; prefer the structured CSV/Python inputs when deciding what is
  canonical.
- `FrancesBurney/CeciliaVol1.docx` is a duplicate format artifact. The `.txt`
  files remain the corpus standard.

## License

- Code: MIT (see `burney-attribution/LICENSE`)
- Literary texts: public domain via Project Gutenberg and other public sources

## Citation

If you cite this repository, name the specific layer you used:

- the root corpus,
- `burney-attribution/`, or
- `gazetteer/`

and record the commit or snapshot date, since the holdings and checked-in
artifacts now evolve independently.

## Author

Daniel Waterfield
