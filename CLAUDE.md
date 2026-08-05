# CLAUDE.md

This file provides guidance to coding agents working in this repository.

## Repository Overview

This repository now has three connected layers:

1. A root literary corpus of eighteenth- and early nineteenth-century texts.
2. `burney-attribution/`, a machine-learning authorship-attribution project.
3. `gazetteer/`, a historical-geography and sensory-evidence project.

Do not treat the repository as a static text dump only. Several checked-in data
files and generated outputs are now part of the working project state.

## Canonical Sources

When accuracy matters, prefer these files over older prose documentation:

- `burney-attribution/data/metadata_v2.csv`
  Canonical list of fiction texts currently used by the attribution pipeline
  and by `gazetteer/extract_fiction.py`.
- `burney-attribution/data/metadata.csv`
  Legacy 7-author metadata used by the older attribution workflow.
- `gazetteer/venues.csv`
  Canonical venue list.
- `gazetteer/sources_catalog.csv`
  Canonical non-fiction source catalog for the gazetteer.
- `gazetteer/events.csv`, `gazetteer/event_venues.csv`,
  `gazetteer/event_instances.csv`
  Canonical event definitions and links.

## Corpus Layout

The root of the repository currently contains 15 author directories. Not every
text present there is wired into the current code.

### Root Literary Holdings

- Mostly Project Gutenberg `.txt` files.
- Multi-volume works remain split by volume.
- `FrancesBurney/CeciliaVol1.docx` is a duplicate-format artifact; `.txt`
  remains the corpus standard.
- `FrancesSheridan/`, `HenryMackenzie/`, and
  `SamuelRichardson/SirCharlesGrandison_Vol4.txt` are supplemental holdings and
  are not currently represented in `metadata_v2.csv`.

### Supplemental Non-Fiction

- `nonfiction/FrancesBurney/` contains three diary-and-letters volumes.
- The gazetteer has a separate non-fiction corpus under `gazetteer/sources/`.
  Do not conflate the two without checking the relevant catalogs.

## Working With Texts

When reading or processing corpus texts:

- Strip Project Gutenberg boilerplate before analysis.
- Expect multi-volume works for Burney, Radcliffe, Fielding, and Richardson.
- Use the metadata files rather than raw directory listings to determine what
  the modeling code currently expects.

## Working With Attribution Results

The repository contains both legacy and current evaluation regimes.

- Legacy v1:
  `burney-attribution/scripts/train_bert.py` and related outputs use chunk-level
  splits within works, overlapping windows, and the 7-author metadata file.
- Current v2:
  `burney-attribution/scripts/train_bert_v2.py` uses work-level splits,
  non-overlapping chunks, and `metadata_v2.csv`.

When discussing performance, explicitly distinguish between them. The older
99.9% chunk-level result is not directly comparable to the newer work-level
holdout result.

## Working With The Gazetteer

`gazetteer/` is a code-and-data project, not just a static map export.

- `sensory.db` is a checked-in SQLite evidence store used by the HTML builders.
- The HTML files are generated deliverables and convenient previews.
- The authoritative structured inputs are the Python scripts and CSV catalogs.

If counts in markdown files disagree with the database or CSVs, trust the
structured data and update the docs.
