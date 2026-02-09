# Burney Authorship Attribution Project

Building a deep learning system for identifying Frances Burney's stylistic fingerprint in 18th-century texts using ECCO-BERT and modern NLP techniques.

## Project Status

**Phase 1: Data Acquisition & Preparation** - ✅ COMPLETE

- ✅ Assembled corpus of Burney and contemporary authors
- ✅ Created preprocessing pipeline for Project Gutenberg texts
- ✅ Generated metadata management system
- ✅ Completed exploratory data analysis

## Corpus Overview

- **Total**: 18 texts, 2.5M words
- **Authors**: 7 (Burney, Austen, Radcliffe, Richardson, Fielding, Smollett, Edgeworth)
- **Date range**: 1740-1814
- **Burney corpus**: 10 texts (4 novels), 1.1M words (44% of total)

### Authors Included

- **Frances Burney**: Evelina (1778), Cecilia (1782), Camilla (1796), The Wanderer (1814)
- **Jane Austen**: Pride and Prejudice (1813)
- **Ann Radcliffe**: A Sicilian Romance (1790), The Mysteries of Udolpho (1794)
- **Samuel Richardson**: Pamela (1740)
- **Henry Fielding**: Tom Jones (1749)
- **Tobias Smollett**: Ferdinand Count Fathom (1753), Humphry Clinker (1771)
- **Maria Edgeworth**: Castle Rackrent (1800)

## Directory Structure

```
burney-attribution/
├── data/
│   ├── raw/              # Original texts from Project Gutenberg
│   ├── processed/        # Cleaned texts (Gutenberg boilerplate removed)
│   └── metadata.csv      # Corpus metadata and statistics
├── notebooks/
│   └── 01_eda.ipynb      # Exploratory data analysis
├── scripts/
│   ├── preprocess.py     # Text cleaning pipeline
│   └── create_metadata.py # Metadata generation
├── models/               # (Future: trained models)
└── outputs/
    └── figures/          # Visualizations
```

## Getting Started

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Preprocessing

```bash
python scripts/preprocess.py
```

### Generate Metadata

```bash
python scripts/create_metadata.py
```

### Explore the Corpus

Open `notebooks/01_eda.ipynb` in Jupyter to see corpus statistics and visualizations.

## Next Steps (Phase 2)

1. Set up ML environment (transformers, torch, datasets)
2. Implement traditional stylometry baseline (Burrows' Delta)
3. Fine-tune ECCO-BERT for authorship classification
4. Establish evaluation framework

## Project Timeline

- **Phase 1** (Weeks 1-2): Data preparation ✅
- **Phase 2** (Weeks 3-4): Baseline implementation
- **Phase 3** (Weeks 5-6): Passage-level attribution
- **Phase 4** (Weeks 7-8): Interpretability analysis
- **Phase 5** (Weeks 9-10): Applied case study
- **Phase 6** (Weeks 11-12): Polish & release

## License

MIT License - See LICENSE file for details

## Data Source

All texts sourced from [Project Gutenberg](https://www.gutenberg.org) under their license terms.
