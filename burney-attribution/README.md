# Burney Authorship Attribution Project

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Digital Humanities](https://img.shields.io/badge/digital-humanities-purple.svg)
![BERT](https://img.shields.io/badge/model-BERT-orange.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

Building a deep learning system for identifying Frances Burney's stylistic fingerprint in 18th-century texts using BERT and modern NLP techniques.

## Project Status

**Phase 1: Data Acquisition & Preparation** - ✅ COMPLETE

- ✅ Assembled corpus of Burney and contemporary authors
- ✅ Created preprocessing pipeline for Project Gutenberg texts
- ✅ Generated metadata management system
- ✅ Completed exploratory data analysis

**Phase 2: Deep Learning Implementation** - ✅ COMPLETE

- ✅ Implemented Burrows' Delta baseline (80% accuracy)
- ✅ Created stratified train/val/test splits with text chunking
- ✅ Fine-tuned BERT for 7-author classification
- ✅ **Achieved 99.9% test accuracy** - substantially outperforming baseline

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

## Key Results

🎯 **99.9% Test Accuracy** - BERT fine-tuning achieved near-perfect authorship attribution across all 7 authors, substantially outperforming the 80% Burrows' Delta baseline.

**Per-Author Performance:**
- Jane Austen: 100% (97 test chunks)
- Frances Burney: 100% (874 test chunks)
- Maria Edgeworth: 100% (35 test chunks)
- Henry Fielding: 99% (265 test chunks)
- Ann Radcliffe: 100% (275 test chunks)
- Samuel Richardson: 100% (169 test chunks)
- Tobias Smollett: 100% (240 test chunks)

See `results/test_results.txt` for detailed metrics.

## Directory Structure

```
burney-attribution/
├── data/
│   ├── raw/              # Original texts from Project Gutenberg
│   ├── processed/        # Cleaned texts (Gutenberg boilerplate removed)
│   ├── bert_data/        # Prepared datasets for BERT training
│   │   └── label_mapping.json  # Author ID mappings
│   └── metadata.csv      # Corpus metadata and statistics
├── scripts/
│   ├── preprocess.py          # Text cleaning pipeline
│   ├── create_metadata.py     # Metadata generation
│   ├── prepare_bert_data.py   # Dataset preparation with stratified splitting
│   └── train_bert.py          # BERT fine-tuning script
├── results/
│   └── test_results.txt       # Final evaluation metrics (99.9% accuracy)
├── notebooks/
│   └── 01_eda.ipynb      # Exploratory data analysis
├── models/               # Trained models (not in git - too large)
└── outputs/
    └── figures/          # Visualizations
```

## Getting Started

### Install Dependencies

```bash
pip install -r requirements.txt
```

### 1. Data Preprocessing

```bash
python scripts/preprocess.py       # Clean texts
python scripts/create_metadata.py  # Generate metadata
```

### 2. Prepare BERT Training Data

```bash
python scripts/prepare_bert_data.py
```

This creates stratified train/validation/test splits (70/15/15) with:
- 512-token chunks with 256-token stride
- All 7 authors represented in each split
- ~13,000 total chunks from 18 novels

### 3. Train BERT Model

**On Google Colab (recommended - free GPU):**
1. Upload `burney_colab_data_fixed.zip` to Colab
2. Unzip and run `!python scripts/train_bert.py`
3. Training takes ~30 minutes on Tesla T4

**Locally (requires GPU):**
```bash
python scripts/train_bert.py
```

### 4. View Results

See `results/test_results.txt` for detailed evaluation metrics.

## Methodology

**Data Preparation:**
- Stratified splitting ensures all authors appear in train/val/test
- Overlapping chunks (512 tokens, 256 stride) create robust training data
- Label mapping preserves author identities across splits

**Model:**
- Base model: `bert-base-uncased` (12 layers, 110M parameters)
- Fine-tuning: 3 epochs, batch size 4, learning rate 2e-5
- Mixed precision training (FP16) for efficiency

**Baseline:**
- Burrows' Delta (stylometric method): 80% accuracy
- BERT improvement: +19.9 percentage points → 99.9%

## Next Steps (Phase 3+)

1. ✅ ~~Set up ML environment~~
2. ✅ ~~Implement traditional stylometry baseline~~
3. ✅ ~~Fine-tune BERT for authorship classification~~
4. 🔄 **Interpretability analysis**: What linguistic features does BERT learn?
5. 🔄 **Gender & authorship study**: Male vs. female stylistic patterns
6. 🔄 **Burney evolution**: Track style changes across 1778-1814
7. 🔄 **Interactive demo**: Web interface for passage attribution

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
