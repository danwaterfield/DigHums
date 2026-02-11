# 18th-Century English Novel Corpus

A digital humanities text corpus containing 18th and early 19th-century English novels sourced from Project Gutenberg, with computational authorship attribution research.

## Overview

This repository contains:
1. **Text Corpus**: 28+ novels from 13 authors (1719-1814)
2. **Authorship Attribution System**: BERT-based deep learning model achieving 99.9% accuracy
3. **Anonymous Attribution Testing**: Tests on works published "By a Lady" and anonymously

## Quick Stats

- **13 authors**: Austen, Burney, Radcliffe, Smith, Haywood, Reeve, Edgeworth, Richardson, Fielding, Smollett, Walpole, Lewis, Beckford
- **28+ texts**: ~4M words total
- **8 anonymous test cases**: Works originally published anonymously
- **99.8% accuracy**: BERT correctly identifies authors from anonymous works

## Repository Structure

```
DigHums/
├── [Author Directories]/     # 13 author folders with texts
│   ├── JaneAusten/
│   ├── FrancesBurney/
│   ├── AnnRadcliffe/
│   └── ... (10 more)
│
├── burney-attribution/        # ML authorship attribution project
│   ├── scripts/               # Training and testing scripts
│   ├── results/               # Test results (99.9% accuracy)
│   ├── README.md              # Detailed project documentation
│   └── ROADMAP.md             # Research phases and timeline
│
├── CORPUS_CATALOG.md          # Complete text catalog with attribution details
├── CORPUS_EXPANSION_REPORT.md # Corpus expansion documentation
└── CLAUDE.md                  # Repository guide for AI assistants
```

## Key Features

### 1. Comprehensive Corpus
- Balanced representation of **women authors** (7 of 13)
- Multiple genres: Gothic, Domestic, Epistolary, Picaresque, Amatory
- Spans formative period of English novel (1719-1814)

### 2. Anonymous Attribution Research
Many texts were originally published anonymously:
- Frances Burney - *Evelina* (1778): "By a Lady"
- Ann Radcliffe - *Castles of Athlin and Dunbayne* (1789): Anonymous
- Maria Edgeworth - *Castle Rackrent* (1800): Anonymous
- Horace Walpole - *Castle of Otranto* (1764): "By Onuphrio Muralto"

Our BERT model achieves **99.8% accuracy** identifying these authors from their anonymous works.

### 3. State-of-the-Art ML System
- Fine-tuned BERT (bert-base-uncased)
- 99.9% test accuracy on 7-author corpus
- Stratified train/val/test splitting
- Robustness testing on out-of-sample authors

## Research Highlights

**Key Finding**: BERT learns authorial style that transcends publication attribution:
- ✅ Identifies Burney from "By a Lady" publication (100% accuracy)
- ✅ Identifies Radcliffe from anonymous debut (88% accuracy)
- ✅ Distinguishes authors within genres (Burney vs Austen)

**Current Limitation**: Model shows genre/author correlation (Gothic → Radcliffe, Domestic → Burney), suggesting it learns both individual style and genre conventions.

## Getting Started

### Using the Corpus

All texts are in UTF-8 plain text format (.txt) from Project Gutenberg:

```bash
# Example: Read Pride and Prejudice
cat JaneAusten/PrideAndPrejudice.txt

# List all texts by author
ls -lh [Author]/*.txt
```

See `CORPUS_CATALOG.md` for complete text listings with publication dates and attribution status.

### Running the Attribution Model

See `burney-attribution/README.md` for detailed instructions on:
- Training the BERT model
- Testing on anonymous works
- Expanding to 13 authors
- Reproducing results

## Use Cases

This corpus is suitable for:
- **Authorship attribution** research
- **Stylometric analysis**
- Gender and authorship studies
- Historical text analysis
- NLP/ML training datasets
- Digital humanities pedagogy
- Comparative literary studies

## Documentation

- `CORPUS_CATALOG.md` - Complete text inventory with attribution details
- `CORPUS_EXPANSION_REPORT.md` - Details on recent corpus expansion
- `burney-attribution/README.md` - ML system documentation
- `burney-attribution/ROADMAP.md` - Research phases and future work
- `CLAUDE.md` - Repository guide and coding conventions

## Citation

If you use this corpus or attribution system in your research, please cite:

```
Waterfield, D. (2025). 18th-Century English Novel Corpus with BERT Authorship Attribution.
GitHub repository. https://github.com/[username]/DigHums
```

## License

- **Code**: MIT License (see LICENSE)
- **Texts**: Public domain via Project Gutenberg

All literary texts are sourced from Project Gutenberg and are in the public domain. See https://www.gutenberg.org/policy/license.html

## Author

**Daniel Waterfield**
- PhD Candidate, History, University of Cambridge
- Research: 18th-century literature, digital humanities, computational text analysis

## Acknowledgments

- Project Gutenberg for providing public domain texts
- Anthropic Claude for development assistance
- Cambridge University History Faculty

---

**Status**: Active research project | Phase 3 Complete | 99.8% anonymous attribution accuracy achieved
