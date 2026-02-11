# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a digital humanities text corpus containing 18th and early 19th-century English novels sourced from Project Gutenberg. The repository contains plain text files organized by author, intended for literary analysis, computational text studies, or teaching purposes.

## Corpus Structure

```
DigHums/
├── JaneAusten/          # 1 novel
├── FrancesBurney/       # 4 novels (some multi-volume)
├── AnnRadcliffe/        # 6 novels (expanded 2025-02-10)
├── MariaEdgeworth/      # 1 novel
├── HenryFielding/       # 1 novel
├── SamuelRichardson/    # 1 novel
├── TobiasSmollett/      # 2 novels
├── CharlotteSmith/      # 1 novel (added 2025-02-10)
├── ElizaHaywood/        # 1 novel (added 2025-02-10)
├── ClaraReeve/          # 1 novel (added 2025-02-10)
├── HoraceWalpole/       # 1 novel (added 2025-02-10)
├── MGLewis/             # 1 novel (added 2025-02-10)
└── WilliamBeckford/     # 1 novel (added 2025-02-10)
```

**Total: 13 authors, 28+ texts, ~19 MB**

## Text File Format

All texts are Project Gutenberg eBooks in UTF-8 plain text format (.txt). Each file includes:
- Project Gutenberg license header at the beginning
- Original publication metadata (author, title, release date)
- Full text of the novel
- Project Gutenberg footer at the end

Note: FrancesBurney/CeciliaVol1.docx is in Word format and should be converted to .txt to match the corpus standard.

## Working with This Corpus

When adding new texts:
- Maintain the author-based directory structure
- Use .txt format (UTF-8 encoding)
- Preserve Project Gutenberg headers and footers
- Name files using the novel title in PascalCase (e.g., PrideAndPrejudice.txt)
- For multi-volume novels, append "Vol" + number (e.g., TheWandererVol1.txt)

When analyzing texts:
- Skip or filter out Project Gutenberg boilerplate (before "*** START OF" and after "*** END OF")
- Be aware that some novels span multiple files (Cecilia, The Wanderer)
- File sizes range from 268KB to 6.3MB per directory

## Corpus Focus

The collection emphasizes the development of the English novel during its formative period, with particular attention to women writers (Burney, Radcliffe, Austen, Edgeworth, Smith, Haywood, Reeve) and Gothic fiction. This corpus is suitable for:
- **Authorship attribution** (especially anonymous/"By a Lady" publications)
- Stylometric analysis
- Character network analysis
- Sentiment analysis
- Comparative studies of 18th-century narrative techniques
- Gender and authorship studies
- Gothic vs. domestic novel stylistic comparison

## Anonymous Attribution Testing

Many texts in this corpus were originally published anonymously, making them ideal for testing authorship attribution models:
- **Frances Burney** - *Evelina* (1778): "By a Lady"
- **Ann Radcliffe** - *Castles of Athlin and Dunbayne* (1789): Completely anonymous
- **Ann Radcliffe** - *A Sicilian Romance* (1790): "By the Authoress of..."
- **Maria Edgeworth** - *Castle Rackrent* (1800): Anonymous
- **Horace Walpole** - *Castle of Otranto* (1764): "By Onuphrio Muralto"
- **Eliza Haywood** - *Love in Excess* (1719-20): Initially anonymous
- **William Beckford** - *Vathek* (1786): Anonymous English edition

See `CORPUS_CATALOG.md` for complete attribution details.
