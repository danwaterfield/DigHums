# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a digital humanities text corpus containing 18th and early 19th-century English novels sourced from Project Gutenberg. The repository contains plain text files organized by author, intended for literary analysis, computational text studies, or teaching purposes.

## Corpus Structure

```
DigHums/
├── JaneAusten/          # 1 novel
├── FrancesBurney/       # 4 novels (some multi-volume)
├── AnnRadcliffe/        # 2 novels
├── HenryFielding/       # 1 novel
├── SamuelRichardson/    # 1 novel
├── TobiasSmollett/      # 2 novels
└── MariaEdgeworth/      # 1 novel
```

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

The collection emphasizes the development of the English novel during its formative period, with particular attention to women writers (Burney, Radcliffe, Austen, Edgeworth). This corpus is suitable for:
- Stylometric analysis
- Character network analysis
- Sentiment analysis
- Comparative studies of 18th-century narrative techniques
- Gender and authorship studies
