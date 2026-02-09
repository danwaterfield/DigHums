#!/usr/bin/env python3
"""
Generate metadata.csv for the corpus.
"""

import csv
from pathlib import Path
from collections import defaultdict


def count_words(file_path: Path) -> int:
    """Count words in a text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return len(text.split())


def create_metadata():
    """Generate metadata CSV for all texts in corpus."""

    # Metadata for each work (author, title, year, genre, volume, source)
    works = [
        # Frances Burney
        ('burney', 'Evelina', 1778, 'novel', None, 'Project Gutenberg', 'burney/Evelina.txt'),
        ('burney', 'Cecilia', 1782, 'novel', 1, 'Project Gutenberg', 'burney/CeciliaVol1.txt'),
        ('burney', 'Cecilia', 1782, 'novel', 2, 'Project Gutenberg', 'burney/CeciliaVol2.txt'),
        ('burney', 'Cecilia', 1782, 'novel', 3, 'Project Gutenberg', 'burney/CeciliaVol3.txt'),
        ('burney', 'Camilla', 1796, 'novel', None, 'Project Gutenberg', 'burney/Camilla.txt'),
        ('burney', 'The Wanderer', 1814, 'novel', 1, 'Project Gutenberg', 'burney/TheWandererVol1.txt'),
        ('burney', 'The Wanderer', 1814, 'novel', 2, 'Project Gutenberg', 'burney/TheWandererVol2.txt'),
        ('burney', 'The Wanderer', 1814, 'novel', 3, 'Project Gutenberg', 'burney/TheWandererVol3.txt'),
        ('burney', 'The Wanderer', 1814, 'novel', 4, 'Project Gutenberg', 'burney/TheWandererVol4.txt'),
        ('burney', 'The Wanderer', 1814, 'novel', 5, 'Project Gutenberg', 'burney/TheWandererVol5.txt'),

        # Jane Austen
        ('austen', 'Pride and Prejudice', 1813, 'novel', None, 'Project Gutenberg', 'austen/PrideAndPrejudice.txt'),

        # Ann Radcliffe
        ('radcliffe', 'A Sicilian Romance', 1790, 'novel', None, 'Project Gutenberg', 'radcliffe/ASicilianRomance.txt'),
        ('radcliffe', 'The Mysteries of Udolpho', 1794, 'novel', None, 'Project Gutenberg', 'radcliffe/Udolpho.txt'),

        # Samuel Richardson
        ('richardson', 'Pamela', 1740, 'novel', None, 'Project Gutenberg', 'richardson/Pamela.txt'),

        # Henry Fielding
        ('fielding', 'Tom Jones', 1749, 'novel', None, 'Project Gutenberg', 'fielding/TomJones.txt'),

        # Tobias Smollett
        ('smollett', 'The Adventures of Ferdinand Count Fathom', 1753, 'novel', None, 'Project Gutenberg', 'smollett/FerdinandFathom.txt'),
        ('smollett', 'The Expedition of Humphry Clinker', 1771, 'novel', None, 'Project Gutenberg', 'smollett/HumphryClinker.txt'),

        # Maria Edgeworth
        ('edgeworth', 'Castle Rackrent', 1800, 'novel', None, 'Project Gutenberg', 'edgeworth/CastleRackrent.txt'),
    ]

    # Base directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    processed_dir = project_dir / 'data' / 'processed'
    output_file = project_dir / 'data' / 'metadata.csv'

    # Collect metadata with word counts
    metadata_rows = []
    for author, title, year, genre, volume, source, rel_path in works:
        file_path = processed_dir / rel_path

        if file_path.exists():
            word_count = count_words(file_path)
            vol_str = str(volume) if volume is not None else ''

            metadata_rows.append({
                'author': author,
                'title': title,
                'year': year,
                'genre': genre,
                'volume': vol_str,
                'word_count': word_count,
                'source': source,
                'file_path': rel_path,
                'notes': ''
            })
        else:
            print(f"Warning: File not found: {file_path}")

    # Write CSV
    fieldnames = ['author', 'title', 'year', 'genre', 'volume', 'word_count', 'source', 'file_path', 'notes']
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    # Print summary
    print(f"Created metadata.csv with {len(metadata_rows)} entries")
    print(f"\nCorpus summary:")

    # Count by author
    author_counts = defaultdict(lambda: {'works': 0, 'words': 0})
    for row in metadata_rows:
        author_counts[row['author']]['works'] += 1
        author_counts[row['author']]['words'] += row['word_count']

    print("\nBy author:")
    for author in sorted(author_counts.keys()):
        counts = author_counts[author]
        print(f"  {author:12} {counts['works']:2} files  {counts['words']:9,} words")

    total_words = sum(row['word_count'] for row in metadata_rows)
    print(f"\n  {'Total':12} {len(metadata_rows):2} files  {total_words:9,} words")


if __name__ == "__main__":
    create_metadata()
