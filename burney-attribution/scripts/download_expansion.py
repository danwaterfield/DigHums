#!/usr/bin/env python3
"""
Download additional novels from Project Gutenberg to expand the corpus.
"""

import urllib.request
import os
import sys
import time
from pathlib import Path

DOWNLOADS = [
    # (author_dir, filename, gutenberg_url)
    # Austen
    ("JaneAusten", "SenseAndSensibility.txt", "https://www.gutenberg.org/ebooks/161.txt.utf-8"),
    ("JaneAusten", "Emma.txt", "https://www.gutenberg.org/ebooks/158.txt.utf-8"),
    ("JaneAusten", "MansfieldPark.txt", "https://www.gutenberg.org/ebooks/141.txt.utf-8"),
    ("JaneAusten", "NorthangerAbbey.txt", "https://www.gutenberg.org/ebooks/121.txt.utf-8"),
    ("JaneAusten", "Persuasion.txt", "https://www.gutenberg.org/ebooks/105.txt.utf-8"),
    # Fielding
    ("HenryFielding", "JosephAndrewsVol1.txt", "https://www.gutenberg.org/ebooks/9611.txt.utf-8"),
    ("HenryFielding", "JosephAndrewsVol2.txt", "https://www.gutenberg.org/ebooks/9609.txt.utf-8"),
    ("HenryFielding", "Amelia.txt", "https://www.gutenberg.org/ebooks/6098.txt.utf-8"),
    # Richardson - Clarissa (9 volumes)
    ("SamuelRichardson", "ClarissaVol1.txt", "https://www.gutenberg.org/ebooks/9296.txt.utf-8"),
    ("SamuelRichardson", "ClarissaVol2.txt", "https://www.gutenberg.org/ebooks/9798.txt.utf-8"),
    ("SamuelRichardson", "ClarissaVol3.txt", "https://www.gutenberg.org/ebooks/9881.txt.utf-8"),
    ("SamuelRichardson", "ClarissaVol4.txt", "https://www.gutenberg.org/ebooks/10462.txt.utf-8"),
    ("SamuelRichardson", "ClarissaVol5.txt", "https://www.gutenberg.org/ebooks/10799.txt.utf-8"),
    ("SamuelRichardson", "ClarissaVol6.txt", "https://www.gutenberg.org/ebooks/11364.txt.utf-8"),
    ("SamuelRichardson", "ClarissaVol7.txt", "https://www.gutenberg.org/ebooks/11889.txt.utf-8"),
    ("SamuelRichardson", "ClarissaVol8.txt", "https://www.gutenberg.org/ebooks/12180.txt.utf-8"),
    ("SamuelRichardson", "ClarissaVol9.txt", "https://www.gutenberg.org/ebooks/12398.txt.utf-8"),
    # Smollett
    ("TobiasSmollett", "RoderickRandom.txt", "https://www.gutenberg.org/ebooks/4085.txt.utf-8"),
    ("TobiasSmollett", "PeregrinePickle.txt", "https://www.gutenberg.org/ebooks/4084.txt.utf-8"),
]


def main():
    corpus_root = Path(__file__).parent.parent.parent
    total = 0
    failed = 0

    for author_dir, filename, url in DOWNLOADS:
        dest_dir = corpus_root / author_dir
        dest_dir.mkdir(exist_ok=True)
        dest_file = dest_dir / filename

        if dest_file.exists():
            size = dest_file.stat().st_size
            print(f"  SKIP {author_dir}/{filename} (already exists, {size:,} bytes)")
            total += 1
            continue

        print(f"  GET  {author_dir}/{filename} ...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, dest_file)
            size = dest_file.stat().st_size
            print(f"{size:,} bytes")
            total += 1
            time.sleep(1)
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

    print(f"\nDownloaded {total} files ({failed} failures)")


if __name__ == "__main__":
    main()
