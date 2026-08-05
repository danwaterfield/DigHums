#!/usr/bin/env python3
"""
Download new textual sources for the sensory evidence pipeline.

Sources include the existing Gutenberg texts plus selected plain-text
or OCR text downloads for phase-2 topography and institutional sources.

Run: python3 gazetteer/download_sources.py
"""

import urllib.request
import time
from pathlib import Path

OUT = Path(__file__).parent / "sources"
OUT.mkdir(exist_ok=True)

GUTENBERG = [
    # (subdir, filename, gutenberg_id)
    ("poetry",      "Gay_Trivia.txt",               "43968"),
    ("topography",  "Defoe_TourVol1.txt",            "4083"),
    ("topography",  "Evelyn_Fumifugium.txt",          "56535"),
    ("topography",  "Pennant_OfLondon.txt",           "42521"),
    ("poetry",      "Anstey_NewBathGuide.txt",        "14448"),
    ("diary",       "Burney_DiaryVol1.txt",           "15905"),
    ("diary",       "Burney_DiaryVol2.txt",           "19941"),
    ("letters",     "Walpole_LettersVol1.txt",        "9948"),
]

# Boswell's London Journal is intentionally not auto-fetched here:
# the previously used Gutenberg id resolved to an unrelated text, and the
# public Internet Archive OCR endpoints need a cleaner validation pass.

DIRECT_URLS = [
    (
        "topography",
        "Hatton_NewViewOfLondonVol1.txt",
        [
            "https://archive.org/download/bim_eighteenth-century_a-new-view-of-london-or_hatton-edward_1708_1_1/"
            "bim_eighteenth-century_a-new-view-of-london-or_hatton-edward_1708_1_1_djvu.txt",
        ],
    ),
    (
        "topography",
        "Hatton_NewViewOfLondonVol2.txt",
        [
            "https://archive.org/download/bim_eighteenth-century_a-new-view-of-london-or_hatton-edward_1708_2_1/"
            "bim_eighteenth-century_a-new-view-of-london-or_hatton-edward_1708_2_1_djvu.txt",
        ],
    ),
    (
        "topography",
        "Strype_SurveyOfLondonVol1.txt",
        [
            "https://archive.org/download/bim_eighteenth-century_a-survey-of-the-cities-o_stow-john_1720_1/"
            "bim_eighteenth-century_a-survey-of-the-cities-o_stow-john_1720_1_djvu.txt",
        ],
    ),
    (
        "topography",
        "Strype_SurveyOfLondonVol2.txt",
        [
            "https://archive.org/download/bim_eighteenth-century_a-survey-of-the-cities-o_stow-john_1720_2/"
            "bim_eighteenth-century_a-survey-of-the-cities-o_stow-john_1720_2_djvu.txt",
        ],
    ),
    (
        "topography",
        "PictureOfLondon_1810.txt",
        [
            "https://archive.org/download/b22026691/b22026691_djvu.txt",
        ],
    ),
    (
        "institutional",
        "Colquhoun_PoliceOfTheMetropolis.txt",
        [
            "https://www.gutenberg.org/cache/epub/35650/pg35650.txt",
        ],
    ),
    (
        "topography",
        "BathAndBristolGuide_1765.txt",
        [
            "https://archive.org/download/bim_eighteenth-century_the-bath-and-bristol-gui_1765/"
            "bim_eighteenth-century_the-bath-and-bristol-gui_1765_djvu.txt",
        ],
    ),
    (
        "topography",
        "Wood_DescriptionOfBathVol1.txt",
        [
            "https://archive.org/download/bim_eighteenth-century_a-description-of-bath-w_wood-john_1765_1/"
            "bim_eighteenth-century_a-description-of-bath-w_wood-john_1765_1_djvu.txt",
        ],
    ),
    (
        "topography",
        "Wood_DescriptionOfBathVol2.txt",
        [
            "https://archive.org/download/bim_eighteenth-century_a-description-of-bath-w_wood-john_1765_2/"
            "bim_eighteenth-century_a-description-of-bath-w_wood-john_1765_2_djvu.txt",
        ],
    ),
    (
        "institutional",
        "Howard_StateOfPrisons1792.txt",
        [
            "https://archive.org/download/bim_eighteenth-century_the-state-of-the-prisons_howard-john_1792/"
            "bim_eighteenth-century_the-state-of-the-prisons_howard-john_1792_djvu.txt",
        ],
    ),
]

BASE = "https://www.gutenberg.org/files/{gid}/{gid}-0.txt"
FALLBACK = "https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"

def fetch(url: str, dest: Path) -> bool:
    try:
        urllib.request.urlretrieve(url, dest)
        size = dest.stat().st_size
        if size < 5000:
            dest.unlink()
            return False
        return True
    except Exception:
        return False


def fetch_any(urls: list[str], dest: Path) -> tuple[bool, str | None]:
    for url in urls:
        if fetch(url, dest):
            return True, url
    return False, None

for subdir, filename, gid in GUTENBERG:
    dest_dir = OUT / subdir
    dest_dir.mkdir(exist_ok=True)
    dest = dest_dir / filename
    if dest.exists():
        print(f"  skip (exists): {filename}")
        continue
    url = BASE.format(gid=gid)
    ok = fetch(url, dest)
    if not ok:
        url = FALLBACK.format(gid=gid)
        ok = fetch(url, dest)
    status = f"{dest.stat().st_size/1024:.0f}KB" if ok else "FAILED"
    print(f"  {'ok' if ok else 'FAIL'} {filename:45s} {status}")
    time.sleep(0.5)

for subdir, filename, urls in DIRECT_URLS:
    dest_dir = OUT / subdir
    dest_dir.mkdir(exist_ok=True)
    dest = dest_dir / filename
    if dest.exists():
        print(f"  skip (exists): {filename}")
        continue
    ok, used_url = fetch_any(urls, dest)
    status = f"{dest.stat().st_size/1024:.0f}KB" if ok else "FAILED"
    if ok and used_url:
        print(f"  ok   {filename:45s} {status}  [{used_url}]")
    else:
        print(f"  FAIL {filename:45s} {status}")
    time.sleep(0.5)

print(f"\nSources directory: {OUT}")
