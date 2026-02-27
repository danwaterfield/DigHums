#!/usr/bin/env python3
"""
Download new textual sources for the sensory evidence pipeline.

Sources: Gay Trivia, Defoe Tour Vol 1, Evelyn Fumifugium, Pennant Of London,
Anstey New Bath Guide, Burney Diary vols 1-2, Boswell London Journal,
Walpole Letters Vol 1.

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
    ("diary",       "Boswell_LondonJournal.txt",      "4059"),
    ("letters",     "Walpole_LettersVol1.txt",        "9948"),
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

print(f"\nSources directory: {OUT}")
