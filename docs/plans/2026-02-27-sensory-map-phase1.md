# Sensory Map — Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the evidence store and ingestion pipeline for the sensory-historical map: download new textual sources, extract sensory passages using a curated lexicon, store results in SQLite, and verify coverage against four key venues.

**Architecture:** A SQLite database (`gazetteer/sensory.db`) holds extracted passages tagged by venue, modality, source type, and date range. A download script fetches new Gutenberg/Archive.org sources into `gazetteer/sources/`. An extraction script scans every source (existing corpus + new) and populates the DB using lexicon-based matching + venue co-occurrence geocoding. Tests use pytest against a small fixture corpus.

**Tech Stack:** Python 3.11, sqlite3 (stdlib), requests, pathlib, re, pytest. All run inside `burney-attribution/venv`. No new packages needed.

---

### Task 1: Create directory structure and DB schema

**Files:**
- Create: `gazetteer/sources/` (directory for new text sources)
- Create: `gazetteer/sensory_db.py` (schema + connection helper)
- Create: `gazetteer/tests/test_sensory_db.py`

**Step 1: Write the failing test**

```python
# gazetteer/tests/test_sensory_db.py
import sqlite3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from sensory_db import init_db, DB_PATH_DEFAULT

def test_schema_creates_tables(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "sensory_evidence" in tables
    assert "sources" in tables
    conn.close()

def test_sensory_evidence_columns(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    cols = {r[1] for r in conn.execute(
        "PRAGMA table_info(sensory_evidence)"
    ).fetchall()}
    for expected in ("venue_id", "modality", "source_type", "author",
                     "pub_year", "date_min", "date_max", "text", "confidence"):
        assert expected in cols, f"missing column: {expected}"
    conn.close()
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/danielwaterfield/Documents/DigHums
source burney-attribution/venv/bin/activate
python -m pytest gazetteer/tests/test_sensory_db.py -v
```
Expected: `ModuleNotFoundError: No module named 'sensory_db'`

**Step 3: Write minimal implementation**

```python
# gazetteer/sensory_db.py
"""SQLite schema and connection helper for the sensory evidence store."""

import sqlite3
from pathlib import Path

DB_PATH_DEFAULT = Path(__file__).parent / "sensory.db"

DDL = """
CREATE TABLE IF NOT EXISTS sources (
    source_id   TEXT PRIMARY KEY,   -- e.g. "defoe_tour", "old_bailey"
    source_type TEXT NOT NULL,       -- fiction|diary|periodical|legal|
                                     --   topography|poetry|environmental
    author      TEXT NOT NULL,
    title       TEXT NOT NULL,
    pub_year    INTEGER,
    date_min    INTEGER,             -- earliest year the content describes
    date_max    INTEGER,             -- latest year the content describes
    file_path   TEXT,               -- relative to gazetteer/sources/
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS sensory_evidence (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   TEXT NOT NULL REFERENCES sources(source_id),
    venue_id    TEXT,               -- e.g. LON001; NULL if ungeocoded
    venue_name  TEXT,
    lat         REAL,
    lon         REAL,
    source_type TEXT NOT NULL,
    author      TEXT NOT NULL,
    title       TEXT NOT NULL,
    pub_year    INTEGER,
    date_min    INTEGER,
    date_max    INTEGER,
    modality    TEXT NOT NULL,      -- auditory|olfactory|visual|tactile|
                                    --   thermal|crowd|economic|unclassified
    text        TEXT NOT NULL,      -- extracted passage (~500 chars)
    context     TEXT,               -- surrounding sentence
    char_offset INTEGER,
    pos         REAL,               -- 0-1 narrative position (fiction only)
    confidence  REAL DEFAULT 1.0,   -- extraction confidence
    notes       TEXT
);

CREATE INDEX IF NOT EXISTS idx_venue    ON sensory_evidence(venue_id);
CREATE INDEX IF NOT EXISTS idx_modality ON sensory_evidence(modality);
CREATE INDEX IF NOT EXISTS idx_year     ON sensory_evidence(pub_year);
CREATE INDEX IF NOT EXISTS idx_source   ON sensory_evidence(source_type);
"""

def init_db(path: Path = DB_PATH_DEFAULT) -> sqlite3.Connection:
    """Create tables if not present; return open connection."""
    conn = sqlite3.connect(path)
    conn.executescript(DDL)
    conn.commit()
    return conn
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest gazetteer/tests/test_sensory_db.py -v
```
Expected: 2 PASSED

**Step 5: Commit**

```bash
git add gazetteer/sensory_db.py gazetteer/tests/test_sensory_db.py
git commit -m "feat: add sensory evidence SQLite schema and init_db helper"
```

---

### Task 2: Download new textual sources

**Files:**
- Create: `gazetteer/download_sources.py`
- Create: `gazetteer/sources/` (populated by script)

No tests needed for a download script — verify by checking file sizes.

**Step 1: Write download script**

```python
#!/usr/bin/env python3
"""
Download new textual sources for the sensory evidence pipeline.

Sources not already in the main corpus:
  - John Gay, Trivia (1716)
  - Daniel Defoe, Tour Through Great Britain vol 1 (1724)
  - John Evelyn, Fumifugium (1661)
  - Thomas Pennant, Of London (1790)
  - Christopher Anstey, New Bath Guide (1766)
  - Frances Burney, Diary and Letters vol 1 (1778-1840)
  - Frances Burney, Diary and Letters vol 2
  - James Boswell, London Journal (1762-63)
  - Horace Walpole, Letters vol 1

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
        if size < 5000:          # suspiciously small — probably an error page
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
```

**Step 2: Run it**

```bash
python3 gazetteer/download_sources.py
```
Expected: 9 files downloaded, each >50KB. Note any FAILs — Gutenberg IDs occasionally change; search gutenberg.org manually and update the ID.

**Step 3: Verify**

```bash
find gazetteer/sources -name "*.txt" | sort | xargs wc -l | tail -1
```
Expected: >100,000 lines total across all files.

**Step 4: Commit**

```bash
git add gazetteer/download_sources.py
# Do NOT add gazetteer/sources/ — add to .gitignore instead
echo "gazetteer/sources/" >> .gitignore
git add .gitignore
git commit -m "feat: add source downloader for sensory evidence pipeline"
```

---

### Task 3: Sensory lexicon

**Files:**
- Create: `gazetteer/sensory_lexicon.py`
- Create: `gazetteer/tests/test_sensory_lexicon.py`

**Step 1: Write the failing tests**

```python
# gazetteer/tests/test_sensory_lexicon.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from sensory_lexicon import tag_modalities, MODALITY_PATTERNS

def test_auditory_match():
    results = tag_modalities("The din of the carriages was insupportable.")
    assert any(m == "auditory" for _, m in results)

def test_olfactory_match():
    results = tag_modalities("A most offensive stench arose from the kennel.")
    assert any(m == "olfactory" for _, m in results)

def test_visual_match():
    results = tag_modalities("The street was narrow and the buildings lofty.")
    assert any(m == "visual" for _, m in results)

def test_no_false_positive():
    # "feeling" is not a sensory term in this context
    results = tag_modalities("She had a feeling of relief.")
    assert results == []

def test_returns_matched_term():
    results = tag_modalities("The clatter of hooves echoed down the street.")
    terms = [t for t, _ in results]
    assert "clatter" in terms

def test_multiple_modalities():
    text = "The smoke was thick and the din of the mob overwhelming."
    results = tag_modalities(text)
    modalities = {m for _, m in results}
    assert "auditory" in modalities
    assert "olfactory" in modalities or "visual" in modalities
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest gazetteer/tests/test_sensory_lexicon.py -v
```
Expected: `ModuleNotFoundError: No module named 'sensory_lexicon'`

**Step 3: Write minimal implementation**

```python
# gazetteer/sensory_lexicon.py
"""
Lexicon-based sensory term detection for Pass 1 extraction.

Returns list of (matched_term, modality) tuples for a text fragment.
Only matches whole words (word-boundary anchored).
"""

import re

# Each entry: (pattern, modality)
# Patterns are word-boundary anchored unless they start with r"
# Period-specific terms drawn from OED citations and corpus reading.
MODALITY_PATTERNS: list[tuple[str, str]] = [
    # ── AUDITORY ──────────────────────────────────────────────────────────
    (r"\bdin\b",          "auditory"),
    (r"\bclatter\b",      "auditory"),
    (r"\bclattering\b",   "auditory"),
    (r"\bbustle\b",       "auditory"),
    (r"\bhubbub\b",       "auditory"),
    (r"\bhuzza\b",        "auditory"),
    (r"\bnoisy\b",        "auditory"),
    (r"\bnoise\b",        "auditory"),
    (r"\bsilence\b",      "auditory"),
    (r"\bstillness\b",    "auditory"),
    (r"\bcry\b",          "auditory"),
    (r"\bcries\b",        "auditory"),
    (r"\brumble\b",       "auditory"),
    (r"\brumbling\b",     "auditory"),
    (r"\btumult\b",       "auditory"),
    (r"\buproar\b",       "auditory"),
    (r"\bdiscord\b",      "auditory"),
    (r"\btolling\b",      "auditory"),
    (r"\bclamour\b",      "auditory"),
    (r"\bclamor\b",       "auditory"),
    (r"\bshriek\b",       "auditory"),
    (r"\bshrieking\b",    "auditory"),
    (r"\bstreet-cries\b", "auditory"),
    (r"\bstreet cries\b", "auditory"),

    # ── OLFACTORY ─────────────────────────────────────────────────────────
    (r"\bstench\b",       "olfactory"),
    (r"\beffluvia\b",     "olfactory"),
    (r"\beffluvium\b",    "olfactory"),
    (r"\bperfume\b",      "olfactory"),
    (r"\bperfumed\b",     "olfactory"),
    (r"\breek\b",         "olfactory"),
    (r"\breeking\b",      "olfactory"),
    (r"\bvapour\b",       "olfactory"),
    (r"\bvapors\b",       "olfactory"),
    (r"\bodour\b",        "olfactory"),
    (r"\bodor\b",         "olfactory"),
    (r"\bfetid\b",        "olfactory"),
    (r"\bfragrant\b",     "olfactory"),
    (r"\bfragrance\b",    "olfactory"),
    (r"\bputrid\b",       "olfactory"),
    (r"\bsmoke\b",        "olfactory"),
    (r"\bsmoky\b",        "olfactory"),
    (r"\bmiasma\b",       "olfactory"),
    (r"\bkennel\b",       "olfactory"),  # street gutter/drain, 18c usage

    # ── VISUAL ────────────────────────────────────────────────────────────
    (r"\bnarrow\b",       "visual"),
    (r"\blofty\b",        "visual"),
    (r"\bglare\b",        "visual"),
    (r"\bgloom\b",        "visual"),
    (r"\bgloomy\b",       "visual"),
    (r"\bdazzling\b",     "visual"),
    (r"\bdazzle\b",       "visual"),
    (r"\bmurky\b",        "visual"),
    (r"\bthronged\b",     "visual"),
    (r"\billuminated\b",  "visual"),
    (r"\billumination\b", "visual"),
    (r"\blamplight\b",    "visual"),
    (r"\btorch-light\b",  "visual"),
    (r"\bdark\b",         "visual"),
    (r"\bdarkness\b",     "visual"),
    (r"\bdirty\b",        "visual"),
    (r"\bfilthy\b",       "visual"),
    (r"\bmuddy\b",        "visual"),
    (r"\bmud\b",          "visual"),

    # ── THERMAL ───────────────────────────────────────────────────────────
    (r"\bsultry\b",       "thermal"),
    (r"\bdamp\b",         "thermal"),
    (r"\braw\b",          "thermal"),
    (r"\bfog\b",          "thermal"),
    (r"\bfoggy\b",        "thermal"),
    (r"\bmist\b",         "thermal"),
    (r"\bmisty\b",        "thermal"),
    (r"\bfrost\b",        "thermal"),
    (r"\bfrosty\b",       "thermal"),
    (r"\bclose\b",        "thermal"),  # "close air" = stuffy; high false-pos risk
    (r"\bstifling\b",     "thermal"),
    (r"\bchilly\b",       "thermal"),

    # ── CROWD / DENSITY ───────────────────────────────────────────────────
    (r"\bpress\b",        "crowd"),    # "the press of people"
    (r"\bmob\b",          "crowd"),
    (r"\bjostle\b",       "crowd"),
    (r"\bjostled\b",      "crowd"),
    (r"\bthrong\b",       "crowd"),
    (r"\bcrowd\b",        "crowd"),
    (r"\bcrowded\b",      "crowd"),
    (r"\bcramm'd\b",      "crowd"),
    (r"\bcrammed\b",      "crowd"),
    (r"\bdeserted\b",     "crowd"),
    (r"\bempty\b",        "crowd"),
    (r"\bsparsely\b",     "crowd"),
]

# Pre-compile for speed
_COMPILED = [(re.compile(pat, re.IGNORECASE), mod)
             for pat, mod in MODALITY_PATTERNS]


def tag_modalities(text: str) -> list[tuple[str, str]]:
    """
    Return list of (matched_term, modality) for all sensory matches in text.
    Deduplicates by (term.lower(), modality).
    """
    seen: set[tuple[str, str]] = set()
    results: list[tuple[str, str]] = []
    for pattern, modality in _COMPILED:
        for m in pattern.finditer(text):
            key = (m.group().lower(), modality)
            if key not in seen:
                seen.add(key)
                results.append((m.group(), modality))
    return results
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest gazetteer/tests/test_sensory_lexicon.py -v
```
Expected: 6 PASSED. If `test_no_false_positive` fails, the "feeling" term has leaked — check the pattern list.

**Step 5: Commit**

```bash
git add gazetteer/sensory_lexicon.py gazetteer/tests/test_sensory_lexicon.py
git commit -m "feat: add lexicon-based sensory modality tagger (Pass 1)"
```

---

### Task 4: Venue geocoder (passage → venue_id)

**Files:**
- Create: `gazetteer/venue_geocoder.py`
- Create: `gazetteer/tests/test_venue_geocoder.py`

The geocoder assigns a venue_id to a text passage by looking for venue alias matches within a ±500-word window around the sensory term. Reuses VENUE_ALIASES from validate_venues.py.

**Step 1: Write the failing tests**

```python
# gazetteer/tests/test_venue_geocoder.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from venue_geocoder import geocode_passage

# Fake venues list for testing
VENUES = [
    {"id": "LON001", "name": "Vauxhall Spring Gardens",
     "lat": "51.4882", "lon": "-0.1228"},
    {"id": "LON006", "name": "Theatre Royal Drury Lane",
     "lat": "51.5133", "lon": "-0.1226"},
]

def test_vauxhall_passage():
    text = ("The company proceeded to Vauxhall, where the illuminations "
            "were remarkably brilliant and the music loud.")
    result = geocode_passage(text, "illuminations were remarkably brilliant", VENUES)
    assert result is not None
    assert result["venue_id"] == "LON001"

def test_no_venue_returns_none():
    text = "The mud was deep and the fog considerable."
    result = geocode_passage(text, "mud was deep", VENUES)
    assert result is None

def test_drury_lane_passage():
    text = "We secured a box at Drury Lane and the noise of the pit was deafening."
    result = geocode_passage(text, "noise of the pit was deafening", VENUES)
    assert result is not None
    assert result["venue_id"] == "LON006"
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest gazetteer/tests/test_venue_geocoder.py -v
```
Expected: `ModuleNotFoundError: No module named 'venue_geocoder'`

**Step 3: Write minimal implementation**

```python
# gazetteer/venue_geocoder.py
"""
Geocode a sensory passage to a venue_id by scanning a context window
for venue alias matches.

Strategy:
  1. Find the passage within the full text (by substring match).
  2. Extract a ±500-word window around it.
  3. Search the window for any venue alias.
  4. Return the first matching venue's metadata, or None.
"""

import re
import sys
from pathlib import Path

# Import VENUE_ALIASES from validate_venues (same directory)
sys.path.insert(0, str(Path(__file__).parent))
from validate_venues import VENUE_ALIASES

WINDOW_WORDS = 500


def _build_alias_map(venues: list[dict]) -> list[tuple[re.Pattern, dict]]:
    """Return list of (compiled_pattern, venue_dict) for all aliases."""
    result = []
    for venue in venues:
        vid = venue["id"]
        if vid not in VENUE_ALIASES:
            continue
        for alias, _city_filter in VENUE_ALIASES[vid]:
            if alias.startswith(r"\b") or alias.startswith("("):
                pat = re.compile(alias, re.IGNORECASE)
            else:
                pat = re.compile(re.escape(alias), re.IGNORECASE)
            result.append((pat, venue))
    return result


def geocode_passage(
    full_text: str,
    passage: str,
    venues: list[dict],
    window_words: int = WINDOW_WORDS,
) -> dict | None:
    """
    Return dict with venue_id, venue_name, lat, lon if a venue alias
    is found within window_words of the passage in full_text.
    Returns None if no venue found or passage not located in text.
    """
    alias_map = _build_alias_map(venues)

    # Find passage location in full text
    idx = full_text.find(passage)
    if idx == -1:
        # Try first 60 chars as anchor
        anchor = passage[:60]
        idx = full_text.find(anchor)
        if idx == -1:
            return None

    # Extract word-bounded window
    before = full_text[:idx].split()[-window_words:]
    after  = full_text[idx:].split()[:window_words]
    window = " ".join(before + after)

    for pattern, venue in alias_map:
        if pattern.search(window):
            return {
                "venue_id":   venue["id"],
                "venue_name": venue["name"],
                "lat":        float(venue["lat"]),
                "lon":        float(venue["lon"]),
            }
    return None
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest gazetteer/tests/test_venue_geocoder.py -v
```
Expected: 3 PASSED.

**Step 5: Commit**

```bash
git add gazetteer/venue_geocoder.py gazetteer/tests/test_venue_geocoder.py
git commit -m "feat: add venue geocoder for sensory passage → venue_id mapping"
```

---

### Task 5: Extraction pipeline

**Files:**
- Create: `gazetteer/extract_sensory.py`
- Create: `gazetteer/tests/test_extract_sensory.py`
- Create: `gazetteer/sources_catalog.csv` (metadata for new sources)

The extractor scans every source file, finds sensory passages, geocodes them, and writes to `sensory.db`.

**Step 1: Create sources_catalog.csv**

```csv
source_id,source_type,author,title,pub_year,date_min,date_max,file_path,notes
fiction_corpus,fiction,various,18c Novel Corpus,,,,,Existing corpus — scanned via corpus.py
gay_trivia,poetry,JohnGay,Trivia or the Art of Walking the Streets of London,1716,1716,1716,poetry/Gay_Trivia.txt,Sensory poem about London streets
defoe_tour,topography,DanielDefoe,Tour Through Great Britain Vol 1,1724,1720,1730,topography/Defoe_TourVol1.txt,Urban survey
evelyn_fumifugium,topography,JohnEvelyn,Fumifugium,1661,1660,1700,topography/Evelyn_Fumifugium.txt,Pre-period; coal smoke baseline
pennant_london,topography,ThomasPennant,Of London,1790,1785,1800,topography/Pennant_OfLondon.txt,Late-century survey
anstey_bath,poetry,ChristopherAnstey,New Bath Guide,1766,1760,1775,poetry/Anstey_NewBathGuide.txt,Bath satire
burney_diary1,diary,FrancesBurney,Early Diary Vol 1,1889,1768,1778,diary/Burney_DiaryVol1.txt,Published posthumously
burney_diary2,diary,FrancesBurney,Early Diary Vol 2,1889,1778,1800,diary/Burney_DiaryVol2.txt,Published posthumously
boswell_journal,diary,JamesBoswell,London Journal,1950,1762,1763,diary/Boswell_LondonJournal.txt,First published 1950
walpole_letters1,letters,HoraceWalpole,Letters Vol 1,1840,1732,1755,letters/Walpole_LettersVol1.txt,
```

**Step 2: Write the failing tests**

```python
# gazetteer/tests/test_extract_sensory.py
import sqlite3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from sensory_db import init_db
from extract_sensory import extract_from_text, WINDOW_CHARS

VENUES = [
    {"id": "LON001", "name": "Vauxhall Spring Gardens",
     "lat": "51.4882", "lon": "-0.1228"},
]

def test_extract_finds_auditory(tmp_path):
    db = init_db(tmp_path / "t.db")
    text = ("We went to Vauxhall last Tuesday. The din of the orchestra "
            "was excessive, and the crowd suffocating.")
    rows = extract_from_text(
        text=text, source_id="test", source_type="diary",
        author="Test", title="Test", pub_year=1760,
        date_min=1758, date_max=1762, venues=VENUES, conn=db
    )
    assert len(rows) > 0
    modalities = {r["modality"] for r in rows}
    assert "auditory" in modalities

def test_extract_geocodes_vauxhall(tmp_path):
    db = init_db(tmp_path / "t.db")
    text = ("We went to Vauxhall last Tuesday. The din of the orchestra "
            "was excessive, and the crowd suffocating.")
    rows = extract_from_text(
        text=text, source_id="test", source_type="diary",
        author="Test", title="Test", pub_year=1760,
        date_min=1758, date_max=1762, venues=VENUES, conn=db
    )
    vauxhall_rows = [r for r in rows if r.get("venue_id") == "LON001"]
    assert len(vauxhall_rows) > 0

def test_extract_writes_to_db(tmp_path):
    db = init_db(tmp_path / "t.db")
    text = ("We went to Vauxhall last Tuesday. The din of the orchestra "
            "was excessive, and the crowd suffocating.")
    extract_from_text(
        text=text, source_id="test_src", source_type="diary",
        author="Test", title="Test", pub_year=1760,
        date_min=1758, date_max=1762, venues=VENUES, conn=db,
        write=True
    )
    count = db.execute(
        "SELECT COUNT(*) FROM sensory_evidence WHERE source_id='test_src'"
    ).fetchone()[0]
    assert count > 0
```

**Step 3: Run test to verify it fails**

```bash
python -m pytest gazetteer/tests/test_extract_sensory.py -v
```
Expected: `ModuleNotFoundError: No module named 'extract_sensory'`

**Step 4: Write minimal implementation**

```python
# gazetteer/extract_sensory.py
"""
Pass 1 sensory extraction pipeline.

For each source, scans the text for sensory terms (lexicon-based),
extracts a context window around each match, geocodes to a venue,
and writes to the sensory_evidence table.

Usage:
    python3 gazetteer/extract_sensory.py            # dry run, print stats
    python3 gazetteer/extract_sensory.py --write    # write to sensory.db
"""

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sensory_db import init_db, DB_PATH_DEFAULT
from sensory_lexicon import tag_modalities
from venue_geocoder import geocode_passage

SOURCES_DIR  = Path(__file__).parent / "sources"
CATALOG_PATH = Path(__file__).parent / "sources_catalog.csv"
VENUES_PATH  = Path(__file__).parent / "venues.csv"
CORPUS_ROOT  = Path(__file__).parent.parent

WINDOW_CHARS = 400   # chars either side of matched term for context
STRIDE_CHARS = 200   # minimum gap between recorded passages (dedup)


def load_venues(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def strip_gutenberg(text: str) -> str:
    """Remove Project Gutenberg header/footer boilerplate."""
    start_markers = ["*** START OF", "***START OF"]
    end_markers   = ["*** END OF",   "***END OF"]
    for m in start_markers:
        idx = text.find(m)
        if idx != -1:
            nl = text.find("\n", idx)
            text = text[nl+1:]
            break
    for m in end_markers:
        idx = text.find(m)
        if idx != -1:
            text = text[:idx]
            break
    return text.strip()


def extract_from_text(
    text: str,
    source_id: str,
    source_type: str,
    author: str,
    title: str,
    pub_year: int,
    date_min: int,
    date_max: int,
    venues: list[dict],
    conn,
    write: bool = False,
) -> list[dict]:
    """
    Scan text for sensory passages. Return list of evidence dicts.
    If write=True, also INSERT into sensory_evidence.
    """
    results = []
    last_offset: dict[str, int] = {}   # modality → last char offset recorded

    words = text.split()
    text_len = len(text)

    # Slide through the text matching sensory terms
    for m in re.finditer(r"\S+", text):
        word_start = m.start()
        fragment = text[max(0, word_start - WINDOW_CHARS):
                        word_start + WINDOW_CHARS]
        matches = tag_modalities(fragment)
        if not matches:
            continue

        for term, modality in matches:
            prev = last_offset.get(modality, -STRIDE_CHARS * 2)
            if word_start - prev < STRIDE_CHARS:
                continue
            last_offset[modality] = word_start

            # Context: sentence containing the match
            ctx_start = max(0, word_start - WINDOW_CHARS)
            ctx_end   = min(text_len, word_start + WINDOW_CHARS)
            passage   = text[ctx_start:ctx_end].strip()

            # Geocode
            geo = geocode_passage(text, passage[:60], venues)

            pos = round(word_start / text_len, 4) if text_len > 0 else 0.0

            row = {
                "source_id":   source_id,
                "venue_id":    geo["venue_id"]   if geo else None,
                "venue_name":  geo["venue_name"] if geo else None,
                "lat":         geo["lat"]        if geo else None,
                "lon":         geo["lon"]        if geo else None,
                "source_type": source_type,
                "author":      author,
                "title":       title,
                "pub_year":    pub_year,
                "date_min":    date_min,
                "date_max":    date_max,
                "modality":    modality,
                "text":        passage[:500],
                "context":     term,
                "char_offset": word_start,
                "pos":         pos,
                "confidence":  1.0,
            }
            results.append(row)

            if write:
                conn.execute("""
                    INSERT INTO sensory_evidence
                    (source_id, venue_id, venue_name, lat, lon,
                     source_type, author, title, pub_year, date_min,
                     date_max, modality, text, context, char_offset,
                     pos, confidence)
                    VALUES
                    (:source_id, :venue_id, :venue_name, :lat, :lon,
                     :source_type, :author, :title, :pub_year, :date_min,
                     :date_max, :modality, :text, :context, :char_offset,
                     :pos, :confidence)
                """, row)

    if write:
        conn.commit()
    return results


def run(write: bool = False):
    venues = load_venues(VENUES_PATH)
    conn   = init_db(DB_PATH_DEFAULT)

    with open(CATALOG_PATH, newline="", encoding="utf-8") as f:
        catalog = list(csv.DictReader(f))

    total = 0
    for entry in catalog:
        sid = entry["source_id"]
        if sid == "fiction_corpus":
            print(f"  [skip] fiction_corpus — use extract_fiction.py")
            continue

        fp = SOURCES_DIR / entry["file_path"]
        if not fp.exists():
            print(f"  [missing] {fp}")
            continue

        text = strip_gutenberg(fp.read_text(encoding="utf-8", errors="replace"))
        rows = extract_from_text(
            text=text,
            source_id=sid,
            source_type=entry["source_type"],
            author=entry["author"],
            title=entry["title"],
            pub_year=int(entry["pub_year"]) if entry["pub_year"] else 0,
            date_min=int(entry["date_min"]) if entry["date_min"] else 0,
            date_max=int(entry["date_max"]) if entry["date_max"] else 0,
            venues=venues,
            conn=conn,
            write=write,
        )
        geocoded = sum(1 for r in rows if r["venue_id"])
        print(f"  {sid:30s}  {len(rows):4d} passages  "
              f"{geocoded:3d} geocoded  "
              f"({entry['source_type']})")
        total += len(rows)

    print(f"\nTotal: {total} passages extracted")
    if not write:
        print("(dry run — pass --write to persist)")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    run(write=args.write)
```

**Step 5: Run test to verify it passes**

```bash
python -m pytest gazetteer/tests/test_extract_sensory.py -v
```
Expected: 3 PASSED.

**Step 6: Commit**

```bash
git add gazetteer/extract_sensory.py gazetteer/sources_catalog.csv \
        gazetteer/tests/test_extract_sensory.py
git commit -m "feat: add sensory extraction pipeline (Pass 1, lexicon-based)"
```

---

### Task 6: Run extraction and verify coverage

**Files:** no new files — running the pipeline and checking output.

**Step 1: Dry run**

```bash
cd /Users/danielwaterfield/Documents/DigHums
source burney-attribution/venv/bin/activate
python3 gazetteer/extract_sensory.py
```
Expected output: per-source passage counts. Gay's Trivia and Defoe's Tour should produce the most geocoded passages. If any source shows 0 passages, the download probably failed — check `gazetteer/sources/`.

**Step 2: Write to DB**

```bash
python3 gazetteer/extract_sensory.py --write
```

**Step 3: Spot-check four key venues**

```bash
python3 - <<'EOF'
import sqlite3
conn = sqlite3.connect("gazetteer/sensory.db")
venues = ["LON001", "LON002", "LON008", "BAT001"]  # Vauxhall, Ranelagh, Opera, Pump Room
for vid in venues:
    rows = conn.execute("""
        SELECT venue_name, modality, author, pub_year, text
        FROM sensory_evidence WHERE venue_id = ?
        ORDER BY pub_year LIMIT 3
    """, (vid,)).fetchall()
    print(f"\n── {vid} ({len(rows)} rows shown) ──")
    for r in rows:
        print(f"  [{r[2]}, {r[3]}, {r[1]}] {r[4][:120]}")
conn.close()
EOF
```
Expected: at least 1 result for Vauxhall (LON001) and the Opera (LON008). If Ranelagh (LON002) and the Pump Room (BAT001) are empty that's plausible — fewer sources describe them explicitly. Note counts for the commit message.

**Step 4: Summary query**

```bash
python3 - <<'EOF'
import sqlite3
conn = sqlite3.connect("gazetteer/sensory.db")
print("By modality:")
for r in conn.execute("""
    SELECT modality, COUNT(*) as n FROM sensory_evidence
    GROUP BY modality ORDER BY n DESC
""").fetchall():
    print(f"  {r[0]:15s} {r[1]}")
print("\nBy source type:")
for r in conn.execute("""
    SELECT source_type, COUNT(*) as n FROM sensory_evidence
    GROUP BY source_type ORDER BY n DESC
""").fetchall():
    print(f"  {r[0]:15s} {r[1]}")
print("\nGeocoded:")
total   = conn.execute("SELECT COUNT(*) FROM sensory_evidence").fetchone()[0]
geocoded = conn.execute("SELECT COUNT(*) FROM sensory_evidence WHERE venue_id IS NOT NULL").fetchone()[0]
print(f"  {geocoded}/{total} passages assigned to a venue")
conn.close()
EOF
```

**Step 5: Commit**

```bash
git add gazetteer/sensory.db
git commit -m "data: initial sensory evidence extraction — [N] passages, [M] geocoded"
```
Replace [N] and [M] with actual numbers from the summary query.

---

### Task 7: Extract from fiction corpus

**Files:**
- Create: `gazetteer/extract_fiction.py`

This is a thin wrapper that feeds the existing corpus through `extract_from_text`.

**Step 1: Write the script**

```python
#!/usr/bin/env python3
"""
Extract sensory passages from the fiction corpus into sensory.db.

Reuses extract_sensory.extract_from_text and the existing corpus.py
metadata loading. Run after extract_sensory.py has already populated
the non-fiction sources.

Usage:
    python3 gazetteer/extract_fiction.py            # dry run
    python3 gazetteer/extract_fiction.py --write    # write to sensory.db
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "burney-attribution" / "scripts"))

from sensory_db    import init_db, DB_PATH_DEFAULT
from extract_sensory import extract_from_text, load_venues, strip_gutenberg
from corpus import load_metadata, load_work_text, get_project_paths

VENUES_PATH = Path(__file__).parent / "venues.csv"


def run(write: bool = False):
    paths   = get_project_paths()
    works   = load_metadata(paths["metadata"])
    venues  = load_venues(VENUES_PATH)
    conn    = init_db(DB_PATH_DEFAULT)

    # Load corpus_dates for date_min/date_max (text_year from corpus_dates.csv)
    dates: dict[tuple[str,str], int] = {}
    dates_path = Path(__file__).parent / "corpus_dates.csv"
    if dates_path.exists():
        with open(dates_path, newline="") as f:
            for row in csv.DictReader(f):
                key = (row["author"], row["title"])
                if row.get("text_year"):
                    dates[key] = int(row["text_year"])

    total = 0
    for work in works:
        text = strip_gutenberg(load_work_text(work, paths["processed"]))
        key  = (work.author, work.title)
        ty   = dates.get(key, work.year)   # text year if known, else pub year
        sid  = f"fiction_{work.author}_{work.title}".replace(" ", "_")[:60]

        rows = extract_from_text(
            text=text,
            source_id=sid,
            source_type="fiction",
            author=work.author,
            title=work.title,
            pub_year=work.year,
            date_min=ty - 5,
            date_max=ty + 5,
            venues=venues,
            conn=conn,
            write=write,
        )
        geocoded = sum(1 for r in rows if r["venue_id"])
        print(f"  {work.author:20s} {work.title:35s}  "
              f"{len(rows):4d} passages  {geocoded:3d} geocoded")
        total += len(rows)

    print(f"\nTotal fiction passages: {total}")
    if not write:
        print("(dry run — pass --write to persist)")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    run(write=args.write)
```

**Step 2: Dry run**

```bash
python3 gazetteer/extract_fiction.py
```
Expected: passage counts per work. Smollett and Fielding should dominate (auditory/olfactory satire); Burney and Austen should show fewer olfactory hits but more crowd/visual.

**Step 3: Write**

```bash
python3 gazetteer/extract_fiction.py --write
```

**Step 4: Final summary**

```bash
python3 - <<'EOF'
import sqlite3
conn = sqlite3.connect("gazetteer/sensory.db")
total = conn.execute("SELECT COUNT(*) FROM sensory_evidence").fetchone()[0]
geocoded = conn.execute(
    "SELECT COUNT(*) FROM sensory_evidence WHERE venue_id IS NOT NULL"
).fetchone()[0]
print(f"Total passages: {total}")
print(f"Geocoded: {geocoded} ({100*geocoded//total}%)")
print("\nTop 10 venues by evidence count:")
for r in conn.execute("""
    SELECT venue_name, COUNT(*) as n, GROUP_CONCAT(DISTINCT modality)
    FROM sensory_evidence WHERE venue_id IS NOT NULL
    GROUP BY venue_id ORDER BY n DESC LIMIT 10
""").fetchall():
    print(f"  {r[0]:35s}  {r[1]:4d}  {r[2]}")
conn.close()
EOF
```

**Step 5: Commit**

```bash
git add gazetteer/extract_fiction.py gazetteer/sensory.db
git commit -m "data: extract fiction corpus into sensory evidence store — [N] total passages"
```

---

## After Phase 1

Phase 1 delivers a populated `sensory.db` with passages from ~10 textual sources tagged by modality and geocoded to the 74-venue gazetteer. The natural next steps are:

- **Phase 2**: Old Bailey API integration (`extract_old_bailey.py`)
- **Phase 3**: Embedding-based Pass 2 extraction (catches oblique sensory language)
- **Phase 5**: Venue explorer UI in the existing Leaflet map
