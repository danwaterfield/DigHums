# Old Bailey Phase 2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ingest Old Bailey Proceedings into `sensory.db`, expand the gazetteer to cover plebeian urban spaces, add a `valence` field to the evidence schema, and surface legal evidence in the venue explorer.

**Architecture:** A new `extract_old_bailey.py` queries the Old Bailey API by venue alias, caches raw JSON at `gazetteer/sources/legal/`, applies the existing lexicon-based sensory tagger, and inserts into `sensory.db` with `venue_id` set directly (no geocoding needed — venue is known from the search alias). The `build_venue_explorer.py` rebuild surfaces the new evidence with a crimson "legal" badge and a valence pip on each card.

**Tech Stack:** Python 3 stdlib only (`urllib.request`, `sqlite3`, `csv`, `json`, `re`); Leaflet 1.9.4 (already in template); pytest for tests.

---

### Task 1: Add new venues to venues.csv and VENUE_ALIASES

Extend the gazetteer with ~22 new London venues covering spaces that legal records describe but fiction largely ignores.

**Files:**
- Modify: `gazetteer/venues.csv`
- Modify: `gazetteer/validate_venues.py` (VENUE_ALIASES dict)

**Step 1: Append new rows to venues.csv**

Open `gazetteer/venues.csv` and append the following rows (keeping the existing header `id,name,city,lat,lon,opened,closed,notes,map_layer`):

```
LON074,Newgate Prison,London,51.5153,-0.1017,1188,,Primary criminal prison of London; adjacent to the Sessions House. Rebuilt 1770-78 (Architects: George Dance the Younger).,horwood_1799
LON075,Fleet Prison,London,51.5144,-0.1048,1197,1844,Debtors' prison on the east bank of the Fleet River. Demolished 1844.,rocque_1746;horwood_1799
LON076,Marshalsea Prison,London,51.5006,-0.0948,1373,1842,Debtors' prison in Southwark. Site now marked in Borough High Street. Cannot appear post-1842.,rocque_1746;horwood_1799
LON077,Bridewell Prison,London,51.5128,-0.1024,1553,1855,House of correction near Blackfriars Bridge. Originally a royal palace.,rocque_1746;horwood_1799
LON078,King's Bench Prison,London,51.5005,-0.0962,1373,1869,Debtors' and misdemeanour prison in Southwark Borough.,rocque_1746;horwood_1799
LON079,Tothill Fields Bridewell,London,51.4961,-0.1335,1618,,Westminster house of correction. Site now occupied by HM Prison Belmarsh predecessor.,rocque_1746;horwood_1799
LON080,Old Bailey Courthouse,London,51.5151,-0.1017,1539,,The Sessions House on Old Bailey street; Central Criminal Court from 1907. Adjacent to Newgate Prison.,rocque_1746;horwood_1799
LON081,Bow Street Magistrates Court,London,51.5133,-0.1228,1739,,Founded by Henry Fielding 1739; home of the Bow Street Runners.,rocque_1746;horwood_1799
LON082,Smithfield Market,London,51.5188,-0.1008,,,Livestock and meat market; notorious for smell and noise. West Smithfield area.,rocque_1746;horwood_1799
LON083,Billingsgate Fish Market,London,51.5072,-0.0843,,,Fish market on Lower Thames Street below London Bridge. Notorious for coarse language and smell.,rocque_1746;horwood_1799
LON084,Leadenhall Market,London,51.5127,-0.0831,1321,,Poultry and general produce market in the City.,rocque_1746;horwood_1799
LON085,Covent Garden Market,London,51.5127,-0.1228,1654,,Fruit and vegetable market; distinct from the Theatre and Piazza entries.,rocque_1746;horwood_1799
LON086,St Giles / Seven Dials,London,51.5162,-0.1276,,,Dense slum district; large Irish population. Centre of Seven Dials rookery.,rocque_1746;horwood_1799
LON087,Whitechapel,London,51.5163,-0.0647,,,East End district; mixed immigrant population; extensive street trade.,rocque_1746;horwood_1799
LON088,Spitalfields,London,51.5195,-0.0741,,,Huguenot weaver district; silk industry; became increasingly impoverished late 18c.,rocque_1746;horwood_1799
LON089,Wapping,London,51.5073,-0.0535,,,Riverside district; sailors and dockhands; Execution Dock for pirates.,rocque_1746;horwood_1799
LON090,Cheapside,London,51.5144,-0.0943,,,Major commercial thoroughfare through the City of London.,rocque_1746;horwood_1799
LON091,Holborn,London,51.5183,-0.1197,,,Mixed district merging lawyers' quarter (Inns of Court) with dense residential areas.,rocque_1746;horwood_1799
LON092,Fleet Street,London,51.5139,-0.1056,,,Print trade and taverns; connects City to Temple Bar.,rocque_1746;horwood_1799
LON093,Southwark / Borough,London,51.5024,-0.0932,,,South bank district; tanneries; brewing; Bankside entertainment.,rocque_1746;horwood_1799
LON094,Tyburn Gallows,London,51.5140,-0.1648,,1783,Public execution site at the north-east corner of Hyde Park (near modern Marble Arch). Cannot appear post-1783.,rocque_1746
LON095,Newgate Gallows,London,51.5151,-0.1017,1783,,Public executions moved outside Newgate Prison after 1783. Cannot appear pre-1783.,horwood_1799
```

**Step 2: Add aliases to VENUE_ALIASES in validate_venues.py**

Open `gazetteer/validate_venues.py`. Find the `VENUE_ALIASES` dict (around line 56). Add the following entries before the closing `}`:

```python
    # PRISONS
    "LON074": [
        ("Newgate", None),
        ("Newgate Prison", None),
        ("Newgate Gaol", None),
    ],
    "LON075": [
        ("Fleet Prison", None),
        ("Fleet Gaol", None),
        ("the Fleet", None),
    ],
    "LON076": [
        ("Marshalsea", None),
        ("Marshalsea Prison", None),
    ],
    "LON077": [
        ("Bridewell", None),
        ("Bridewell Prison", None),
        ("House of Correction", None),
    ],
    "LON078": [
        ("King's Bench", None),
        ("King's Bench Prison", None),
        ("Kings Bench", None),
    ],
    "LON079": [
        ("Tothill Fields", None),
        ("Tothill", None),
    ],
    # COURTS
    "LON080": [
        ("Old Bailey", None),
        ("Sessions House", None),
        ("the Sessions", None),
    ],
    "LON081": [
        ("Bow Street", None),
        ("Bow-Street", None),
    ],
    # MARKETS
    "LON082": [
        ("Smithfield", None),
        ("West Smithfield", None),
        ("Bartholomew Fair", None),
    ],
    "LON083": [
        ("Billingsgate", None),
        ("Billings-gate", None),
    ],
    "LON084": [
        ("Leadenhall", None),
        ("Leaden-hall", None),
    ],
    "LON085": [
        ("Covent Garden Market", None),
        ("the Market", "London"),
    ],
    # ROOKERIES AND STREET AREAS
    "LON086": [
        ("St Giles", None),
        ("Saint Giles", None),
        ("Seven Dials", None),
    ],
    "LON087": [
        ("Whitechapel", None),
        ("White-chapel", None),
    ],
    "LON088": [
        ("Spitalfields", None),
        ("Spital-fields", None),
    ],
    "LON089": [
        ("Wapping", None),
        ("Execution Dock", None),
    ],
    "LON090": [
        ("Cheapside", None),
        ("Cheap-side", None),
    ],
    "LON091": [
        ("Holborn", None),
        ("Hol-born", None),
    ],
    "LON092": [
        ("Fleet Street", None),
        ("Fleet-Street", None),
    ],
    "LON093": [
        ("Southwark", None),
        ("Borough", "London"),
        ("Bankside", None),
    ],
    # EXECUTION SITES
    "LON094": [
        ("Tyburn", None),
        ("Tyburn Tree", None),
        ("Tyburn Gallows", None),
    ],
    "LON095": [
        ("Newgate Gallows", None),
        ("outside Newgate", None),
    ],
```

**Step 3: Verify row count**

```bash
python3 -c "
import csv
with open('gazetteer/venues.csv') as f:
    rows = list(csv.DictReader(f))
print(f'{len(rows)} venues total')
assert len(rows) == 95, f'Expected 95, got {len(rows)}'
"
```

Expected output: `95 venues total`

**Step 4: Commit**

```bash
git add gazetteer/venues.csv gazetteer/validate_venues.py
git commit -m "feat: expand gazetteer with 22 new legal/plebeian London venues"
```

---

### Task 2: Add valence column to sensory_evidence schema

Add a `valence TEXT` column (nullable; values: `'pleasant'`, `'neutral'`, `'unpleasant'`, or `NULL`). Uses `ALTER TABLE` migration so existing `sensory.db` is updated without recreation.

**Files:**
- Modify: `gazetteer/sensory_db.py`
- Test: `gazetteer/tests/test_sensory_db.py`

**Step 1: Write the failing test**

Open `gazetteer/tests/test_sensory_db.py` and add:

```python
def test_valence_column_exists(tmp_path):
    """init_db() must create the valence column on a fresh DB."""
    conn = init_db(tmp_path / "v.db")
    cols = {row[1] for row in conn.execute("PRAGMA table_info(sensory_evidence)")}
    assert "valence" in cols

def test_valence_column_migrated(tmp_path):
    """init_db() adds valence column to an existing DB that lacks it."""
    import sqlite3
    # Create a DB without the valence column
    old_ddl = """
    CREATE TABLE sensory_evidence (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id TEXT NOT NULL,
        modality TEXT NOT NULL,
        text TEXT NOT NULL,
        char_offset INTEGER,
        UNIQUE(source_id, char_offset, modality)
    );
    CREATE TABLE sources (
        source_id TEXT PRIMARY KEY, source_type TEXT NOT NULL,
        author TEXT NOT NULL, title TEXT NOT NULL
    );
    """
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(old_ddl)
    conn.close()
    # Now call init_db — it should add the column
    conn2 = init_db(db_path)
    cols = {row[1] for row in conn2.execute("PRAGMA table_info(sensory_evidence)")}
    assert "valence" in cols
```

**Step 2: Run to verify it fails**

```bash
cd /Users/danielwaterfield/Documents/DigHums
python3 -m pytest gazetteer/tests/test_sensory_db.py::test_valence_column_exists gazetteer/tests/test_sensory_db.py::test_valence_column_migrated -v
```

Expected: FAIL — `valence` not in cols.

**Step 3: Update sensory_db.py**

In `gazetteer/sensory_db.py`:

1. Add `valence TEXT` to the DDL's `sensory_evidence` table (after the `notes TEXT` column, before the `UNIQUE` constraint):

```python
    notes       TEXT,
    valence     TEXT,
    UNIQUE(source_id, char_offset, modality)
```

2. Add a migration block at the end of `init_db()`, after `conn.executescript(DDL)`:

```python
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(DDL)
    # Migrate: add valence column to existing DBs that pre-date this field
    cols = {row[1] for row in conn.execute("PRAGMA table_info(sensory_evidence)")}
    if "valence" not in cols:
        conn.execute("ALTER TABLE sensory_evidence ADD COLUMN valence TEXT")
        conn.commit()
    conn.commit()
    return conn
```

**Step 4: Run tests to verify they pass**

```bash
python3 -m pytest gazetteer/tests/test_sensory_db.py -v
```

Expected: all tests PASS.

**Step 5: Run migration against live sensory.db**

```bash
python3 -c "
import sys; sys.path.insert(0, 'gazetteer')
from sensory_db import init_db, DB_PATH_DEFAULT
conn = init_db(DB_PATH_DEFAULT)
cols = {row[1] for row in conn.execute('PRAGMA table_info(sensory_evidence)')}
print('valence' in cols, '— valence column present')
"
```

Expected: `True — valence column present`

**Step 6: Commit**

```bash
git add gazetteer/sensory_db.py gazetteer/tests/test_sensory_db.py
git commit -m "feat: add valence column to sensory_evidence schema with migration"
```

---

### Task 3: Add tag_valence() to sensory_lexicon.py

Add a function that classifies a sensory passage as `'pleasant'`, `'unpleasant'`, or `'neutral'` based on a small period-specific lexicon.

**Files:**
- Modify: `gazetteer/sensory_lexicon.py`
- Test: `gazetteer/tests/test_sensory_lexicon.py`

**Step 1: Write the failing tests**

Open `gazetteer/tests/test_sensory_lexicon.py` and add:

```python
from sensory_lexicon import tag_valence

def test_unpleasant_terms():
    assert tag_valence("the stench was insupportable") == "unpleasant"
    assert tag_valence("a most fetid and noisome vapour") == "unpleasant"
    assert tag_valence("the mob jostled and reeked") == "unpleasant"

def test_pleasant_terms():
    assert tag_valence("the fragrance of the roses was delightful") == "pleasant"
    assert tag_valence("soft music floated through the air") == "pleasant"
    assert tag_valence("a most charming and elegant prospect") == "pleasant"

def test_neutral():
    assert tag_valence("the crowd moved slowly along the street") == "neutral"

def test_unpleasant_beats_pleasant():
    """When both occur, unpleasant wins (more specific signal)."""
    assert tag_valence("fragrant smoke and stench mingled together") == "unpleasant"
```

**Step 2: Run to verify failure**

```bash
python3 -m pytest gazetteer/tests/test_sensory_lexicon.py::test_unpleasant_terms -v
```

Expected: FAIL — `ImportError: cannot import name 'tag_valence'`

**Step 3: Implement tag_valence() in sensory_lexicon.py**

Add the following at the end of `gazetteer/sensory_lexicon.py`:

```python
# ── VALENCE LEXICON ────────────────────────────────────────────────────────
# Period-specific terms. Unpleasant is checked first; if any unpleasant term
# matches, the passage is classified 'unpleasant' regardless of pleasant hits.

_UNPLEASANT_TERMS = [
    r"\bstench\b", r"\breek\b", r"\breeking\b", r"\bfetid\b",
    r"\bnoisome\b", r"\bputrid\b", r"\beffluvia\b", r"\bmiasma\b",
    r"\bsqualor\b", r"\bfilth\b", r"\bfilthy\b", r"\bdirt\b",
    r"\bdirty\b", r"\bfoul\b", r"\bnoxious\b", r"\bkennel\b",
    r"\bdin\b", r"\bhubbub\b", r"\buproar\b", r"\btumult\b",
    r"\bdiscord\b", r"\bshriek\b", r"\bclamour\b", r"\bjostle\b",
    r"\bjostled\b", r"\bmob\b", r"\bsqualid\b",
]

_PLEASANT_TERMS = [
    r"\bfragrant\b", r"\bfragrance\b", r"\bperfume\b", r"\bperfumed\b",
    r"\bmusic\b", r"\bmelody\b", r"\bharmony\b", r"\bcharming\b",
    r"\belegant\b", r"\bdelightful\b", r"\bpleasant\b", r"\bsweet\b",
    r"\bgay\b", r"\bbright\b", r"\billuminated\b", r"\bdazzling\b",
]

_UNPLEASANT_RE = [re.compile(p, re.IGNORECASE) for p in _UNPLEASANT_TERMS]
_PLEASANT_RE   = [re.compile(p, re.IGNORECASE) for p in _PLEASANT_TERMS]


def tag_valence(text: str) -> str:
    """Return 'unpleasant', 'pleasant', or 'neutral' for a text passage.

    Unpleasant takes precedence when both occur.
    """
    for pat in _UNPLEASANT_RE:
        if pat.search(text):
            return "unpleasant"
    for pat in _PLEASANT_RE:
        if pat.search(text):
            return "pleasant"
    return "neutral"
```

**Step 4: Run all lexicon tests**

```bash
python3 -m pytest gazetteer/tests/test_sensory_lexicon.py -v
```

Expected: all PASS.

**Step 5: Commit**

```bash
git add gazetteer/sensory_lexicon.py gazetteer/tests/test_sensory_lexicon.py
git commit -m "feat: add tag_valence() to sensory lexicon with pleasant/unpleasant classification"
```

---

### Task 4: Thread valence through extract_sensory.py

Update `extract_from_text()` to compute and store `valence` for each passage.

**Files:**
- Modify: `gazetteer/extract_sensory.py`
- Test: `gazetteer/tests/test_extract_sensory.py`

**Step 1: Write the failing test**

Add to `gazetteer/tests/test_extract_sensory.py`:

```python
def test_extract_includes_valence(tmp_path):
    db = init_db(tmp_path / "t.db")
    text = ("We went to Vauxhall last Tuesday. The stench was insupportable "
            "and the din quite unbearable.")
    rows = extract_from_text(
        text=text, source_id="test_valence", source_type="diary",
        author="Test", title="Test", pub_year=1760,
        date_min=1758, date_max=1762, venues=VENUES, conn=db
    )
    assert len(rows) > 0
    for r in rows:
        assert "valence" in r
        assert r["valence"] in ("pleasant", "neutral", "unpleasant")

def test_extract_writes_valence_to_db(tmp_path):
    db = init_db(tmp_path / "t.db")
    text = ("We went to Vauxhall. The stench was insupportable.")
    extract_from_text(
        text=text, source_id="test_val_write", source_type="diary",
        author="Test", title="Test", pub_year=1760,
        date_min=1758, date_max=1762, venues=VENUES, conn=db,
        write=True
    )
    rows = db.execute(
        "SELECT valence FROM sensory_evidence WHERE source_id='test_val_write'"
    ).fetchall()
    assert len(rows) > 0
    assert rows[0][0] in ("pleasant", "neutral", "unpleasant")
```

**Step 2: Run to verify failure**

```bash
python3 -m pytest gazetteer/tests/test_extract_sensory.py::test_extract_includes_valence -v
```

Expected: FAIL — `'valence' not in r`

**Step 3: Update extract_sensory.py**

1. Import `tag_valence` at the top of the imports block:

```python
from sensory_lexicon import tag_modalities, tag_valence
```

2. In `extract_from_text()`, after `"confidence": 1.0,` in the row dict, add:

```python
                "valence":     tag_valence(passage),
```

3. In the `INSERT OR IGNORE INTO sensory_evidence` statement, add `valence` to both the column list and VALUES:

```python
                conn.execute("""
                    INSERT OR IGNORE INTO sensory_evidence
                    (source_id, venue_id, venue_name, lat, lon,
                     source_type, author, title, pub_year, date_min,
                     date_max, modality, text, context, char_offset,
                     pos, confidence, valence)
                    VALUES
                    (:source_id, :venue_id, :venue_name, :lat, :lon,
                     :source_type, :author, :title, :pub_year, :date_min,
                     :date_max, :modality, :text, :context, :char_offset,
                     :pos, :confidence, :valence)
                """, row)
```

**Step 4: Run all extract tests**

```bash
python3 -m pytest gazetteer/tests/test_extract_sensory.py -v
```

Expected: all PASS.

**Step 5: Commit**

```bash
git add gazetteer/extract_sensory.py gazetteer/tests/test_extract_sensory.py
git commit -m "feat: thread valence field through extract_from_text() and DB insert"
```

---

### Task 5: Create extract_old_bailey.py

Write the Old Bailey API extraction script with caching, date filtering, and sensory tagging.

**Files:**
- Create: `gazetteer/extract_old_bailey.py`
- Create: `gazetteer/sources/legal/.gitkeep`
- Create: `gazetteer/tests/test_extract_old_bailey.py`

**Step 1: Create cache directory**

```bash
mkdir -p gazetteer/sources/legal
touch gazetteer/sources/legal/.gitkeep
```

**Step 2: Write the failing tests**

Create `gazetteer/tests/test_extract_old_bailey.py`:

```python
"""Tests for extract_old_bailey.py"""

import json
import sys
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from sensory_db import init_db
from extract_old_bailey import (
    parse_date, alias_slug, extract_trial, fetch_page
)


# ── unit tests ──────────────────────────────────────────────────────────────

def test_parse_date():
    assert parse_date(17650116) == 1765
    assert parse_date(17001231) == 1700
    assert parse_date(18200101) == 1820


def test_alias_slug():
    assert alias_slug("Newgate Prison") == "newgate_prison"
    assert alias_slug("King's Bench") == "king_s_bench"
    assert alias_slug("St Giles / Seven Dials") == "st_giles_seven_dials"


def test_extract_trial_finds_sensory_terms():
    text = ("The prisoner was seen near Newgate. The stench of the gaol "
            "was insupportable, and the din of the crowd overwhelming.")
    rows = extract_trial(
        text=text,
        venue_id="LON074",
        venue_name="Newgate Prison",
        lat=51.5153, lon=-0.1017,
        reference="t17650116-1",
        year=1765,
    )
    assert len(rows) > 0
    modalities = {r["modality"] for r in rows}
    assert "olfactory" in modalities or "auditory" in modalities


def test_extract_trial_sets_venue_directly():
    text = "The stench near the Sessions House was intolerable."
    rows = extract_trial(
        text=text,
        venue_id="LON080",
        venue_name="Old Bailey Courthouse",
        lat=51.5151, lon=-0.1017,
        reference="t17700601-2",
        year=1770,
    )
    assert all(r["venue_id"] == "LON080" for r in rows)
    assert all(r["source_type"] == "legal" for r in rows)


def test_extract_trial_valence_is_unpleasant():
    text = "The stench of Newgate was insupportable."
    rows = extract_trial(
        text=text,
        venue_id="LON074", venue_name="Newgate Prison",
        lat=51.5153, lon=-0.1017,
        reference="t17650116-3", year=1765,
    )
    assert len(rows) > 0
    assert all(r["valence"] == "unpleasant" for r in rows)


def test_extract_trial_source_id_format():
    text = "The mob jostled near the Fleet."
    rows = extract_trial(
        text=text,
        venue_id="LON075", venue_name="Fleet Prison",
        lat=51.5144, lon=-0.1048,
        reference="t17800501-5", year=1780,
    )
    assert len(rows) > 0
    assert all(r["source_id"] == "old_bailey_t17800501-5" for r in rows)


# ── caching test ─────────────────────────────────────────────────────────────

def test_fetch_page_uses_cache(tmp_path):
    """fetch_page() reads cache file and does not call urlopen."""
    cache_data = {"records": [], "total": 0}
    cache_file = tmp_path / "test_p0.json"
    cache_file.write_text(json.dumps(cache_data), encoding="utf-8")

    with patch("urllib.request.urlopen") as mock_url:
        result = fetch_page("Newgate", 0, cache_file)
        mock_url.assert_not_called()

    assert result == cache_data


def test_fetch_page_writes_cache(tmp_path):
    """fetch_page() writes response to cache on first fetch."""
    cache_file = tmp_path / "newgate_p0.json"
    fake_response = json.dumps({"records": [], "total": 0}).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.read.return_value = fake_response
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with patch("time.sleep"):
            result = fetch_page("Newgate", 0, cache_file)

    assert cache_file.exists()
    assert result == {"records": [], "total": 0}


# ── integration test ──────────────────────────────────────────────────────────

def test_ingest_single_trial(tmp_path):
    """End-to-end: fake API response → rows in sensory_evidence."""
    import extract_old_bailey as eob

    # Patch the CACHE_DIR and API
    fake_trial = {
        "records": [
            {
                "reference": "t17650116-99",
                "date": 17650116,
                "text": "Near Newgate. The stench of the place was insupportable "
                        "and the din of the condemned dreadful.",
            }
        ],
        "total": 1,
    }

    db = init_db(tmp_path / "test.db")

    with patch.object(eob, "fetch_page", return_value=fake_trial):
        with patch.object(eob, "CACHE_DIR", tmp_path):
            eob.ingest_venue(
                venue={"id": "LON074", "name": "Newgate Prison",
                       "lat": "51.5153", "lon": "-0.1017"},
                aliases=[("Newgate", None)],
                conn=db,
                write=True,
            )

    count = db.execute(
        "SELECT COUNT(*) FROM sensory_evidence WHERE source_type='legal'"
    ).fetchone()[0]
    assert count > 0
```

**Step 3: Run to verify failure**

```bash
python3 -m pytest gazetteer/tests/test_extract_old_bailey.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'extract_old_bailey'`

**Step 4: Implement extract_old_bailey.py**

Create `gazetteer/extract_old_bailey.py`:

```python
#!/usr/bin/env python3
"""
Phase 2: Old Bailey Proceedings extraction.

Queries the Old Bailey API by venue alias, caches raw JSON responses,
extracts sensory passages using the existing lexicon tagger, and writes
to sensory.db with venue_id set directly (no geocoding needed).

Usage:
    python3 gazetteer/extract_old_bailey.py            # dry run
    python3 gazetteer/extract_old_bailey.py --write    # write to sensory.db
"""

import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sensory_db import init_db, DB_PATH_DEFAULT
from sensory_lexicon import tag_modalities, tag_valence
from validate_venues import VENUE_ALIASES

VENUES_PATH = Path(__file__).parent / "venues.csv"
CACHE_DIR   = Path(__file__).parent / "sources" / "legal"
API_BASE    = "https://www.dhi.ac.uk/api/data/oldbailey_record"

DATE_MIN  = 1660
DATE_MAX  = 1820
PAGE_SIZE = 10
SLEEP_SECS = 0.3

# IDs of the new venues added in Phase 2 (the ones to query against Old Bailey)
TARGET_IDS = {
    "LON074", "LON075", "LON076", "LON077", "LON078", "LON079",
    "LON080", "LON081",
    "LON082", "LON083", "LON084", "LON085",
    "LON086", "LON087", "LON088", "LON089", "LON090", "LON091",
    "LON092", "LON093",
    "LON094", "LON095",
}


def parse_date(date_int: int) -> int:
    """Extract year from YYYYMMDD integer."""
    return date_int // 10000


def alias_slug(alias: str) -> str:
    """Convert alias to a safe filename fragment."""
    return re.sub(r"[^a-z0-9]+", "_", alias.lower()).strip("_")


def fetch_page(alias: str, offset: int, cache_path: Path) -> dict:
    """Fetch one API page, returning cached JSON if available."""
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    encoded = urllib.parse.quote(alias)
    url = f"{API_BASE}?text={encoded}&_limit={PAGE_SIZE}&_offset={offset}"
    with urllib.request.urlopen(url, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    time.sleep(SLEEP_SECS)
    return data


def extract_trial(
    text: str,
    venue_id: str,
    venue_name: str,
    lat: float,
    lon: float,
    reference: str,
    year: int,
) -> list[dict]:
    """Extract sensory passages from a single trial text.

    Venue is known from the search alias — no geocoding required.
    All legal passages default to valence='unpleasant'.
    """
    results: list[dict] = []
    seen_offsets: set[int] = set()

    for term, modality in tag_modalities(text):
        term_pos = text.lower().find(term.lower())
        if term_pos < 0 or term_pos in seen_offsets:
            continue
        seen_offsets.add(term_pos)

        ctx_start = max(0, term_pos - 200)
        ctx_end   = min(len(text), term_pos + 200)
        passage   = text[ctx_start:ctx_end].strip()

        results.append({
            "source_id":   f"old_bailey_{reference}",
            "venue_id":    venue_id,
            "venue_name":  venue_name,
            "lat":         float(lat),
            "lon":         float(lon),
            "source_type": "legal",
            "author":      "Old Bailey Proceedings",
            "title":       f"Trial {reference}",
            "pub_year":    year,
            "date_min":    year,
            "date_max":    year,
            "modality":    modality,
            "text":        passage[:500],
            "context":     term,
            "char_offset": term_pos,
            "pos":         round(term_pos / max(len(text), 1), 4),
            "confidence":  1.0,
            "valence":     "unpleasant",
        })

    return results


def ingest_venue(
    venue: dict,
    aliases: list[tuple[str, str | None]],
    conn,
    write: bool = False,
) -> list[dict]:
    """Query Old Bailey API for all aliases of one venue."""
    vid  = venue["id"]
    lat  = venue["lat"]
    lon  = venue["lon"]
    name = venue["name"]
    all_rows: list[dict] = []

    for alias, _city_filter in aliases:
        offset = 0
        while True:
            slug       = alias_slug(alias)
            cache_path = CACHE_DIR / f"{vid}_{slug}_p{offset // PAGE_SIZE}.json"
            data       = fetch_page(alias, offset, cache_path)

            records = data.get("records", [])
            if not records:
                break

            for record in records:
                date_int = record.get("date", 0)
                year     = parse_date(date_int)
                if not (DATE_MIN <= year <= DATE_MAX):
                    continue

                ref  = record.get("reference", f"unknown_{date_int}")
                text = record.get("text", "")
                if not text:
                    continue

                rows = extract_trial(text, vid, name, lat, lon, ref, year)
                all_rows.extend(rows)

                if write:
                    for row in rows:
                        conn.execute("""
                            INSERT OR IGNORE INTO sources
                            (source_id, source_type, author, title,
                             pub_year, date_min, date_max)
                            VALUES
                            (:source_id, :source_type, :author, :title,
                             :pub_year, :date_min, :date_max)
                        """, row)
                        conn.execute("""
                            INSERT OR IGNORE INTO sensory_evidence
                            (source_id, venue_id, venue_name, lat, lon,
                             source_type, author, title, pub_year, date_min,
                             date_max, modality, text, context, char_offset,
                             pos, confidence, valence)
                            VALUES
                            (:source_id, :venue_id, :venue_name, :lat, :lon,
                             :source_type, :author, :title, :pub_year, :date_min,
                             :date_max, :modality, :text, :context, :char_offset,
                             :pos, :confidence, :valence)
                        """, row)

            if len(records) < PAGE_SIZE:
                break
            offset += PAGE_SIZE

    if write:
        conn.commit()

    return all_rows


def run(write: bool = False) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with open(VENUES_PATH, newline="", encoding="utf-8") as f:
        venues = {row["id"]: row for row in csv.DictReader(f)}

    conn  = init_db(DB_PATH_DEFAULT)
    total = 0

    for vid in sorted(TARGET_IDS):
        venue   = venues.get(vid)
        aliases = VENUE_ALIASES.get(vid, [])
        if not venue or not aliases:
            print(f"  [skip] {vid} — no venue or no aliases")
            continue

        rows = ingest_venue(venue, aliases, conn, write=write)
        geocoded = sum(1 for r in rows if r["venue_id"])
        print(f"  {vid:8s} {venue['name']:35s} {len(rows):4d} passages")
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

**Step 5: Run all tests**

```bash
python3 -m pytest gazetteer/tests/test_extract_old_bailey.py -v
```

Expected: all PASS.

**Step 6: Commit**

```bash
git add gazetteer/extract_old_bailey.py \
        gazetteer/sources/legal/.gitkeep \
        gazetteer/tests/test_extract_old_bailey.py
git commit -m "feat: add extract_old_bailey.py with API caching and sensory extraction"
```

---

### Task 6: Update build_venue_explorer.py for legal evidence

Add the "Legal" source pill, crimson badge colour, valence pip, and pull `valence` from the DB.

**Files:**
- Modify: `gazetteer/build_venue_explorer.py`
- Test: `gazetteer/tests/test_build_venue_explorer.py`

**Step 1: Write the failing tests**

Add to `gazetteer/tests/test_build_venue_explorer.py`:

```python
def test_legal_pill_present():
    """Generated HTML must contain a Legal source filter pill."""
    html = (Path(__file__).parent.parent / "venue_explorer.html").read_text()
    assert 'data-v="legal"' in html


def test_legal_badge_colour_present():
    """Generated HTML must define the src-legal CSS class."""
    html = (Path(__file__).parent.parent / "venue_explorer.html").read_text()
    assert "src-legal" in html


def test_valence_pip_in_js():
    """renderCard must include valence pip logic."""
    html = (Path(__file__).parent.parent / "venue_explorer.html").read_text()
    assert "valence-pip" in html
```

**Step 2: Run to verify failure**

```bash
python3 -m pytest gazetteer/tests/test_build_venue_explorer.py::test_legal_pill_present \
                   gazetteer/tests/test_build_venue_explorer.py::test_legal_badge_colour_present \
                   gazetteer/tests/test_build_venue_explorer.py::test_valence_pip_in_js -v
```

Expected: FAIL — items not found in HTML.

**Step 3: Update load_data() to include valence**

In `gazetteer/build_venue_explorer.py`, update the `conn.execute` SELECT in `load_data()` to include `valence`:

```python
        for row in conn.execute("""
            SELECT venue_id, source_type, author, title, pub_year,
                   date_min, date_max, modality, text, context, valence
            FROM   sensory_evidence
            WHERE  venue_id IS NOT NULL
            ORDER  BY date_min
        """):
```

And add `"valence"` to the evidence dict:

```python
                venues[vid]["evidence"].append({
                    "source_type": row["source_type"],
                    "author":      fmt_author(row["author"] or ""),
                    "title":       row["title"] or "",
                    "pub_year":    row["pub_year"],
                    "date_min":    row["date_min"],
                    "date_max":    row["date_max"],
                    "modality":    row["modality"],
                    "text":        row["text"] or "",
                    "context":     row["context"] or "",
                    "valence":     row["valence"],
                })
```

**Step 4: Update HTML_TEMPLATE**

Make the following targeted changes to `HTML_TEMPLATE` in `build_venue_explorer.py`:

**4a. Add `.src-legal` CSS after `.src-letters`:**
```css
.src-letters    { background: #e8e8e8; color: #444; }
.src-legal      { background: #f5e0e0; color: #8b1a1a; }
```

**4b. Add `.valence-pip` CSS after `.context-chip`:**
```css
.valence-pip {
  width: 8px; height: 8px; border-radius: 50%;
  display: inline-block; flex-shrink: 0;
  margin-left: 4px; align-self: center;
}
.valence-pip.unpleasant { background: #c0392b; opacity: 0.55; }
.valence-pip.pleasant   { background: #9a6f2a; opacity: 0.55; }
```

**4c. Add "Legal" pill to the source filter group in the HTML body** (after the Letters pill):
```html
    <button class="pill active" data-f="source" data-v="letters">Letters</button>
    <button class="pill active" data-f="source" data-v="legal">Legal</button>
```

**4d. Add `legal` to `SOURCE_CLASSES` in the JS:**
```js
const SOURCE_CLASSES = {
  fiction: 'src-fiction', diary: 'src-diary',
  topography: 'src-topography', poetry: 'src-poetry',
  letters: 'src-letters', legal: 'src-legal',
};
```

**4e. Add `'legal'` to `state.sources` Set:**
```js
  sources: new Set(['fiction','diary','topography','poetry','letters','legal']),
```

**4f. Add valence pip to `renderCard()`** — add after the `context-chip` span, inside `ev-footer`:
```js
  return '<div class="ev-card">'
    + '<div class="ev-card-head">'
    + '<span class="source-badge ' + cls + '">' + ev.source_type + '</span>'
    + '<span class="ev-author">' + esc(ev.author) + '</span>'
    + '<span class="ev-title">' + esc(ev.title) + '</span>'
    + (ev.valence === 'unpleasant' ? '<span class="valence-pip unpleasant" title="unpleasant"></span>'
       : ev.valence === 'pleasant'  ? '<span class="valence-pip pleasant" title="pleasant"></span>'
       : '')
    + '</div>'
    + '<div class="ev-date">' + dateStr + '</div>'
    + '<div class="ev-text">\u201c' + esc(text) + '\u201d</div>'
    + '<div class="ev-footer"><span class="context-chip">' + esc(ev.context) + '</span></div>'
    + '</div>';
```

Also add `'legal'` to the panel modality pills reset inside `openPanel()`:

The panel pills are *modality* pills, not source pills — no change needed there. But add `'legal'` to the `renderPanel()` local pills if they show source filtering. Looking at the existing code, panel pills filter by modality, not source. No change needed.

**Step 5: Rebuild the HTML**

```bash
python3 gazetteer/build_venue_explorer.py
```

Expected output:
```
Venue explorer -> gazetteer/venue_explorer.html
  95 venues  N with evidence  M passages
```

**Step 6: Run all tests**

```bash
python3 -m pytest gazetteer/tests/test_build_venue_explorer.py -v
```

Expected: all PASS.

**Step 7: Commit**

```bash
git add gazetteer/build_venue_explorer.py gazetteer/tests/test_build_venue_explorer.py
git commit -m "feat: add legal badge, valence pip, and Legal filter pill to venue explorer"
```

---

### Task 7: Ingest Old Bailey data and final rebuild

Run the extraction script against the live DB, verify counts, rebuild the HTML.

**Files:**
- Run: `gazetteer/extract_old_bailey.py --write`
- Run: `gazetteer/build_venue_explorer.py`
- Modify: `gazetteer/venue_explorer.html` (regenerated)

**Step 1: Dry run first**

```bash
python3 gazetteer/extract_old_bailey.py
```

Check: output shows passage counts per venue. No errors. Verify the API is reachable and returns results for at least Newgate, Smithfield, Tyburn.

**Step 2: Write to DB**

```bash
python3 gazetteer/extract_old_bailey.py --write
```

**Step 3: Verify DB counts**

```bash
python3 -c "
import sys; sys.path.insert(0, 'gazetteer')
import sqlite3
conn = sqlite3.connect('gazetteer/sensory.db')
total  = conn.execute('SELECT COUNT(*) FROM sensory_evidence').fetchone()[0]
legal  = conn.execute(\"SELECT COUNT(*) FROM sensory_evidence WHERE source_type='legal'\").fetchone()[0]
geocod = conn.execute('SELECT COUNT(*) FROM sensory_evidence WHERE venue_id IS NOT NULL').fetchone()[0]
print(f'Total:   {total}')
print(f'Legal:   {legal}')
print(f'Geocoded: {geocod}')
"
```

Expected: `Legal` > 0; `Total` > previous 8,099.

**Step 4: Rebuild the venue explorer**

```bash
python3 gazetteer/build_venue_explorer.py
```

**Step 5: Smoke-test the HTML in browser**

```bash
open gazetteer/venue_explorer.html
```

Verify:
- "Legal" pill visible in top bar
- Clicking Newgate Prison (LON074) shows crimson legal badges in panel
- Valence pips visible on unpleasant cards
- Existing fiction/diary evidence unaffected

**Step 6: Commit**

```bash
git add gazetteer/venue_explorer.html
git commit -m "feat: ingest Old Bailey evidence and rebuild venue explorer with legal evidence"
```

---

### Task 8: Update README

**Files:**
- Modify: `README.md`

**Step 1: Update stats and add Old Bailey section**

In `README.md`, find the existing stats (8,099 passages, 565 geocoded) and update with new numbers from the DB query in Task 7 Step 3.

Add a brief note under the Venue Explorer section:

```markdown
**Sources:** 18th-century fiction (13 authors, 28 texts), diaries, topography, poetry, letters, and
Old Bailey Proceedings (legal records, 1660–1820). Legal evidence surfaces in the Venue Explorer
under the "Legal" filter; crimson badges distinguish trial records from literary sources.
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README with Old Bailey phase 2 stats and sources"
```
