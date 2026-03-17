# Correspondent Network Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an interactive D3 force-directed network visualisation of Burney's correspondents from the OUP *Journals and Letters* headers.

**Architecture:** Python build script parses letter headers from the OUP text, extracts correspondents/dates/types, and emits a self-contained HTML file with inlined D3.js and all data as JSON. The HTML renders a force graph with a dual-handle timeline slider and a slide-in detail panel.

**Tech Stack:** Python 3 (stdlib only), D3.js v7 (inlined), pytest

**Spec:** `docs/superpowers/specs/2026-03-17-correspondent-network-design.md`

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `gazetteer/build_correspondent_network.py` | Parse headers, normalise names, assign communities/phases, emit HTML |
| Create | `gazetteer/tests/test_build_correspondent_network.py` | All unit/integration tests |
| Create | `gazetteer/correspondent_network.html` | Generated output (not committed) |

The build script follows the `build_comparison.py` pattern: a `parse_headers()` function for data extraction, a `build()` function for orchestration, and an `HTML_TEMPLATE` string with doubled JS braces.

---

## Task 1: Header Parsing — Test & Implement

**Files:**
- Create: `gazetteer/tests/test_build_correspondent_network.py`
- Create: `gazetteer/build_correspondent_network.py`

- [ ] **Step 1: Write failing tests for header parsing**

```python
"""Tests for gazetteer/build_correspondent_network.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from build_correspondent_network import parse_headers


# --- Fixture: a minimal subset of real headers ---

SAMPLE_TEXT = """\
1. Journal 27 March 1768

To have some account of my thoughts...

9. Verse Letter to Dr Charles Burney 23 June 1769

To Doctor Last...

17. From Journal Letters to Susanna Burney
(Teignmouth Journal) August–September 1773

We set out from Teignmouth...

44. From Letter to Susanna Burney 5 July 1778

My dearest Susy...

79. From Journal Letter to Susanna Burney and Charlotte Ann Burney June 1781

My dearest girls...

103. From Letter to Charlotte Cambridge 12June 1786

My dearest Charlotte...

209. Waterloo Journal 27 April and 13 May 1815

My best friend left me...

26. From Letter to Samuel Crisp 2 March 1775 and Journal 1775

My dear Daddy Crisp...

36. Letters to Thomas Lowndes 25 and 26 December 1776

Sir, I take the liberty...

214. Journal for 22 July 1815

The day began...
"""


def test_parse_headers_counts():
    entries = parse_headers(SAMPLE_TEXT)
    # 10 numbered selections total
    assert len(entries) == 10


def test_pure_journal_has_no_correspondent():
    entries = parse_headers(SAMPLE_TEXT)
    journal = [e for e in entries if e["number"] == 1][0]
    assert journal["correspondents"] == []
    assert journal["type"] == "journal"
    assert journal["year"] == 1768


def test_letter_extracts_correspondent():
    entries = parse_headers(SAMPLE_TEXT)
    letter = [e for e in entries if e["number"] == 44][0]
    assert letter["correspondents"] == ["Susanna Burney"]
    assert letter["type"] == "letter"
    assert letter["year"] == 1778


def test_verse_letter_extracts_correspondent():
    entries = parse_headers(SAMPLE_TEXT)
    verse = [e for e in entries if e["number"] == 9][0]
    assert verse["correspondents"] == ["Dr Charles Burney"]
    assert verse["type"] == "verse letter"


def test_multi_recipient_splits():
    entries = parse_headers(SAMPLE_TEXT)
    multi = [e for e in entries if e["number"] == 79][0]
    assert set(multi["correspondents"]) == {
        "Susanna Burney", "Charlotte Ann Burney"
    }


def test_location_journal_excluded():
    entries = parse_headers(SAMPLE_TEXT)
    waterloo = [e for e in entries if e["number"] == 209][0]
    assert waterloo["correspondents"] == []
    assert waterloo["type"] == "journal"


def test_compound_entry_extracts_letter_only():
    entries = parse_headers(SAMPLE_TEXT)
    compound = [e for e in entries if e["number"] == 26][0]
    assert compound["correspondents"] == ["Samuel Crisp"]
    assert compound["year"] == 1775


def test_plural_letters():
    entries = parse_headers(SAMPLE_TEXT)
    plural = [e for e in entries if e["number"] == 36][0]
    assert plural["correspondents"] == ["Thomas Lowndes"]
    assert plural["type"] == "letter"


def test_journal_for_excluded():
    entries = parse_headers(SAMPLE_TEXT)
    jf = [e for e in entries if e["number"] == 214][0]
    assert jf["correspondents"] == []
    assert jf["type"] == "journal"


def test_multi_line_header():
    """Entry 17 has its date on a continuation line."""
    entries = parse_headers(SAMPLE_TEXT)
    e17 = [e for e in entries if e["number"] == 17][0]
    assert e17["correspondents"] == ["Susanna Burney"]
    assert e17["year"] == 1773


def test_month_range_date():
    """August–September 1773 should extract year 1773, month 8."""
    entries = parse_headers(SAMPLE_TEXT)
    e17 = [e for e in entries if e["number"] == 17][0]
    assert e17["month"] == 8


def test_no_space_date():
    """'12June 1786' should still parse correctly."""
    entries = parse_headers(SAMPLE_TEXT)
    e103 = [e for e in entries if e["number"] == 103][0]
    assert e103["year"] == 1786
    assert e103["correspondents"] == ["Charlotte Cambridge"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest gazetteer/tests/test_build_correspondent_network.py -v`
Expected: FAIL — `ImportError: cannot import name 'parse_headers'`

- [ ] **Step 3: Implement `parse_headers()`**

In `gazetteer/build_correspondent_network.py`:

```python
#!/usr/bin/env python3
"""
Build the self-contained correspondent network HTML.

Reads the OUP Journals and Letters text file, parses structured headers
to extract correspondents, dates, and entry types, and writes an
interactive D3 force-directed network visualisation.

Usage:
    python3 gazetteer/build_correspondent_network.py
    open gazetteer/correspondent_network.html
"""

import json
import re
from pathlib import Path

TEXT_PATH = Path(__file__).resolve().parent.parent / (
    "nonfiction/FrancesBurney/JournalsAndLetters.txt"
)
OUT_PATH = Path(__file__).parent / "correspondent_network.html"

# Regex: numbered header line — must start with a known keyword after
# the number to avoid matching footnote lines like "1.  EJL, i. xv".
_HEADER_RE = re.compile(
    r"^(\d+)\.\s+"                           # selection number
    r"(From\s+)?"                            # optional "From"
    r"((?:Journal|Letter|Verse|Letters)\b"   # must start with keyword
    r".+)",                                  # rest of header
    re.MULTILINE,
)

# Secondary pattern: also match lines starting with a known location
# name followed by "Journal" (e.g., "209. Waterloo Journal ...")
_LOC_HEADER_RE = re.compile(
    r"^(\d+)\.\s+"
    r"(From\s+)?"
    r"((?:" + "|".join(["Waterloo", "Ilfracombe"]) + r")\s+Journal\b.+)",
    re.MULTILINE,
)

# Known location-prefixed journals (word before "Journal")
_LOCATION_JOURNALS = {"Waterloo", "Ilfracombe"}

# Date qualifiers to strip
_DATE_QUAL_RE = re.compile(
    r"\b(?:c\.\s*|post\s+|pre-|late\s+|early\s+|mid\s+(?:or\s+late\s+)?|"
    r"between\s+\d+\s+and\s+)"
)

_YEAR_RE = re.compile(r"\b(1[6-8]\d{2})\b")

_MONTH_NAMES = (
    "January|February|March|April|May|June|July|August|"
    "September|October|November|December"
)
_MONTH_RE = re.compile(rf"\b({_MONTH_NAMES})\b")
_MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}


def _parse_date(s: str) -> dict:
    """Extract year and optional month from a date string."""
    cleaned = _DATE_QUAL_RE.sub("", s)
    year_m = _YEAR_RE.search(cleaned)
    month_m = _MONTH_RE.search(cleaned)
    return {
        "year": int(year_m.group(1)) if year_m else None,
        "month": _MONTH_MAP[month_m.group(1)] if month_m else None,
    }


def _classify_and_extract(rest: str) -> tuple[str, list[str], str]:
    """
    Given the rest of a header (after number and optional 'From'),
    return (entry_type, correspondent_list, date_portion).
    """
    # Compound: "Letter to X DATE and Journal DATE" — take letter part
    compound_m = re.match(
        r"((?:Journal\s+)?Letters?\s+to\s+.+?\d{4})\s+and\s+Journal\b",
        rest,
    )
    if compound_m:
        rest = compound_m.group(1)

    # Location-prefixed journal: "Waterloo Journal ..."
    loc_m = re.match(r"(\w+)\s+Journal\b", rest)
    if loc_m and loc_m.group(1) in _LOCATION_JOURNALS:
        date_part = rest[loc_m.end():]
        return "journal", [], date_part.strip()

    # "Journal for DATE" — no correspondent
    if re.match(r"Journal\s+for\b", rest):
        date_part = re.sub(r"^Journal\s+for\s*", "", rest)
        return "journal", [], date_part.strip()

    # Pure journal (no "to"): "Journal DATE" or "Journal Letter(s) to ..."
    # Check if there is a "to" clause
    to_match = re.search(r"\bto\s+", rest, re.IGNORECASE)
    if not to_match:
        # Pure journal
        date_part = re.sub(r"^Journal\s*", "", rest)
        return "journal", [], date_part.strip()

    # Has a "to" — extract type, correspondent, date
    before_to = rest[:to_match.start()].strip()
    after_to = rest[to_match.end():].strip()

    # Determine type
    bt_lower = before_to.lower().rstrip("s")
    if "verse letter" in bt_lower:
        entry_type = "verse letter"
    elif "journal letter" in bt_lower:
        entry_type = "journal letter"
    elif "letter" in bt_lower:
        entry_type = "letter"
    else:
        entry_type = "letter"

    # Split after_to into correspondent(s) and date
    # Date starts at first month name or year
    date_split = re.search(
        rf"(?:\b(?:c\.\s*|post\s+|pre-|late\s+|early\s+|mid\s+))*"
        rf"\b(?:{_MONTH_NAMES}|\d{{1,2}}\s+(?:{_MONTH_NAMES})|1[6-8]\d{{2}})\b",
        after_to,
    )
    if date_split:
        corr_part = after_to[:date_split.start()].strip()
        date_part = after_to[date_split.start():].strip()
    else:
        corr_part = after_to
        date_part = ""

    # Split correspondents on " and " — but only real splits
    correspondents = _split_correspondents(corr_part)

    return entry_type, correspondents, date_part


def _split_correspondents(s: str) -> list[str]:
    """Split 'A and B' into ['A', 'B'], handling compound names.

    After splitting on ' and ', if a fragment is a bare first name
    (single word), try appending the surname from the next fragment
    to produce a recognisable name (checked against NAME_ALIASES and
    COMMUNITIES). E.g. "William and Frederica Locke" → ["William Locke",
    "Frederica Locke"].
    """
    if not s:
        return []
    parts = [p.strip().rstrip(",") for p in re.split(r"\s+and\s+", s) if p.strip()]
    if len(parts) <= 1:
        return parts

    # Check for bare-first-name fragments and try surname grafting
    result = []
    for i, p in enumerate(parts):
        if " " not in p and i + 1 < len(parts):
            # Bare first name — try adding surname from next part
            next_words = parts[i + 1].split()
            if len(next_words) >= 2:
                candidate = p + " " + next_words[-1]
                if candidate in NAME_ALIASES or candidate in COMMUNITIES:
                    result.append(candidate)
                    continue
        if p:
            result.append(p)
    return result


def _find_headers(text: str) -> list[tuple[int, str, int]]:
    """
    Find all header matches, returning (number, rest_text, char_offset).
    Merges _HEADER_RE and _LOC_HEADER_RE results.
    """
    seen = set()
    results = []
    for rx in (_HEADER_RE, _LOC_HEADER_RE):
        for m in rx.finditer(text):
            num = int(m.group(1))
            if num in seen:
                continue
            seen.add(num)
            results.append((num, m.group(3).strip(), m.end()))
    results.sort(key=lambda x: x[2])  # sort by position in text
    return results


def parse_headers(text: str) -> list[dict]:
    """
    Parse all numbered selection headers from the OUP text.

    Returns a list of dicts with keys:
        number, type, correspondents, year, month

    Handles multi-line headers by peeking at the next line when the
    header line itself contains no year.
    """
    lines = text.split("\n")
    entries = []

    for num, rest, char_offset in _find_headers(text):
        entry_type, correspondents, date_part = _classify_and_extract(rest)
        date = _parse_date(date_part)

        # If no year found, check the next non-blank line for a
        # continuation (e.g., entry 17 has date on line 2)
        if date["year"] is None:
            # Find the line index of this match
            preceding = text[:char_offset]
            line_idx = preceding.count("\n")
            for next_idx in range(line_idx, min(line_idx + 3, len(lines))):
                next_line = lines[next_idx].strip()
                if not next_line:
                    continue
                continuation_date = _parse_date(next_line)
                if continuation_date["year"] is not None:
                    date = continuation_date
                    break

        # Skip if still no year (not a real header)
        if date["year"] is None:
            continue

        entries.append({
            "number": num,
            "type": entry_type,
            "correspondents": correspondents,
            "year": date["year"],
            "month": date["month"],
        })

    return entries
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest gazetteer/tests/test_build_correspondent_network.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Run against real data to check parse count**

Run: `python3 -c "from pathlib import Path; import sys; sys.path.insert(0,'gazetteer'); from build_correspondent_network import parse_headers, TEXT_PATH; entries = parse_headers(TEXT_PATH.read_text()); print(f'{len(entries)} entries, {sum(1 for e in entries if e[\"correspondents\"])} with correspondents'); print('No-year:', [e for e in entries if e['year'] is None])"`

Inspect output and adjust the header regex if the count is off. Fix any entries that fail to parse. This is an exploratory step — adjust the regex and re-run until all headers are captured.

- [ ] **Step 6: Commit**

```bash
git add gazetteer/build_correspondent_network.py gazetteer/tests/test_build_correspondent_network.py
git commit -m "feat: add header parsing for correspondent network (Task 1)"
```

---

## Task 2: Name Normalisation & Community Assignment

**Files:**
- Modify: `gazetteer/build_correspondent_network.py`
- Modify: `gazetteer/tests/test_build_correspondent_network.py`

- [ ] **Step 1: Write failing tests**

Add to the test file:

```python
from build_correspondent_network import (
    normalise_name, assign_community, assign_phase,
    NAME_ALIASES, COMMUNITIES,
)


def test_normalise_susanna():
    assert normalise_name("Susanna Burney") == "Susanna Burney Phillips"
    assert normalise_name("Susanna Phillips") == "Susanna Burney Phillips"


def test_normalise_thrale_piozzi():
    assert normalise_name("Hester Lynch Thrale") == "Hester Thrale Piozzi"
    assert normalise_name("Hester Lynch Piozzi") == "Hester Thrale Piozzi"


def test_normalise_dr_burney():
    assert normalise_name("Dr Burney") == "Dr Charles Burney"
    assert normalise_name("Dr Charles Burney") == "Dr Charles Burney"


def test_normalise_charlotte_broome():
    assert normalise_name("Charlotte Cambridge") == "Charlotte Broome"


def test_alexandre_vs_alexander_distinct():
    assert normalise_name("Alexandre d'Arblay") != normalise_name(
        "Alexander d'Arblay"
    )


def test_publisher_normalisation():
    assert normalise_name("Longman, Hurst, Rees, Orme and Brown") == "Longman & Co"
    assert normalise_name("Messrs Longman and Company") == "Longman & Co"


def test_community_family():
    assert assign_community("Dr Charles Burney") == "Family"
    assert assign_community("Susanna Burney Phillips") == "Family"
    assert assign_community("Alexandre d'Arblay") == "Family"
    assert assign_community("Alexander d'Arblay") == "Family"


def test_community_literary():
    assert assign_community("Samuel Crisp") == "Literary"
    assert assign_community("Hester Thrale Piozzi") == "Literary"


def test_community_court():
    assert assign_community("Queen Charlotte") == "Court"


def test_community_publishers():
    assert assign_community("Thomas Lowndes") == "Publishers"
    assert assign_community("Longman & Co") == "Publishers"


def test_community_intimate():
    assert assign_community("Frederica Locke") == "Intimate circle"


def test_community_unknown_flagged(capsys):
    result = assign_community("Unknown Person")
    assert result == "Unknown"
    assert "Unknown Person" in capsys.readouterr().err


def test_phase_apprentice():
    assert assign_phase(1770, None) == "Apprentice Years"


def test_phase_court():
    assert assign_phase(1786, None) == "Court Years"


def test_phase_boundary_1786():
    # 1786 belongs to Court, not Cecilia
    assert assign_phase(1786, None) == "Court Years"


def test_phase_france_mid_1802():
    # July 1802 onward = France
    assert assign_phase(1802, 7) == "France"
    # Before July 1802 = Camilla & Camilla Cottage
    assert assign_phase(1802, 3) == "Camilla & Camilla Cottage"
    # No month, 1802 = Camilla (conservative)
    assert assign_phase(1802, None) == "Camilla & Camilla Cottage"


def test_phase_widowhood():
    assert assign_phase(1820, None) == "Widowhood"
    assert assign_phase(1839, None) == "Widowhood"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest gazetteer/tests/test_build_correspondent_network.py -v -k "normalise or community or phase"`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement normalisation, communities, and phases**

Add to `build_correspondent_network.py`:

```python
NAME_ALIASES = {
    "Susanna Burney": "Susanna Burney Phillips",
    "Susanna Phillips": "Susanna Burney Phillips",
    "Dr Burney": "Dr Charles Burney",
    "Dr Charles Burney": "Dr Charles Burney",
    "Hester Lynch Thrale": "Hester Thrale Piozzi",
    "Hester Lynch Piozzi": "Hester Thrale Piozzi",
    "Charlotte Cambridge": "Charlotte Broome",
    "Charlotte Broome": "Charlotte Broome",
    "Longman, Hurst, Rees, Orme and Brown": "Longman & Co",
    "Messrs Longman and Company": "Longman & Co",
}

COMMUNITIES = {
    "Dr Charles Burney": "Family",
    "Susanna Burney Phillips": "Family",
    "Esther Burney": "Family",
    "Charlotte Broome": "Family",
    "Charlotte Ann Burney": "Family",
    "Charlotte Barrett": "Family",
    "James Burney": "Family",
    "Charles Burney": "Family",
    "Charles Parr Burney": "Family",
    "Maria Rishton": "Family",
    "Alexandre d'Arblay": "Family",
    "Alexander d'Arblay": "Family",
    "Samuel Crisp": "Literary",
    "Samuel Johnson": "Literary",
    "Hester Thrale Piozzi": "Literary",
    "Georgiana Waddington": "Literary",
    "Queen Charlotte": "Court",
    "Princess Elizabeth": "Court",
    "Margaret Planta": "Court",
    "William Lowndes": "Court",
    "Thomas Lowndes": "Publishers",
    "Longman & Co": "Publishers",
    "Frederica Locke": "Intimate circle",
    "Amelia Angerstein": "Intimate circle",
    "William Locke": "Intimate circle",
    "Viscountess Keith": "Intimate circle",
    "William Wilberforce": "Intimate circle",
}

# Phase boundaries: (start_year, start_month, label)
# Sorted newest-first so first match wins.
_PHASES = [
    (1819, 1,  "Widowhood"),
    (1815, 1,  "Final Years with d'Arblay"),
    (1814, 1,  "Waterloo"),
    (1812, 1,  "Interlude / The Wanderer"),
    (1802, 7,  "France"),
    (1796, 1,  "Camilla & Camilla Cottage"),
    (1793, 1,  "Courtship & Marriage"),
    (1791, 1,  "London & Western Tour"),
    (1786, 1,  "Court Years"),
    (1782, 1,  "Cecilia & Prelude to Court"),
    (1778, 1,  "Evelina & Streatham"),
    (1768, 1,  "Apprentice Years"),
]


def normalise_name(name: str) -> str:
    return NAME_ALIASES.get(name, name)


def assign_community(name: str) -> str:
    if name in COMMUNITIES:
        return COMMUNITIES[name]
    import sys
    print(f"WARNING: no community for '{name}'", file=sys.stderr)
    return "Unknown"


def assign_phase(year: int, month: int | None) -> str:
    ym = (year, month or 0)
    for start_y, start_m, label in _PHASES:
        if ym >= (start_y, start_m):
            return label
    return "Apprentice Years"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest gazetteer/tests/test_build_correspondent_network.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add gazetteer/build_correspondent_network.py gazetteer/tests/test_build_correspondent_network.py
git commit -m "feat: add name normalisation, communities, phase assignment (Task 2)"
```

---

## Task 3: JSON Data Assembly

**Files:**
- Modify: `gazetteer/build_correspondent_network.py`
- Modify: `gazetteer/tests/test_build_correspondent_network.py`

- [ ] **Step 1: Write failing tests**

```python
from build_correspondent_network import build_network_data


def test_build_network_data_structure():
    data = build_network_data(SAMPLE_TEXT)
    assert "nodes" in data
    assert "edges" in data
    assert "letters" in data
    # Burney is always a node
    burney = [n for n in data["nodes"] if n["id"] == "Frances Burney"]
    assert len(burney) == 1


def test_build_network_data_edges():
    data = build_network_data(SAMPLE_TEXT)
    # Sample has letters to: Dr Charles Burney, Susanna Burney,
    # Charlotte Ann Burney, Samuel Crisp, Thomas Lowndes
    correspondent_ids = {e["target"] for e in data["edges"]}
    assert "Dr Charles Burney" in correspondent_ids
    assert "Susanna Burney Phillips" in correspondent_ids


def test_build_network_data_no_journal_edges():
    data = build_network_data(SAMPLE_TEXT)
    # Pure journals should not create edges
    for edge in data["edges"]:
        assert edge["target"] != "Frances Burney"


def test_letters_list_has_required_fields():
    data = build_network_data(SAMPLE_TEXT)
    for letter in data["letters"]:
        assert "number" in letter
        assert "correspondent" in letter
        assert "year" in letter
        assert "type" in letter
        assert "phase" in letter
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest gazetteer/tests/test_build_correspondent_network.py -v -k "network_data"`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement `build_network_data()`**

```python
def build_network_data(text: str) -> dict:
    """
    Parse the OUP text and return a dict ready for JSON serialisation:
    {
        "nodes": [{"id": str, "community": str, "count": int, ...}, ...],
        "edges": [{"source": "Frances Burney", "target": str, "weight": int}, ...],
        "letters": [{"number": int, "correspondent": str, "year": int, ...}, ...],
        "journals": [{"number": int, "year": int, "month": int|null}, ...],
        "phases": [{"label": str, "start": int, "end": int}, ...]
    }
    """
    entries = parse_headers(text)

    # Count letters per normalised correspondent
    counts: dict[str, int] = {}
    letters = []
    journals = []

    for e in entries:
        if not e["correspondents"]:
            journals.append({
                "number": e["number"],
                "year": e["year"],
                "month": e["month"],
            })
            continue
        for raw_name in e["correspondents"]:
            name = normalise_name(raw_name)
            counts[name] = counts.get(name, 0) + 1
            letters.append({
                "number": e["number"],
                "correspondent": name,
                "year": e["year"],
                "month": e["month"],
                "type": e["type"],
                "phase": assign_phase(e["year"], e["month"]),
            })

    # Build nodes
    nodes = [{"id": "Frances Burney", "community": "Centre", "count": len(letters)}]
    for name, count in sorted(counts.items(), key=lambda x: -x[1]):
        nodes.append({
            "id": name,
            "community": assign_community(name),
            "count": count,
        })

    # Build edges
    edges = [
        {"source": "Frances Burney", "target": name, "weight": count}
        for name, count in counts.items()
    ]

    # Phase definitions for UI
    phase_defs = [
        {"label": label, "start": sy, "end": 0}
        for sy, _, label in reversed(_PHASES)
    ]
    # Fill end years — use next phase's start year (not start-1)
    # so that the UI slider range matches the spec's overlapping boundaries
    for i, p in enumerate(phase_defs):
        if i + 1 < len(phase_defs):
            p["end"] = phase_defs[i + 1]["start"]
        else:
            p["end"] = 1839

    return {
        "nodes": nodes,
        "edges": edges,
        "letters": letters,
        "journals": journals,
        "phases": phase_defs,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest gazetteer/tests/test_build_correspondent_network.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add gazetteer/build_correspondent_network.py gazetteer/tests/test_build_correspondent_network.py
git commit -m "feat: add network data assembly (Task 3)"
```

---

## Task 4: HTML Template & Build Function

**Files:**
- Modify: `gazetteer/build_correspondent_network.py`
- Modify: `gazetteer/tests/test_build_correspondent_network.py`

- [ ] **Step 1: Write failing tests for `build()`**

```python
import subprocess
import tempfile


def test_build_produces_html():
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        out = Path(f.name)
    try:
        from build_correspondent_network import build
        build(out_path=out)
        html = out.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html.lower() or "<!doctype html>" in html.lower()
        assert "Frances Burney" in html
        assert "d3" in html.lower()
    finally:
        out.unlink(missing_ok=True)


def test_build_html_has_no_unsubstituted_placeholders():
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        out = Path(f.name)
    try:
        from build_correspondent_network import build
        build(out_path=out)
        html = out.read_text(encoding="utf-8")
        # Single braces around UPPER_SNAKE should not remain
        import re
        unsubst = re.findall(r"(?<!\{)\{[A-Z_]+\}(?!\})", html)
        assert unsubst == [], f"Unsubstituted placeholders: {unsubst}"
    finally:
        out.unlink(missing_ok=True)


def test_build_cli(tmp_path):
    out = tmp_path / "network.html"
    result = subprocess.run(
        [sys.executable, "-c",
         f"from pathlib import Path; import sys; sys.path.insert(0,'gazetteer');"
         f"from build_correspondent_network import build; build(out_path=Path('{out}'))"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert out.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest gazetteer/tests/test_build_correspondent_network.py -v -k "build"`
Expected: FAIL — `ImportError` for `build`

- [ ] **Step 3: Download D3.js v7 minified for inlining**

Run: `curl -sL https://d3js.org/d3.v7.min.js -o gazetteer/.d3.v7.min.js && wc -c gazetteer/.d3.v7.min.js`

Expected: ~280KB file. This caches D3 locally so builds work offline.
The file is gitignored (added in Task 6). If absent, the build script
fetches it automatically via `urllib.request`.

- [ ] **Step 4: Implement `HTML_TEMPLATE` and `build()`**

The HTML template is the largest piece. It contains:
- Inlined D3.js (read from a cached local copy or fetched at build time)
- The `{NETWORK_JSON}` placeholder
- All CSS and JS for the force graph, timeline slider, detail panel, and legend
- Professional styling: `#fafafa` background, system sans-serif, muted palette

The template structure (key sections — full implementation at build time):

```python
def _get_d3_source() -> str:
    """Return D3 v7 minified source for inlining."""
    cache = Path(__file__).parent / ".d3.v7.min.js"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    import urllib.request
    url = "https://d3js.org/d3.v7.min.js"
    data = urllib.request.urlopen(url).read().decode("utf-8")
    cache.write_text(data, encoding="utf-8")
    return data


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Burney Correspondent Network</title>
<style>
/* --- Professional analyst aesthetic --- */
/* System sans-serif, #fafafa bg, muted palette */
/* Force graph SVG, timeline slider, detail panel, legend */
/* All JS braces doubled: {{ }} */
</style>
<script>{D3_SOURCE}</script>
</head>
<body>
<div id="app">
  <header><!-- title, subtitle --></header>
  <div id="graph-container">
    <svg id="network"></svg>
    <div id="detail-panel"></div>
  </div>
  <div id="timeline">
    <!-- dual-handle slider, phase presets -->
  </div>
  <div id="legend"><!-- community colour key --></div>
</div>
<script>
const DATA = {NETWORK_JSON};
/* Force simulation, slider, interactions — all with doubled braces */
</script>
</body>
</html>"""


def build(
    text_path: Path = TEXT_PATH,
    out_path: Path = OUT_PATH,
) -> None:
    text = text_path.read_text(encoding="utf-8")
    data = build_network_data(text)
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    d3_src = _get_d3_source()
    html = HTML_TEMPLATE.replace("{D3_SOURCE}", d3_src).replace(
        "{NETWORK_JSON}", data_json
    )
    out_path.write_text(html, encoding="utf-8")
    n_letters = len(data["letters"])
    n_corr = len(data["nodes"]) - 1  # exclude Burney
    print(f"Correspondent network -> {out_path}")
    print(f"  {n_corr} correspondents, {n_letters} letters")


if __name__ == "__main__":
    build()
```

**Important:** The full HTML_TEMPLATE must be written with:
- All JS `{` and `}` doubled to `{{` and `}}`
- Only `{D3_SOURCE}` and `{NETWORK_JSON}` as single-brace placeholders
- Complete CSS for the professional aesthetic
- Complete JS for D3 force simulation, drag, zoom, tooltip, detail panel, timeline slider, phase presets

This is the bulk of implementation work. The JS should follow this structure:

1. **Colour scale**: map community names to the palette defined in the spec
2. **Force simulation**: `d3.forceSimulation` with center, charge, link, collision forces. Burney node fixed at centre.
3. **SVG rendering**: links as lines, nodes as circles, labels for high-count nodes
4. **Tooltip**: `<div>` positioned on hover with name, community, count, date range
5. **Detail panel**: slide-in `<div>` on click, showing full letter list filtered by current time range
6. **Timeline slider**: custom dual-handle slider (two `<input type="range">` overlaid, or a custom SVG/canvas slider). Filter nodes/edges by year range on drag.
7. **Phase preset buttons**: one per life-phase, styled as pills. Click sets slider range.
8. **Legend**: bottom-left, small community colour key

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest gazetteer/tests/test_build_correspondent_network.py -v`
Expected: All tests PASS

- [ ] **Step 6: Build and visually inspect**

Run: `python3 gazetteer/build_correspondent_network.py && open gazetteer/correspondent_network.html`

Verify in browser:
- Graph renders with Burney at centre
- Nodes are coloured by community
- Hover shows tooltip
- Click shows detail panel
- Timeline slider filters the graph
- Phase preset buttons work
- Professional, clean aesthetic — no toy colours

- [ ] **Step 7: Commit**

```bash
git add gazetteer/build_correspondent_network.py gazetteer/tests/test_build_correspondent_network.py
git commit -m "feat: add HTML template and build function (Task 4)"
```

---

## Task 5: Full Integration Test Against Real Data

**Files:**
- Modify: `gazetteer/tests/test_build_correspondent_network.py`

- [ ] **Step 1: Write integration tests**

```python
def test_real_data_parse_count():
    """Verify we parse all headers from the real OUP text."""
    from build_correspondent_network import parse_headers, TEXT_PATH
    text = TEXT_PATH.read_text(encoding="utf-8")
    entries = parse_headers(text)
    # Should capture all numbered selections
    # The exact count is determined by the source, not hardcoded
    assert len(entries) >= 240, f"Only parsed {len(entries)} entries (expected ~243)"
    # Every entry should have a year
    for e in entries:
        assert e["year"] is not None, f"Entry {e['number']} has no year"


def test_real_data_all_correspondents_have_communities():
    """Every correspondent in the real data should be assigned a community."""
    from build_correspondent_network import (
        build_network_data, TEXT_PATH,
    )
    text = TEXT_PATH.read_text(encoding="utf-8")
    data = build_network_data(text)
    unknowns = [
        n["id"] for n in data["nodes"]
        if n["community"] == "Unknown"
    ]
    assert unknowns == [], f"Unassigned correspondents: {unknowns}"


def test_real_data_susanna_is_top_correspondent():
    """Susanna should have the most letters — sanity check."""
    from build_correspondent_network import build_network_data, TEXT_PATH
    text = TEXT_PATH.read_text(encoding="utf-8")
    data = build_network_data(text)
    nodes = sorted(data["nodes"], key=lambda n: -n["count"])
    # Burney herself is first; Susanna should be second
    assert nodes[0]["id"] == "Frances Burney"
    assert "Susanna" in nodes[1]["id"]
```

- [ ] **Step 2: Run tests**

Run: `pytest gazetteer/tests/test_build_correspondent_network.py -v`
Expected: All PASS. If `test_real_data_all_correspondents_have_communities` fails, add the missing names to `COMMUNITIES` dict and re-run.

- [ ] **Step 3: Commit**

```bash
git add gazetteer/tests/test_build_correspondent_network.py
git commit -m "test: add integration tests against real OUP data (Task 5)"
```

---

## Task 6: Polish & Final Build

- [ ] **Step 1: Run full test suite**

Run: `pytest gazetteer/tests/test_build_correspondent_network.py -v`
Expected: All PASS

- [ ] **Step 2: Final build and visual QA**

Run: `python3 gazetteer/build_correspondent_network.py && open gazetteer/correspondent_network.html`

Check:
- [ ] Graph renders, Burney centred
- [ ] Node sizes vary by letter count
- [ ] Community colours match spec palette
- [ ] Hover tooltip shows correct data
- [ ] Click opens detail panel with letter list
- [ ] Timeline slider filters nodes/edges
- [ ] Phase preset buttons set correct ranges
- [ ] Faded nodes outside range at ~10% opacity
- [ ] Legend visible, correct colours
- [ ] Clean professional aesthetic (no toy colours, no parchment)
- [ ] Works at 1200px+ width

- [ ] **Step 3: Add `.d3.v7.min.js` to `.gitignore`**

Run: `echo 'gazetteer/.d3.v7.min.js' >> .gitignore`

- [ ] **Step 4: Final commit**

```bash
git add -A gazetteer/build_correspondent_network.py gazetteer/tests/test_build_correspondent_network.py .gitignore
git commit -m "feat: correspondent network visualisation complete"
```
