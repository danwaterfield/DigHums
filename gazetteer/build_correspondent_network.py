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
import sys
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


# ── Name normalisation ─────────────────────────────────────────────

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
    "Longman, Hurst, Rees, Orme": "Longman & Co",
    "Messrs Longman and Company": "Longman & Co",
    "Messrs Longman": "Longman & Co",
    "Hester Maria Thrale": "Hester Thrale Piozzi",
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


# ── Helper functions ────────────────────────────────────────────────

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
    # Date starts at first month name, day+month, day-range, day-and-day,
    # comma-separated days, or bare year
    date_split = re.search(
        rf"(?:(?:c\.\s*|post\s+|pre-|late\s+|early\s+|"
        rf"mid\s+(?:or\s+late\s+)?|between\s+)\s*)?"
        rf"(?:{_MONTH_NAMES}"
        rf"|\d{{1,2}}\s+and\s+\d{{1,2}}\s+(?:{_MONTH_NAMES})"
        rf"|\d{{1,2}}[,\u2013-]\s*\d{{1,2}}"   # "14, 16" or "27-28" or "27–28"
        rf"|\d{{1,2}}\s*(?:{_MONTH_NAMES})"
        rf"|1[6-8]\d{{2}})\b",
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


# Pre-split patterns: multi-word correspondent names containing " and "
# that should NOT be split. Maps full string to single normalised name.
_PRESPLIT_NAMES = {
    "Longman, Hurst, Rees, Orme and Brown": "Longman & Co",
    "Messrs Longman and Company": "Longman & Co",
}


def _split_correspondents(s: str) -> list[str]:
    """Split 'A and B' into ['A', 'B'], handling compound names.

    After splitting on ' and ', if a fragment is a bare first name
    (single word), try appending the surname from the next fragment
    to produce a recognisable name (checked against NAME_ALIASES and
    COMMUNITIES). E.g. "William and Frederica Locke" -> ["William Locke",
    "Frederica Locke"].
    """
    if not s:
        return []

    # Check for known multi-word names before splitting
    for pattern, normalised in _PRESPLIT_NAMES.items():
        if pattern in s:
            return [normalised]

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
    # Normalise curly quotes to straight for consistent matching
    text = text.replace("\u2018", "'").replace("\u2019", "'")
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


# ── Public API ──────────────────────────────────────────────────────

def normalise_name(name: str) -> str:
    return NAME_ALIASES.get(name, name)


def assign_community(name: str) -> str:
    if name in COMMUNITIES:
        return COMMUNITIES[name]
    print(f"WARNING: no community for '{name}'", file=sys.stderr)
    return "Unknown"


def assign_phase(year: int, month: int | None) -> str:
    ym = (year, month or 1)
    for start_y, start_m, label in _PHASES:
        if ym >= (start_y, start_m):
            return label
    return "Apprentice Years"


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


if __name__ == "__main__":
    text = TEXT_PATH.read_text(encoding="utf-8")
    data = build_network_data(text)
    print(json.dumps(data, indent=2, ensure_ascii=False))
