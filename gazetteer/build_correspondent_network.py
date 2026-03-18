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

import csv
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
    "Charlotte Francis": "Charlotte Broome",
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
    "Sarah Harriet Burney": "Family",
    "Richard Thomas Burney": "Family",
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


D3_CACHE = Path(__file__).resolve().parent / ".d3.v7.min.js"
D3_URL = "https://d3js.org/d3.v7.min.js"


def _get_d3_source() -> str:
    """Return D3 v7 minified JS, fetching and caching if needed."""
    if D3_CACHE.exists():
        return D3_CACHE.read_text(encoding="utf-8")
    import urllib.request
    print(f"Fetching D3 v7 from {D3_URL} ...")
    req = urllib.request.Request(D3_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        src = resp.read().decode("utf-8")
    D3_CACHE.write_text(src, encoding="utf-8")
    print(f"Cached -> {D3_CACHE}")
    return src


# ── HTML template ─────────────────────────────────────────────────
# Convention: ALL JS braces are doubled.  Only {D3_SOURCE},
# {NETWORK_JSON}, and {CATALOGUE_JSON} use single braces
# (Python .replace() targets).

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Burney Family — Correspondent Networks</title>
<style>
*,*::before,*::after {{ box-sizing:border-box; margin:0; padding:0; }}
html,body {{ height:100%; overflow:hidden; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #fafafa;
  color: #1a1a1a;
  display: flex;
  flex-direction: column;
}}

/* ── Header ─────────────────────────────────────── */
.header {{
  padding: 10px 24px;
  background: #fff;
  border-bottom: 1px solid #e0e0e0;
  display: flex;
  align-items: center;
  gap: 16px;
  flex-shrink: 0;
  z-index: 10;
  flex-wrap: wrap;
}}
.header h1 {{
  font-size: 16px;
  font-weight: 600;
  letter-spacing: -0.02em;
  color: #2d3436;
}}
.header .subtitle {{
  font-size: 12px;
  color: #888;
  font-weight: 400;
}}
.stats {{
  margin-left: auto;
  font-size: 11px;
  color: #999;
  display: flex;
  gap: 16px;
}}
.stats span {{ font-variant-numeric: tabular-nums; }}

/* ── Tab switcher ───────────────────────────────── */
.tab-bar {{
  display: flex;
  gap: 0;
  margin-left: 12px;
}}
.tab-btn {{
  font-size: 12px;
  font-weight: 500;
  padding: 5px 14px;
  border: 1px solid #d0d0d0;
  background: #fff;
  color: #666;
  cursor: pointer;
  transition: all 0.15s;
  white-space: nowrap;
}}
.tab-btn:first-child {{ border-radius: 4px 0 0 4px; }}
.tab-btn:last-child {{ border-radius: 0 4px 4px 0; }}
.tab-btn:not(:first-child) {{ margin-left: -1px; }}
.tab-btn:hover {{ color: #333; background: #f5f5f5; }}
.tab-btn.active {{
  background: #2d3436;
  color: #fff;
  border-color: #2d3436;
  z-index: 1;
  position: relative;
}}
.tab-count {{
  font-size: 10px;
  font-weight: 400;
  color: #999;
  margin-left: 4px;
}}
.tab-btn.active .tab-count {{ color: rgba(255,255,255,0.7); }}

/* ── Threshold slider ───────────────────────────── */
.threshold-bar {{
  display: none;
  align-items: center;
  gap: 10px;
  padding: 6px 24px 8px;
  background: #fff;
  border-bottom: 1px solid #e8e8e8;
  font-size: 11px;
  color: #666;
  flex-shrink: 0;
}}
.threshold-bar.visible {{ display: flex; }}
.threshold-bar label {{ white-space: nowrap; }}
.threshold-bar input[type="range"] {{
  width: 180px;
  -webkit-appearance: none;
  appearance: none;
  height: 4px;
  background: #e0e0e0;
  border-radius: 2px;
  outline: none;
}}
.threshold-bar input[type="range"]::-webkit-slider-thumb {{
  -webkit-appearance: none;
  width: 14px; height: 14px;
  border-radius: 50%;
  background: #2d3436;
  border: 2px solid #fff;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
  cursor: pointer;
}}
.threshold-bar input[type="range"]::-moz-range-thumb {{
  width: 14px; height: 14px;
  border-radius: 50%;
  background: #2d3436;
  border: 2px solid #fff;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
  cursor: pointer;
}}
.threshold-val {{
  font-weight: 600;
  color: #2d3436;
  min-width: 20px;
  text-align: center;
}}

/* ── Main layout ────────────────────────────────── */
.main {{
  flex: 1;
  display: flex;
  position: relative;
  overflow: hidden;
}}
.graph-area {{
  flex: 1;
  position: relative;
}}
svg {{ display: block; width: 100%; height: 100%; }}
.tooltip {{
  position: absolute;
  pointer-events: none;
  background: rgba(255,255,255,0.97);
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 8px 12px;
  font-size: 12px;
  line-height: 1.5;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  max-width: 300px;
  z-index: 20;
  display: none;
}}
.tooltip .tt-name {{ font-weight: 600; font-size: 13px; }}
.tooltip .tt-community {{ font-size: 11px; color: #666; }}
.tooltip .tt-detail {{ margin-top: 4px; font-size: 11px; color: #555; }}
.tooltip .tt-direction {{ font-size: 11px; color: #777; }}
.tooltip .tt-edge {{ font-size: 11px; color: #555; margin-top: 2px; }}

/* ── Legend ──────────────────────────────────────── */
.legend {{
  position: absolute;
  bottom: 12px;
  left: 12px;
  background: rgba(255,255,255,0.92);
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  padding: 8px 12px;
  font-size: 10px;
  line-height: 1.8;
  z-index: 5;
}}
.legend-item {{
  display: flex;
  align-items: center;
  gap: 6px;
}}
.legend-swatch {{
  width: 10px; height: 10px;
  border-radius: 2px;
  flex-shrink: 0;
}}

/* ── Detail panel ───────────────────────────────── */
.detail-panel {{
  width: 340px;
  background: #fff;
  border-left: 1px solid #e0e0e0;
  overflow-y: auto;
  flex-shrink: 0;
  transform: translateX(340px);
  transition: transform 0.25s ease;
  position: absolute;
  top: 0; right: 0; bottom: 0;
  z-index: 15;
}}
.detail-panel.open {{ transform: translateX(0); }}
.dp-header {{
  padding: 16px 20px;
  border-bottom: 1px solid #eee;
}}
.dp-close {{
  float: right;
  cursor: pointer;
  color: #999;
  font-size: 18px;
  line-height: 1;
  border: none;
  background: none;
}}
.dp-close:hover {{ color: #333; }}
.dp-name {{
  font-size: 18px;
  font-weight: 600;
  color: #1a1a1a;
  margin-bottom: 4px;
}}
.dp-community {{
  font-size: 12px;
  color: #666;
  display: flex;
  align-items: center;
  gap: 6px;
}}
.dp-chip {{
  display: inline-block;
  width: 10px; height: 10px;
  border-radius: 2px;
}}
.dp-stats {{
  padding: 12px 20px;
  border-bottom: 1px solid #eee;
  font-size: 12px;
  color: #555;
  line-height: 1.8;
}}
.dp-stats strong {{ color: #1a1a1a; font-weight: 600; }}
.dp-direction-bar {{
  display: flex;
  height: 8px;
  border-radius: 4px;
  overflow: hidden;
  margin: 6px 0;
  background: #f0f0f0;
}}
.dp-direction-bar .dp-bar-to {{
  background: #4a6fa5;
  height: 100%;
}}
.dp-direction-bar .dp-bar-from {{
  background: #a07855;
  height: 100%;
}}
.dp-cross {{
  padding: 8px 20px 12px;
  border-bottom: 1px solid #eee;
  font-size: 11px;
  color: #555;
}}
.dp-cross-link {{
  cursor: pointer;
  color: #4a6fa5;
  text-decoration: underline;
  text-underline-offset: 2px;
}}
.dp-cross-link:hover {{ color: #2d3436; }}
.dp-letters {{
  padding: 12px 20px;
}}
.dp-letters h3 {{
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #999;
  margin-bottom: 8px;
}}
.dp-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 11px;
}}
.dp-table th {{
  text-align: left;
  font-weight: 500;
  color: #888;
  padding: 4px 6px;
  border-bottom: 1px solid #eee;
}}
.dp-table td {{
  padding: 4px 6px;
  border-bottom: 1px solid #f5f5f5;
  color: #444;
}}

/* ── Timeline ───────────────────────────────────── */
.timeline {{
  flex-shrink: 0;
  background: #fff;
  border-top: 1px solid #e0e0e0;
  padding: 12px 24px 16px;
}}
.timeline-label {{
  font-size: 12px;
  color: #555;
  margin-bottom: 6px;
  font-variant-numeric: tabular-nums;
}}
.timeline-label strong {{ color: #1a1a1a; }}
.slider-wrap {{
  position: relative;
  height: 28px;
  margin-bottom: 8px;
}}
.slider-wrap input[type="range"] {{
  -webkit-appearance: none;
  appearance: none;
  position: absolute;
  width: 100%;
  top: 0; left: 0;
  margin: 0;
  background: transparent;
  pointer-events: none;
  height: 28px;
}}
.slider-wrap input[type="range"]::-webkit-slider-runnable-track {{
  height: 4px;
  background: #e0e0e0;
  border-radius: 2px;
}}
.slider-wrap input[type="range"]::-webkit-slider-thumb {{
  -webkit-appearance: none;
  width: 16px; height: 16px;
  border-radius: 50%;
  background: #2d3436;
  border: 2px solid #fff;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
  margin-top: -6px;
  pointer-events: all;
  cursor: pointer;
}}
.slider-wrap input[type="range"]::-moz-range-track {{
  height: 4px;
  background: #e0e0e0;
  border-radius: 2px;
  border: none;
}}
.slider-wrap input[type="range"]::-moz-range-thumb {{
  width: 16px; height: 16px;
  border-radius: 50%;
  background: #2d3436;
  border: 2px solid #fff;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
  pointer-events: all;
  cursor: pointer;
}}
.phase-pills {{
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}}
.phase-pill {{
  font-size: 10px;
  padding: 3px 8px;
  border-radius: 12px;
  border: 1px solid #ddd;
  background: #fff;
  color: #666;
  cursor: pointer;
  transition: all 0.15s;
  white-space: nowrap;
}}
.phase-pill:hover {{ border-color: #999; color: #333; }}
.phase-pill.active {{
  background: #2d3436;
  color: #fff;
  border-color: #2d3436;
}}

/* ── Narrow viewport ────────────────────────────── */
@media (max-width: 999px) {{
  .detail-panel {{
    width: 100%;
    transform: translateX(100%);
    box-shadow: -4px 0 16px rgba(0,0,0,0.1);
  }}
  .detail-panel.open {{ transform: translateX(0); }}
}}
</style>
</head>
<body>

<div class="header">
  <h1>Burney Family</h1>
  <span class="subtitle">Correspondent Networks</span>
  <div class="tab-bar" id="tab-bar"></div>
  <div class="stats">
    <span id="stat-nodes"></span>
    <span id="stat-edges"></span>
    <span id="stat-letters"></span>
  </div>
</div>

<div class="threshold-bar" id="threshold-bar">
  <label>Show correspondents with &ge; <span class="threshold-val" id="threshold-val">5</span> letters</label>
  <input type="range" id="threshold-slider" min="1" max="50" value="5">
</div>

<div class="main">
  <div class="graph-area" id="graph-area">
    <svg id="network-svg">
      <defs>
        <marker id="arrow-to" viewBox="0 0 10 6" refX="10" refY="3"
          markerWidth="8" markerHeight="6" orient="auto">
          <path d="M0,0 L10,3 L0,6" fill="#4a6fa5" opacity="0.5"/>
        </marker>
      </defs>
    </svg>
    <div class="tooltip" id="tooltip"></div>
    <div class="legend" id="legend"></div>
  </div>
  <div class="detail-panel" id="detail-panel">
    <div class="dp-header">
      <button class="dp-close" id="dp-close">&times;</button>
      <div class="dp-name" id="dp-name"></div>
      <div class="dp-community" id="dp-community"></div>
    </div>
    <div class="dp-stats" id="dp-stats"></div>
    <div class="dp-cross" id="dp-cross"></div>
    <div class="dp-letters">
      <h3>Letters</h3>
      <table class="dp-table">
        <thead><tr><th>Date</th><th>Dir</th><th>Phase</th></tr></thead>
        <tbody id="dp-tbody"></tbody>
      </table>
    </div>
  </div>
</div>

<div class="timeline">
  <div class="timeline-label" id="tl-label"></div>
  <div class="slider-wrap">
    <input type="range" id="slider-lo" min="1749" max="1839" value="1749">
    <input type="range" id="slider-hi" min="1749" max="1839" value="1839">
  </div>
  <div class="phase-pills" id="phase-pills"></div>
</div>

<script>
{D3_SOURCE}
</script>
<script>
(function() {{
"use strict";

var OUP_DATA = {NETWORK_JSON};
var CATALOGUE = {CATALOGUE_JSON};

var COMMUNITY_COLOUR = {{
  "Centre":           "#2d3436",
  "Family":           "#4a6fa5",
  "Literary":         "#a07855",
  "Court":            "#5a8a7a",
  "Publishers":       "#7a7a8a",
  "Intimate circle":  "#a07080",
  "French circle":    "#6a5a8a",
  "Musical circle":   "#8a6a50",
  "Scholarly/Church": "#5a7a6a",
  "Royal":            "#8a7a5a",
  "Unknown":          "#b0b0b0"
}};

/* ── Tab definitions ──────────────────────────────── */
var TABS = [
  {{ key: "frances", label: "Frances", data: CATALOGUE.frances }},
  {{ key: "cb_sr",   label: "Charles Sr", data: CATALOGUE.cb_sr }},
  {{ key: "cb_jr",   label: "Charles Jr", data: CATALOGUE.cb_jr }}
];

var activeTab = "frances";
var rangeLo, rangeHi;
var selectedNode = null;
var threshold = 5;
var simulation = null;
var link = null, node = null;
var currentData = null;

/* Cross-network lookup: for each node name, which other tabs contain it */
var crossNetwork = {{}};
TABS.forEach(function(tab) {{
  var nodeIds = new Set(tab.data.nodes.map(function(n) {{ return n.id; }}));
  nodeIds.forEach(function(id) {{
    if (!crossNetwork[id]) crossNetwork[id] = {{}};
    crossNetwork[id][tab.key] = tab.data.nodes.find(function(n) {{ return n.id === id; }});
  }});
}});

/* ── Build tabs ────────────────────────────────────── */
(function() {{
  var bar = document.getElementById("tab-bar");
  TABS.forEach(function(tab) {{
    var btn = document.createElement("button");
    btn.className = "tab-btn" + (tab.key === activeTab ? " active" : "");
    btn.dataset.key = tab.key;
    var nameSpan = document.createTextNode(tab.label);
    btn.appendChild(nameSpan);
    var countSpan = document.createElement("span");
    countSpan.className = "tab-count";
    countSpan.textContent = "(" + tab.data.letters.length.toLocaleString() + ")";
    btn.appendChild(countSpan);
    btn.addEventListener("click", function() {{
      switchTab(tab.key);
    }});
    bar.appendChild(btn);
  }});
}})();

function switchTab(key) {{
  activeTab = key;
  var btns = document.querySelectorAll(".tab-btn");
  btns.forEach(function(b) {{ b.classList.toggle("active", b.dataset.key === key); }});

  /* Show/hide threshold bar for Frances only */
  var tbar = document.getElementById("threshold-bar");
  tbar.classList.toggle("visible", key === "frances");

  dismissDetail();
  buildGraph();
}}

/* ── Threshold slider ──────────────────────────────── */
var thresholdSlider = document.getElementById("threshold-slider");
var thresholdValEl = document.getElementById("threshold-val");
thresholdSlider.addEventListener("input", function() {{
  threshold = +this.value;
  thresholdValEl.textContent = threshold;
  buildGraph();
}});

/* ── SVG setup ─────────────────────────────────────── */
var svgEl = document.getElementById("network-svg");
var width = svgEl.clientWidth || 900;
var height = svgEl.clientHeight || 600;
var svg = d3.select("#network-svg");
var g = svg.select("g.graph-root");
if (g.empty()) g = svg.append("g").attr("class", "graph-root");

var zoom = d3.zoom()
  .scaleExtent([0.2, 6])
  .on("zoom", function(event) {{ g.attr("transform", event.transform); }});
svg.call(zoom);
svg.on("click", function(event) {{
  if (event.target === svgEl) dismissDetail();
}});

/* ── Tooltip element ───────────────────────────────── */
var tooltip = document.getElementById("tooltip");

/* ── Main graph builder ────────────────────────────── */
function buildGraph() {{
  var tab = TABS.find(function(t) {{ return t.key === activeTab; }});
  var data = tab.data;
  currentData = data;

  /* Apply threshold for Frances */
  var filteredNodes, filteredEdges;
  if (activeTab === "frances" && threshold > 1) {{
    var keepSet = new Set();
    keepSet.add(data.subject);
    data.nodes.forEach(function(n) {{
      if (n.id === data.subject || n.count >= threshold) keepSet.add(n.id);
    }});
    filteredNodes = data.nodes.filter(function(n) {{ return keepSet.has(n.id); }});
    filteredEdges = data.edges.filter(function(e) {{ return keepSet.has(e.target); }});
  }} else {{
    filteredNodes = data.nodes.slice();
    filteredEdges = data.edges.slice();
  }}

  /* Deep-clone nodes for simulation (avoid mutating source) */
  var simNodes = filteredNodes.map(function(n) {{
    return {{ id: n.id, community: n.community, count: n.count,
              to_count: n.to_count || 0, from_count: n.from_count || 0 }};
  }});
  var simEdges = filteredEdges.map(function(e) {{
    return {{ source: e.source, target: e.target, weight: e.weight,
              to_weight: e.to_weight || 0, from_weight: e.from_weight || 0 }};
  }});

  /* Update stats */
  document.getElementById("stat-nodes").textContent = simNodes.length + " correspondents";
  document.getElementById("stat-edges").textContent = simEdges.length + " connections";
  document.getElementById("stat-letters").textContent = data.letters.length.toLocaleString() + " letters";

  /* Update timeline range */
  var years = data.letters.map(function(l) {{ return l.year; }}).filter(Boolean);
  var minYear = d3.min(years) || 1749;
  var maxYear = d3.max(years) || 1839;
  rangeLo = minYear;
  rangeHi = maxYear;
  var sliderLo = document.getElementById("slider-lo");
  var sliderHi = document.getElementById("slider-hi");
  sliderLo.min = minYear; sliderLo.max = maxYear; sliderLo.value = minYear;
  sliderHi.min = minYear; sliderHi.max = maxYear; sliderHi.value = maxYear;

  /* Rebuild phase pills */
  var pillContainer = document.getElementById("phase-pills");
  while (pillContainer.firstChild) pillContainer.removeChild(pillContainer.firstChild);
  data.phases.forEach(function(p) {{
    var btn = document.createElement("button");
    btn.className = "phase-pill";
    btn.textContent = p.label;
    btn.addEventListener("click", function() {{
      rangeLo = p.start;
      rangeHi = p.end;
      document.getElementById("slider-lo").value = rangeLo;
      document.getElementById("slider-hi").value = rangeHi;
      updatePills();
      filterGraph();
    }});
    pillContainer.appendChild(btn);
  }});

  /* Rebuild legend */
  var legendEl = document.getElementById("legend");
  while (legendEl.firstChild) legendEl.removeChild(legendEl.firstChild);
  var order = ["Centre","Family","Literary","Court","Publishers","Intimate circle",
               "French circle","Musical circle","Scholarly/Church","Royal","Unknown"];
  var present = new Set(simNodes.map(function(n) {{ return n.community; }}));
  order.forEach(function(c) {{
    if (!present.has(c)) return;
    var row = document.createElement("div");
    row.className = "legend-item";
    var swatch = document.createElement("div");
    swatch.className = "legend-swatch";
    swatch.style.background = COMMUNITY_COLOUR[c] || "#b0b0b0";
    var label = document.createElement("span");
    label.textContent = c;
    row.appendChild(swatch);
    row.appendChild(label);
    legendEl.appendChild(row);
  }});

  /* Clear previous graph */
  g.selectAll("*").remove();
  if (simulation) simulation.stop();

  /* Scales */
  var maxCount = d3.max(simNodes, function(n) {{ return n.count; }}) || 1;
  var rScale = d3.scaleSqrt().domain([1, maxCount]).range([4, 28]);
  var maxWeight = d3.max(simEdges, function(e) {{ return e.weight; }}) || 1;
  var wScale = d3.scaleSqrt().domain([1, maxWeight]).range([0.5, 6]);

  /* Node lookup */
  var nodeMap = {{}};
  simNodes.forEach(function(n) {{ nodeMap[n.id] = n; }});

  /* Links — directional thickness */
  var linkG = g.append("g").attr("class", "links");

  /* Draw two lines per edge: a thicker "to" and thinner "from" offset slightly */
  link = linkG.selectAll("g.edge-group")
    .data(simEdges)
    .join("g")
    .attr("class", "edge-group");

  /* Main combined line (stroke = community colour) */
  link.append("line")
    .attr("class", "edge-main")
    .attr("stroke", function(d) {{
      var t = nodeMap[d.target] || nodeMap[(d.target && d.target.id) || ""];
      return COMMUNITY_COLOUR[t ? t.community : "Unknown"] || "#ccc";
    }})
    .attr("stroke-opacity", 0.25)
    .attr("stroke-width", function(d) {{ return wScale(d.weight); }});

  /* Directional "to" overlay (blue, thicker on the target side via marker) */
  link.filter(function(d) {{ return d.to_weight > 0 && d.from_weight > 0; }})
    .append("line")
    .attr("class", "edge-dir")
    .attr("stroke", "#4a6fa5")
    .attr("stroke-opacity", 0.15)
    .attr("stroke-width", function(d) {{
      var ratio = d.to_weight / (d.to_weight + d.from_weight);
      return Math.max(0.5, wScale(d.weight) * ratio * 1.5);
    }})
    .attr("stroke-dasharray", "4,3");

  /* Edge tooltips */
  link.on("mouseenter", function(event, d) {{
    while (tooltip.firstChild) tooltip.removeChild(tooltip.firstChild);
    var subj = data.subject;
    var edgeDiv = document.createElement("div");
    edgeDiv.className = "tt-edge";
    var toStr = (d.to_weight || 0) + " to " + (typeof d.target === "object" ? d.target.id : d.target);
    var fromStr = (d.from_weight || 0) + " from";
    edgeDiv.textContent = toStr + ", " + fromStr;
    tooltip.appendChild(edgeDiv);
    tooltip.style.display = "block";
  }}).on("mousemove", function(event) {{
    var rect = svgEl.getBoundingClientRect();
    tooltip.style.left = (event.clientX - rect.left + 14) + "px";
    tooltip.style.top  = (event.clientY - rect.top  - 10) + "px";
  }}).on("mouseleave", function() {{
    tooltip.style.display = "none";
  }});

  /* Nodes */
  var subjectId = data.subject;
  node = g.append("g").attr("class", "nodes")
    .selectAll("g.node-group")
    .data(simNodes)
    .join("g")
    .attr("class", "node-group")
    .attr("cursor", "pointer");

  /* Cross-network badge ring */
  node.filter(function(d) {{
    if (d.id === subjectId) return false;
    var cn = crossNetwork[d.id];
    if (!cn) return false;
    var keys = Object.keys(cn).filter(function(k) {{ return k !== activeTab; }});
    return keys.length > 0;
  }}).append("circle")
    .attr("class", "cross-badge")
    .attr("r", function(d) {{ return rScale(d.count) + 4; }})
    .attr("fill", "none")
    .attr("stroke", "#e0c070")
    .attr("stroke-width", 2)
    .attr("stroke-dasharray", "3,2")
    .attr("opacity", 0.7);

  /* Main circle */
  node.append("circle")
    .attr("class", "node-circle")
    .attr("r", function(d) {{ return d.id === subjectId ? 30 : rScale(d.count); }})
    .attr("fill", function(d) {{ return COMMUNITY_COLOUR[d.community] || "#b0b0b0"; }})
    .attr("stroke", function(d) {{ return d.id === subjectId ? "#1a1a1a" : "#fff"; }})
    .attr("stroke-width", function(d) {{ return d.id === subjectId ? 2.5 : 1.5; }});

  /* Centre label */
  var subjectInitials = subjectId.split(" ").map(function(w) {{ return w[0]; }}).join("").substring(0,3);
  node.filter(function(d) {{ return d.id === subjectId; }})
    .append("text")
    .text(subjectInitials)
    .attr("text-anchor", "middle")
    .attr("dy", "0.35em")
    .attr("fill", "#fff")
    .attr("font-size", "11px")
    .attr("font-weight", "600")
    .style("pointer-events", "none");

  /* Labels for prominent nodes */
  var countThreshold = d3.quantile(
    simNodes.filter(function(n) {{ return n.id !== subjectId; }})
      .map(function(n) {{ return n.count; }}).sort(d3.ascending),
    0.8
  ) || 1;

  node.filter(function(d) {{ return d.id !== subjectId && d.count >= countThreshold; }})
    .append("text")
    .text(function(d) {{
      var parts = d.id.split(" ");
      return parts.length > 1 ? parts[parts.length - 1] : d.id;
    }})
    .attr("dx", function(d) {{ return rScale(d.count) + 4; }})
    .attr("dy", "0.35em")
    .attr("font-size", "10px")
    .attr("fill", "#555")
    .style("pointer-events", "none");

  /* Node tooltip */
  node.on("mouseenter", function(event, d) {{
    if (d.id === subjectId) return;
    var letters = data.letters.filter(function(l) {{ return l.correspondent === d.id; }});
    var years = letters.map(function(l) {{ return l.year; }}).filter(Boolean);
    var yMin = d3.min(years), yMax = d3.max(years);

    while (tooltip.firstChild) tooltip.removeChild(tooltip.firstChild);
    var nameDiv = document.createElement("div");
    nameDiv.className = "tt-name";
    nameDiv.textContent = d.id;
    tooltip.appendChild(nameDiv);

    var commDiv = document.createElement("div");
    commDiv.className = "tt-community";
    commDiv.textContent = d.community;
    tooltip.appendChild(commDiv);

    var detDiv = document.createElement("div");
    detDiv.className = "tt-detail";
    detDiv.textContent = d.count + " letters" + (yMin ? " \u00b7 " + yMin + "\u2013" + yMax : "");
    tooltip.appendChild(detDiv);

    if (d.to_count || d.from_count) {{
      var dirDiv = document.createElement("div");
      dirDiv.className = "tt-direction";
      dirDiv.textContent = (d.to_count || 0) + " sent \u00b7 " + (d.from_count || 0) + " received";
      tooltip.appendChild(dirDiv);
    }}

    /* Cross-network note */
    var cn = crossNetwork[d.id];
    if (cn) {{
      var otherKeys = Object.keys(cn).filter(function(k) {{ return k !== activeTab; }});
      if (otherKeys.length > 0) {{
        var crossDiv = document.createElement("div");
        crossDiv.className = "tt-detail";
        crossDiv.style.color = "#b08030";
        var parts = otherKeys.map(function(k) {{
          var t = TABS.find(function(tb) {{ return tb.key === k; }});
          return t ? t.label : k;
        }});
        crossDiv.textContent = "Also in: " + parts.join(", ");
        tooltip.appendChild(crossDiv);
      }}
    }}

    tooltip.style.display = "block";
  }}).on("mousemove", function(event) {{
    var rect = svgEl.getBoundingClientRect();
    tooltip.style.left = (event.clientX - rect.left + 14) + "px";
    tooltip.style.top  = (event.clientY - rect.top  - 10) + "px";
  }}).on("mouseleave", function() {{
    tooltip.style.display = "none";
  }});

  /* Click -> detail panel */
  node.on("click", function(event, d) {{
    event.stopPropagation();
    showDetail(d, data, rScale);
  }});

  /* Force simulation */
  width = svgEl.clientWidth || 900;
  height = svgEl.clientHeight || 600;

  simulation = d3.forceSimulation(simNodes)
    .force("link", d3.forceLink(simEdges).id(function(d) {{ return d.id; }})
      .distance(function(d) {{ return 50 + 140 / Math.sqrt(d.weight || 1); }})
      .strength(function(d) {{ return 0.3 + 0.3 * Math.min(d.weight / maxWeight, 1); }})
    )
    .force("charge", d3.forceManyBody().strength(function() {{ return simNodes.length > 100 ? -80 : -120; }}))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide().radius(function(d) {{
      return (d.id === subjectId ? 34 : rScale(d.count) + 3);
    }}))
    .on("tick", ticked);

  var centreNode = simNodes.find(function(n) {{ return n.id === subjectId; }});
  if (centreNode) {{
    centreNode.fx = width / 2;
    centreNode.fy = height / 2;
  }}

  function ticked() {{
    link.selectAll("line")
      .attr("x1", function(d) {{ return d.source.x; }})
      .attr("y1", function(d) {{ return d.source.y; }})
      .attr("x2", function(d) {{ return d.target.x; }})
      .attr("y2", function(d) {{ return d.target.y; }});
    node.attr("transform", function(d) {{ return "translate(" + d.x + "," + d.y + ")"; }});
  }}

  /* Drag */
  node.call(d3.drag()
    .on("start", function(event, d) {{
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x; d.fy = d.y;
    }})
    .on("drag", function(event, d) {{
      d.fx = event.x; d.fy = event.y;
    }})
    .on("end", function(event, d) {{
      if (!event.active) simulation.alphaTarget(0);
      if (d.id !== subjectId) {{ d.fx = null; d.fy = null; }}
    }})
  );

  /* Reset zoom */
  svg.call(zoom.transform, d3.zoomIdentity);

  updateLabel();
}}

/* ── Detail panel ──────────────────────────────────── */
function showDetail(d, data, rScale) {{
  selectedNode = d;
  var panel = document.getElementById("detail-panel");
  document.getElementById("dp-name").textContent = d.id;

  var comm = document.getElementById("dp-community");
  while (comm.firstChild) comm.removeChild(comm.firstChild);
  var chip = document.createElement("span");
  chip.className = "dp-chip";
  chip.style.background = COMMUNITY_COLOUR[d.community] || "#b0b0b0";
  comm.appendChild(chip);
  comm.appendChild(document.createTextNode(" " + d.community));

  var letters = data.letters.filter(function(l) {{ return l.correspondent === d.id; }});
  var inRange = letters.filter(function(l) {{ return l.year >= rangeLo && l.year <= rangeHi; }});
  var years = letters.map(function(l) {{ return l.year; }}).filter(Boolean);
  var yMin = d3.min(years), yMax = d3.max(years);

  var statsEl = document.getElementById("dp-stats");
  while (statsEl.firstChild) statsEl.removeChild(statsEl.firstChild);

  /* Direction breakdown */
  var toCount = d.to_count || 0;
  var fromCount = d.from_count || 0;
  var total = toCount + fromCount;

  if (total > 0 && (toCount > 0 || fromCount > 0)) {{
    var dirLine = document.createElement("div");
    var b1 = document.createElement("strong");
    b1.textContent = toCount;
    dirLine.appendChild(b1);
    dirLine.appendChild(document.createTextNode(" sent \u00b7 "));
    var b2 = document.createElement("strong");
    b2.textContent = fromCount;
    dirLine.appendChild(b2);
    dirLine.appendChild(document.createTextNode(" received"));
    statsEl.appendChild(dirLine);

    /* Direction bar */
    var barOuter = document.createElement("div");
    barOuter.className = "dp-direction-bar";
    var barTo = document.createElement("div");
    barTo.className = "dp-bar-to";
    barTo.style.width = (toCount / total * 100) + "%";
    var barFrom = document.createElement("div");
    barFrom.className = "dp-bar-from";
    barFrom.style.width = (fromCount / total * 100) + "%";
    barOuter.appendChild(barTo);
    barOuter.appendChild(barFrom);
    statsEl.appendChild(barOuter);
  }} else {{
    var countLine = document.createElement("div");
    var b3 = document.createElement("strong");
    b3.textContent = letters.length;
    countLine.appendChild(b3);
    countLine.appendChild(document.createTextNode(" total letters"));
    statsEl.appendChild(countLine);
  }}

  if (inRange.length !== letters.length) {{
    var rangeLine = document.createElement("div");
    var b4 = document.createElement("strong");
    b4.textContent = inRange.length;
    rangeLine.appendChild(b4);
    rangeLine.appendChild(document.createTextNode(" in selected range"));
    statsEl.appendChild(rangeLine);
  }}

  var dateLine = document.createElement("div");
  dateLine.textContent = (yMin || "?") + " \u2013 " + (yMax || "?");
  statsEl.appendChild(dateLine);

  /* Cross-network links */
  var crossEl = document.getElementById("dp-cross");
  while (crossEl.firstChild) crossEl.removeChild(crossEl.firstChild);
  var cn = crossNetwork[d.id];
  if (cn) {{
    var otherKeys = Object.keys(cn).filter(function(k) {{ return k !== activeTab; }});
    otherKeys.forEach(function(k) {{
      var otherNode = cn[k];
      var t = TABS.find(function(tb) {{ return tb.key === k; }});
      if (!t || !otherNode) return;
      var line = document.createElement("div");
      line.style.marginBottom = "4px";
      line.textContent = "Also in ";
      var lnk = document.createElement("span");
      lnk.className = "dp-cross-link";
      lnk.textContent = t.label + ": " + otherNode.count + " letters";
      lnk.addEventListener("click", function() {{
        switchTab(k);
      }});
      line.appendChild(lnk);
      crossEl.appendChild(line);
    }});
  }}

  /* Letters table */
  var tbody = document.getElementById("dp-tbody");
  while (tbody.firstChild) tbody.removeChild(tbody.firstChild);
  letters.forEach(function(l) {{
    var tr = document.createElement("tr");
    if (l.year && (l.year < rangeLo || l.year > rangeHi)) tr.style.opacity = "0.3";
    var tdDate = document.createElement("td");
    var monthStr = l.month ? String(l.month).padStart(2, "0") + "/" : "";
    tdDate.textContent = monthStr + (l.year || "?");
    var tdDir = document.createElement("td");
    tdDir.textContent = l.direction || l.type || "";
    tdDir.style.fontSize = "10px";
    var tdPhase = document.createElement("td");
    tdPhase.style.fontSize = "10px";
    tdPhase.textContent = l.phase || "";
    tr.appendChild(tdDate);
    tr.appendChild(tdDir);
    tr.appendChild(tdPhase);
    tbody.appendChild(tr);
  }});

  panel.classList.add("open");
}}

function dismissDetail() {{
  selectedNode = null;
  document.getElementById("detail-panel").classList.remove("open");
}}

document.getElementById("dp-close").addEventListener("click", dismissDetail);

/* ── Timeline controls ─────────────────────────────── */
function updatePills() {{
  var tab = TABS.find(function(t) {{ return t.key === activeTab; }});
  var pills = document.querySelectorAll(".phase-pill");
  pills.forEach(function(pill, i) {{
    if (i < tab.data.phases.length) {{
      var p = tab.data.phases[i];
      pill.classList.toggle("active", rangeLo === p.start && rangeHi === p.end);
    }}
  }});
}}

var sliderLo = document.getElementById("slider-lo");
var sliderHi = document.getElementById("slider-hi");
sliderLo.addEventListener("input", function() {{
  rangeLo = +this.value;
  if (rangeLo > rangeHi) {{ rangeHi = rangeLo; sliderHi.value = rangeHi; }}
  updatePills();
  filterGraph();
}});
sliderHi.addEventListener("input", function() {{
  rangeHi = +this.value;
  if (rangeHi < rangeLo) {{ rangeLo = rangeHi; sliderLo.value = rangeLo; }}
  updatePills();
  filterGraph();
}});

function updateLabel() {{
  var tab = TABS.find(function(t) {{ return t.key === activeTab; }});
  var inRange = tab.data.letters.filter(function(l) {{ return l.year >= rangeLo && l.year <= rangeHi; }});
  var el = document.getElementById("tl-label");
  while (el.firstChild) el.removeChild(el.firstChild);
  var strong = document.createElement("strong");
  strong.textContent = rangeLo + " \u2013 " + rangeHi;
  el.appendChild(strong);
  el.appendChild(document.createTextNode(" \u00b7 " + inRange.length.toLocaleString() + " letters in range"));
}}

/* ── Filter by range ───────────────────────────────── */
function filterGraph() {{
  updateLabel();
  var tab = TABS.find(function(t) {{ return t.key === activeTab; }});
  var data = tab.data;
  var rangeCounts = {{}};
  data.letters.forEach(function(l) {{
    if (l.year >= rangeLo && l.year <= rangeHi) {{
      rangeCounts[l.correspondent] = (rangeCounts[l.correspondent] || 0) + 1;
    }}
  }});

  if (!node || !link) return;

  var maxCount = d3.max(data.nodes, function(n) {{ return n.count; }}) || 1;
  var rScale = d3.scaleSqrt().domain([1, maxCount]).range([4, 28]);
  var maxWeight = d3.max(data.edges, function(e) {{ return e.weight; }}) || 1;
  var wScale = d3.scaleSqrt().domain([1, maxWeight]).range([0.5, 6]);
  var subjectId = data.subject;

  node.select(".node-circle")
    .transition().duration(300)
    .attr("r", function(d) {{
      if (d.id === subjectId) return 30;
      var c = rangeCounts[d.id] || 0;
      return c > 0 ? rScale(c) : rScale(1) * 0.6;
    }})
    .attr("opacity", function(d) {{
      if (d.id === subjectId) return 1;
      return (rangeCounts[d.id] || 0) > 0 ? 1 : 0.1;
    }});

  node.select(".cross-badge")
    .transition().duration(300)
    .attr("opacity", function(d) {{
      return (rangeCounts[d.id] || 0) > 0 ? 0.7 : 0.05;
    }});

  node.selectAll("text")
    .transition().duration(300)
    .attr("opacity", function(d) {{
      if (d.id === subjectId) return 1;
      return (rangeCounts[d.id] || 0) > 0 ? 1 : 0.08;
    }});

  link.selectAll("line").transition().duration(300)
    .attr("stroke-opacity", function(d) {{
      var tid = typeof d.target === "object" ? d.target.id : d.target;
      return (rangeCounts[tid] || 0) > 0 ? 0.25 : 0.03;
    }})
    .attr("stroke-width", function(d) {{
      var tid = typeof d.target === "object" ? d.target.id : d.target;
      var c = rangeCounts[tid] || 0;
      return c > 0 ? wScale(c) : 0.3;
    }});

  if (selectedNode) {{
    var tab2 = TABS.find(function(t) {{ return t.key === activeTab; }});
    var maxCount2 = d3.max(tab2.data.nodes, function(n) {{ return n.count; }}) || 1;
    var rScale2 = d3.scaleSqrt().domain([1, maxCount2]).range([4, 28]);
    showDetail(selectedNode, tab2.data, rScale2);
  }}
}}

/* ── Resize ────────────────────────────────────────── */
window.addEventListener("resize", function() {{
  width = svgEl.clientWidth;
  height = svgEl.clientHeight;
  if (simulation) {{
    simulation.force("center", d3.forceCenter(width / 2, height / 2));
    var tab = TABS.find(function(t) {{ return t.key === activeTab; }});
    var subjectId = tab.data.subject;
    var centreNode = simulation.nodes().find(function(n) {{ return n.id === subjectId; }});
    if (centreNode) {{ centreNode.fx = width / 2; centreNode.fy = height / 2; }}
    simulation.alpha(0.3).restart();
  }}
}});

/* ── Init ──────────────────────────────────────────── */
document.getElementById("threshold-bar").classList.add("visible");
buildGraph();

}})();
</script>
</body>
</html>
"""


# ── Catalogue-based pipeline ──────────────────────────────────────

from burney_names import normalise as _bn_normalise
from burney_names import community as _bn_community
from burney_names import is_artefact as _bn_is_artefact

_CATALOGUE_DIR = Path(__file__).resolve().parent.parent / "nonfiction/FrancesBurney"

_CATALOGUE_FILES = {
    "frances": "catalogue_frances_darblay.csv",
    "cb_sr":   "catalogue_cb_sr.csv",
    "cb_jr":   "catalogue_cb_jr.csv",
}

_CATALOGUE_SUBJECTS = {
    "frances": "Frances Burney d'Arblay",
    "cb_sr":   "Dr Charles Burney",
    "cb_jr":   "Charles Burney Jr",
}

_CATALOGUE_PHASES = {
    "frances": _PHASES,
    "cb_sr": [
        (1800, 1,  "Final Years"),
        (1789, 1,  "Late Career"),
        (1776, 1,  "Streatham & Literary Fame"),
        (1770, 1,  "Continental Tours"),
        (1760, 1,  "London Establishment"),
        (1749, 1,  "Lynn & Early Career"),
    ],
    "cb_jr": [
        (1800, 1,  "Late Career & DD"),
        (1786, 1,  "Schoolmaster & Greek Scholar"),
        (1767, 1,  "Early Life & Cambridge"),
    ],
}

# Self-reference names for each catalogue subject — these are all names
# that refer to the letter-writer themselves and should be excluded.
_SELF_NAMES = {
    "frances": {"Frances Burney", "Frances Burney d'Arblay"},
    "cb_sr":   {"Charles Burney", "Dr Charles Burney"},
    "cb_jr":   {"Charles Burney Jr"},
}

# Year extraction regex for catalogue dates
_CAT_YEAR_RE = re.compile(r"\b(1[5-8]\d{2})\b")
_CAT_MONTH_RE = re.compile(
    r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?"
    r"|July?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b",
    re.IGNORECASE,
)
_CAT_MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2,
    "mar": 3, "march": 3, "apr": 4, "april": 4,
    "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def _parse_catalogue_date(date_str: str) -> tuple[int | None, int | None]:
    """Extract (year, month) from a catalogue date string."""
    ym = _CAT_YEAR_RE.search(date_str)
    mm = _CAT_MONTH_RE.search(date_str)
    year = int(ym.group(1)) if ym else None
    month = _CAT_MONTH_MAP.get(mm.group(1).lower()) if mm else None
    return year, month


def _assign_catalogue_phase(year: int, month: int | None, phases: list) -> str:
    """Assign a phase label using the given phase list."""
    ym = (year, month or 1)
    for start_y, start_m, label in phases:
        if ym >= (start_y, start_m):
            return label
    return phases[-1][2] if phases else "Unknown"


def _split_compound_correspondent(raw: str) -> list[str]:
    """Split compound correspondent fields like 'SBP & Frederica Locke'.

    Splits on ' & ', ' and ', and ' to ' (when used to indicate a
    forwarded letter, e.g. 'Rosette (Rose) Burney to CPB').
    Returns a list of individual correspondent strings.
    """
    # Split on & first
    parts = re.split(r'\s+&\s+', raw)
    # Further split on ' to ' only when it looks like forwarding
    expanded = []
    for p in parts:
        # "X to Y" where both are short => forwarded letter notation
        sub = re.split(r'\s+to\s+', p, maxsplit=1)
        expanded.extend(sub)
    return [s.strip() for s in expanded if s.strip()]


def load_catalogue(key: str) -> list[dict]:
    """Load a catalogue CSV and return a list of row dicts.

    *key* is one of ``'frances'``, ``'cb_sr'``, ``'cb_jr'``.
    Each dict has keys: ``date``, ``direction``, ``correspondent``,
    ``repository``, ``first_line``.
    """
    if key not in _CATALOGUE_FILES:
        raise ValueError(f"Unknown catalogue key {key!r}; "
                         f"choose from {sorted(_CATALOGUE_FILES)}")
    path = _CATALOGUE_DIR / _CATALOGUE_FILES[key]
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_catalogue_network(key: str) -> dict:
    """Build a network-data dict from a catalogue CSV.

    Returns::

        {
            "subject": str,
            "nodes": [{"id", "community", "count", "to_count", "from_count"}, ...],
            "edges": [{"source", "target", "weight", "to_weight", "from_weight"}, ...],
            "letters": [...],
            "phases": [...]
        }
    """
    rows = load_catalogue(key)
    subject = _CATALOGUE_SUBJECTS[key]
    phases_def = _CATALOGUE_PHASES[key]

    # Per-correspondent direction counts
    to_counts: dict[str, int] = {}
    from_counts: dict[str, int] = {}
    letters: list[dict] = []

    for row in rows:
        raw_corr = row["correspondent"].strip()
        direction = row.get("direction", "").strip().lower()
        date_str = row.get("date", "")
        year, month = _parse_catalogue_date(date_str)

        if not raw_corr or _bn_is_artefact(raw_corr):
            continue

        # Split compound correspondents
        individuals = _split_compound_correspondent(raw_corr)
        for ind in individuals:
            if _bn_is_artefact(ind):
                continue
            canon = _bn_normalise(ind)
            # Skip self-references (subject writing to themselves)
            if canon in _SELF_NAMES.get(key, set()):
                continue

            if direction == "to":
                to_counts[canon] = to_counts.get(canon, 0) + 1
            elif direction == "from":
                from_counts[canon] = from_counts.get(canon, 0) + 1
            else:
                # Unknown direction — count in both
                to_counts[canon] = to_counts.get(canon, 0) + 1

            phase = _assign_catalogue_phase(year, month, phases_def) if year else "Unknown"
            letters.append({
                "correspondent": canon,
                "year": year,
                "month": month,
                "direction": direction,
                "phase": phase,
            })

    # All correspondents
    all_names = sorted(set(to_counts) | set(from_counts))

    # Build nodes
    nodes = []
    nodes.append({
        "id": subject,
        "community": "Centre",
        "count": len(letters),
        "to_count": 0,
        "from_count": 0,
    })
    for name in all_names:
        tc = to_counts.get(name, 0)
        fc = from_counts.get(name, 0)
        nodes.append({
            "id": name,
            "community": _bn_community(name),
            "count": tc + fc,
            "to_count": tc,
            "from_count": fc,
        })

    # Sort non-subject nodes by count descending
    nodes[1:] = sorted(nodes[1:], key=lambda n: -n["count"])

    # Build edges
    edges = []
    for name in all_names:
        tc = to_counts.get(name, 0)
        fc = from_counts.get(name, 0)
        edges.append({
            "source": subject,
            "target": name,
            "weight": tc + fc,
            "to_weight": tc,
            "from_weight": fc,
        })

    # Phase definitions for UI
    phase_defs = [
        {"label": label, "start": sy, "end": 0}
        for sy, _, label in reversed(phases_def)
    ]
    for i, p in enumerate(phase_defs):
        if i + 1 < len(phase_defs):
            p["end"] = phase_defs[i + 1]["start"]
        else:
            # End at a sensible year
            if key == "frances":
                p["end"] = 1839
            elif key == "cb_sr":
                p["end"] = 1814
            elif key == "cb_jr":
                p["end"] = 1817

    return {
        "subject": subject,
        "nodes": nodes,
        "edges": edges,
        "letters": letters,
        "phases": phase_defs,
    }


# ── Build function ────────────────────────────────────────────────

def build(
    text_path: Path = TEXT_PATH,
    out_path: Path = OUT_PATH,
) -> None:
    """Read the OUP text, build network data, render HTML, write file."""
    text = text_path.read_text(encoding="utf-8")
    data = build_network_data(text)
    network_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    # Build catalogue networks for all three family members
    catalogue = {}
    for key in ("frances", "cb_sr", "cb_jr"):
        catalogue[key] = build_catalogue_network(key)
    catalogue_json = json.dumps(catalogue, ensure_ascii=False, separators=(",", ":"))

    d3_src = _get_d3_source()
    # Un-double braces first (CSS/JS use {{ }} in the template to avoid
    # colliding with the Python placeholders), then insert D3 and data
    # so their natural braces are not affected.
    html = HTML_TEMPLATE.replace("{{", "{").replace("}}", "}")
    html = html.replace("{D3_SOURCE}", d3_src, 1)
    html = html.replace("{NETWORK_JSON}", network_json, 1)
    html = html.replace("{CATALOGUE_JSON}", catalogue_json, 1)
    out_path.write_text(html, encoding="utf-8")
    print(f"Correspondent network -> {out_path}")
    print(f"  OUP: {len(data['nodes'])} nodes  {len(data['edges'])} edges  "
          f"{len(data['letters'])} letters  {len(data['journals'])} journals")
    for key in ("frances", "cb_sr", "cb_jr"):
        cat = catalogue[key]
        print(f"  {cat['subject']}: {len(cat['nodes'])} nodes  "
              f"{len(cat['edges'])} edges  {len(cat['letters'])} letters")


if __name__ == "__main__":
    build()
