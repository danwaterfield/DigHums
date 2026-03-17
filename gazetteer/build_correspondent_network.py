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
# Convention: ALL JS braces are doubled.  Only {D3_SOURCE} and
# {NETWORK_JSON} use single braces (Python .replace() targets).

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Frances Burney — Correspondent Network</title>
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
  padding: 12px 24px;
  background: #fff;
  border-bottom: 1px solid #e0e0e0;
  display: flex;
  align-items: baseline;
  gap: 16px;
  flex-shrink: 0;
  z-index: 10;
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
  max-width: 260px;
  z-index: 20;
  display: none;
}}
.tooltip .tt-name {{ font-weight: 600; font-size: 13px; }}
.tooltip .tt-community {{ font-size: 11px; color: #666; }}
.tooltip .tt-detail {{ margin-top: 4px; font-size: 11px; color: #555; }}

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
  width: 320px;
  background: #fff;
  border-left: 1px solid #e0e0e0;
  overflow-y: auto;
  flex-shrink: 0;
  transform: translateX(320px);
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
  <h1>Frances Burney</h1>
  <span class="subtitle">Correspondent Network, 1768 &ndash; 1839</span>
  <div class="stats">
    <span id="stat-nodes"></span>
    <span id="stat-edges"></span>
    <span id="stat-letters"></span>
  </div>
</div>

<div class="main">
  <div class="graph-area" id="graph-area">
    <svg id="network-svg"></svg>
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
    <div class="dp-letters">
      <h3>Letters</h3>
      <table class="dp-table">
        <thead><tr><th>No.</th><th>Date</th><th>Type</th><th>Phase</th></tr></thead>
        <tbody id="dp-tbody"></tbody>
      </table>
    </div>
  </div>
</div>

<div class="timeline">
  <div class="timeline-label" id="tl-label"></div>
  <div class="slider-wrap">
    <input type="range" id="slider-lo" min="1768" max="1839" value="1768">
    <input type="range" id="slider-hi" min="1768" max="1839" value="1839">
  </div>
  <div class="phase-pills" id="phase-pills"></div>
</div>

<script>
{D3_SOURCE}
</script>
<script>
(function() {{
"use strict";

var DATA = {NETWORK_JSON};

var COMMUNITY_COLOUR = {{
  "Centre":         "#2d3436",
  "Family":         "#4a6fa5",
  "Literary":       "#a07855",
  "Court":          "#5a8a7a",
  "Publishers":     "#7a7a8a",
  "Intimate circle":"#a07080",
  "Unknown":        "#b0b0b0"
}};

var rangeLo = 1768, rangeHi = 1839;
var selectedNode = null;

document.getElementById("stat-nodes").textContent = DATA.nodes.length + " correspondents";
document.getElementById("stat-edges").textContent = DATA.edges.length + " connections";
document.getElementById("stat-letters").textContent = DATA.letters.length + " letters";

/* Legend */
(function() {{
  var el = document.getElementById("legend");
  var frag = document.createDocumentFragment();
  var order = ["Centre","Family","Literary","Court","Publishers","Intimate circle","Unknown"];
  var present = new Set(DATA.nodes.map(function(n){{ return n.community; }}));
  order.forEach(function(c) {{
    if (!present.has(c)) return;
    var row = document.createElement("div");
    row.className = "legend-item";
    var swatch = document.createElement("div");
    swatch.className = "legend-swatch";
    swatch.style.background = COMMUNITY_COLOUR[c];
    var label = document.createElement("span");
    label.textContent = c;
    row.appendChild(swatch);
    row.appendChild(label);
    frag.appendChild(row);
  }});
  el.appendChild(frag);
}})();

/* Phase pills */
(function() {{
  var el = document.getElementById("phase-pills");
  DATA.phases.forEach(function(p) {{
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
    el.appendChild(btn);
  }});
}})();

function updatePills() {{
  var pills = document.querySelectorAll(".phase-pill");
  pills.forEach(function(pill, i) {{
    var p = DATA.phases[i];
    pill.classList.toggle("active", rangeLo === p.start && rangeHi === p.end);
  }});
}}

/* Sliders */
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
  var inRange = DATA.letters.filter(function(l) {{ return l.year >= rangeLo && l.year <= rangeHi; }});
  var el = document.getElementById("tl-label");
  el.textContent = "";
  var strong = document.createElement("strong");
  strong.textContent = rangeLo + " \u2013 " + rangeHi;
  el.appendChild(strong);
  el.appendChild(document.createTextNode(" \u00b7 " + inRange.length + " letters in range"));
}}

/* SVG setup */
var svgEl = document.getElementById("network-svg");
var width = svgEl.clientWidth || 900;
var height = svgEl.clientHeight || 600;

var svg = d3.select("#network-svg");
var g = svg.append("g");

var zoom = d3.zoom()
  .scaleExtent([0.3, 5])
  .on("zoom", function(event) {{ g.attr("transform", event.transform); }});
svg.call(zoom);
svg.on("click", function(event) {{
  if (event.target === svgEl) dismissDetail();
}});

/* Scales */
var maxCount = d3.max(DATA.nodes, function(n){{ return n.count; }}) || 1;
var rScale = d3.scaleSqrt().domain([1, maxCount]).range([4, 28]);
var maxWeight = d3.max(DATA.edges, function(e){{ return e.weight; }}) || 1;
var wScale = d3.scaleSqrt().domain([1, maxWeight]).range([0.5, 5]);

/* Build node lookup */
var nodeMap = {{}};
DATA.nodes.forEach(function(n) {{ nodeMap[n.id] = n; }});

/* Links */
var link = g.append("g").attr("class","links")
  .selectAll("line")
  .data(DATA.edges)
  .join("line")
    .attr("stroke", function(d) {{
      var t = nodeMap[d.target] || nodeMap[(d.target && d.target.id) || ""];
      return COMMUNITY_COLOUR[t ? t.community : "Unknown"] || "#ccc";
    }})
    .attr("stroke-opacity", 0.3)
    .attr("stroke-width", function(d) {{ return wScale(d.weight); }});

/* Nodes */
var node = g.append("g").attr("class","nodes")
  .selectAll("g")
  .data(DATA.nodes)
  .join("g")
    .attr("cursor","pointer");

node.append("circle")
  .attr("r", function(d) {{ return d.id === "Frances Burney" ? 30 : rScale(d.count); }})
  .attr("fill", function(d) {{ return COMMUNITY_COLOUR[d.community] || "#b0b0b0"; }})
  .attr("stroke", function(d) {{ return d.id === "Frances Burney" ? "#1a1a1a" : "#fff"; }})
  .attr("stroke-width", function(d) {{ return d.id === "Frances Burney" ? 2.5 : 1.5; }});

node.filter(function(d) {{ return d.id === "Frances Burney"; }})
  .append("text")
  .text("FB")
  .attr("text-anchor","middle")
  .attr("dy","0.35em")
  .attr("fill","#fff")
  .attr("font-size","12px")
  .attr("font-weight","600")
  .style("pointer-events","none");

/* Labels for prominent nodes */
var countThreshold = d3.quantile(
  DATA.nodes.filter(function(n){{ return n.id !== "Frances Burney"; }})
    .map(function(n){{ return n.count; }}).sort(d3.ascending),
  0.75
) || 1;

node.filter(function(d) {{ return d.id !== "Frances Burney" && d.count >= countThreshold; }})
  .append("text")
  .text(function(d) {{ return d.id.split(" ").slice(-1)[0]; }})
  .attr("dx", function(d) {{ return rScale(d.count) + 4; }})
  .attr("dy","0.35em")
  .attr("font-size","10px")
  .attr("fill","#555")
  .style("pointer-events","none");

/* Tooltip */
var tooltip = document.getElementById("tooltip");

node.on("mouseenter", function(event, d) {{
  var letters = DATA.letters.filter(function(l) {{ return l.correspondent === d.id; }});
  var years = letters.map(function(l) {{ return l.year; }});
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
  showDetail(d);
}});

function showDetail(d) {{
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

  var letters = DATA.letters.filter(function(l) {{ return l.correspondent === d.id; }});
  var inRange = letters.filter(function(l) {{ return l.year >= rangeLo && l.year <= rangeHi; }});
  var years = letters.map(function(l) {{ return l.year; }});
  var yMin = d3.min(years), yMax = d3.max(years);

  var statsEl = document.getElementById("dp-stats");
  while (statsEl.firstChild) statsEl.removeChild(statsEl.firstChild);
  var countLine = document.createElement("div");
  var b1 = document.createElement("strong");
  b1.textContent = letters.length;
  countLine.appendChild(b1);
  countLine.appendChild(document.createTextNode(" total letters"));
  if (inRange.length !== letters.length) {{
    countLine.appendChild(document.createTextNode(" \u00b7 "));
    var b2 = document.createElement("strong");
    b2.textContent = inRange.length;
    countLine.appendChild(b2);
    countLine.appendChild(document.createTextNode(" in range"));
  }}
  statsEl.appendChild(countLine);
  var dateLine = document.createElement("div");
  dateLine.textContent = (yMin || "?") + " \u2013 " + (yMax || "?");
  statsEl.appendChild(dateLine);

  var tbody = document.getElementById("dp-tbody");
  while (tbody.firstChild) tbody.removeChild(tbody.firstChild);
  letters.forEach(function(l) {{
    var tr = document.createElement("tr");
    if (l.year < rangeLo || l.year > rangeHi) tr.style.opacity = "0.3";
    var tdNum = document.createElement("td");
    tdNum.textContent = l.number;
    var tdDate = document.createElement("td");
    var monthStr = l.month ? String(l.month).padStart(2,"0") + "/" : "";
    tdDate.textContent = monthStr + l.year;
    var tdType = document.createElement("td");
    tdType.textContent = l.type;
    var tdPhase = document.createElement("td");
    tdPhase.style.fontSize = "10px";
    tdPhase.textContent = l.phase;
    tr.appendChild(tdNum);
    tr.appendChild(tdDate);
    tr.appendChild(tdType);
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

/* Force simulation */
var simulation = d3.forceSimulation(DATA.nodes)
  .force("link", d3.forceLink(DATA.edges).id(function(d){{ return d.id; }})
    .distance(function(d) {{ return 60 + 120 / Math.sqrt(d.weight || 1); }})
    .strength(function(d) {{ return 0.3 + 0.3 * Math.min(d.weight / maxWeight, 1); }})
  )
  .force("charge", d3.forceManyBody().strength(-120))
  .force("center", d3.forceCenter(width / 2, height / 2))
  .force("collision", d3.forceCollide().radius(function(d) {{
    return (d.id === "Frances Burney" ? 34 : rScale(d.count) + 3);
  }}))
  .on("tick", ticked);

var burneyNode = DATA.nodes.find(function(n){{ return n.id === "Frances Burney"; }});
if (burneyNode) {{
  burneyNode.fx = width / 2;
  burneyNode.fy = height / 2;
}}

function ticked() {{
  link
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
    if (d.id !== "Frances Burney") {{ d.fx = null; d.fy = null; }}
  }})
);

/* Filter by range */
function filterGraph() {{
  updateLabel();
  var rangeCounts = {{}};
  DATA.letters.forEach(function(l) {{
    if (l.year >= rangeLo && l.year <= rangeHi) {{
      rangeCounts[l.correspondent] = (rangeCounts[l.correspondent] || 0) + 1;
    }}
  }});

  node.select("circle")
    .transition().duration(300)
    .attr("r", function(d) {{
      if (d.id === "Frances Burney") return 30;
      var c = rangeCounts[d.id] || 0;
      return c > 0 ? rScale(c) : rScale(1) * 0.6;
    }})
    .attr("opacity", function(d) {{
      if (d.id === "Frances Burney") return 1;
      return (rangeCounts[d.id] || 0) > 0 ? 1 : 0.1;
    }});

  node.selectAll("text")
    .transition().duration(300)
    .attr("opacity", function(d) {{
      if (d.id === "Frances Burney") return 1;
      return (rangeCounts[d.id] || 0) > 0 ? 1 : 0.08;
    }});

  link.transition().duration(300)
    .attr("stroke-opacity", function(d) {{
      var tid = typeof d.target === "object" ? d.target.id : d.target;
      return (rangeCounts[tid] || 0) > 0 ? 0.3 : 0.03;
    }})
    .attr("stroke-width", function(d) {{
      var tid = typeof d.target === "object" ? d.target.id : d.target;
      var c = rangeCounts[tid] || 0;
      return c > 0 ? wScale(c) : 0.3;
    }});

  if (selectedNode) showDetail(selectedNode);
}}

/* Resize */
window.addEventListener("resize", function() {{
  width = svgEl.clientWidth;
  height = svgEl.clientHeight;
  simulation.force("center", d3.forceCenter(width / 2, height / 2));
  if (burneyNode) {{ burneyNode.fx = width / 2; burneyNode.fy = height / 2; }}
  simulation.alpha(0.3).restart();
}});

/* Init */
updateLabel();

}})();
</script>
</body>
</html>
"""


# ── Build function ────────────────────────────────────────────────

def build(
    text_path: Path = TEXT_PATH,
    out_path: Path = OUT_PATH,
) -> None:
    """Read the OUP text, build network data, render HTML, write file."""
    text = text_path.read_text(encoding="utf-8")
    data = build_network_data(text)
    network_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    d3_src = _get_d3_source()
    # Un-double braces first (CSS/JS use {{ }} in the template to avoid
    # colliding with the Python placeholders), then insert D3 and data
    # so their natural braces are not affected.
    html = HTML_TEMPLATE.replace("{{", "{").replace("}}", "}")
    html = html.replace("{D3_SOURCE}", d3_src, 1)
    html = html.replace("{NETWORK_JSON}", network_json, 1)
    out_path.write_text(html, encoding="utf-8")
    print(f"Correspondent network -> {out_path}")
    print(f"  {len(data['nodes'])} nodes  {len(data['edges'])} edges  "
          f"{len(data['letters'])} letters  {len(data['journals'])} journals")


if __name__ == "__main__":
    build()
