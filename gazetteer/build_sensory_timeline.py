#!/usr/bin/env python3
"""
Build the self-contained sensory timeline HTML.

Reads venues.csv and sensory.db, writes sensory_timeline.html.

Usage:
    python3 gazetteer/build_sensory_timeline.py
    open gazetteer/sensory_timeline.html
"""

import csv
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

VENUES_PATH = Path(__file__).parent / "venues.csv"
DB_PATH     = Path(__file__).parent / "sensory.db"
OUT_PATH    = Path(__file__).parent / "sensory_timeline.html"

MIN_PASSAGES = 20
MIN_SOURCES  = 3


# ── helpers ────────────────────────────────────────────────────────────────


def fmt_author(s: str) -> str:
    """'FrancesBurney' -> 'Frances Burney'; 'burney' -> 'Burney'."""
    overrides = {"MGLewis": "M. G. Lewis"}
    if s in overrides:
        return overrides[s]
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    return spaced.capitalize() if " " not in spaced else spaced


def assign_decade(date_min: int) -> int:
    """Return the decade for a given year: assign_decade(1725) == 1720."""
    return (date_min // 10) * 10


def venue_qualifies(
    passages: list[dict],
    min_passages: int = MIN_PASSAGES,
    min_sources: int  = MIN_SOURCES,
) -> bool:
    """Return True if the passage list meets the evidence threshold."""
    if len(passages) < min_passages:
        return False
    distinct_sources = {p["source_id"] for p in passages}
    return len(distinct_sources) >= min_sources


def select_highlight(passages: list[dict]) -> dict:
    """
    Choose the best representative passage from a list.

    Preference order:
      1. Fiction over non-fiction (when length is within 20% of the longest).
      2. Longest text otherwise.
    """
    if not passages:
        raise ValueError("passages must be non-empty")
    longest_len = max(len(p["text"]) for p in passages)
    threshold   = longest_len * 0.8

    # Candidates whose text length >= threshold
    candidates = [p for p in passages if len(p["text"]) >= threshold]

    # Prefer fiction among candidates
    fiction = [p for p in candidates if p.get("source_type") == "fiction"]
    if fiction:
        return max(fiction, key=lambda p: len(p["text"]))
    return max(candidates, key=lambda p: len(p["text"]))


# ── data loading ───────────────────────────────────────────────────────────


def load_venues(venues_path: Path) -> dict[str, dict]:
    """Load LON-prefix venues from CSV, keyed by id."""
    with open(venues_path, newline="", encoding="utf-8") as f:
        return {
            row["id"]: {
                "id":            row["id"],
                "name":          row["name"],
                "lat":           float(row["lat"]),
                "lon":           float(row["lon"]),
                "enclosure":     row.get("enclosure", ""),
                "building_type": row.get("building_type", ""),
                "material":      row.get("material", ""),
                "capacity":      row.get("capacity", ""),
            }
            for row in csv.DictReader(f)
            if row["id"].startswith("LON")
        }


def load_evidence(db_path: Path) -> dict[str, list[dict]]:
    """
    Query sensory_evidence for LON venues.

    Returns dict keyed by venue_id, each value a list of passage dicts.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    by_venue: dict[str, list[dict]] = defaultdict(list)
    try:
        for row in conn.execute("""
            SELECT venue_id, source_id, source_type, author, title,
                   pub_year, date_min, date_max, modality, valence, text
            FROM   sensory_evidence
            WHERE  venue_id LIKE 'LON%'
              AND  date_min IS NOT NULL
            ORDER  BY date_min
        """):
            by_venue[row["venue_id"]].append({
                "source_id":   row["source_id"],
                "source_type": row["source_type"],
                "author":      fmt_author(row["author"] or ""),
                "title":       row["title"] or "",
                "pub_year":    row["pub_year"],
                "date_min":    row["date_min"],
                "date_max":    row["date_max"],
                "modality":    row["modality"],
                "valence":     row["valence"],
                "text":        row["text"] or "",
            })
    finally:
        conn.close()
    return dict(by_venue)


# ── computation ────────────────────────────────────────────────────────────


def compute_decade_summaries(
    evidence_by_venue: dict[str, list[dict]],
) -> dict[str, dict[int, dict]]:
    """
    For each qualifying venue, compute per-decade summary dicts.

    Returns:
        {venue_id: {decade: {"passage_count": int, "source_count": int,
                             "modality_breakdown": {modality: int}}}}
    """
    result: dict[str, dict[int, dict]] = {}
    for venue_id, passages in evidence_by_venue.items():
        if not venue_qualifies(passages):
            continue
        decades: dict[int, dict] = {}
        for p in passages:
            decade = assign_decade(p["date_min"])
            if decade not in decades:
                decades[decade] = {
                    "passage_count": 0,
                    "source_ids":    set(),
                    "modality_breakdown": defaultdict(int),
                }
            decades[decade]["passage_count"] += 1
            decades[decade]["source_ids"].add(p["source_id"])
            decades[decade]["modality_breakdown"][p["modality"]] += 1

        # Serialise sets and defaultdicts
        serialised: dict[int, dict] = {}
        for decade, summary in decades.items():
            serialised[decade] = {
                "passage_count":      summary["passage_count"],
                "source_count":       len(summary["source_ids"]),
                "modality_breakdown": dict(summary["modality_breakdown"]),
            }
        result[venue_id] = serialised
    return result


def compute_highlights(
    evidence_by_venue: dict[str, list[dict]],
) -> dict[str, dict[int, dict]]:
    """
    For each qualifying venue and decade, select one highlight passage.

    Returns:
        {venue_id: {decade: passage_dict}}
    """
    result: dict[str, dict[int, dict]] = {}
    for venue_id, passages in evidence_by_venue.items():
        if not venue_qualifies(passages):
            continue
        by_decade: dict[int, list[dict]] = defaultdict(list)
        for p in passages:
            by_decade[assign_decade(p["date_min"])].append(p)
        result[venue_id] = {
            decade: select_highlight(ps)
            for decade, ps in by_decade.items()
        }
    return result


# ── build ──────────────────────────────────────────────────────────────────


def build(
    venues_path: Path = VENUES_PATH,
    db_path:     Path = DB_PATH,
    out_path:    Path = OUT_PATH,
) -> None:
    """Build sensory_timeline.html."""
    venues_all        = load_venues(venues_path)
    evidence_by_venue = load_evidence(db_path)

    # Filter venues to qualifying ones only
    decade_summaries = compute_decade_summaries(evidence_by_venue)
    highlights       = compute_highlights(evidence_by_venue)
    qualifying_ids   = set(decade_summaries.keys())

    venues_out   = [v for vid, v in venues_all.items() if vid in qualifying_ids]
    evidence_out = {vid: ps for vid, ps in evidence_by_venue.items() if vid in qualifying_ids}

    html = HTML_TEMPLATE.format(
        VENUES_JSON           = json.dumps(venues_out, ensure_ascii=False, separators=(",", ":")),
        EVIDENCE_JSON         = json.dumps(evidence_out, ensure_ascii=False, separators=(",", ":")),
        DECADE_SUMMARIES_JSON = json.dumps(
            {vid: {str(k): v for k, v in ds.items()} for vid, ds in decade_summaries.items()},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        HIGHLIGHTS_JSON       = json.dumps(
            {vid: {str(k): v for k, v in hl.items()} for vid, hl in highlights.items()},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )
    out_path.write_text(html, encoding="utf-8")

    total_ev = sum(len(ps) for ps in evidence_out.values())
    print(f"Sensory timeline -> {out_path}")
    print(f"  {len(venues_out)} qualifying venues  {total_ev} passages")


# ── template ───────────────────────────────────────────────────────────────
# Note: literal JS braces are doubled ({{ }}) so str.format() works.

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sensory Timeline &middot; 18c London</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Inter', system-ui, sans-serif; background: #f4f1eb; color: #1a1816; }}
#header {{
  padding: 16px 20px;
  background: rgba(255,255,255,0.97);
  border-bottom: 1px solid #ccc;
  box-shadow: 0 2px 8px rgba(0,0,0,0.10);
}}
#header h1 {{ font-size: 15px; font-weight: 700; }}
#header p  {{ font-size: 12px; color: #888; margin-top: 3px; }}
#content {{ padding: 20px; max-width: 1100px; margin: 0 auto; }}
.venue-block {{ margin-bottom: 28px; }}
.venue-name {{ font-size: 14px; font-weight: 700; margin-bottom: 8px; }}
.decade-row {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.decade-cell {{
  background: white; border: 1px solid #e0dbd4; border-radius: 4px;
  padding: 8px 10px; min-width: 110px; font-size: 11px; line-height: 1.5;
}}
.decade-label {{ font-weight: 700; color: #1e3c6e; margin-bottom: 3px; }}
.decade-stats {{ color: #888; font-size: 10px; }}
.highlight-text {{ color: #444; font-style: italic; margin-top: 4px; font-size: 11px; line-height: 1.4; }}
</style>
</head>
<body>
<div id="header">
  <h1>Sensory Timeline &middot; 18th-Century London</h1>
  <p>Qualifying venues with 20+ passages from 3+ distinct sources, summarised by decade.</p>
</div>
<div id="content"></div>
<script>
// ── embedded data ──────────────────────────────────────────────────────────
const VENUES           = {VENUES_JSON};
const EVIDENCE         = {EVIDENCE_JSON};
const DECADE_SUMMARIES = {DECADE_SUMMARIES_JSON};
const HIGHLIGHTS       = {HIGHLIGHTS_JSON};

// ── helpers ────────────────────────────────────────────────────────────────
function escHtml(s) {{
  return String(s || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}}

// ── render ─────────────────────────────────────────────────────────────────
var container = document.getElementById('content');

VENUES.forEach(function(v) {{
  var summaries  = DECADE_SUMMARIES[v.id] || {{}};
  var highlights = HIGHLIGHTS[v.id] || {{}};
  var decades    = Object.keys(summaries).map(Number).sort(function(a, b) {{ return a - b; }});
  if (!decades.length) return;

  var block = document.createElement('div');
  block.className = 'venue-block';

  var nameEl = document.createElement('div');
  nameEl.className = 'venue-name';
  nameEl.textContent = v.name;
  block.appendChild(nameEl);

  var row = document.createElement('div');
  row.className = 'decade-row';

  decades.forEach(function(d) {{
    var s   = summaries[d];
    var hl  = highlights[d];
    var cell = document.createElement('div');
    cell.className = 'decade-cell';

    var label = document.createElement('div');
    label.className = 'decade-label';
    label.textContent = d + 's';
    cell.appendChild(label);

    var stats = document.createElement('div');
    stats.className = 'decade-stats';
    stats.textContent = s.passage_count + ' passages \u00b7 ' + s.source_count + ' sources';
    cell.appendChild(stats);

    if (hl && hl.text) {{
      var snippet = hl.text.length > 120 ? hl.text.slice(0, 120) + '\u2026' : hl.text;
      var hlEl = document.createElement('div');
      hlEl.className = 'highlight-text';
      hlEl.textContent = '\u201c' + snippet + '\u201d';
      cell.appendChild(hlEl);
    }}

    row.appendChild(cell);
  }});

  block.appendChild(row);
  container.appendChild(block);
}});
</script>
</body>
</html>"""


if __name__ == "__main__":
    build()
