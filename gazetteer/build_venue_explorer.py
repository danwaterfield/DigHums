#!/usr/bin/env python3
"""
Build the self-contained venue explorer HTML.

Reads venues.csv and sensory.db, writes venue_explorer.html.

Usage:
    python3 gazetteer/build_venue_explorer.py
    open gazetteer/venue_explorer.html
"""

import csv
import json
import re
import sqlite3
from pathlib import Path

VENUES_PATH = Path(__file__).parent / "venues.csv"
DB_PATH     = Path(__file__).parent / "sensory.db"
OUT_PATH    = Path(__file__).parent / "venue_explorer.html"


def fmt_author(s: str) -> str:
    """'FrancesBurney' -> 'Frances Burney'; 'burney' -> 'Burney'."""
    overrides = {"MGLewis": "M. G. Lewis"}
    if s in overrides:
        return overrides[s]
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    # All DB authors are single-word lowercase or CamelCase; no multi-word lowercase names exist.
    return spaced.capitalize() if " " not in spaced else spaced


def load_data(venues_path: Path, db_path: Path) -> list[dict]:
    """
    Return list of venue dicts, each with an 'evidence' array.
    All 73 venues are included; venues with no evidence have evidence=[].
    """
    with open(venues_path, newline="", encoding="utf-8") as f:
        venues = {
            row["id"]: {
                "id":       row["id"],
                "name":     row["name"],
                "lat":      float(row["lat"]),
                "lon":      float(row["lon"]),
                "evidence": [],
            }
            for row in csv.DictReader(f)
        }

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    for row in conn.execute("""
        SELECT venue_id, source_type, author, title, pub_year,
               date_min, date_max, modality, text, context
        FROM   sensory_evidence
        WHERE  venue_id IS NOT NULL
        ORDER  BY date_min
    """):
        vid = row["venue_id"]
        if vid in venues:
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
            })
    conn.close()
    return list(venues.values())


def build(
    venues_path: Path = VENUES_PATH,
    db_path: Path     = DB_PATH,
    out_path: Path    = OUT_PATH,
) -> None:
    """Build venue_explorer.html."""
    venues  = load_data(venues_path, db_path)
    data_js = json.dumps(venues, ensure_ascii=False, separators=(",", ":"))
    html    = _render(data_js)
    out_path.write_text(html, encoding="utf-8")
    geocoded = sum(1 for v in venues if v["evidence"])
    total_ev = sum(len(v["evidence"]) for v in venues)
    print(f"Venue explorer -> {out_path}")
    print(f"  {len(venues)} venues  {geocoded} with evidence  {total_ev} passages")


def _render(data_js: str) -> str:
    return HTML_TEMPLATE.replace("__VENUES_DATA__", data_js)


HTML_TEMPLATE = "<html><body>TODO __VENUES_DATA__</body></html>"


if __name__ == "__main__":
    build()
