"""Canonical figures for the public landing page.

The landing page used to carry hand-typed counts that drifted away from the
data as the gazetteer grew. Everything advertised on index.html is computed
here instead, from the same CSVs and database the builders read, so the copy
cannot get ahead of the tools again.

Run via build_all.py, or on its own to inspect the numbers:

    python3 gazetteer/site_stats.py
"""

from __future__ import annotations

import csv
import json
import sqlite3
import subprocess
from pathlib import Path

BASE = Path(__file__).parent
ROOT = BASE.parent

DB_PATH               = BASE / "sensory.db"
VENUES_PATH           = BASE / "venues.csv"
BOOKSELLERS_PATH      = BASE / "booksellers.csv"
BOOKSELLER_LOCS_PATH  = BASE / "bookseller_locations.csv"
GRAPH_PATH            = BASE / "unified_correspondence_graph.csv"
NARRATIVE_PATH        = BASE / "narrative_mentions.json"

# Directories holding the published plain-text corpus.
CORPUS_EXCLUDE_PREFIXES = ("gazetteer/", "burney-attribution/", "analysis/",
                           "docs/", "tmp/", "output/", "venv/")


def _rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def corpus_stats() -> dict:
    """Size of the plain-text corpus that is actually published.

    Counts tracked files only: gitignored texts exist locally but are not on
    the public site, so advertising them would overstate what a visitor gets.
    """
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "*.txt"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"texts": None, "size_mb": None}

    files = [p for p in tracked
             if not p.startswith(CORPUS_EXCLUDE_PREFIXES)]
    total = sum((ROOT / p).stat().st_size for p in files if (ROOT / p).exists())
    return {
        "texts": len(files),
        "size_mb": round(total / 1_000_000),
    }


def fiction_stats() -> dict:
    """Venue, evidence and novel counts behind the sensory tools."""
    venues = _rows(VENUES_PATH)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    venue_linked, = cur.execute(
        "SELECT COUNT(*) FROM sensory_evidence WHERE venue_id IS NOT NULL AND venue_id != ''"
    ).fetchone()
    total_evidence, = cur.execute("SELECT COUNT(*) FROM sensory_evidence").fetchone()
    venues_with_evidence, = cur.execute(
        "SELECT COUNT(DISTINCT venue_id) FROM sensory_evidence "
        "WHERE venue_id IS NOT NULL AND venue_id != ''"
    ).fetchone()

    novels, authors, y_min, y_max = cur.execute(
        "SELECT COUNT(DISTINCT title), COUNT(DISTINCT author), "
        "MIN(pub_year), MAX(pub_year) FROM sources WHERE source_type='fiction'"
    ).fetchone()

    span_min, span_max = cur.execute(
        "SELECT MIN(date_min), MAX(date_max) FROM sources WHERE date_min > 0"
    ).fetchone()

    source_types, = cur.execute(
        "SELECT COUNT(DISTINCT source_type) FROM sources"
    ).fetchone()
    con.close()

    # Novels that actually carry a plotted path in the narrative map. The
    # builder drops anything below MIN_EVENTS, so apply the same threshold
    # rather than counting texts the selector never offers.
    novels_with_paths = None
    if NARRATIVE_PATH.exists():
        from build_narrative_map import MIN_EVENTS
        data = json.loads(NARRATIVE_PATH.read_text(encoding="utf-8"))
        novels_with_paths = sum(1 for t in data if t.get("event_count", 0) >= MIN_EVENTS)

    return {
        "venues": len(venues),
        "venues_with_evidence": venues_with_evidence,
        "passages": venue_linked,
        "passages_total": total_evidence,
        "novels": novels,
        "authors": authors,
        "novels_with_paths": novels_with_paths,
        "novel_year_min": y_min,
        "novel_year_max": y_max,
        "span_min": span_min,
        "span_max": span_max,
        "source_types": source_types,
    }


def correspondence_stats() -> dict:
    """Network and bookseller counts behind the correspondence tools.

    Counts come from the builder's own load_graph(), not from a raw read of
    the CSV. The builder drops unresolvable names and catalogue artefacts, so
    the CSV holds more edges than the page ever renders — quoting the CSV is
    how the landing page came to advertise 1,468 connections for a graph that
    draws 1,254.
    """
    from build_full_network import load_graph

    graph = load_graph()
    lo, hi = graph["year_min"], graph["year_max"]

    # The network view filters every edge against the timeline window, so an
    # edge with no usable year is invisible at any slider position. Headline
    # figures report what the page can actually draw; the *_total keys keep
    # the dataset size available for the method note.
    renderable = [e for e in graph["edges"]
                  if not (e["year_max"] < lo or e["year_min"] > hi)]
    renderable_ids = ({e["source"] for e in renderable}
                      | {e["target"] for e in renderable})

    booksellers = _rows(BOOKSELLERS_PATH)
    locations   = _rows(BOOKSELLER_LOCS_PATH)
    located_venues = {r["venue_id"] for r in locations if r.get("venue_id")}

    return {
        "connections": len(renderable),
        "people": len(renderable_ids),
        "letters": sum(int(e.get("weight") or 0) for e in renderable),
        "connections_total": len(graph["edges"]),
        "people_total": len(graph["nodes"]),
        "undated_connections": len(graph["edges"]) - len(renderable),
        "year_min": lo,
        "year_max": hi,
        "booksellers": len(booksellers),
        "bookseller_locations": len(locations),
        "bookseller_venues": len(located_venues),
    }


def all_stats() -> dict:
    return {
        "corpus": corpus_stats(),
        "fiction": fiction_stats(),
        "correspondence": correspondence_stats(),
    }


def flat_stats() -> dict[str, str]:
    """Flatten to `group.key` -> preformatted string, for index.html injection.

    Counts get thousands separators; years must not (1719, never 1,719).
    """
    out: dict[str, str] = {}
    for group, values in all_stats().items():
        for key, value in values.items():
            if value is None:
                continue
            is_year = key.endswith(("year_min", "year_max", "span_min", "span_max"))
            if isinstance(value, int) and not is_year:
                out[f"{group}.{key}"] = f"{value:,}"
            else:
                out[f"{group}.{key}"] = str(value)
    return out


if __name__ == "__main__":
    for group, values in all_stats().items():
        print(f"[{group}]")
        for key, value in values.items():
            print(f"  {key:24} {value}")
