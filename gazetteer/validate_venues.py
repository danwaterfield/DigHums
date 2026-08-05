#!/usr/bin/env python3
"""
Validate venue mentions against publication dates and opening/closing dates.

For each text in the corpus, check whether named venues existed during
the text's setting period. Flags anachronisms and produces a report.

Also determines which base map layer is appropriate for each text.

Usage:
    python3 validate_venues.py            # validation report only
    python3 validate_venues.py --export   # report + write mentions.json
"""

import argparse
import csv
import json
import re
from pathlib import Path
from collections import defaultdict


CORPUS_ROOT = Path(__file__).parent.parent
ALIAS_SUPPLEMENT_PATH = Path(__file__).with_name("venue_aliases_supplement.csv")


def load_venues(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_corpus(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# Search terms for each venue.
#
# Format: venue_id -> list of (pattern, city_filter) tuples.
# city_filter: restrict match to texts whose primary_cities includes this
#              string. None = apply to all texts.
#
# Changes from v1:
#   LON001  removed "Spring Garden/s" — ambiguous with Charing Cross gardens
#   LON003  removed "Mary-le-bone" — catches district name, not the Gardens
#   LON007  merged LON016 (Covent Garden piazza) here; removed "the Piazza"
#           which is too ambiguous in Gothic texts set in Italy
#   LON008  removed "the Opera" — catches "operation"; replaced with
#           word-boundary regex and specific house names
#   LON009  removed "James's Park" — too short, too many false contexts
#   LON010  removed "the Park" — 100% false positives (private estate parks)
#   BAT001  removed "the Rooms" — too generic without city filter; kept as
#           city-filtered alias for Bath-only texts
#   BAT003  removed "the Pump" — keep only hyphenated / full forms
#   BAT005  removed "the Crescent" — kept as city-filtered Bath-only alias
#   BAT004  kept "the Circus" but city-filtered to Bath
VENUE_ALIASES = {
    # LONDON
    "LON001": [
        ("Vauxhall", None),
        ("Vaux-hall", None),
        ("Vauxhall Gardens", None),
    ],
    "LON002": [
        ("Ranelagh", None),
        ("Ranelagh Gardens", None),
    ],
    "LON003": [
        ("Marylebone Gardens", None),
        ("Marybone Gardens", None),
        ("Mary-le-bone Gardens", None),
    ],
    "LON004": [
        ("the Pantheon", None),
        ("Pantheon in Oxford", None),       # distinguishes from generic "pantheon"
    ],
    "LON005": [
        ("Almack's", None),
        ("Almacks", None),
    ],
    "LON006": [
        ("Drury Lane", None),
        ("Theatre Royal, Drury", None),
    ],
    "LON007": [                             # Covent Garden (theatre + piazza merged)
        ("Covent Garden", None),
        ("Covent-Garden", None),
    ],
    "LON016": [                             # Covent Garden Piazza
        ("Covent-Garden Piazza", None),
        (r"\bthe Piazza\b", "London"),      # city-filtered: avoids Italian piazzas
    ],
    "LON008": [                             # King's Theatre / Opera House, Haymarket
        ("King's Theatre", None),
        ("Opera-house", None),              # Burney's spelling
        ("Opera House", None),
        (r"\bthe opera\b", None),           # regex — word boundary stops "operation"
    ],
    "LON009": [
        ("St James's Park", None),
        ("St. James's Park", None),
        ("Saint James's Park", None),
    ],
    "LON010": [
        ("Hyde Park", None),
        ("Rotten Row", None),
    ],
    "LON011": [
        ("Grosvenor Square", None),
        ("Grosvenor-Square", None),
    ],
    "LON012": [
        ("Hanover Square", None),
        ("Hanover-Square", None),
    ],
    "LON013": [
        ("Berkeley Square", None),
        ("Berkley Square", None),
    ],
    "LON014": [
        ("St James's Square", None),
        ("St. James's Square", None),
    ],
    "LON015": [
        ("the Strand", None),
    ],
    "LON018": [
        ("White's Club", None),
        ("White's in St", None),            # "White's in St James's Street"
    ],
    "LON024": [("Portman Square", None)],
    "LON025": [("Portland Place", None)],
    "LON026": [("Bedford Square", None)],

    # --- STREETS ---
    "LON027": [                             # Bond Street
        ("Bond Street", None),
        ("Bond-Street", None),
        ("Bond-street", None),
        ("New Bond Street", None),
        ("Old Bond Street", None),
    ],
    "LON028": [                             # Pall Mall
        ("Pall Mall", None),
        ("Pall-Mall", None),
        ("Pall-mall", None),
    ],
    "LON029": [("Piccadilly", None)],
    "LON030": [                             # St James's Street
        ("St James's Street", None),
        ("St. James's Street", None),
        ("James's Street", None),           # short form used in context
    ],
    "LON031": [                             # Fleet Street
        ("Fleet Street", None),
        ("Fleet-Street", None),
        ("Fleet-street", None),
    ],
    "LON032": [                             # Cheapside
        ("Cheapside", None),
        ("Cheap-side", None),
    ],
    "LON033": [                             # Oxford Street / Oxford Road
        ("Oxford Street", None),
        ("Oxford-Street", None),
        (r"\bOxford Road\b", None),
        (r"\bOxford-Road\b", None),
    ],
    "LON034": [("Ludgate Hill", None), ("Ludgate-Hill", None)],
    "LON035": [("Cornhill", None), ("Corn-hill", None)],
    "LON036": [("Charing Cross", None), ("Charing-Cross", None)],
    "LON037": [                             # Leicester Fields / Square
        ("Leicester Square", None),
        ("Leicester-Square", None),
        ("Leicester Fields", None),
        ("Leicester-Fields", None),
    ],
    "LON038": [("Cavendish Square", None), ("Cavendish-Square", None)],
    "LON039": [                             # Harley St / Wimpole St medical quarter
        ("Harley Street", None),
        ("Harley-Street", None),
        ("Wimpole Street", None),
        ("Wimpole-Street", None),
    ],
    "LON040": [("Brook Street", None), ("Brook-Street", None)],

    # --- PLEBEIAN LEISURE VENUES ---
    "LON054": [("Saltero", None), ("Don Saltero", None)],
    "LON055": [("White-Conduit", None), ("White Conduit", None)],
    "LON056": [("Bagnigge", None)],
    "LON057": [("Mother Red Cap", None), ("Mother Redcap", None)],

    # --- COFFEE HOUSES ---
    # Aliases kept specific to avoid false positives (full name preferred).
    "LON041": [("Will's Coffee", None), ("Wills Coffee", None)],
    "LON042": [("Button's Coffee", None), ("Buttons Coffee", None)],
    "LON043": [("Tom's Coffee", None)],
    "LON044": [                             # Slaughter's — distinctive enough unqualified
        ("Slaughter's Coffee", None),
        ("Slaughter's", None),
    ],
    "LON045": [("Lloyd's Coffee", None), ("Lloyds Coffee", None)],
    "LON046": [("Chapter Coffee", None)],
    "LON047": [("Garraway's", None), ("Garaway's", None)],

    # --- DISTRICTS ---
    # Word-boundary regex on "the City" prevents catching "city of Bath" etc.
    "LON048": [
        (r"\bthe City\b", "London"),        # city-filtered: "the city" in Gothic texts = Rome/Seville/Samarah
        ("City of London", None),
        ("Lombard Street", None),          # emblematic City address
    ],
    "LON049": [("Bloomsbury", None)],
    "LON050": [("Holborn", None)],
    "LON051": [("Soho", None)],
    "LON052": [                             # Southwark — "the Borough" city-filtered
        ("Southwark", None),
        (r"\bthe Borough\b", None),         # could be generic; watch for false positives
    ],
    "LON053": [("Mayfair", None), ("May Fair", None), ("May-Fair", None)],

    # BATH  (city-filtered aliases marked with "Bath")
    "BAT001": [
        ("Lower Rooms", "Bath"),
        ("Harrison's Rooms", "Bath"),
        ("Lindsey's Rooms", "Bath"),
        ("Lower Assembly Rooms", "Bath"),
        ("the Rooms", "Bath"),              # city-filtered: safe in Bath-only texts
    ],
    "BAT002": [
        ("Upper Rooms", "Bath"),
        ("New Rooms", "Bath"),
        ("Upper Assembly Rooms", "Bath"),
    ],
    "BAT003": [
        ("Pump Room", "Bath"),
        ("Pump-Room", "Bath"),
        ("Pump-room", "Bath"),
    ],
    "BAT004": [
        ("the Circus", "Bath"),             # city-filtered: avoids Italian circus refs
        ("King's Circus", "Bath"),
    ],
    "BAT005": [
        ("Royal Crescent", "Bath"),
        ("the Crescent", "Bath"),           # city-filtered: safe in Bath-only texts
    ],
    "BAT006": [
        ("Pulteney Bridge", "Bath"),
        ("Pulteney-Bridge", "Bath"),
    ],
    "BAT007": [
        ("Sydney Gardens", "Bath"),
        ("Sydney-Gardens", "Bath"),
    ],
    "BAT008": [
        ("Queen Square", "Bath"),
        ("Queen's Square", "Bath"),
    ],
    "BAT009": [
        ("Milsom Street", "Bath"),
        ("Milsom-Street", "Bath"),
    ],
    "BAT010": [                             # Bath Abbey
        ("Bath Abbey", None),
    ],
    "BAT011": [("North Parade", "Bath")],
    "BAT012": [("South Parade", "Bath")],

    # BATH STREETS (all city-filtered)
    "BAT013": [("Gay Street", "Bath"), ("Gay-Street", "Bath")],
    "BAT014": [("Cheap Street", "Bath"), ("Cheap-Street", "Bath")],
    "BAT015": [
        ("Argyle Street", "Bath"),
        ("Argyle-Street", "Bath"),
        ("Argyll Street", "Bath"),
    ],
    "BAT016": [("Pulteney Street", "Bath"), ("Pulteney-Street", "Bath")],

    # MONUMENTS AND LANDMARKS
    "LON021": [                             # Tower of London
        ("Tower of London", None),
        (r"\bthe Tower\b", "London"),       # city-filtered: avoids false positives
    ],
    "LON022": [                             # St Paul's Cathedral
        ("St Paul's Cathedral", None),
        ("St. Paul's Cathedral", None),
        ("St Paul's", None),
        ("St. Paul's", None),
    ],
    "LON023": [                             # Westminster Abbey
        ("Westminster Abbey", None),
        (r"\bthe Abbey\b", "London"),       # city-filtered: avoids Gothic abbey refs
    ],
    "LON096": [                             # Greenwich Park
        ("Greenwich Park", None),
        ("Greenwich", None),
    ],
    # PRISONS
    "LON074": [
        ("Newgate", None),
        ("Newgate Prison", None),
        ("Newgate Gaol", None),
    ],
    "LON075": [
        ("Fleet Prison", None),
        ("Fleet Gaol", None),
    ],
    "LON076": [
        ("Marshalsea", None),
        ("Marshalsea Prison", None),
    ],
    "LON077": [
        ("Bridewell", None),
        ("Bridewell Prison", None),
        ("House of Correction", "London"),
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
    "LON093": [
        ("Bankside", None),
        ("Borough", "London"),
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
}


def _load_alias_supplement(path: Path) -> dict[str, list[tuple[str, str | None]]]:
    """Load optional venue aliases from a CSV supplement file."""
    extra_aliases: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
    if not path.exists():
        return extra_aliases

    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            venue_id = (row.get("venue_id") or "").strip()
            pattern = (row.get("pattern") or "").strip()
            if not venue_id or not pattern:
                continue
            city_filter = (row.get("city_filter") or "").strip() or None
            extra_aliases[venue_id].append((pattern, city_filter))
    return extra_aliases


def _merge_alias_supplement(
    base_aliases: dict[str, list[tuple[str, str | None]]],
    path: Path,
) -> dict[str, list[tuple[str, str | None]]]:
    """Append deduplicated supplement aliases to the base alias map."""
    merged = {venue_id: list(aliases) for venue_id, aliases in base_aliases.items()}
    seen = {
        (venue_id, pattern, (city_filter or "").casefold())
        for venue_id, aliases in merged.items()
        for pattern, city_filter in aliases
    }

    for venue_id, aliases in _load_alias_supplement(path).items():
        dest = merged.setdefault(venue_id, [])
        for pattern, city_filter in aliases:
            key = (venue_id, pattern, (city_filter or "").casefold())
            if key in seen:
                continue
            dest.append((pattern, city_filter))
            seen.add(key)
    return merged


VENUE_ALIASES = _merge_alias_supplement(VENUE_ALIASES, ALIAS_SUPPLEMENT_PATH)


def find_mentions(text: str, aliases: list[tuple[str, str | None]],
                  primary_cities: str = "") -> list[tuple[int, str]]:
    """Return list of (char_position, matched_string) for each alias found.

    Each alias is a (pattern, city_filter) tuple.
    city_filter: if not None, only match when primary_cities contains that string.
    Patterns containing \\b are used as-is (regex); others are escaped literally.
    """
    mentions = []
    for pattern, city_filter in aliases:
        if city_filter and city_filter not in primary_cities:
            continue
        regex = pattern if r"\b" in pattern else re.escape(pattern)
        for m in re.finditer(regex, text, re.IGNORECASE):
            mentions.append((m.start(), m.group()))
    return sorted(mentions)


def get_context(text: str, pos: int, window: int = 80) -> str:
    start = max(0, pos - window)
    end = min(len(text), pos + window)
    return "..." + text[start:end].replace("\n", " ") + "..."


def collect(venues: list[dict], corpus: list[dict]) -> tuple[list, dict]:
    """Scan corpus and return (anachronisms, mention_data).

    mention_data: venue_id -> list of per-text dicts with accurate counts
                  (no cap; counts reflect all occurrences in each text).
    """
    venue_map = {v["id"]: v for v in venues}
    anachronisms = []
    mention_data = defaultdict(list)  # venue_id -> [{author, title, ...count}]

    for entry in corpus:
        fpath = CORPUS_ROOT / entry["file_path"]
        if not fpath.exists():
            continue

        text = fpath.read_text(encoding="utf-8", errors="replace")
        pub_year = int(entry["pub_year"]) if entry["pub_year"] else None
        setting_start = int(entry["setting_period_start"]) if entry["setting_period_start"] else pub_year
        setting_end = int(entry["setting_period_end"]) if entry["setting_period_end"] else pub_year
        title = entry["title"]
        author = entry["author"]
        primary_cities = entry.get("primary_cities", "")
        text_year = setting_start or pub_year

        for venue_id, aliases in VENUE_ALIASES.items():
            venue = venue_map.get(venue_id)
            if not venue:
                continue

            mentions = find_mentions(text, aliases, primary_cities)
            if not mentions:
                continue

            opened = int(venue["opened"]) if venue["opened"] else 0
            closed = int(venue["closed"]) if venue["closed"] else 9999

            mention_data[venue_id].append({
                "author": author,
                "title": title,
                "pub_year": pub_year,
                "text_year": text_year,
                "count": len(mentions),
            })

            if text_year and text_year < opened:
                anachronisms.append({
                    "author": author,
                    "title": title,
                    "pub_year": pub_year,
                    "text_year": text_year,
                    "venue": venue["name"],
                    "venue_opened": opened,
                    "count": len(mentions),
                    "problem": f"mentioned in text set c.{text_year} but opened {opened}",
                })
            if text_year and text_year > closed:
                anachronisms.append({
                    "author": author,
                    "title": title,
                    "pub_year": pub_year,
                    "text_year": text_year,
                    "venue": venue["name"],
                    "venue_closed": closed,
                    "count": len(mentions),
                    "problem": f"mentioned in text set c.{text_year} but closed {closed}",
                })

    return anachronisms, mention_data


def report(venues: list[dict], corpus: list[dict],
           anachronisms: list, mention_data: dict) -> None:
    venue_map = {v["id"]: v for v in venues}

    print("=" * 70)
    print("VENUE VALIDATION REPORT")
    print("=" * 70)

    # --- Anachronisms ---
    print(f"\nANACHRONISTIC MENTIONS")
    print("-" * 70)
    if anachronisms:
        for a in sorted(anachronisms, key=lambda x: x["text_year"] or 0):
            print(f"  [{a['author']} — {a['title']}, c.{a['text_year']}]")
            print(f"    {a['venue']}: {a['problem']} ({a['count']} mention(s))")
    else:
        print("  None found.")

    # --- Venue mention counts ---
    print(f"\n\nVENUE MENTION COUNTS (fiction corpus)")
    print("-" * 70)
    print(f"  {'Venue':<35} {'Mentions':>8}  {'Authors'}")
    print(f"  {'-'*35} {'-'*8}  {'-'*20}")
    for venue_id, text_list in sorted(mention_data.items(),
                                      key=lambda x: -sum(t["count"] for t in x[1])):
        venue = venue_map[venue_id]
        total = sum(t["count"] for t in text_list)
        authors = sorted(set(t["author"] for t in text_list))
        print(f"  {venue['name']:<35} {total:>8}  {', '.join(authors)}")

    # --- Map layer assignments ---
    print(f"\n\nMAP LAYER ASSIGNMENTS")
    print("-" * 70)
    for entry in sorted(corpus, key=lambda x: int(x["pub_year"]) if x["pub_year"] else 0):
        if not (CORPUS_ROOT / entry["file_path"]).exists():
            continue
        print(f"  {entry['pub_year']}  {entry['author']:<20} {entry['title']:<35} → {entry['map_layer']}")

    print("\n" + "=" * 70)


def export_json(venues: list[dict], mention_data: dict, out_path: Path) -> None:
    """Write geocoded mention data to JSON for the visualisation layer."""
    venue_map = {v["id"]: v for v in venues}
    output = []

    for venue_id, text_list in mention_data.items():
        venue = venue_map[venue_id]
        total = sum(t["count"] for t in text_list)
        output.append({
            "venue_id": venue_id,
            "name": venue["name"],
            "city": venue["city"],
            "lat": float(venue["lat"]),
            "lon": float(venue["lon"]),
            "opened": int(venue["opened"]) if venue["opened"] else None,
            "closed": int(venue["closed"]) if venue["closed"] else None,
            "map_layer": venue["map_layer"],
            "total_mentions": total,
            "mentions": sorted(text_list, key=lambda x: x["text_year"] or 0),
        })

    output.sort(key=lambda x: -x["total_mentions"])

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nExported {len(output)} venues → {out_path}")


def collect_narrative(venues: list[dict], corpus: list[dict]) -> list[dict]:
    """Return per-text, position-ordered venue events for the narrative path map.

    Each entry is one text with its full event sequence sorted by character
    offset (i.e. reading order).  Events store a 0–1 narrative position so
    a slider can filter them without knowing the raw text length.
    """
    venue_map = {v["id"]: v for v in venues}
    results = []

    for entry in corpus:
        fpath = CORPUS_ROOT / entry["file_path"]
        if not fpath.exists():
            continue

        text = fpath.read_text(encoding="utf-8", errors="replace")
        text_len = max(len(text), 1)
        pub_year = int(entry["pub_year"]) if entry["pub_year"] else None
        setting_start = int(entry["setting_period_start"]) if entry["setting_period_start"] else pub_year
        title = entry["title"]
        author = entry["author"]
        primary_cities = entry.get("primary_cities", "")
        text_year = setting_start or pub_year

        events = []
        for venue_id, aliases in VENUE_ALIASES.items():
            venue = venue_map.get(venue_id)
            if not venue:
                continue
            for char_pos, matched in find_mentions(text, aliases, primary_cities):
                start = max(0, char_pos - 90)
                end   = min(text_len, char_pos + 90)
                ctx   = "\u2026" + text[start:end].replace("\n", " ").strip() + "\u2026"
                events.append({
                    "pos":        round(char_pos / text_len, 4),
                    "venue_id":   venue_id,
                    "venue_name": venue["name"],
                    "venue_type": venue.get("map_layer", ""),
                    "lat":        float(venue["lat"]),
                    "lon":        float(venue["lon"]),
                    "matched":    matched,
                    "context":    ctx,
                })

        if not events:
            continue

        events.sort(key=lambda e: e["pos"])
        results.append({
            "author":         author,
            "title":          title,
            "pub_year":       pub_year,
            "text_year":      text_year,
            "primary_cities": primary_cities,
            "event_count":    len(events),
            "events":         events,
        })

    return sorted(results, key=lambda t: t["pub_year"] or 0)


def export_narrative_json(data: list[dict], out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    total_events = sum(t["event_count"] for t in data)
    print(f"\nExported {len(data)} texts, {total_events} events → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", action="store_true",
                        help="Write mentions.json alongside this script")
    parser.add_argument("--narrative", action="store_true",
                        help="Write narrative_mentions.json for the path map")
    args = parser.parse_args()

    here = Path(__file__).parent
    venues = load_venues(here / "venues.csv")
    corpus = load_corpus(here / "corpus_dates.csv")

    anachronisms, mention_data = collect(venues, corpus)
    report(venues, corpus, anachronisms, mention_data)

    if args.export:
        export_json(venues, mention_data, here / "mentions.json")

    if args.narrative:
        narrative = collect_narrative(venues, corpus)
        export_narrative_json(narrative, here / "narrative_mentions.json")
