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

sys.path.insert(0, str(Path(__file__).parent))
from validate_venues import VENUE_ALIASES

WINDOW_WORDS = 500

_alias_map_cache: dict = {}


def _build_alias_map(
    venues: list[dict],
    primary_cities: str = "",
) -> list[tuple[re.Pattern, dict]]:
    """Return list of (compiled_pattern, venue_dict) for all aliases.

    Aliases with a city_filter are only included when primary_cities
    contains that filter string (case-insensitive), matching the
    behaviour of validate_venues.find_mentions.
    """
    result = []
    for venue in venues:
        vid = venue["id"]
        if vid not in VENUE_ALIASES:
            continue
        for alias, city_filter in VENUE_ALIASES[vid]:
            if city_filter and city_filter.lower() not in primary_cities.lower():
                continue
            # Use same regex-detection logic as validate_venues.find_mentions
            if r"\b" in alias:
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
    primary_cities: str = "",
    anchor_pos: int | None = None,
) -> dict | None:
    """
    Return dict with venue_id, venue_name, lat, lon if a venue alias
    is found within window_words of the passage in full_text.
    Returns None if no venue found or passage not located in text.

    primary_cities: used to apply city_filter on aliases (e.g. "London",
    "Bath"). Pass the text's primary_cities field from corpus_dates.csv.
    anchor_pos: optional absolute character offset to use directly instead of
    re-locating the passage substring inside full_text.
    """
    cache_key = (id(venues), primary_cities)
    if cache_key not in _alias_map_cache:
        _alias_map_cache[cache_key] = _build_alias_map(venues, primary_cities)
    alias_map = _alias_map_cache[cache_key]

    idx = anchor_pos if anchor_pos is not None else full_text.find(passage)
    if idx == -1:
        anchor = passage[:60]
        idx = full_text.find(anchor)
        if idx == -1:
            return None

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
