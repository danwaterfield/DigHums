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


def _build_alias_map(venues: list[dict]) -> list[tuple[re.Pattern, dict]]:
    """Return list of (compiled_pattern, venue_dict) for all aliases."""
    result = []
    for venue in venues:
        vid = venue["id"]
        if vid not in VENUE_ALIASES:
            continue
        for alias, _city_filter in VENUE_ALIASES[vid]:
            if alias.startswith(r"\b") or alias.startswith("("):
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
) -> dict | None:
    """
    Return dict with venue_id, venue_name, lat, lon if a venue alias
    is found within window_words of the passage in full_text.
    Returns None if no venue found or passage not located in text.
    """
    alias_map = _build_alias_map(venues)

    idx = full_text.find(passage)
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
