"""
Lexicon-based sensory term detection for Pass 1 extraction.

Returns list of (matched_term, modality) tuples for a text fragment.
Only matches whole words (word-boundary anchored).
"""

import re

# Each entry: (pattern, modality)
# All patterns are word-boundary anchored (the r"\b" prefix).
# Period-specific terms drawn from OED citations and corpus reading.
MODALITY_PATTERNS: list[tuple[str, str]] = [
    # ── AUDITORY ──────────────────────────────────────────────────────────
    (r"\bdin\b",          "auditory"),
    (r"\bclatter\b",      "auditory"),
    (r"\bclattering\b",   "auditory"),
    (r"\bbustle\b",       "auditory"),
    (r"\bhubbub\b",       "auditory"),
    (r"\bhuzza\b",        "auditory"),
    (r"\bnoisy\b",        "auditory"),
    (r"\bnoise\b",        "auditory"),
    (r"\bsilence\b",      "auditory"),
    (r"\bstillness\b",    "auditory"),
    (r"\bcry\b",          "auditory"),
    (r"\bcries\b",        "auditory"),
    (r"\brumble\b",       "auditory"),
    (r"\brumbling\b",     "auditory"),
    (r"\btumult\b",       "auditory"),
    (r"\buproar\b",       "auditory"),
    (r"\bdiscord\b",      "auditory"),
    (r"\btolling\b",      "auditory"),
    (r"\bclamour\b",      "auditory"),
    (r"\bclamor\b",       "auditory"),
    (r"\bshriek\b",       "auditory"),
    (r"\bshrieking\b",    "auditory"),
    (r"\bstreet-cries\b", "auditory"),
    (r"\bstreet cries\b", "auditory"),
    # ── OLFACTORY ─────────────────────────────────────────────────────────
    (r"\bstench\b",       "olfactory"),
    (r"\beffluvia\b",     "olfactory"),
    (r"\beffluvium\b",    "olfactory"),
    (r"\bperfume\b",      "olfactory"),
    (r"\bperfumed\b",     "olfactory"),
    (r"\breek\b",         "olfactory"),
    (r"\breeking\b",      "olfactory"),
    (r"\bvapour\b",       "olfactory"),
    (r"\bvapors\b",       "olfactory"),
    (r"\bodour\b",        "olfactory"),
    (r"\bodor\b",         "olfactory"),
    (r"\bfetid\b",        "olfactory"),
    (r"\bfragrant\b",     "olfactory"),
    (r"\bfragrance\b",    "olfactory"),
    (r"\bputrid\b",       "olfactory"),
    (r"\bsmoke\b",        "olfactory"),
    (r"\bsmoky\b",        "olfactory"),
    (r"\bmiasma\b",       "olfactory"),
    (r"\bkennel\b",       "olfactory"),  # street gutter/drain, 18c usage
    # ── VISUAL ────────────────────────────────────────────────────────────
    (r"\bnarrow\b",       "visual"),
    (r"\blofty\b",        "visual"),
    (r"\bglare\b",        "visual"),
    (r"\bgloom\b",        "visual"),
    (r"\bgloomy\b",       "visual"),
    (r"\bdazzling\b",     "visual"),
    (r"\bdazzle\b",       "visual"),
    (r"\bmurky\b",        "visual"),
    (r"\bthronged\b",     "visual"),
    (r"\billuminated\b",  "visual"),
    (r"\billumination\b", "visual"),
    (r"\blamplight\b",    "visual"),
    (r"\btorch-light\b",  "visual"),
    (r"\bdark\b",         "visual"),
    (r"\bdarkness\b",     "visual"),
    (r"\bdirty\b",        "visual"),
    (r"\bfilthy\b",       "visual"),
    (r"\bmuddy\b",        "visual"),
    (r"\bmud\b",          "visual"),
    # ── THERMAL ───────────────────────────────────────────────────────────
    (r"\bsultry\b",       "thermal"),
    (r"\bdamp\b",         "thermal"),
    (r"\braw\b",          "thermal"),
    (r"\bfog\b",          "thermal"),
    (r"\bfoggy\b",        "thermal"),
    (r"\bmist\b",         "thermal"),
    (r"\bmisty\b",        "thermal"),
    (r"\bfrost\b",        "thermal"),
    (r"\bfrosty\b",       "thermal"),
    (r"\bclose air\b",    "thermal"),  # "close air" = stuffy; phrase-level avoids false positives
    (r"\bstifling\b",     "thermal"),
    (r"\bchilly\b",       "thermal"),
    # ── CROWD / DENSITY ───────────────────────────────────────────────────
    (r"\bpress\b",        "crowd"),
    (r"\bmob\b",          "crowd"),
    (r"\bjostle\b",       "crowd"),
    (r"\bjostled\b",      "crowd"),
    (r"\bthrong\b",       "crowd"),
    (r"\bcrowd\b",        "crowd"),
    (r"\bcrowded\b",      "crowd"),
    (r"\bcramm'd\b",      "crowd"),
    (r"\bcrammed\b",      "crowd"),
    (r"\bdeserted\b",     "crowd"),
    (r"\bempty\b",        "crowd"),
    (r"\bsparsely\b",     "crowd"),
]

# Pre-compile for speed
_COMPILED = [(re.compile(pat, re.IGNORECASE), mod)
             for pat, mod in MODALITY_PATTERNS]


def tag_modalities(text: str) -> list[tuple[str, str]]:
    """
    Return list of (matched_term, modality) for all sensory matches in text.
    Deduplicates by (term.lower(), modality).
    """
    seen: set[tuple[str, str]] = set()
    results: list[tuple[str, str]] = []
    for pattern, modality in _COMPILED:
        for m in pattern.finditer(text):
            key = (m.group().lower(), modality)
            if key not in seen:
                seen.add(key)
                results.append((m.group(), modality))
    return results
