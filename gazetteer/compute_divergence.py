"""
Compute divergence between passage sensory description and event model baseline.

For each passage that has both a venue_id and an event_id, compares the
passage's valence / modality against the event's load scores and writes
a 'divergence' label:

  "confirms"  — sensory description matches the event's dominant load
                (e.g. unpleasant smell passage at high smell_load event)
  "diverges"  — sensory description contradicts the event's dominant load
                (e.g. pleasant smell at high smell_load event;
                       silence/quiet terms at high noise_load event)
  NULL        — not enough information to decide (event has no strong load,
                passage has no valence, or no event_id)

Modality → load mapping:
  smell   → smell_load
  sound   → noise_load
  sight   → visual_load
  touch / taste → crowd_load (proxy for crowd/physical contact intensity)

Usage:
    python3 gazetteer/compute_divergence.py           # dry run
    python3 gazetteer/compute_divergence.py --write   # write to sensory.db
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sensory_db import init_db, DB_PATH_DEFAULT

# Load threshold: a load score above this is considered "dominant"
HIGH_LOAD_THRESHOLD = 0.65

# Regex patterns that indicate quiet / absence of sound in text
SILENCE_PATTERNS = [
    re.compile(r"\bsilent\b",      re.I),
    re.compile(r"\bsilence\b",     re.I),
    re.compile(r"\bquiet\b",       re.I),
    re.compile(r"\bstill\b",       re.I),
    re.compile(r"\bhush(ed)?\b",   re.I),
    re.compile(r"\bpeaceful\b",    re.I),
    re.compile(r"\bnoiseless\b",   re.I),
    re.compile(r"\bno sound\b",    re.I),
]

# Modality → event load column (DB uses "olfactory", "auditory", "visual", etc.)
MODALITY_LOAD = {
    "olfactory": "smell_load",
    "auditory":  "noise_load",
    "visual":    "visual_load",
    "touch":     "crowd_load",
    "thermal":   "crowd_load",
    "taste":     "crowd_load",
    "crowd":     "crowd_load",
    # legacy aliases
    "smell":  "smell_load",
    "sound":  "noise_load",
    "sight":  "visual_load",
}


def _is_silent(text: str) -> bool:
    return any(p.search(text) for p in SILENCE_PATTERNS)


def _compute_label(
    modality: str,
    valence: str,
    text: str,
    event_loads: dict,
) -> str | None:
    """
    Return 'confirms', 'diverges', or None for a single passage.

    Parameters
    ----------
    modality    : passage modality (smell, sound, sight, touch, taste)
    valence     : 'pleasant', 'unpleasant', or 'neutral'
    text        : raw passage text
    event_loads : dict with keys smell_load, noise_load, visual_load, crowd_load
    """
    load_col = MODALITY_LOAD.get(modality)
    if load_col is None:
        return None

    load = event_loads.get(load_col) or 0.0
    if load < HIGH_LOAD_THRESHOLD:
        return None  # event load too weak to draw a conclusion

    if modality in ("auditory", "sound"):
        if _is_silent(text):
            return "diverges"
        if valence == "unpleasant":
            return "confirms"
        return None

    if valence == "unpleasant":
        return "confirms"
    if valence == "pleasant":
        return "diverges"

    return None


def compute(
    write: bool = False,
    db_path: Path = DB_PATH_DEFAULT,
) -> dict:
    """
    Compute divergence labels for all passages with both venue_id and event_id.
    Existing non-NULL divergence values are preserved (idempotent).
    """
    conn = init_db(db_path)

    # Load event load scores indexed by event_id
    event_rows = conn.execute(
        "SELECT event_id, smell_load, noise_load, visual_load, crowd_load "
        "FROM events"
    ).fetchall()
    events = {
        r[0]: {
            "smell_load":  r[1] or 0.0,
            "noise_load":  r[2] or 0.0,
            "visual_load": r[3] or 0.0,
            "crowd_load":  r[4] or 0.0,
        }
        for r in event_rows
    }

    # Fetch passages with event_id set and divergence not yet computed
    passages = conn.execute(
        """
        SELECT id, modality, valence, text, event_id
        FROM   sensory_evidence
        WHERE  event_id IS NOT NULL
        AND    venue_id IS NOT NULL
        AND    divergence IS NULL
        """
    ).fetchall()

    confirms = 0
    diverges = 0
    nulls    = 0

    updates = []
    for row_id, modality, valence, text, event_id in passages:
        loads = events.get(event_id)
        if loads is None:
            nulls += 1
            continue

        label = _compute_label(modality, valence or "", text or "", loads)
        if label == "confirms":
            confirms += 1
        elif label == "diverges":
            diverges += 1
        else:
            nulls += 1

        if label is not None:
            updates.append((label, row_id))

    if write and updates:
        conn.executemany(
            "UPDATE sensory_evidence SET divergence = ? WHERE id = ?",
            updates,
        )
        conn.commit()

    conn.close()
    return {
        "passages_examined": len(passages),
        "confirms":  confirms,
        "diverges":  diverges,
        "undecided": nulls,
        "updated":   len(updates),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--write", action="store_true",
                        help="Write divergence labels to the database")
    args = parser.parse_args()

    stats = compute(write=args.write)
    print(f"Passages examined  : {stats['passages_examined']}")
    print(f"  confirms         : {stats['confirms']}")
    print(f"  diverges         : {stats['diverges']}")
    print(f"  undecided        : {stats['undecided']}")
    if args.write:
        print(f"Labels written     : {stats['updated']}")
        print("✓ Divergence labels written to sensory.db")
    else:
        print(f"Labels to write    : {stats['updated']}")
        print("(Dry run — pass --write to commit changes)")


if __name__ == "__main__":
    main()
