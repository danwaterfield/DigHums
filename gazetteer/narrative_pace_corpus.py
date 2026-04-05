"""narrative_pace_corpus.py — corpus loading module for narrative pace analysis.

Reads novels from the processed corpus directory, concatenates multi-volume
works, and records volume boundary positions normalised to [0, 1].
"""

from __future__ import annotations

import csv
import re
from itertools import groupby
from pathlib import Path
from typing import List, Tuple, Dict, Any

_DEFAULT_METADATA = Path(__file__).parent.parent / "burney-attribution" / "data" / "metadata_v2.csv"
_DEFAULT_ROOT = Path(__file__).parent.parent / "burney-attribution" / "data" / "processed"


def concatenate_volumes(texts: List[str]) -> Tuple[str, List[float]]:
    """Concatenate volume texts.

    Returns
    -------
    combined_text : str
        All volumes joined with a single newline separator.
    boundary_positions : list[float]
        Normalised 0.0–1.0 positions where volume breaks occur (one fewer
        than the number of volumes).  Empty for single-volume works.
    """
    if not texts:
        return "", []

    if len(texts) == 1:
        return texts[0], []

    combined = "\n".join(texts)
    total = len(combined)

    boundaries: List[float] = []
    offset = 0
    for vol in texts[:-1]:
        offset += len(vol) + 1  # +1 for the "\n" separator
        if total > 0:
            boundaries.append(offset / total)

    return combined, boundaries


def _title_to_snake(title: str) -> str:
    """Convert a title to lowercase snake_case for use in novel IDs."""
    slug = title.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug


def load_novel_texts(rows: List[Dict[str, Any]], root: Path) -> List[Dict[str, Any]]:
    """Group metadata rows by (author, title), load and concatenate volumes.

    Parameters
    ----------
    rows : list[dict]
        Dicts with keys matching the metadata CSV columns.
    root : Path
        Root directory for resolving ``file_path`` values.

    Returns
    -------
    list[dict] sorted by (year, title), each with keys:
        id, author, title, year, genre, text, word_count, volume_boundaries.
    """
    # Group by (author, title) preserving natural CSV order for volume sorting.
    def group_key(row: Dict[str, Any]) -> Tuple[str, str]:
        return (row["author"], row["title"])

    # Sort by author+title first so groupby works, but we need to handle
    # volumes within a group in volume order.
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = group_key(row)
        grouped.setdefault(key, []).append(row)

    novels: List[Dict[str, Any]] = []
    for (author, title), group_rows in grouped.items():
        # Sort volumes numerically (empty string → treat as 0 so single-vol comes first)
        def vol_sort_key(r: Dict[str, Any]) -> int:
            v = r.get("volume", "")
            try:
                return int(v) if v else 0
            except ValueError:
                return 0

        group_rows.sort(key=vol_sort_key)

        texts: List[str] = []
        for r in group_rows:
            fp = root / r["file_path"]
            texts.append(fp.read_text(encoding="utf-8"))

        combined, boundaries = concatenate_volumes(texts)
        word_count = len(combined.split())

        # Use metadata from the first row (all rows in group share author/title/year/genre)
        first = group_rows[0]
        novel_id = f"{author}_{_title_to_snake(title)}_{first['year']}"

        novels.append(
            {
                "id": novel_id,
                "author": author,
                "title": title,
                "year": first["year"],
                "genre": first["genre"],
                "text": combined,
                "word_count": word_count,
                "volume_boundaries": boundaries,
            }
        )

    # Sort output by (year, title)
    novels.sort(key=lambda n: (int(n["year"]), n["title"]))
    return novels


def load_corpus(
    metadata_path: Path | str = _DEFAULT_METADATA,
    root: Path | str = _DEFAULT_ROOT,
) -> List[Dict[str, Any]]:
    """Load the full corpus from the canonical metadata CSV.

    Parameters
    ----------
    metadata_path : path-like
        Path to the metadata CSV (default: burney-attribution/data/metadata_v2.csv).
    root : path-like
        Root directory for resolving file_path values
        (default: burney-attribution/data/processed/).

    Returns
    -------
    list[dict] — same structure as :func:`load_novel_texts`.
    """
    metadata_path = Path(metadata_path)
    root = Path(root)

    rows: List[Dict[str, Any]] = []
    with metadata_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(dict(row))

    return load_novel_texts(rows, root)
