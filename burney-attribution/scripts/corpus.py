#!/usr/bin/env python3
"""
Shared corpus loading and leave-one-work-out (LOWO) splitting.

All evaluation scripts import from here to ensure consistent splits.
"""

import csv
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Work:
    author: str
    title: str
    year: int
    genre: str
    file_paths: list = field(default_factory=list)
    notes: str = ""

    @property
    def key(self):
        return (self.author, self.title)


def load_metadata(metadata_file: Path) -> list[Work]:
    """Load metadata_v2.csv and group multi-volume works."""
    works_dict = {}

    with open(metadata_file) as f:
        for row in csv.DictReader(f):
            key = (row["author"], row["title"])
            if key not in works_dict:
                works_dict[key] = Work(
                    author=row["author"],
                    title=row["title"],
                    year=int(row["year"]),
                    genre=row["genre"],
                    notes=row.get("notes", ""),
                )
            works_dict[key].file_paths.append(row["file_path"])

    return list(works_dict.values())


def load_work_text(work: Work, processed_dir: Path) -> str:
    """Load and concatenate all volumes of a work into one string."""
    parts = []
    for fp in sorted(work.file_paths):
        full_path = processed_dir / fp
        with open(full_path, "r", encoding="utf-8") as f:
            parts.append(f.read())
    return "\n\n".join(parts)


def get_project_paths():
    """Return standard project directory paths."""
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    return {
        "project": project_dir,
        "processed": project_dir / "data" / "processed",
        "metadata": project_dir / "data" / "metadata_v2.csv",
        "results": project_dir / "results",
        "models": project_dir / "models",
        "outputs": project_dir / "outputs",
    }


def lowo_splits(works: list[Work]):
    """
    Generate leave-one-work-out cross-validation folds.

    Only yields folds where the held-out author has at least one other
    work in the training set (otherwise there's nothing to learn from).

    Yields:
        (fold_name, held_out_work, train_works)
    """
    author_works = {}
    for w in works:
        author_works.setdefault(w.author, []).append(w)

    for work in works:
        if len(author_works[work.author]) < 2:
            continue

        train = [w for w in works if w is not work]
        fold_name = f"{work.author}_{work.title}".replace(" ", "_")
        yield fold_name, work, train


def holdout_split(works: list[Work]):
    """
    Single train/test split: for each author with 2+ works, hold out
    one work for testing. Authors with only one work go to train only.

    Returns:
        (train_works, test_works)
    """
    author_works = {}
    for w in works:
        author_works.setdefault(w.author, []).append(w)

    train, test = [], []
    for author, aw in sorted(author_works.items()):
        if len(aw) == 1:
            train.append(aw[0])
        else:
            sorted_aw = sorted(aw, key=lambda w: w.year)
            test.append(sorted_aw[-1])
            train.extend(sorted_aw[:-1])

    return train, test
