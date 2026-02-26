#!/usr/bin/env python3
"""
Ingest the full 13-author corpus into data/raw/ and rebuild metadata.

Maps corpus root directory names to the lowercase author keys used
throughout the pipeline.
"""

import shutil
import csv
from pathlib import Path

CORPUS_MAP = {
    "JaneAusten": "austen",
    "WilliamBeckford": "beckford",
    "FrancesBurney": "burney",
    "MariaEdgeworth": "edgeworth",
    "HenryFielding": "fielding",
    "ElizaHaywood": "haywood",
    "MGLewis": "lewis",
    "AnnRadcliffe": "radcliffe",
    "ClaraReeve": "reeve",
    "SamuelRichardson": "richardson",
    "CharlotteSmith": "smith",
    "TobiasSmollett": "smollett",
    "HoraceWalpole": "walpole",
}


def main():
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    corpus_root = project_dir.parent
    raw_dir = project_dir / "data" / "raw"

    total_copied = 0

    for corpus_dirname, author_key in sorted(CORPUS_MAP.items()):
        src_dir = corpus_root / corpus_dirname
        dst_dir = raw_dir / author_key

        if not src_dir.exists():
            print(f"  SKIP {corpus_dirname} (not found)")
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)

        txt_files = sorted(src_dir.glob("*.txt"))
        if not txt_files:
            print(f"  SKIP {corpus_dirname} (no .txt files)")
            continue

        for src_file in txt_files:
            dst_file = dst_dir / src_file.name
            shutil.copy2(src_file, dst_file)
            total_copied += 1

        print(f"  {author_key:12} <- {corpus_dirname:20} ({len(txt_files)} files)")

    print(f"\nCopied {total_copied} files into {raw_dir}")

    # Verify against metadata_v2
    metadata_file = project_dir / "data" / "metadata_v2.csv"
    with open(metadata_file) as f:
        reader = csv.DictReader(f)
        missing = []
        for row in reader:
            fp = raw_dir / row["file_path"]
            if not fp.exists():
                missing.append(row["file_path"])

    if missing:
        print(f"\nWARNING: {len(missing)} files in metadata_v2.csv not found in raw/:")
        for m in missing:
            print(f"  {m}")
    else:
        print(f"\nAll metadata_v2.csv entries verified in raw/")


if __name__ == "__main__":
    main()
