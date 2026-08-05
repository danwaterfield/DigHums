#!/usr/bin/env python3
"""Rebuild every published page and resync the landing-page figures.

The public site drifted once because the HTML tools and the counts advertised
on index.html were updated at different times: the gazetteer grew from 99 to
167 venues, but the built pages were committed from an older snapshot and the
landing page quoted the new CSVs. This script makes that failure mode
impossible by doing both in one pass.

    python3 build_all.py              # rebuild everything, resync index.html
    python3 build_all.py --core       # only the pages linked from index.html
    python3 build_all.py --check      # rebuild nothing; fail if index.html is stale

It rebuilds HTML from the existing database and CSVs. It does not re-run the
extraction pipeline (extract_sensory.py, extract_fiction.py, extract_events.py)
— run those first when the underlying evidence has changed.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
GAZETTEER = ROOT / "gazetteer"
INDEX = ROOT / "index.html"

sys.path.insert(0, str(GAZETTEER))

# Pages linked from index.html. A failure here breaks the public site.
CORE_BUILDERS = [
    "build_venue_explorer",
    "build_narrative_map",
    "build_sensory_time_map",
    "build_comparison",
    "build_sensory_timeline",
    "build_full_network",
    "build_correspondent_network",
    "build_tour_map",
]

# Published but not linked from the landing page.
EXTRA_BUILDERS = [
    "build_map",
    "build_gordon_riots",
    "build_gordon_riots_pulse",
    "build_narrative_pace",
]

STAT_RE = re.compile(r'(<span[^>]*\bdata-stat="([^"]+)"[^>]*>)(.*?)(</span>)', re.S)


def run_builder(name: str) -> tuple[str, bool, str]:
    """Run one builder script; return (name, ok, last line of output)."""
    proc = subprocess.run(
        [sys.executable, str(GAZETTEER / f"{name}.py")],
        capture_output=True, text=True,
    )
    ok = proc.returncode == 0
    stream = proc.stdout if ok else (proc.stderr or proc.stdout)
    lines = [ln for ln in stream.strip().splitlines() if ln.strip()]
    return name, ok, (lines[-1] if lines else "")


def collect_stats() -> dict[str, str]:
    import site_stats
    stats = site_stats.flat_stats()
    stats["build.date"] = _dt.date.today().isoformat()
    return stats


def sync_index(stats: dict[str, str], *, write: bool) -> list[str]:
    """Inject stats into index.html. Returns a list of stale/unknown keys."""
    html = INDEX.read_text(encoding="utf-8")
    problems: list[str] = []
    changed: list[str] = []

    def replace(match: re.Match) -> str:
        open_tag, key, current, close_tag = match.groups()
        if key not in stats:
            problems.append(f"unknown stat key: {key}")
            return match.group(0)
        new = stats[key]
        if current != new:
            changed.append(f"{key}: {current!r} -> {new!r}")
        return f"{open_tag}{new}{close_tag}"

    updated = STAT_RE.sub(replace, html)

    if write and updated != html:
        INDEX.write_text(updated, encoding="utf-8")

    for line in changed:
        problems.append(f"stale figure -> {line}" if not write else f"updated {line}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", action="store_true",
                        help="build only the pages linked from index.html")
    parser.add_argument("--check", action="store_true",
                        help="do not build or write; exit non-zero if index.html is stale")
    args = parser.parse_args()

    if args.check:
        problems = sync_index(collect_stats(), write=False)
        if problems:
            print("index.html is out of sync with the data:")
            for p in problems:
                print(f"  {p}")
            return 1
        print("index.html figures match the data.")
        return 0

    builders = CORE_BUILDERS + ([] if args.core else EXTRA_BUILDERS)
    failures: list[str] = []

    print(f"Building {len(builders)} pages\n")
    for name in builders:
        label, ok, detail = run_builder(name)
        mark = "ok  " if ok else "FAIL"
        print(f"  [{mark}] {label:32} {detail}")
        if not ok:
            failures.append(label)

    print()
    for line in sync_index(collect_stats(), write=True):
        print(f"  {line}")

    if failures:
        core_failed = [f for f in failures if f in CORE_BUILDERS]
        print(f"\n{len(failures)} builder(s) failed: {', '.join(failures)}")
        if core_failed:
            print("These pages are linked from index.html — the site is incomplete.")
            return 1
        print("These pages are not linked from index.html.")
    else:
        print("\nAll pages rebuilt; index.html figures match the data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
