#!/usr/bin/env python3
"""Populate bookseller_publications.csv linking booksellers to works in bibliography files.

Based on known imprint data from Harris (Hume biography), general knowledge of
18c publishing, and title matching against bibliography CSVs.
"""

import csv
from pathlib import Path

OUT_PATH = Path(__file__).parent / "bookseller_publications.csv"

# Known bookseller-publication links from imprint data and general knowledge.
# Each entry: (bookseller_id, role, author_substring, title_substring, pub_year or None)
# These will be matched against all bibliography files.

KNOWN_LINKS = [
    # === From Harris imprints (Hume's works) ===
    ("BS_NOON", "publisher", "Hume", "Treatise of Human Nature, Book 1", 1739),
    ("BS_LONGMAN", "publisher", "Hume", "Treatise of Human Nature, Books 2", 1740),
    ("BS_BORBET", "publisher", "Hume", "Abstract of a Book", 1740),
    ("BS_MILLAR", "publisher", "Hume", "Three Essays, Moral and Political", 1748),
    ("BS_MILLAR", "publisher", "Hume", "Philosophical Essays Concerning Human Understanding", 1748),
    ("BS_MILLAR", "publisher", "Hume", "Enquiry Concerning the Principles of Morals", 1751),
    ("BS_MILLAR", "publisher", "Hume", "Four Dissertations", 1757),
    ("BS_MILLAR", "publisher", "Hume", "History of Great Britain, Vol 2", 1757),
    ("BS_MILLAR", "publisher", "Hume", "History of England under the House of Tudor", 1759),
    ("BS_CADELL", "publisher", "Hume", "My Own Life", 1777),
    ("BS_COOPER", "publisher", "Hume", "True Account of the Behaviour", 1748),

    # === Millar's other major publications ===
    ("BS_MILLAR", "publisher", "Fielding", "Tom Jones", None),
    ("BS_MILLAR", "publisher", "Fielding", "Joseph Andrews", None),
    ("BS_MILLAR", "publisher", "Fielding", "Amelia", None),
    ("BS_MILLAR", "publisher", "Fielding", "Enquiry into the Causes of the Late Increase of Robbers", 1751),
    ("BS_MILLAR", "publisher", "Johnson", "Dictionary of the English Language", 1755),
    ("BS_MILLAR", "publisher", "Smith, Adam", "Theory of Moral Sentiments", 1759),
    ("BS_MILLAR", "publisher", "Robertson", "History of Scotland", 1759),
    ("BS_MILLAR", "publisher", "Thomson", None, None),  # James Thomson's Seasons
    ("BS_MILLAR", "publisher", "Fielding, Sarah", "Adventures of David Simple", 1744),
    ("BS_MILLAR", "publisher", "Fielding, Sarah", "Governess", 1749),

    # === Strahan as printer (often alongside Millar as publisher) ===
    ("BS_STRAHAN", "printer", "Johnson", "Dictionary of the English Language", 1755),
    ("BS_STRAHAN", "printer", "Gibbon", "Decline and Fall", None),
    ("BS_STRAHAN", "printer", "Smith, Adam", "Wealth of Nations", 1776),
    ("BS_STRAHAN", "printer", "Robertson", "History of Scotland", 1759),

    # === Cadell (successor to Millar) ===
    ("BS_CADELL", "publisher", "Gibbon", "Decline and Fall", None),
    ("BS_CADELL", "publisher", "Smith, Adam", "Wealth of Nations", 1776),
    ("BS_CADELL", "publisher", "Robertson", "History of the Reign of the Emperor Charles V", 1769),
    ("BS_CADELL", "publisher", "Burke", "Philosophical Enquiry", 1757),
    ("BS_CADELL", "publisher", "Johnson", "Lives of the Most Eminent English Poets", 1781),
    ("BS_CADELL", "publisher", "Burney", "Evelina", 1778),
    ("BS_CADELL", "publisher", "Burney", "Cecilia", 1782),
    ("BS_CADELL", "publisher", "Burney", "Camilla", 1796),
    ("BS_CADELL", "publisher", "Radcliffe", "Mysteries of Udolpho", None),
    ("BS_CADELL", "publisher", "Smith, Charlotte", "Emmeline", None),
    ("BS_CADELL", "publisher", "Inchbald", "Simple Story", None),
    ("BS_CADELL", "publisher", "More, Hannah", "Strictures", None),
    ("BS_CADELL", "publisher", "Chapone", "Letters on the Improvement", None),

    # === Dodsley ===
    ("BS_DODSLEY", "publisher", "Burke", "Philosophical Enquiry", 1757),
    ("BS_DODSLEY", "publisher", "Johnson", "Vanity of Human Wishes", None),
    ("BS_DODSLEY", "publisher", "Johnson", "Rasselas", None),
    ("BS_DODSLEY", "publisher", "Sterne", None, None),
    ("BS_DODSLEY", "publisher", "Goldsmith", "Vicar of Wakefield", None),
    ("BS_DODSLEY", "publisher", "Goldsmith", "Deserted Village", None),
    ("BS_DODSLEY", "publisher", "Shaftesbury", "Characteristicks", None),
    ("BS_DODSLEY", "publisher", "Young", None, None),

    # === Tonson Sr — major publisher of Restoration/Augustan period ===
    ("BS_TONSON_SR", "publisher", "Dryden", None, None),
    ("BS_TONSON_SR", "publisher", "Milton", "Paradise Lost", None),
    ("BS_TONSON_SR", "publisher", "Locke", "Essay Concerning Human Understanding", None),
    ("BS_TONSON_SR", "publisher", "Addison", None, None),
    ("BS_TONSON_SR", "publisher", "Manley", "New Atalantis", None),
    ("BS_TONSON_SR", "publisher", "Shaftesbury", "Characteristicks", 1711),

    # === Tonson Jr ===
    ("BS_TONSON_JR", "publisher", "Addison", "Spectator", None),
    ("BS_TONSON_JR", "publisher", "Pope", "Essay on Man", None),
    ("BS_TONSON_JR", "publisher", "Steele", None, None),

    # === Rivington ===
    ("BS_RIVINGTON", "publisher", "Richardson", "Pamela", None),
    ("BS_RIVINGTON", "publisher", "Richardson", "Clarissa", None),

    # === Longman ===
    ("BS_LONGMAN", "publisher", "Hume", "Treatise of Human Nature, Books 2", 1740),
    ("BS_LONGMAN", "publisher", "Fordyce", "Sermons to Young Women", None),

    # === Murray I ===
    ("BS_MURRAY_I", "publisher", "Priestley", None, None),
    ("BS_MURRAY_I", "publisher", "Lennox", "Female Quixote", None),

    # === Edinburgh booksellers (linking to Harris entries) ===
    ("BS_KINCAID", "publisher", "Hume", "Essays, Moral and Political, Volume the First", 1741),
    ("BS_KINCAID", "publisher", "Hume", "Essays, Moral and Political, Volume the Second", 1742),
    ("BS_KINCAID", "publisher", "Hume", "Political Discourses", 1752),
    ("BS_DONALDSON", "publisher", "Hume", "Political Discourses", 1752),
    ("BS_FLEMING_ALISON", "printer", "Hume", "Essays, Moral and Political, Volume the First", 1741),
    ("BS_FLEMING_ALISON", "printer", "Hume", "Essays, Moral and Political, Volume the Second", 1742),

    # === Colquhoun / Howard — institutional works ===
    ("BS_CADELL", "publisher", "Colquhoun", "Police of the Metropolis", None),

    # === Cooper ===
    ("BS_COOPER", "publisher", "Hume", "True Account", 1748),
]

# Load all bibliography files
BIB_FILES = [
    "harris_hume_bibliography.csv",
    "white_bibliography_batch1.csv",
    "white_bibliography_batch2.csv",
    "almeroth_williams_bibliography.csv",
    "perry_novel_relations_bibliography.csv",
]

bib_dir = Path(__file__).parent
all_bib_entries = []
for bib_file in BIB_FILES:
    path = bib_dir / bib_file
    if not path.exists():
        continue
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["_file"] = bib_file
            all_bib_entries.append(row)

# Also check sources_catalog
sources_path = bib_dir / "sources_catalog.csv"
if sources_path.exists():
    with open(sources_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            all_bib_entries.append({
                "author": row.get("author", ""),
                "title": row.get("title", ""),
                "date": row.get("pub_year", ""),
                "cited_by": "sources_catalog",
                "_file": "sources_catalog.csv",
            })

print(f"Loaded {len(all_bib_entries)} bibliography entries from {len(BIB_FILES) + 1} files")


def match_bib(author_sub, title_sub, year, active_max=None):
    """Find matching bibliography entries."""
    matches = []
    for entry in all_bib_entries:
        e_author = entry.get("author", "")
        e_title = entry.get("title", "")
        e_date = entry.get("date", "")

        # Author match (substring, case-insensitive)
        if author_sub and author_sub.lower() not in e_author.lower():
            continue

        # Title match (substring, case-insensitive)
        if title_sub and title_sub.lower() not in e_title.lower():
            continue

        # Year match (exact, if specified)
        if year and str(year) != str(e_date).strip():
            continue

        # Filter out publications after bookseller's active period
        if active_max and e_date:
            try:
                pub_year = int(str(e_date).strip()[:4])
                if pub_year > active_max + 5:  # small grace period
                    continue
            except ValueError:
                pass

        matches.append(entry)
    return matches


# Load bookseller active dates for filtering
bs_active = {}
bs_path = bib_dir / "booksellers.csv"
if bs_path.exists():
    with open(bs_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                bs_active[row["bookseller_id"]] = int(row["active_max"])
            except (ValueError, KeyError):
                pass

# Build publication links
publications = []
seen = set()  # deduplicate

for bs_id, role, author_sub, title_sub, year in KNOWN_LINKS:
    active_max = bs_active.get(bs_id)
    matches = match_bib(author_sub, title_sub, year, active_max)
    for m in matches:
        key = (bs_id, role, m["_file"], m["author"], m["title"])
        if key in seen:
            continue
        seen.add(key)
        publications.append({
            "bookseller_id": bs_id,
            "role": role,
            "bibliography_file": m["_file"],
            "author": m["author"],
            "title": m["title"],
            "pub_year": m.get("date", ""),
        })

# Sort by bookseller, then year
publications.sort(key=lambda x: (x["bookseller_id"], x.get("pub_year", "")))

# Write
with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "bookseller_id", "role", "bibliography_file", "author", "title", "pub_year",
    ])
    writer.writeheader()
    writer.writerows(publications)

print(f"Wrote {len(publications)} publication links to {OUT_PATH}")

# Summary
from collections import Counter
by_bs = Counter(p["bookseller_id"] for p in publications)
print("\nLinks per bookseller:")
for bs_id, count in by_bs.most_common():
    print(f"  {bs_id}: {count}")

by_file = Counter(p["bibliography_file"] for p in publications)
print("\nLinks per bibliography file:")
for f, count in by_file.most_common():
    print(f"  {f}: {count}")
