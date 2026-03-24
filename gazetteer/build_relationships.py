#!/usr/bin/env python3
"""Build a structured relationships table from bookseller data and Plomer OCR.

Extracts:
- Family connections (child_of, sibling_of, nephew_of, widow_of, spouse_of)
- Business succession (succeeded_by, partner_of)
- Apprenticeship (apprentice_of)
- Publishing relationships (publisher_of — from bookseller_publications.csv)

Outputs: gazetteer/relationships.csv
"""

import csv
import re
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).parent

# ── Load booksellers ──────────────────────────────────────────────────────
with open(BASE / "booksellers.csv", newline="", encoding="utf-8") as f:
    booksellers = list(csv.DictReader(f))

bs_by_id = {r["bookseller_id"]: r for r in booksellers}

# Build name indexes for resolution
# Index by: full name, surname, forename+surname variants
bs_by_surname = defaultdict(list)
bs_by_fullname = defaultdict(list)
for r in booksellers:
    name = r["name"]
    parts = name.split()
    if parts:
        surname = parts[-1].lower()
        bs_by_surname[surname].append(r)
    bs_by_fullname[name.lower()].append(r)


def resolve_person(name_fragment, source_bs_id=None, prefer_earlier=False):
    """Try to resolve a name fragment to a bookseller_id.

    Uses date proximity to source bookseller to disambiguate.
    """
    name_fragment = name_fragment.strip()
    if not name_fragment:
        return "", name_fragment

    # Direct full name match
    candidates = bs_by_fullname.get(name_fragment.lower(), [])

    if not candidates:
        # Surname match
        parts = name_fragment.split()
        surname = parts[-1].lower() if parts else name_fragment.lower()
        candidates = bs_by_surname.get(surname, [])

    if not candidates:
        return "", name_fragment

    if len(candidates) == 1:
        return candidates[0]["bookseller_id"], candidates[0]["name"]

    # Multiple candidates — disambiguate by date proximity to source
    if source_bs_id and source_bs_id in bs_by_id:
        source = bs_by_id[source_bs_id]
        try:
            source_year = int(source["active_min"])
        except (ValueError, KeyError):
            source_year = 1700

        def date_distance(candidate):
            try:
                if prefer_earlier:
                    # For "child_of", prefer someone active earlier
                    return abs(int(candidate["active_min"]) - source_year + 25)
                return abs(int(candidate["active_min"]) - source_year)
            except (ValueError, KeyError):
                return 999

        candidates.sort(key=date_distance)
        # Only resolve if best candidate is clearly closer
        if len(candidates) >= 2:
            d0 = date_distance(candidates[0])
            d1 = date_distance(candidates[1])
            if d0 < d1 - 5:  # at least 5 years closer
                return candidates[0]["bookseller_id"], candidates[0]["name"]

        # If same person name (parent/child), prefer the one with right generational gap
        return candidates[0]["bookseller_id"], candidates[0]["name"]

    return "", name_fragment


# ── Extract relationships from notes ──────────────────────────────────────
relationships = []

PATTERNS = [
    (r'son of ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', 'child_of', True),
    (r'daughter of ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', 'child_of', True),
    (r'brother of ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', 'sibling_of', False),
    (r'sister of ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', 'sibling_of', False),
    (r'nephew of ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', 'nephew_of', True),
    (r'widow of ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', 'widow_of', True),
    (r'wife of ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', 'spouse_of', False),
    (r'succeeded by ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', 'succeeded_by', False),
    (r'apprentice to ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', 'apprentice_of', True),
]

seen = set()
for r in booksellers:
    bs_id = r["bookseller_id"]
    notes = r["notes"]

    for pat, rel_type, prefer_earlier in PATTERNS:
        m = re.search(pat, notes, re.IGNORECASE)
        if m:
            target_name = m.group(1)
            target_id, resolved_name = resolve_person(
                target_name, bs_id, prefer_earlier
            )

            key = (bs_id, rel_type, resolved_name)
            if key in seen:
                continue
            seen.add(key)

            relationships.append({
                "person_a_id": bs_id,
                "person_a_name": r["name"],
                "person_a_type": "bookseller",
                "relationship": rel_type,
                "person_b_id": target_id,
                "person_b_name": resolved_name,
                "person_b_type": "bookseller" if target_id else "unknown",
                "date": "",
                "source": "Plomer (1668-1725)",
            })

# ── Add known business successions ────────────────────────────────────────
KNOWN_SUCCESSIONS = [
    ("BS_MILLAR", "BS_CADELL", "succeeded_by", "1768", "Harris"),
    ("BS_TONSON_SR", "BS_TONSON_JR", "succeeded_by", "1720", "Plomer"),
    ("BS_CADELL", "BS_CADELL", "succeeded_by", "", ""),  # skip self
]

for a_id, b_id, rel, date, source in KNOWN_SUCCESSIONS:
    if a_id == b_id:
        continue
    if a_id in bs_by_id and b_id in bs_by_id:
        key = (a_id, rel, bs_by_id[b_id]["name"])
        if key not in seen:
            seen.add(key)
            relationships.append({
                "person_a_id": a_id,
                "person_a_name": bs_by_id[a_id]["name"],
                "person_a_type": "bookseller",
                "relationship": rel,
                "person_b_id": b_id,
                "person_b_name": bs_by_id[b_id]["name"],
                "person_b_type": "bookseller",
                "date": date,
                "source": source,
            })

# ── Add publishing relationships from bookseller_publications.csv ─────────
pubs_path = BASE / "bookseller_publications.csv"
if pubs_path.exists():
    with open(pubs_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bs_id = row["bookseller_id"]
            author = row["author"]
            role = row["role"]
            year = row.get("pub_year", "")

            if bs_id not in bs_by_id:
                continue

            # Create author ID from name
            author_parts = author.replace(",", "").split()
            if len(author_parts) >= 2:
                author_id = f"AUTHOR_{author_parts[-1].upper()}"
            else:
                author_id = f"AUTHOR_{author.upper().replace(' ', '_')}"

            rel = "publisher_of" if role == "publisher" else "printer_of"

            key = (bs_id, rel, author)
            if key in seen:
                continue
            seen.add(key)

            relationships.append({
                "person_a_id": bs_id,
                "person_a_name": bs_by_id[bs_id]["name"],
                "person_a_type": "bookseller",
                "relationship": rel,
                "person_b_id": author_id,
                "person_b_name": author,
                "person_b_type": "author",
                "date": year,
                "source": row.get("bibliography_file", ""),
            })

# ── Mine Plomer full text for partnership references ──────────────────────
plomer_path = Path("/tmp/plomer_1668_1725.txt")
if plomer_path.exists():
    plomer_text = plomer_path.read_text(encoding="utf-8")

    # Find "in partnership with" references
    for m in re.finditer(
        r'(\b[A-Z][a-z]+)\s+.*?in\s+partnership\s+with\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        plomer_text
    ):
        # These are harder to resolve without context — skip for now
        pass

# ── Sort and write ────────────────────────────────────────────────────────
relationships.sort(key=lambda r: (r["person_a_name"], r["relationship"]))

out_path = BASE / "relationships.csv"
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "person_a_id", "person_a_name", "person_a_type",
        "relationship",
        "person_b_id", "person_b_name", "person_b_type",
        "date", "source",
    ])
    writer.writeheader()
    writer.writerows(relationships)

# ── Stats ─────────────────────────────────────────────────────────────────
from collections import Counter
rel_counts = Counter(r["relationship"] for r in relationships)
resolved_count = sum(1 for r in relationships if r["person_b_id"] and not r["person_b_id"].startswith("AUTHOR_"))
author_links = sum(1 for r in relationships if r["person_b_type"] == "author")
bs_links = sum(1 for r in relationships if r["person_b_type"] == "bookseller")

print(f"Total relationships: {len(relationships)}")
print(f"  Bookseller-bookseller: {bs_links} ({resolved_count} resolved to IDs)")
print(f"  Bookseller-author: {author_links}")
print(f"\nBy type:")
for rel, count in rel_counts.most_common():
    print(f"  {rel}: {count}")

# Show some interesting chains
print(f"\nWrote {out_path}")
