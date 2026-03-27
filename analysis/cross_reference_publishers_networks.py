"""Cross-reference corpus analysis with publishers, networks, and locations."""
import re
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/Users/danielwaterfield/Documents/DigHums")

# ============================================================
# LOAD STRUCTURED DATA
# ============================================================

# Booksellers
booksellers = {}
bs_path = ROOT / "gazetteer/booksellers.csv"
if bs_path.exists():
    with open(bs_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            booksellers[row.get("bookseller_id", "")] = row

# Bookseller locations
bs_locs = []
bsloc_path = ROOT / "gazetteer/bookseller_locations.csv"
if bsloc_path.exists():
    with open(bsloc_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            bs_locs.append(row)

# Bookseller publications
bs_pubs = []
bspub_path = ROOT / "gazetteer/bookseller_publications.csv"
if bspub_path.exists():
    with open(bspub_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            bs_pubs.append(row)

# Relationships
rels = []
rel_path = ROOT / "gazetteer/relationships.csv"
if rel_path.exists():
    with open(rel_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rels.append(row)

# Unified correspondence graph
corr = []
corr_path = ROOT / "gazetteer/unified_correspondence_graph.csv"
if corr_path.exists():
    with open(corr_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            corr.append(row)

# Person info
persons = {}
pi_path = ROOT / "gazetteer/person_info.json"
if pi_path.exists():
    with open(pi_path) as f:
        persons = json.load(f)

# Venues
venues = []
v_path = ROOT / "gazetteer/venues.csv"
if v_path.exists():
    with open(v_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            venues.append(row)

print(f"Loaded: {len(booksellers)} booksellers, {len(bs_locs)} locations, "
      f"{len(bs_pubs)} publications, {len(rels)} relationships, "
      f"{len(corr)} correspondence edges, {len(persons)} persons, {len(venues)} venues\n")

# ============================================================
# 1. PUBLISHER GEOGRAPHY — where are the books published?
# ============================================================
print("=" * 100)
print("1. PUBLISHER GEOGRAPHY — bookseller locations and their authors")
print("=" * 100)

# Map booksellers to their venues
bs_venue = defaultdict(list)
for loc in bs_locs:
    bs_id = loc.get("bookseller_id", "")
    venue = loc.get("venue_id", "")
    addr = loc.get("address_detail", "")
    bs_venue[bs_id].append((venue, addr))

# Map booksellers to their publications
bs_works = defaultdict(list)
for pub in bs_pubs:
    bs_id = pub.get("bookseller_id", "")
    author = pub.get("author", "")
    title = pub.get("title", "")
    year = pub.get("pub_year", "")
    bs_works[bs_id].append((author, title, year))

# Find booksellers who published texts now in our corpus analysis
corpus_authors = ["Burney", "Richardson", "Fielding", "Smollett", "Sterne",
                  "Austen", "Radcliffe", "Goldsmith", "Johnson", "Pope",
                  "Swift", "Hume", "Locke", "Newton", "Burke", "Gibbon",
                  "Smith", "Wollstonecraft", "More", "Edgeworth",
                  "Godwin", "Paine", "Priestley", "Cheyne", "Gregory",
                  "Fordyce", "Chapone", "Rousseau", "Montesquieu"]

print("\nPublishers linked to corpus authors:")
for bs_id, works in bs_works.items():
    matching = []
    for author, title, year in works:
        for ca in corpus_authors:
            if ca.lower() in author.lower():
                matching.append((author, title, year))
                break
    if matching:
        bs_info = booksellers.get(bs_id, {})
        name = bs_info.get("name", bs_id)
        sign = bs_info.get("sign", "")
        locs = bs_venue.get(bs_id, [])
        loc_str = "; ".join(f"{addr}" for _, addr in locs[:3]) if locs else "unknown"
        print(f"\n  {name} ({bs_id})")
        if sign:
            print(f"    Sign: {sign}")
        print(f"    Location: {loc_str}")
        for author, title, year in matching:
            print(f"    → {author}, {title} ({year})")

# ============================================================
# 2. THE STRAND/FLEET STREET AXIS — publisher clustering
# ============================================================
print("\n\n" + "=" * 100)
print("2. PUBLISHER CLUSTERING — which streets dominate?")
print("=" * 100)

street_counts = Counter()
street_authors = defaultdict(set)
for loc in bs_locs:
    addr = loc.get("address_detail", "").lower()
    bs_id = loc.get("bookseller_id", "")
    # Extract street name
    for street in ["strand", "fleet street", "paternoster row", "pall mall",
                    "cornhill", "ludgate", "st paul's churchyard", "cheapside",
                    "temple bar", "covent garden", "piccadilly", "bond street",
                    "holborn", "newgate street", "ave maria lane"]:
        if street in addr:
            street_counts[street] += 1
            # Link to published authors
            for author, title, year in bs_works.get(bs_id, []):
                street_authors[street].add(author)
            break

print(f"\n{'Street':<25} {'Booksellers':>12} {'Authors published':>40}")
print("-" * 80)
for street, count in street_counts.most_common(15):
    authors = sorted(street_authors[street])[:5]
    author_str = ", ".join(authors) if authors else "—"
    print(f"  {street:<25} {count:>10}   {author_str}")

# ============================================================
# 3. CORRESPONDENT NETWORK — who knows whom?
# ============================================================
print("\n\n" + "=" * 100)
print("3. CORRESPONDENT NETWORK — connections between corpus authors")
print("=" * 100)

# Find corpus-relevant people in the correspondence graph
corpus_people = ["burney", "johnson", "burke", "hume", "smith", "richardson",
                 "fielding", "goldsmith", "garrick", "reynolds", "walpole",
                 "montagu", "chapone", "more", "thrale", "piozzi",
                 "priestley", "sterne", "gibbon", "boswell"]

print("\nCorrespondence edges between corpus-relevant figures:")
relevant_edges = []
for edge in corr:
    a = edge.get("person_a", "").lower()
    b = edge.get("person_b", "").lower()
    weight = edge.get("weight", "1")
    year_min = edge.get("year_min", "")
    year_max = edge.get("year_max", "")
    sources = edge.get("sources", "")
    for pa in corpus_people:
        for pb in corpus_people:
            if pa != pb and pa in a and pb in b:
                relevant_edges.append((edge.get("person_a", ""),
                                       edge.get("person_b", ""),
                                       weight, year_min, year_max, sources))

# Deduplicate and show
seen = set()
for a, b, w, ymin, ymax, src in sorted(relevant_edges, key=lambda x: x[3]):
    key = tuple(sorted([a.lower(), b.lower()]))
    if key in seen:
        continue
    seen.add(key)
    print(f"  {a:<30} ↔ {b:<30} weight={w:<4} {ymin}-{ymax}  [{src}]")

# ============================================================
# 4. RELATIONSHIP NETWORK — venue co-presence
# ============================================================
print("\n\n" + "=" * 100)
print("4. VENUE CO-PRESENCE — who frequented the same spaces?")
print("=" * 100)

# Group relationships by venue
venue_members = defaultdict(list)
for rel in rels:
    rel_type = rel.get("relationship", "")
    if rel_type in ("member_of", "frequented", "patron_of"):
        person = rel.get("person_a_name", "")
        venue = rel.get("person_b_name", "")
        date = rel.get("date", "")
        venue_members[venue].append((person, date))

# Show venues with multiple corpus-relevant members
print("\nVenues where corpus authors intersect:")
for venue, members in sorted(venue_members.items()):
    relevant = [(p, d) for p, d in members
                 if any(cp in p.lower() for cp in corpus_people)]
    if len(relevant) >= 2:
        print(f"\n  {venue}:")
        for person, date in sorted(relevant, key=lambda x: x[1] if x[1] else "9999"):
            print(f"    {person:<35} ({date})")

# ============================================================
# 5. CROSS-REFERENCE: LOCKE'S PUBLISHERS
# ============================================================
print("\n\n" + "=" * 100)
print("5. LOCKE'S PUBLISHING NETWORK — who published the dominant conduct authority?")
print("=" * 100)

locke_publishers = []
for pub in bs_pubs:
    if "locke" in pub.get("author", "").lower():
        bs_id = pub.get("bookseller_id", "")
        bs_info = booksellers.get(bs_id, {})
        name = bs_info.get("name", bs_id)
        locs = bs_venue.get(bs_id, [])
        locke_publishers.append((name, pub.get("title", ""), pub.get("pub_year", ""),
                                  locs[0][1] if locs else "unknown"))

if locke_publishers:
    for name, title, year, loc in locke_publishers:
        print(f"  {name}: {title} ({year}) — {loc}")
else:
    print("  No Locke publications found in bookseller_publications.csv")
    print("  Checking bookseller notes for Locke connections...")
    for bs_id, bs in booksellers.items():
        notes = bs.get("notes", "").lower()
        if "locke" in notes:
            locs = bs_venue.get(bs_id, [])
            loc_str = locs[0][1] if locs else "unknown"
            print(f"  {bs.get('name', bs_id)}: {bs.get('notes', '')} — {loc_str}")

# ============================================================
# 6. CADELL'S EMPIRE — the publisher who links the most corpus authors
# ============================================================
print("\n\n" + "=" * 100)
print("6. PUBLISHER HUB ANALYSIS — which publishers link the most corpus authors?")
print("=" * 100)

publisher_author_count = Counter()
publisher_authors = defaultdict(list)
for pub in bs_pubs:
    bs_id = pub.get("bookseller_id", "")
    author = pub.get("author", "")
    title = pub.get("title", "")
    publisher_author_count[bs_id] += 1
    publisher_authors[bs_id].append(f"{author}: {title}")

print(f"\n{'Publisher':<40} {'Works':>6}  Authors")
print("-" * 100)
for bs_id, count in publisher_author_count.most_common(15):
    bs_info = booksellers.get(bs_id, {})
    name = bs_info.get("name", bs_id)
    authors = publisher_authors[bs_id][:5]
    author_str = "; ".join(authors)
    print(f"  {name:<40} {count:>4}   {author_str}")

# ============================================================
# 7. CHARLES BURNEY'S LIBRARY — what did he own?
# ============================================================
print("\n\n" + "=" * 100)
print("7. CHARLES BURNEY'S LIBRARY — corpus authors in the auction catalogue")
print("=" * 100)

auction_path = ROOT / "nonfiction/Charles Burney/library_provenance_tracking.csv"
if auction_path.exists():
    corpus_in_library = []
    with open(auction_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            author = row.get("author", "").lower()
            title = row.get("title", "")
            lot = row.get("lot", "")
            buyer = row.get("buyer_identified", row.get("buyer_raw", ""))
            dest = row.get("probable_destination", "")
            for ca in corpus_authors:
                if ca.lower() in author:
                    corpus_in_library.append((ca, title, lot, buyer, dest))
                    break

    # Group by author
    by_author = defaultdict(list)
    for ca, title, lot, buyer, dest in corpus_in_library:
        by_author[ca].append((title, lot, buyer, dest))

    for author in sorted(by_author.keys()):
        items = by_author[author]
        print(f"\n  {author} ({len(items)} lots):")
        for title, lot, buyer, dest in items[:5]:
            print(f"    Lot {lot}: {title[:60]}")
            if buyer:
                print(f"      Buyer: {buyer} → {dest}")
else:
    print("  Auction catalogue not found")

# ============================================================
# 8. TEMPORAL OVERLAP — who was active simultaneously?
# ============================================================
print("\n\n" + "=" * 100)
print("8. TEMPORAL OVERLAP — active periods of key figures")
print("=" * 100)

key_figures = {
    "Locke": (1632, 1704),
    "Newton": (1643, 1727),
    "Shaftesbury": (1671, 1713),
    "Addison": (1672, 1719),
    "Swift": (1667, 1745),
    "Pope": (1688, 1744),
    "Richardson": (1689, 1761),
    "Hume": (1711, 1776),
    "Fielding": (1707, 1754),
    "Johnson": (1709, 1784),
    "Sterne": (1713, 1768),
    "Smith": (1723, 1790),
    "Burke": (1729, 1797),
    "Goldsmith": (1728, 1774),
    "Gibbon": (1737, 1794),
    "Cheyne": (1671, 1743),
    "Gregory": (1724, 1773),
    "Fordyce": (1720, 1796),
    "F. Burney": (1752, 1840),
    "Wollstonecraft": (1759, 1797),
    "Rousseau": (1712, 1778),
    "Chapone": (1727, 1801),
    "H. More": (1745, 1833),
    "Edgeworth": (1768, 1849),
    "Austen": (1775, 1817),
    "Godwin": (1756, 1836),
    "Priestley": (1733, 1804),
    "A. Young": (1741, 1820),
}

# Find the densest overlap period
year_counts = Counter()
for name, (birth, death) in key_figures.items():
    for y in range(max(birth+20, 1700), min(death, 1820)):
        year_counts[y] += 1

peak_years = year_counts.most_common(5)
print(f"\nPeak overlap years (most key figures active simultaneously):")
for year, count in peak_years:
    active = [name for name, (b, d) in key_figures.items() if b+20 <= year <= d]
    print(f"  {year}: {count} figures active")
    print(f"    {', '.join(sorted(active))}")

# Who overlaps with Locke in his lifetime?
print(f"\nFigures active during Locke's mature career (1670-1704):")
for name, (b, d) in sorted(key_figures.items()):
    if b + 20 <= 1704 and d >= 1670 and name != "Locke":
        overlap = f"{max(b+20, 1670)}-{min(d, 1704)}"
        print(f"  {name:<20} (active {b+20}-{d}, overlap with Locke: {overlap})")

# Who overlaps with Frances Burney?
print(f"\nFigures active during Frances Burney's writing career (1778-1814):")
for name, (b, d) in sorted(key_figures.items()):
    if b + 20 <= 1814 and d >= 1778 and name != "F. Burney":
        overlap_start = max(b+20, 1778)
        overlap_end = min(d, 1814)
        if overlap_end > overlap_start:
            print(f"  {name:<20} (overlap: {overlap_start}-{overlap_end})")
