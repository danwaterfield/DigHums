"""Build typed network: correspondence vs personal vs discursive edges, with dates and places."""
import re
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/Users/danielwaterfield/Documents/DigHums")

def strip_gutenberg(text):
    for m in ["*** START OF", "***START OF"]:
        idx = text.find(m)
        if idx != -1:
            nl = text.find("\n", idx)
            text = text[nl+1:]
            break
    for m in ["*** END OF", "***END OF", "End of the Project Gutenberg"]:
        idx = text.find(m)
        if idx != -1:
            text = text[:idx]
            break
    return text

def load(path):
    return strip_gutenberg(path.read_text(errors="replace")).replace("ſ", "s")

# ============================================================
# BIOGRAPHICAL DATA — birth, death, active locations
# ============================================================
bio = {
    # (birth, death, primary_city, secondary_cities)
    "Samuel Johnson": (1709, 1784, "London", ["Lichfield", "Edinburgh"]),
    "James Boswell": (1740, 1795, "Edinburgh", ["London"]),
    "David Garrick": (1717, 1779, "London", ["Lichfield"]),
    "Joshua Reynolds": (1723, 1792, "London", ["Plymouth"]),
    "Edmund Burke": (1729, 1797, "London", ["Dublin", "Beaconsfield"]),
    "Oliver Goldsmith": (1728, 1774, "London", ["Dublin"]),
    "Edward Gibbon": (1737, 1794, "London", ["Lausanne"]),
    "Adam Smith": (1723, 1790, "Edinburgh", ["Glasgow", "London", "Kirkcaldy"]),
    "David Hume": (1711, 1776, "Edinburgh", ["London", "Paris"]),
    "Laurence Sterne": (1713, 1768, "York", ["London"]),
    "Samuel Richardson": (1689, 1761, "London", []),
    "Henry Fielding": (1707, 1754, "London", ["Lisbon"]),
    "Frances Burney": (1752, 1840, "London", ["Bath", "Paris", "Windsor"]),
    "Charles Burney Sr": (1726, 1814, "London", ["King's Lynn"]),
    "Hester Thrale": (1741, 1821, "London", ["Streatham", "Bath"]),
    "Elizabeth Montagu": (1718, 1800, "London", ["Sandleford"]),
    "Hester Chapone": (1727, 1801, "London", []),
    "Hannah More": (1745, 1833, "London", ["Bristol", "Bath"]),
    "Horace Walpole": (1717, 1797, "London", ["Strawberry Hill"]),
    "Alexander Pope": (1688, 1744, "London", ["Twickenham"]),
    "Jonathan Swift": (1667, 1745, "Dublin", ["London"]),
    "Joseph Addison": (1672, 1719, "London", []),
    "Richard Steele": (1672, 1729, "London", []),
    "John Dryden": (1631, 1700, "London", []),
    "John Milton": (1608, 1674, "London", []),
    "William Shakespeare": (1564, 1616, "London", ["Stratford"]),
    "John Locke": (1632, 1704, "London", ["Oxford", "Oates"]),
    "Isaac Newton": (1643, 1727, "London", ["Cambridge", "Woolsthorpe"]),
    "Robert Boyle": (1627, 1691, "London", ["Oxford"]),
    "Voltaire": (1694, 1778, "Paris", ["London", "Ferney"]),
    "Jean-Jacques Rousseau": (1712, 1778, "Geneva", ["Paris", "London"]),
    "Montesquieu": (1689, 1755, "Bordeaux", ["Paris", "London"]),
    "Mary Wollstonecraft": (1759, 1797, "London", ["Paris"]),
    "William Godwin": (1756, 1836, "London", []),
    "Thomas Paine": (1737, 1809, "London", ["Paris", "Philadelphia"]),
    "Joseph Priestley": (1733, 1804, "Birmingham", ["London", "Leeds"]),
    "Benjamin Franklin": (1706, 1790, "Philadelphia", ["London", "Paris"]),
    "John Wilkes": (1725, 1797, "London", []),
    "R.B. Sheridan": (1751, 1816, "London", []),
    "Maria Edgeworth": (1768, 1849, "Edgeworthstown", ["London"]),
    "Anna Laetitia Barbauld": (1743, 1825, "London", ["Warrington"]),
    "Anna Seward": (1742, 1809, "Lichfield", []),
    "Elizabeth Inchbald": (1753, 1821, "London", []),
    "Catharine Macaulay": (1731, 1791, "London", ["Bath"]),
    "Earl of Shaftesbury": (1671, 1713, "London", ["Naples"]),
    "Francis Hutcheson": (1694, 1746, "Glasgow", ["Dublin"]),
    "Thomas Reid": (1710, 1796, "Glasgow", ["Aberdeen"]),
    "David Hartley": (1705, 1757, "Bath", ["London"]),
    "Condillac": (1714, 1780, "Paris", []),
    "Bernard Mandeville": (1670, 1733, "London", []),
    "Descartes": (1596, 1650, "Paris", ["Amsterdam", "Stockholm"]),
    "Samuel Pepys": (1633, 1703, "London", []),
    "John Evelyn": (1620, 1706, "London", ["Deptford"]),
    "Arthur Young": (1741, 1820, "London", ["Bradfield"]),
}

# ============================================================
# EDGE TYPE CLASSIFICATION
# ============================================================

def could_have_met(a, b):
    """Did their lifetimes overlap enough for personal contact?"""
    if a not in bio or b not in bio:
        return None  # unknown
    a_birth, a_death = bio[a][0], bio[a][1]
    b_birth, b_death = bio[b][0], bio[b][1]
    # Both alive and adult (age 18+) at the same time
    a_adult = a_birth + 18
    b_adult = b_birth + 18
    overlap_start = max(a_adult, b_adult)
    overlap_end = min(a_death, b_death)
    if overlap_end > overlap_start:
        return (overlap_start, overlap_end)
    return None

def shared_city(a, b):
    """Did they share a primary or secondary city?"""
    if a not in bio or b not in bio:
        return []
    a_cities = {bio[a][2]} | set(bio[a][3])
    b_cities = {bio[b][2]} | set(bio[b][3])
    return sorted(a_cities & b_cities)

# ============================================================
# LOAD EXISTING CORRESPONDENCE (TYPE 1)
# ============================================================
type1_edges = []  # (a, b, weight, year_min, year_max, source)
corr_path = ROOT / "gazetteer/unified_correspondence_graph.csv"
with open(corr_path) as f:
    for row in csv.DictReader(f):
        a = row.get("person_a", "").strip()
        b = row.get("person_b", "").strip()
        w = row.get("weight", "1")
        ymin = row.get("year_min", "")
        ymax = row.get("year_max", "")
        src = row.get("sources", "")
        type1_edges.append((a, b, int(w) if w else 1, ymin, ymax, src))

# ============================================================
# LOAD EXISTING RELATIONSHIPS (TYPE 2)
# ============================================================
type2_edges = []
rels_path = ROOT / "gazetteer/relationships.csv"
if rels_path.exists():
    with open(rels_path) as f:
        for row in csv.DictReader(f):
            a = row.get("person_a_name", "").strip()
            b = row.get("person_b_name", "").strip()
            rel = row.get("relationship", "")
            date = row.get("date", "")
            src = row.get("source", "")
            type2_edges.append((a, b, rel, date, src))

# ============================================================
# BUILD TYPE 3: DISCURSIVE CO-OCCURRENCE (from recalc script data)
# Re-run the disambiguated co-occurrence but tag each hit with
# the source text's date and category
# ============================================================

# Figures with disambiguated patterns (abbreviated — same as recalc)
figures = [
    ([r'\bboswell\b'], "James Boswell", 1740, 1795),
    ([r'\bgibbon\b'], "Edward Gibbon", 1737, 1794),
    ([r'\bgoldsmith\b'], "Oliver Goldsmith", 1728, 1774),
    ([r'\bsterne\b'], "Laurence Sterne", 1713, 1768),
    ([r'\brichardson\b'], "Samuel Richardson", 1689, 1761),
    ([r'\bwollstonecraft\b'], "Mary Wollstonecraft", 1759, 1797),
    ([r'\bbarbauld\b'], "Anna Laetitia Barbauld", 1743, 1825),
    ([r'\bchapone\b'], "Hester Chapone", 1727, 1801),
    ([r'\bedgeworth\b'], "Maria Edgeworth", 1768, 1849),
    ([r'\binchbald\b'], "Elizabeth Inchbald", 1753, 1821),
    ([r'\bseward\b'], "Anna Seward", 1742, 1809),
    ([r'\bmontagu\b'], "Elizabeth Montagu", 1718, 1800),
    ([r'\bvoltaire\b'], "Voltaire", 1694, 1778),
    ([r'\brousseau\b'], "Jean-Jacques Rousseau", 1712, 1778),
    ([r'\bmontesquieu\b'], "Montesquieu", 1689, 1755),
    ([r'\bpriestley\b'], "Joseph Priestley", 1733, 1804),
    ([r'\bdescartes\b'], "Descartes", 1596, 1650),
    ([r'\bmandeville\b'], "Bernard Mandeville", 1670, 1733),
    ([r'\bhutcheson\b'], "Francis Hutcheson", 1694, 1746),
    ([r'\bshaftesbury\b'], "Earl of Shaftesbury", 1671, 1713),
    ([r'\bgodwin\b'], "William Godwin", 1756, 1836),
    ([r'\bmilton\b'], "John Milton", 1608, 1674),
    ([r'\bshakespeare\b'], "William Shakespeare", 1564, 1616),
    ([r'\bdryden\b'], "John Dryden", 1631, 1700),
    ([r'\bmacaulay\b'], "Catharine Macaulay", 1731, 1791),
    ([r'\bsheridan\b'], "R.B. Sheridan", 1751, 1816),
    ([r'\bgarrick\b'], "David Garrick", 1717, 1779),
    ([r'\breynolds\b'], "Joshua Reynolds", 1723, 1792),
    ([r'\bpepys\b'], "Samuel Pepys", 1633, 1703),
    ([r'\bevelyn\b'], "John Evelyn", 1620, 1706),
    ([r'\bwilkes\b'], "John Wilkes", 1725, 1797),
    ([r'\blocke\b(?!d\b|r\b|t\b|smith)'], "John Locke", 1632, 1704),
    ([r'\bnewton\b'], "Isaac Newton", 1643, 1727),
    ([r'\bhume\b'], "David Hume", 1711, 1776),
    ([r'\bburke\b'], "Edmund Burke", 1729, 1797),
    ([r'\bjohnson\b'], "Samuel Johnson", 1709, 1784),
    ([r'\baddison\b'], "Joseph Addison", 1672, 1719),
    ([r'\bsteele\b(?!d\b|y\b|yard)'], "Richard Steele", 1672, 1729),
    ([r'\bswift\b(?!ly|ness|er|est)'], "Jonathan Swift", 1667, 1745),
    ([r"mr\.? pope\b", r"pope's (?:essay|dunciad|rape|homer|iliad|works|poems|epitaph|imitation)",
      r"alexander pope"], "Alexander Pope", 1688, 1744),
    ([r"adam smith", r"smith's (?:wealth|theory|inquiry|moral)"], "Adam Smith", 1723, 1790),
    ([r'\bpaine\b'], "Thomas Paine", 1737, 1809),
    ([r'\bfranklin\b'], "Benjamin Franklin", 1706, 1790),
    ([r'\bthrale\b'], "Hester Thrale", 1741, 1821),
    ([r'\bburney\b'], "Frances Burney", 1752, 1840),
    ([r'\bboyle\b'], "Robert Boyle", 1627, 1691),
    ([r'\breid\b'], "Thomas Reid", 1710, 1796),
    ([r'\bhartley\b'], "David Hartley", 1705, 1757),
    ([r'\bcondillac\b'], "Condillac", 1714, 1780),
]

# Load texts
nf_file_dates = {
    "Hobbes_Leviathan": 1651, "Locke_EssayVol1": 1690, "Locke_EssayConcerning": 1690,
    "Berkeley_Principles": 1710, "Hume_TreatiseOfHuman": 1739, "Hume_Essays": 1741,
    "Hume_EnquiryConcerning": 1751, "Reid_Inquiry": 1764, "Hartley_Observations": 1749,
    "Condillac_Origin": 1746, "Paine_CommonSense": 1776, "Paine_RightsOfMan": 1791,
    "Godwin_PoliticalJustice": 1793, "Montesquieu_Spirit": 1748,
    "Rousseau_SocialContract": 1762,
    "Smith_TheoryOfMoral": 1759, "Smith_WealthOfNations": 1776,
    "Burke_Sublime": 1757, "Burke_Reflections": 1790, "Burke_Conciliation": 1775,
    "Newton_Opticks": 1704, "Newton_Principia": 1687,
    "Boyle_Sceptical": 1661, "Priestley_Electricity": 1767,
    "Cheyne_English": 1733, "Battie_Treatise": 1758,
    "Whytt_Nervous": 1765, "Cullen_Practice": 1777,
    "Crichton_Mental": 1798, "Stillingfleet_Irenicum": 1659,
    "Tillotson_Sermons": 1694, "Shaftesbury_Characteristics": 1711,
    "Hutcheson_Beauty": 1725, "Law_SeriousCall": 1729,
    "Swift_TaleOfATub": 1704, "Ward_LondonSpy": 1698,
    "Gibbon_DeclineAndFall": 1776, "Hume_HistoryOfEngland": 1754,
    "Pepys_Diary": 1669, "Evelyn_Diary": 1706,
    "Wraxall_Historical": 1815, "Walpole_Memoirs": 1845,
    "Stewart_PhilosophyOfHumanMind": 1792, "Bentham_PrinciplesMorals": 1789,
    "Ricardo_PoliticalEconomy": 1817, "Mill_EssayOnGovernment": 1820,
    "Blackstone_Commentaries": 1765, "Blair_Rhetoric": 1783,
    "Priestley_EnglishGrammar": 1761, "Johnson_Rambler": 1750,
    "Johnson_Idler": 1758, "Johnson_Rasselas": 1759,
    "Swift_Gullivers": 1726, "Franklin_Autobiography": 1791,
    "Gibbon_Memoirs": 1796, "Boswell_TourToHebrides": 1785,
    "Dryden_Poems": 1693, "Pope_Poems": 1714, "Swift_Poems": 1711,
    "Johnson_Poems": 1749, "Thomson_TheSeasons": 1730,
    "Gray_Poems": 1751, "Collins_Odes": 1747, "Goldsmith_Poems": 1770,
    "Cowper_TheTask": 1785, "Crabbe_Poems": 1783, "Burns_Poems": 1786,
    "Churchill_Poems": 1761, "PeterPindar_Poems": 1785,
    "Barbauld_Poems": 1773, "More_Poems": 1786, "Finch_Poems": 1713,
    "Malthus_Population": 1798, "Darwin_Zoonomia": 1794,
    "Voltaire_Age": 1751, "Goldsmith_HistoryOfEngland": 1771,
    "Arnold_Insanity": 1782, "Davy_ChemicalPhilosophy": 1812,
}

fiction_map = {
    "FrancesBurney": 1778, "SamuelRichardson": 1740, "HenryFielding": 1742,
    "TobiasSmollett": 1748, "LaurenceSterne": 1759, "JaneAusten": 1811,
    "AnnRadcliffe": 1789, "CharlotteSmith": 1788, "OliverGoldsmith": 1766,
    "ElizabethInchbald": 1791, "WilliamGodwin": 1794, "MariaEdgeworth": 1800,
    "ElizaHaywood": 1719, "HenryMackenzie": 1771, "CharlotteLennox": 1752,
}

texts = {}
for author, date in fiction_map.items():
    d = ROOT / author
    if not d.exists(): continue
    all_txt = sorted(d.glob("*.txt"))
    if all_txt:
        texts[author] = (date, "fiction", "\n".join(load(f) for f in all_txt))

for subdir in ["conduct", "state", "medical", "philosophy", "science",
               "philology", "history", "travel", "misc", "poetry"]:
    base = ROOT / "nonfiction" / subdir
    if not base.exists(): continue
    for f in base.rglob("*.txt"):
        text = load(f)
        if len(text) < 3000: continue
        stem = f.stem
        date = None
        for key, d in nf_file_dates.items():
            if key.lower() in stem.lower():
                date = d
                break
        if date is None: date = 1750
        texts[stem] = (date, subdir, text)

print(f"Loaded {len(texts)} texts\n")

# ============================================================
# BUILD TYPE 3 EDGES WITH METADATA
# ============================================================
type3_raw = []  # (nameA, nameB, source_label, source_date, source_cat)

for label, (date, cat, text) in texts.items():
    tl = text.lower()
    if len(tl) < 3000: continue

    fig_positions = {}
    for patterns, name, birth, death in figures:
        if date < birth: continue
        positions = []
        for pat in patterns:
            for m in re.finditer(pat, tl):
                positions.append(m.start())
        if positions:
            fig_positions[name] = positions

    names_found = list(fig_positions.keys())
    for i, name_a in enumerate(names_found):
        for name_b in names_found[i+1:]:
            found = False
            for pos_a in fig_positions[name_a]:
                if found: break
                for pos_b in fig_positions[name_b]:
                    if abs(pos_a - pos_b) < 500:
                        type3_raw.append((name_a, name_b, label, date, cat))
                        found = True
                        break

# Aggregate type 3 by pair
type3_agg = defaultdict(list)  # (a,b) -> [(source, date, cat)]
for a, b, src, date, cat in type3_raw:
    key = tuple(sorted([a, b]))
    type3_agg[key].append((src, date, cat))

# ============================================================
# CLASSIFY AND OUTPUT
# ============================================================

print("=" * 120)
print("TYPED NETWORK — three layers")
print("=" * 120)

# ── TYPE 1: CORRESPONDENCE ──
print(f"\n{'─'*120}")
print(f"TYPE 1: CORRESPONDENCE (actual letters)")
print(f"{'─'*120}")
print(f"  {len(type1_edges)} edges from unified_correspondence_graph.csv")

# Summarise by decade
corr_by_decade = Counter()
for a, b, w, ymin, ymax, src in type1_edges:
    if ymin and ymin.isdigit():
        dec = int(ymin) // 10 * 10
        corr_by_decade[dec] += w

print(f"\n  Letters by decade:")
for dec in sorted(corr_by_decade):
    if 1700 <= dec <= 1820:
        print(f"    {dec}s: {corr_by_decade[dec]:>5} letters")

# ── TYPE 2: PERSONAL CONNECTION ──
print(f"\n{'─'*120}")
print(f"TYPE 2: PERSONAL CONNECTION (venue co-presence, membership, friendship)")
print(f"{'─'*120}")
print(f"  {len(type2_edges)} edges from relationships.csv")

rel_types = Counter(rel for _, _, rel, _, _ in type2_edges)
for rt, c in rel_types.most_common():
    print(f"    {rt}: {c}")

# ── TYPE 3: DISCURSIVE CO-OCCURRENCE ──
print(f"\n{'─'*120}")
print(f"TYPE 3: DISCURSIVE CO-OCCURRENCE (mentioned together in a third-party text)")
print(f"{'─'*120}")

# Filter to 2+ texts
type3_solid = {k: v for k, v in type3_agg.items() if len(v) >= 2}
print(f"  {len(type3_solid)} pairs appearing in 2+ texts (from {len(type3_agg)} total pairs)")

# Classify each type 3 pair: could they have actually met?
type3_classified = []
for (a, b), evidence in sorted(type3_solid.items(), key=lambda x: -len(x[1])):
    overlap = could_have_met(a, b)
    cities = shared_city(a, b)
    n = len(evidence)
    dates = sorted(set(d for _, d, _ in evidence))
    cats = Counter(c for _, _, c in evidence)
    earliest = min(dates)
    latest = max(dates)

    if overlap:
        meeting = f"overlap {overlap[0]}-{overlap[1]}"
    else:
        meeting = "NO OVERLAP — purely discursive"

    type3_classified.append((a, b, n, meeting, cities, earliest, latest, cats, overlap is not None))

# ── PRINT TYPE 3 SPLIT ──
print(f"\n  --- Could have met (contemporaries, discursive evidence may reflect real contact) ---")
print(f"  {'Person A':<25} {'Person B':<25} {'Texts':>5} {'Overlap':>20} {'Shared cities':<20} {'Disc. range'}")
print("  " + "-" * 115)
for a, b, n, meeting, cities, earliest, latest, cats, met in type3_classified:
    if met and n >= 4:
        city_str = ", ".join(cities) if cities else "—"
        print(f"  {a:<25} {b:<25} {n:>5}   {meeting:>20}   {city_str:<20} {earliest}-{latest}")

print(f"\n  --- Could NOT have met (purely discursive — one or both dead) ---")
print(f"  {'Person A':<25} {'Person B':<25} {'Texts':>5} {'Gap':>30} {'Disc. range'}")
print("  " + "-" * 100)
for a, b, n, meeting, cities, earliest, latest, cats, met in type3_classified:
    if not met and n >= 4:
        print(f"  {a:<25} {b:<25} {n:>5}   {meeting:>30}   {earliest}-{latest}")

# ============================================================
# DISCURSIVE SPREAD OVER TIME
# ============================================================
print(f"\n\n{'='*120}")
print("DISCURSIVE SPREAD — when do pairs start being discussed together?")
print(f"{'='*120}")

print(f"\n  {'Pair':<50} {'First':>6} {'Peak decade':>12} {'Genres'}")
print("  " + "-" * 90)
for a, b, n, meeting, cities, earliest, latest, cats, met in type3_classified:
    if n >= 5:
        # Find peak decade
        dec_counts = Counter()
        for _, d, _ in type3_agg[tuple(sorted([a, b]))]:
            dec_counts[d // 10 * 10] += 1
        peak_dec = dec_counts.most_common(1)[0][0] if dec_counts else 0
        genre_str = ", ".join(f"{c}({n})" for c, n in cats.most_common(3))
        pair_label = f"{a} ↔ {b}"
        print(f"  {pair_label:<50} {earliest:>6}   {peak_dec}s        [{genre_str}]")

# ============================================================
# IDEA TRANSMISSION — genre pathway for key pairs
# ============================================================
print(f"\n\n{'='*120}")
print("IDEA TRANSMISSION — in which genres do pairs first appear together?")
print(f"{'='*120}")

# For pairs with 5+ texts, show the chronological genre sequence
for a, b, n, meeting, cities, earliest, latest, cats, met in type3_classified:
    if n >= 8:
        pair_label = f"{a} ↔ {b}"
        evidence = sorted(type3_agg[tuple(sorted([a, b]))], key=lambda x: x[1])
        print(f"\n  {pair_label} ({n} texts):")
        for src, date, cat in evidence:
            marker = " *** FIRST" if date == earliest else ""
            print(f"    {date}  {cat:<12}  {src[:50]}{marker}")

# ============================================================
# GEOGRAPHIC ANALYSIS — where do the edges concentrate?
# ============================================================
print(f"\n\n{'='*120}")
print("GEOGRAPHIC CONCENTRATION — shared cities for contemporaneous pairs")
print(f"{'='*120}")

city_pairs = Counter()
for a, b, n, meeting, cities, earliest, latest, cats, met in type3_classified:
    if met and n >= 3:
        for city in cities:
            city_pairs[city] += 1

print(f"\n  {'City':<20} {'Pairs with shared presence':>10}")
print("  " + "-" * 35)
for city, count in city_pairs.most_common():
    print(f"  {city:<20} {count:>10}")

# ============================================================
# CROSS-TYPE CORRELATION
# ============================================================
print(f"\n\n{'='*120}")
print("CROSS-TYPE: pairs that appear in ALL THREE layers")
print(f"{'='*120}")

# Build sets for comparison
type1_pairs = set()
for a, b, w, ymin, ymax, src in type1_edges:
    type1_pairs.add(tuple(sorted([a.lower(), b.lower()])))

type2_pairs = set()
for a, b, rel, date, src in type2_edges:
    type2_pairs.add(tuple(sorted([a.lower(), b.lower()])))

type3_pairs = set()
for (a, b) in type3_solid:
    type3_pairs.add(tuple(sorted([a.lower(), b.lower()])))

# Find overlaps
all_three = set()
for p3a, p3b in type3_pairs:
    for p1a, p1b in type1_pairs:
        if (p3a in p1a or p1a in p3a) and (p3b in p1b or p1b in p3b):
            for p2a, p2b in type2_pairs:
                if (p3a in p2a or p2a in p3a) and (p3b in p2b or p2b in p3b):
                    all_three.add((p3a, p3b))
                    break
            break

print(f"\n  Pairs present in correspondence + personal + discursive:")
for a, b in sorted(all_three):
    t3_count = len(type3_solid.get(tuple(sorted([a, b])), []))
    # find t1 weight
    t1_w = 0
    for ea, eb, w, _, _, _ in type1_edges:
        if a in ea.lower() and b in eb.lower():
            t1_w += w
        elif b in ea.lower() and a in eb.lower():
            t1_w += w
    print(f"  {a:<25} ↔ {b:<25} letters={t1_w:<5} discursive_texts={t3_count}")

# Pairs in type 1 + type 3 but NOT type 2
t1_t3_not_t2 = set()
for p3a, p3b in type3_pairs:
    in_t1 = False
    for p1a, p1b in type1_pairs:
        if (p3a in p1a or p1a in p3a) and (p3b in p1b or p1b in p3b):
            in_t1 = True
            break
    in_t2 = False
    for p2a, p2b in type2_pairs:
        if (p3a in p2a or p2a in p3a) and (p3b in p2b or p2b in p3b):
            in_t2 = True
            break
    if in_t1 and not in_t2:
        t1_t3_not_t2.add((p3a, p3b))

print(f"\n  Pairs in correspondence + discursive but NOT personal co-presence:")
for a, b in sorted(t1_t3_not_t2):
    print(f"  {a:<25} ↔ {b:<25}")
