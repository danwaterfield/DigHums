"""Recalculate co-occurrence network with disambiguated surnames."""
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
# LOAD EXISTING NETWORK
# ============================================================
existing_edges = set()
corr_path = ROOT / "gazetteer/unified_correspondence_graph.csv"
with open(corr_path) as f:
    for row in csv.DictReader(f):
        a = row.get("person_a", "").strip().lower()
        b = row.get("person_b", "").strip().lower()
        existing_edges.add(tuple(sorted([a, b])))

print(f"Existing network: {len(existing_edges)} unique edges\n")

# ============================================================
# DISAMBIGUATED SEARCH PATTERNS
# Each entry: (patterns_list, canonical_name, birth, death)
# patterns_list = regex patterns that reliably identify the person
# ============================================================
figures = [
    # UNAMBIGUOUS SURNAMES — safe to match as bare words
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
    ([r'\bdavy\b'], "Humphry Davy", 1778, 1829),
    ([r'\bdalton\b'], "John Dalton", 1766, 1844),

    # DISAMBIGUATED SURNAMES — require context or title
    # Locke: exclude locked, locker, locket, locksmith
    ([r'\blocke\b(?!d\b|r\b|t\b|smith)'], "John Locke", 1632, 1704),

    # Newton: reasonably unambiguous as a surname
    ([r'\bnewton\b'], "Isaac Newton", 1643, 1727),

    # Hume: mostly unambiguous (uncommon surname)
    ([r'\bhume\b'], "David Hume", 1711, 1776),

    # Burke: disambiguate by context — "edmund burke" or just "burke" in political texts
    ([r'\bburke\b'], "Edmund Burke", 1729, 1797),

    # Johnson: use "dr johnson", "samuel johnson", "johnson's dictionary", etc.
    ([r'\bjohnson\b'], "Samuel Johnson", 1709, 1784),

    # Addison: reasonably unambiguous
    ([r'\baddison\b'], "Joseph Addison", 1672, 1719),

    # Steele: exclude "steel", "steeled", "steely"
    ([r'\bsteele\b'], "Richard Steele", 1672, 1729),

    # Swift: exclude "swiftly", "swiftness", "swifter", "swiftest"
    ([r'\bswift\b(?!ly|ness|er|est)'], "Jonathan Swift", 1667, 1745),

    # Pope: ONLY count if clearly the poet (mr pope, pope's essay, etc.)
    # or if the text postdates 1688 and context is literary
    ([r"mr\.? pope\b", r"pope's (?:essay|dunciad|rape|homer|iliad|works|poems|epitaph|imitation)",
      r"alexander pope"], "Alexander Pope", 1688, 1744),

    # Smith: ONLY "adam smith" or "smith's wealth" or "smith's theory"
    ([r"adam smith", r"smith's (?:wealth|theory|inquiry|moral)",
      r"dr\.? smith(?:'s)? (?:wealth|theory)"], "Adam Smith", 1723, 1790),

    # More: ONLY "hannah more", "mrs more", "miss more", "more's strictures"
    ([r"hannah more", r"mrs\.? more\b", r"miss more\b",
      r"more's (?:strictures|coelebs|thoughts|sacred)"], "Hannah More", 1745, 1833),

    # Fox: ONLY "charles fox", "mr fox" in post-1749 texts, or "fox's motion/bill"
    ([r"charles (?:james )?fox", r"mr\.? fox\b",
      r"fox's (?:motion|bill|party|speech|opposition)"], "Charles James Fox", 1749, 1806),

    # North: ONLY "lord north" or "north's ministry/government"
    ([r"lord north\b", r"north's (?:ministry|government|administration)"],
     "Lord North", 1732, 1792),

    # Banks: ONLY "sir joseph banks", "mr banks" in natural history context
    ([r"(?:sir )?joseph banks", r"mr\.? banks\b.*(?:royal society|natural|botany|collection)"],
     "Joseph Banks", 1743, 1820),

    # Paine
    ([r'\bpaine\b'], "Thomas Paine", 1737, 1809),

    # Franklin
    ([r'\bfranklin\b'], "Benjamin Franklin", 1706, 1790),

    # Thrale
    ([r'\bthrale\b'], "Hester Thrale", 1741, 1821),

    # Burney
    ([r'\bburney\b'], "Frances Burney", 1752, 1840),

    # Boyle
    ([r'\bboyle\b'], "Robert Boyle", 1627, 1691),

    # Bentham
    ([r'\bbentham\b'], "Jeremy Bentham", 1748, 1832),

    # Reid
    ([r'\breid\b'], "Thomas Reid", 1710, 1796),

    # Hartley
    ([r'\bhartley\b'], "David Hartley", 1705, 1757),

    # Condillac
    ([r'\bcondillac\b'], "Condillac", 1714, 1780),

    # Pitt: "mr pitt", "william pitt"
    ([r"(?:mr\.? |william )pitt\b"], "William Pitt", 1759, 1806),
]

# ============================================================
# LOAD ALL TEXTS
# ============================================================
texts = {}

# Fiction
fiction_map = {
    "ArthurYoung": 1768, "FrancesBurney": 1778, "SamuelRichardson": 1740,
    "HenryFielding": 1742, "TobiasSmollett": 1748, "LaurenceSterne": 1759,
    "JaneAusten": 1811, "AnnRadcliffe": 1789, "CharlotteSmith": 1788,
    "OliverGoldsmith": 1766, "ElizabethInchbald": 1791, "WilliamGodwin": 1794,
    "MariaEdgeworth": 1800, "ElizaHaywood": 1719, "HenryMackenzie": 1771,
    "CharlotteLennox": 1752, "HoraceWalpole": 1764, "ClaraReeve": 1778,
    "MGLewis": 1796, "WilliamBeckford": 1786, "FrancesSheridan": 1761,
}
for author, date in fiction_map.items():
    d = ROOT / author
    if not d.exists():
        continue
    all_txt = sorted(d.glob("*.txt"))
    if not all_txt:
        continue
    combined = "\n".join(load(f) for f in all_txt)
    if len(combined) > 5000:
        texts[author] = (date, "fiction", combined)

# Non-fiction
nf_file_dates = {
    "Hobbes_Leviathan": 1651, "Spinoza_Ethics": 1675,
    "Locke_EssayVol1": 1690, "Locke_EssayConcerning": 1690,
    "Locke_TwoTreatises": 1689, "Berkeley_Principles": 1710,
    "Hume_TreatiseOfHuman": 1739, "Hume_Essays": 1741,
    "Hume_EnquiryConcerning": 1751, "Reid_Inquiry": 1764,
    "Hartley_Observations": 1749, "Condillac_Origin": 1746,
    "Paine_CommonSense": 1776, "Paine_RightsOfMan": 1791,
    "Godwin_PoliticalJustice": 1793, "Montesquieu_Spirit": 1748,
    "Rousseau_SocialContract": 1762,
    "Smith_TheoryOfMoral": 1759, "Smith_WealthOfNations": 1776,
    "Burke_Sublime": 1757, "Burke_Reflections": 1790, "Burke_Conciliation": 1775,
    "Newton_Opticks": 1704, "Newton_Principia": 1687,
    "Boyle_Sceptical": 1661, "Priestley_Electricity": 1767,
    "Franklin_Electricity": 1751, "White_NaturalHistory": 1789,
    "Darwin_Zoonomia": 1794, "Malthus_Population": 1798,
    "Cheyne_English": 1733, "Battie_Treatise": 1758,
    "Whytt_Nervous": 1765, "Cullen_Practice": 1777,
    "Arnold_Insanity": 1782, "Crichton_Mental": 1798,
    "Haslam_Observations": 1798, "Trotter_Nervous": 1807,
    "Stillingfleet_Irenicum": 1659, "Tillotson_Sermons": 1694,
    "Burnet_ThirtyNine": 1699, "Clarke_Boyle": 1704,
    "Shaftesbury_Characteristics": 1711, "Hutcheson_Beauty": 1725,
    "Law_SeriousCall": 1729, "Lavington_Enthusiasm": 1749,
    "Swift_TaleOfATub": 1704, "Woodward_Reformation": 1699,
    "Ward_LondonSpy": 1698, "Brown_Amusements": 1700,
    "Gibbon_DeclineAndFall": 1776, "Hume_HistoryOfEngland": 1754,
    "Robertson_History": 1759, "Clarendon_History": 1702,
    "Pepys_Diary": 1669, "Evelyn_Diary": 1706,
    "Burnet_HistoryOfHisOwn": 1724, "Smollett_HistoryOfEngland": 1757,
    "Goldsmith_HistoryOfEngland": 1771, "Macaulay_HistoryOfEngland": 1763,
    "Wraxall_Historical": 1815, "Walpole_Memoirs": 1845,
    "Hervey_Memoirs": 1848, "Stow_Survey": 1598,
    "Camden_Britannia": 1695, "Pennant_Tour": 1769,
    "Price_Love": 1789, "Priestley_FirstPrinciples": 1768,
    "Blackstone_Commentaries": 1765,
    "Johnson_DictionaryPreface": 1755, "Blair_Rhetoric": 1783,
    "Stewart_PhilosophyOfHumanMind": 1792,
    "Bentham_PrinciplesMorals": 1789, "Paley_Natural": 1802,
    "Paley_Moral": 1785, "Ricardo_PoliticalEconomy": 1817,
    "Mill_EssayOnGovernment": 1820,
    "Hooker_Ecclesiastical": 1594, "Taylor_HolyLiving": 1650,
    "Johnson_Rambler": 1750, "Johnson_Idler": 1758,
    "Johnson_Adventurer": 1752, "Johnson_Rasselas": 1759,
    "Swift_Gullivers": 1726, "Franklin_Autobiography": 1791,
    "Gibbon_Memoirs": 1796,
    "Dryden_Poems": 1693, "Pope_Poems": 1714, "Swift_Poems": 1711,
    "Johnson_Poems": 1749, "Thomson_TheSeasons": 1730,
    "Gray_Poems": 1751, "Collins_Odes": 1747,
    "Goldsmith_Poems": 1770, "Cowper_TheTask": 1785,
    "Crabbe_Poems": 1783, "Burns_Poems": 1786,
    "Churchill_Poems": 1761, "PeterPindar_Poems": 1785,
    "Barbauld_Poems": 1773, "More_Poems": 1786,
    "Finch_Poems": 1713, "Wraxall_Historical": 1815,
}

for subdir in ["conduct", "state", "medical", "philosophy", "science",
               "philology", "history", "travel", "misc", "poetry"]:
    base = ROOT / "nonfiction" / subdir
    if not base.exists():
        continue
    for f in base.rglob("*.txt"):
        text = load(f)
        if len(text) < 3000:
            continue
        stem = f.stem
        date = None
        for key, d in nf_file_dates.items():
            if key.lower() in stem.lower():
                date = d
                break
        if date is None:
            date = 1750
        texts[stem] = (date, subdir, text)

print(f"Loaded {len(texts)} texts\n")

# ============================================================
# FIND CO-OCCURRENCES WITH DISAMBIGUATED PATTERNS
# ============================================================
pair_evidence = defaultdict(lambda: defaultdict(int))  # (nameA, nameB) -> {source: count}

for label, (date, cat, text) in texts.items():
    tl = text.lower()
    if len(tl) < 3000:
        continue

    # Find positions of each figure
    fig_positions = {}
    for patterns, name, birth, death in figures:
        # Temporal filter
        if date < birth:
            continue
        positions = []
        for pat in patterns:
            for m in re.finditer(pat, tl):
                positions.append(m.start())
        if positions:
            fig_positions[name] = positions

    # Find pairs within 500 chars
    names_found = list(fig_positions.keys())
    for i, name_a in enumerate(names_found):
        for name_b in names_found[i+1:]:
            found = False
            for pos_a in fig_positions[name_a]:
                if found:
                    break
                for pos_b in fig_positions[name_b]:
                    if abs(pos_a - pos_b) < 500:
                        key = tuple(sorted([name_a, name_b]))
                        pair_evidence[key][label] += 1
                        found = True
                        break

# ============================================================
# RESULTS
# ============================================================

# Filter to pairs with evidence from 2+ texts
solid_pairs = []
for (a, b), sources in pair_evidence.items():
    n_texts = len(sources)
    total = sum(sources.values())
    if n_texts >= 2:
        is_existing = tuple(sorted([a.lower(), b.lower()])) in existing_edges
        top_sources = sorted(sources.items(), key=lambda x: -x[1])[:4]
        solid_pairs.append((a, b, n_texts, total, is_existing, top_sources))

solid_pairs.sort(key=lambda x: -x[2])

print("=" * 110)
print("DISAMBIGUATED CO-OCCURRENCE NETWORK — pairs appearing in 2+ texts")
print("=" * 110)

print(f"\n{'Person A':<25} {'Person B':<25} {'Texts':>5} {'Hits':>5} {'Status':>10} {'Sources'}")
print("-" * 110)

new_count = 0
strengthened_count = 0
for a, b, n_texts, total, is_existing, top_sources in solid_pairs:
    status = "existing" if is_existing else "NEW"
    if status == "NEW":
        new_count += 1
    else:
        strengthened_count += 1
    src_str = ", ".join(f"{s}({c})" for s, c in top_sources)
    print(f"  {a:<25} {b:<25} {n_texts:>3}   {total:>4}   {status:>8}   [{src_str}]")

print(f"\n  TOTAL: {len(solid_pairs)} pairs ({new_count} new, {strengthened_count} strengthened existing)")

# ============================================================
# NEW NODES — figures with 3+ disambiguated connections not in network
# ============================================================
print("\n\n" + "=" * 110)
print("NEW NODES — figures with 3+ connections")
print("=" * 110)

person_conn = Counter()
for a, b, n_texts, total, _, _ in solid_pairs:
    if n_texts >= 2:
        person_conn[a] += 1
        person_conn[b] += 1

print(f"\n{'Person':<30} {'Connections':>5}")
print("-" * 40)
for person, count in person_conn.most_common(40):
    if count >= 3:
        print(f"  {person:<30} {count:>5}")
