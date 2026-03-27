"""Graph-theoretic analysis of the three-layer intellectual network."""
import re
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

# NetworkX check
try:
    import networkx as nx
    from networkx.algorithms import community as nx_community
    HAS_NX = True
except ImportError:
    HAS_NX = False

ROOT = Path("/Users/danielwaterfield/Documents/DigHums")

if not HAS_NX:
    print("NetworkX not installed. Installing...")
    import subprocess
    subprocess.check_call(["pip3", "install", "networkx"])
    import networkx as nx
    from networkx.algorithms import community as nx_community

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
# BIOGRAPHICAL DATA
# ============================================================
bio = {
    "Samuel Johnson": (1709, 1784, "London"),
    "James Boswell": (1740, 1795, "Edinburgh"),
    "David Garrick": (1717, 1779, "London"),
    "Joshua Reynolds": (1723, 1792, "London"),
    "Edmund Burke": (1729, 1797, "London"),
    "Oliver Goldsmith": (1728, 1774, "London"),
    "Edward Gibbon": (1737, 1794, "London"),
    "Adam Smith": (1723, 1790, "Edinburgh"),
    "David Hume": (1711, 1776, "Edinburgh"),
    "Laurence Sterne": (1713, 1768, "York"),
    "Samuel Richardson": (1689, 1761, "London"),
    "Henry Fielding": (1707, 1754, "London"),
    "Frances Burney": (1752, 1840, "London"),
    "Charles Burney Sr": (1726, 1814, "London"),
    "Hester Thrale": (1741, 1821, "London"),
    "Elizabeth Montagu": (1718, 1800, "London"),
    "Hester Chapone": (1727, 1801, "London"),
    "Hannah More": (1745, 1833, "Bristol"),
    "Horace Walpole": (1717, 1797, "London"),
    "Alexander Pope": (1688, 1744, "London"),
    "Jonathan Swift": (1667, 1745, "Dublin"),
    "Joseph Addison": (1672, 1719, "London"),
    "Richard Steele": (1672, 1729, "London"),
    "John Dryden": (1631, 1700, "London"),
    "John Milton": (1608, 1674, "London"),
    "William Shakespeare": (1564, 1616, "London"),
    "John Locke": (1632, 1704, "London"),
    "Isaac Newton": (1643, 1727, "London"),
    "Robert Boyle": (1627, 1691, "London"),
    "Voltaire": (1694, 1778, "Paris"),
    "Jean-Jacques Rousseau": (1712, 1778, "Geneva"),
    "Montesquieu": (1689, 1755, "Bordeaux"),
    "Mary Wollstonecraft": (1759, 1797, "London"),
    "William Godwin": (1756, 1836, "London"),
    "Thomas Paine": (1737, 1809, "London"),
    "Joseph Priestley": (1733, 1804, "Birmingham"),
    "Benjamin Franklin": (1706, 1790, "Philadelphia"),
    "John Wilkes": (1725, 1797, "London"),
    "R.B. Sheridan": (1751, 1816, "London"),
    "Maria Edgeworth": (1768, 1849, "Edgeworthstown"),
    "Anna Laetitia Barbauld": (1743, 1825, "London"),
    "Anna Seward": (1742, 1809, "Lichfield"),
    "Elizabeth Inchbald": (1753, 1821, "London"),
    "Catharine Macaulay": (1731, 1791, "London"),
    "Earl of Shaftesbury": (1671, 1713, "London"),
    "Francis Hutcheson": (1694, 1746, "Glasgow"),
    "Thomas Reid": (1710, 1796, "Glasgow"),
    "David Hartley": (1705, 1757, "Bath"),
    "Condillac": (1714, 1780, "Paris"),
    "Bernard Mandeville": (1670, 1733, "London"),
    "Descartes": (1596, 1650, "Paris"),
    "Samuel Pepys": (1633, 1703, "London"),
    "John Evelyn": (1620, 1706, "London"),
    "Arthur Young": (1741, 1820, "London"),
}

# Role categories for node colouring
roles = {
    "Samuel Johnson": "literary", "James Boswell": "literary",
    "David Garrick": "theatre", "Joshua Reynolds": "art",
    "Edmund Burke": "political", "Oliver Goldsmith": "literary",
    "Edward Gibbon": "history", "Adam Smith": "philosophy",
    "David Hume": "philosophy", "Laurence Sterne": "fiction",
    "Samuel Richardson": "fiction", "Henry Fielding": "fiction",
    "Frances Burney": "fiction", "Charles Burney Sr": "music",
    "Hester Thrale": "literary", "Elizabeth Montagu": "literary",
    "Hester Chapone": "conduct", "Hannah More": "conduct",
    "Horace Walpole": "literary", "Alexander Pope": "poetry",
    "Jonathan Swift": "literary", "Joseph Addison": "literary",
    "Richard Steele": "literary", "John Dryden": "poetry",
    "John Milton": "poetry", "William Shakespeare": "poetry",
    "John Locke": "philosophy", "Isaac Newton": "science",
    "Robert Boyle": "science", "Voltaire": "philosophy",
    "Jean-Jacques Rousseau": "philosophy", "Montesquieu": "philosophy",
    "Mary Wollstonecraft": "conduct", "William Godwin": "philosophy",
    "Thomas Paine": "political", "Joseph Priestley": "science",
    "Benjamin Franklin": "science", "John Wilkes": "political",
    "R.B. Sheridan": "theatre", "Maria Edgeworth": "fiction",
    "Anna Laetitia Barbauld": "poetry", "Anna Seward": "poetry",
    "Elizabeth Inchbald": "fiction", "Catharine Macaulay": "history",
    "Earl of Shaftesbury": "philosophy", "Francis Hutcheson": "philosophy",
    "Thomas Reid": "philosophy", "David Hartley": "philosophy",
    "Condillac": "philosophy", "Bernard Mandeville": "philosophy",
    "Descartes": "philosophy", "Samuel Pepys": "literary",
    "John Evelyn": "literary", "Arthur Young": "science",
}

# ============================================================
# LOAD THE THREE EDGE LAYERS
# ============================================================

# Layer 1: Correspondence
G1 = nx.Graph()
corr_path = ROOT / "gazetteer/unified_correspondence_graph.csv"
corr_decade_edges = defaultdict(list)
with open(corr_path) as f:
    for row in csv.DictReader(f):
        a = row.get("person_a", "").strip()
        b = row.get("person_b", "").strip()
        w = int(row.get("weight", "1") or "1")
        ymin = row.get("year_min", "")
        # Normalise names to match bio dict
        G1.add_edge(a, b, weight=w, layer="correspondence")
        if ymin and ymin.isdigit():
            dec = int(ymin) // 10 * 10
            corr_decade_edges[dec].append((a, b, w))

# Layer 2: Personal/institutional
G2 = nx.Graph()
rels_path = ROOT / "gazetteer/relationships.csv"
if rels_path.exists():
    with open(rels_path) as f:
        for row in csv.DictReader(f):
            a = row.get("person_a_name", "").strip()
            b = row.get("person_b_name", "").strip()
            rel = row.get("relationship", "")
            if a and b:
                G2.add_edge(a, b, relationship=rel, layer="personal")

# Layer 3: Discursive co-occurrence (rebuild from typed_network output)
# Re-run the disambiguated co-occurrence inline
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

nf_file_dates = {
    "Locke_EssayVol1": 1690, "Locke_EssayConcerning": 1690,
    "Hume_TreatiseOfHuman": 1739, "Hume_Essays": 1741,
    "Hume_EnquiryConcerning": 1751, "Reid_Inquiry": 1764,
    "Hartley_Observations": 1749, "Condillac_Origin": 1746,
    "Paine_RightsOfMan": 1791, "Godwin_PoliticalJustice": 1793,
    "Montesquieu_Spirit": 1748, "Rousseau_SocialContract": 1762,
    "Smith_TheoryOfMoral": 1759, "Smith_WealthOfNations": 1776,
    "Burke_Sublime": 1757, "Burke_Reflections": 1790,
    "Burke_Conciliation": 1775, "Newton_Principia": 1687,
    "Shaftesbury_Characteristics": 1711, "Hutcheson_Beauty": 1725,
    "Swift_TaleOfATub": 1704, "Gibbon_DeclineAndFall": 1776,
    "Hume_HistoryOfEngland": 1754, "Pepys_Diary": 1669,
    "Evelyn_Diary": 1706, "Wraxall_Historical": 1815,
    "Stewart_PhilosophyOfHumanMind": 1792,
    "Crichton_Mental": 1798, "Blackstone_Commentaries": 1765,
    "Blair_Rhetoric": 1783, "Johnson_Rambler": 1750,
    "Johnson_Idler": 1758, "Johnson_Rasselas": 1759,
    "Gibbon_Memoirs": 1796, "Boswell_TourToHebrides": 1785,
    "Dryden_Poems": 1693, "Pope_Poems": 1714, "Swift_Poems": 1711,
    "Johnson_Poems": 1749, "Thomson_TheSeasons": 1730,
    "Gray_Poems": 1751, "Collins_Odes": 1747,
    "Goldsmith_Poems": 1770, "Churchill_Poems": 1761,
    "PeterPindar_Poems": 1785, "Barbauld_Poems": 1773,
    "More_Poems": 1786, "Finch_Poems": 1713,
    "Malthus_Population": 1798, "Darwin_Zoonomia": 1794,
    "Franklin_Autobiography": 1791, "Arnold_Insanity": 1782,
    "Priestley_EnglishGrammar": 1761, "Lowth_EnglishGrammar": 1762,
    "Voltaire_Age": 1751, "Goldsmith_HistoryOfEngland": 1771,
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

print(f"Loaded {len(texts)} texts for co-occurrence\n")

# Build G3
G3 = nx.Graph()
disc_decade_pairs = defaultdict(list)

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
                        if G3.has_edge(name_a, name_b):
                            G3[name_a][name_b]["weight"] += 1
                            G3[name_a][name_b]["texts"].append(label)
                            G3[name_a][name_b]["dates"].append(date)
                            G3[name_a][name_b]["genres"].append(cat)
                        else:
                            G3.add_edge(name_a, name_b, weight=1, layer="discursive",
                                       texts=[label], dates=[date], genres=[cat])
                        dec = date // 10 * 10
                        disc_decade_pairs[dec].append((name_a, name_b))
                        found = True
                        break

# Filter G3 to edges with 2+ texts
edges_to_remove = [(u, v) for u, v, d in G3.edges(data=True) if d["weight"] < 2]
G3.remove_edges_from(edges_to_remove)
G3.remove_nodes_from(list(nx.isolates(G3)))

print(f"G1 (correspondence): {G1.number_of_nodes()} nodes, {G1.number_of_edges()} edges")
print(f"G2 (personal):       {G2.number_of_nodes()} nodes, {G2.number_of_edges()} edges")
print(f"G3 (discursive 2+):  {G3.number_of_nodes()} nodes, {G3.number_of_edges()} edges")

# ============================================================
# COMBINED GRAPH
# ============================================================
G_combined = nx.Graph()

# Add all nodes with bio attributes
for name, (birth, death, city) in bio.items():
    role = roles.get(name, "unknown")
    G_combined.add_node(name, birth=birth, death=death, city=city, role=role)

# Add edges from all layers with type tags
for u, v, d in G1.edges(data=True):
    key = tuple(sorted([u, v]))
    if G_combined.has_edge(u, v):
        G_combined[u][v]["layers"].add("correspondence")
        G_combined[u][v]["corr_weight"] = d.get("weight", 1)
    else:
        G_combined.add_edge(u, v, layers={"correspondence"},
                           corr_weight=d.get("weight", 1),
                           disc_weight=0, disc_texts=0)

for u, v, d in G2.edges(data=True):
    if G_combined.has_edge(u, v):
        G_combined[u][v]["layers"].add("personal")
    else:
        G_combined.add_edge(u, v, layers={"personal"},
                           corr_weight=0, disc_weight=0, disc_texts=0)

for u, v, d in G3.edges(data=True):
    if G_combined.has_edge(u, v):
        G_combined[u][v]["layers"].add("discursive")
        G_combined[u][v]["disc_weight"] = d["weight"]
        G_combined[u][v]["disc_texts"] = d["weight"]
    else:
        G_combined.add_edge(u, v, layers={"discursive"},
                           corr_weight=0, disc_weight=d["weight"],
                           disc_texts=d["weight"])

# Keep only nodes in bio dict for cleaner analysis
nodes_to_keep = set(bio.keys()) & set(G_combined.nodes())
G_analysis = G_combined.subgraph(nodes_to_keep).copy()

print(f"\nG_analysis (combined, bio-matched): {G_analysis.number_of_nodes()} nodes, {G_analysis.number_of_edges()} edges")

# ============================================================
# 1. BETWEENNESS CENTRALITY
# ============================================================
print("\n" + "=" * 100)
print("1. BETWEENNESS CENTRALITY — who bridges clusters?")
print("=" * 100)

bc = nx.betweenness_centrality(G_analysis)
print(f"\n{'Rank':>4} {'Person':<30} {'Betweenness':>12} {'Role':<12} {'City':<12}")
print("-" * 75)
for i, (name, score) in enumerate(sorted(bc.items(), key=lambda x: -x[1])[:25]):
    role = roles.get(name, "?")
    city = bio.get(name, (0, 0, "?"))[2]
    print(f"  {i+1:>2}  {name:<30} {score:>12.4f}   {role:<12} {city}")

# ============================================================
# 2. DEGREE CENTRALITY BY LAYER
# ============================================================
print("\n" + "=" * 100)
print("2. DEGREE CENTRALITY BY LAYER — different importance in different networks")
print("=" * 100)

# Build per-layer subgraphs from G_analysis
G1_sub = nx.Graph()
G3_sub = nx.Graph()
for u, v, d in G_analysis.edges(data=True):
    if "correspondence" in d.get("layers", set()):
        G1_sub.add_edge(u, v)
    if "discursive" in d.get("layers", set()):
        G3_sub.add_edge(u, v)

dc1 = nx.degree_centrality(G1_sub) if G1_sub.number_of_nodes() > 0 else {}
dc3 = nx.degree_centrality(G3_sub) if G3_sub.number_of_nodes() > 0 else {}

print(f"\n{'Person':<30} {'Corr.degree':>12} {'Disc.degree':>12} {'Difference':>12}")
print("-" * 70)
all_names = set(list(dc1.keys()) + list(dc3.keys()))
diffs = [(name, dc1.get(name, 0), dc3.get(name, 0), dc3.get(name, 0) - dc1.get(name, 0))
         for name in all_names if name in bio]
diffs.sort(key=lambda x: -abs(x[3]))
for name, d1, d3, diff in diffs[:25]:
    marker = "← more discursive" if diff > 0.05 else "← more correspondence" if diff < -0.05 else ""
    print(f"  {name:<30} {d1:>12.3f} {d3:>12.3f} {diff:>+12.3f}  {marker}")

# ============================================================
# 3. COMMUNITY DETECTION — Louvain on each layer
# ============================================================
print("\n" + "=" * 100)
print("3. COMMUNITY DETECTION — do the layers produce different clusters?")
print("=" * 100)

def detect_communities(G, label):
    if G.number_of_nodes() < 3:
        print(f"\n  {label}: too few nodes ({G.number_of_nodes()})")
        return {}
    comms = nx_community.louvain_communities(G, seed=42)
    print(f"\n  {label}: {len(comms)} communities")
    comm_map = {}
    for i, comm in enumerate(sorted(comms, key=len, reverse=True)):
        members = sorted(comm, key=lambda n: -G.degree(n))
        bio_members = [m for m in members if m in bio]
        if bio_members:
            print(f"    C{i}: {', '.join(bio_members[:8])}" +
                  (f" (+{len(bio_members)-8} more)" if len(bio_members) > 8 else ""))
            for m in bio_members:
                comm_map[m] = i
    return comm_map

comm_disc = detect_communities(G3_sub, "Discursive (G3)")
comm_combined = detect_communities(G_analysis, "Combined")

# ============================================================
# 4. TEMPORAL CENTRALITY EVOLUTION
# ============================================================
print("\n" + "=" * 100)
print("4. TEMPORAL CENTRALITY — who is most central per decade?")
print("=" * 100)

# Build decade-specific discursive subgraphs
print(f"\n{'Decade':<8} {'Top 5 by discursive degree':}")
print("-" * 80)
for dec in sorted(disc_decade_pairs.keys()):
    if dec < 1690 or dec > 1810:
        continue
    G_dec = nx.Graph()
    for a, b in disc_decade_pairs[dec]:
        if a in bio and b in bio:
            if G_dec.has_edge(a, b):
                G_dec[a][b]["weight"] += 1
            else:
                G_dec.add_edge(a, b, weight=1)
    if G_dec.number_of_nodes() < 3:
        continue
    dc = nx.degree_centrality(G_dec)
    top5 = sorted(dc.items(), key=lambda x: -x[1])[:5]
    names = ", ".join(f"{n}({s:.2f})" for n, s in top5)
    print(f"  {dec}s   {names}")

# ============================================================
# 5. EIGENVECTOR CENTRALITY — who knows the well-connected?
# ============================================================
print("\n" + "=" * 100)
print("5. EIGENVECTOR CENTRALITY — prestige (connected to well-connected people)")
print("=" * 100)

try:
    ec = nx.eigenvector_centrality(G_analysis, max_iter=1000)
    print(f"\n{'Rank':>4} {'Person':<30} {'Eigenvector':>12} {'Role':<12}")
    print("-" * 62)
    for i, (name, score) in enumerate(sorted(ec.items(), key=lambda x: -x[1])[:20]):
        role = roles.get(name, "?")
        print(f"  {i+1:>2}  {name:<30} {score:>12.4f}   {role}")
except nx.PowerIterationFailedConvergence:
    print("  Eigenvector centrality did not converge")

# ============================================================
# 6. BRIDGES — edges whose removal disconnects components
# ============================================================
print("\n" + "=" * 100)
print("6. BRIDGE EDGES — connections whose removal would disconnect the network")
print("=" * 100)

bridges = list(nx.bridges(G_analysis))
print(f"\n  {len(bridges)} bridge edges found:")
for u, v in bridges[:15]:
    layers = G_analysis[u][v].get("layers", set())
    print(f"    {u:<30} ↔ {v:<30} [{', '.join(layers)}]")

# ============================================================
# 7. CROSS-LAYER COMPARISON — same person, different roles
# ============================================================
print("\n" + "=" * 100)
print("7. LAYER MISMATCH — people important in one layer but not another")
print("=" * 100)

print(f"\n  High correspondence, low discursive (private influencers):")
for name, d1, d3, diff in sorted(diffs, key=lambda x: x[3]):
    if d1 > 0.05 and d3 < 0.05 and name in bio:
        print(f"    {name:<30} corr={d1:.3f}  disc={d3:.3f}")

print(f"\n  High discursive, low/no correspondence (canonical but not personally connected):")
for name, d1, d3, diff in sorted(diffs, key=lambda x: -x[3]):
    if d3 > 0.1 and d1 < 0.02 and name in bio:
        print(f"    {name:<30} corr={d1:.3f}  disc={d3:.3f}")

# ============================================================
# 8. GENRE DIFFUSION — how fast do pairings spread across genres?
# ============================================================
print("\n" + "=" * 100)
print("8. GENRE DIFFUSION — time from first co-occurrence to 3+ genres")
print("=" * 100)

print(f"\n{'Pair':<50} {'First':>6} {'3rd genre':>10} {'Lag':>5} {'Path'}")
print("-" * 100)
for u, v, d in sorted(G3.edges(data=True), key=lambda x: -x[2]["weight"]):
    if d["weight"] < 5:
        continue
    dates = d.get("dates", [])
    genres = d.get("genres", [])
    if not dates:
        continue
    # Build chronological genre sequence
    genre_seq = []
    seen_genres = set()
    for date, genre in sorted(zip(dates, genres)):
        if genre not in seen_genres:
            genre_seq.append((date, genre))
            seen_genres.add(genre)
    if len(genre_seq) >= 3:
        first_date = genre_seq[0][0]
        third_date = genre_seq[2][0]
        lag = third_date - first_date
        path = " → ".join(f"{g}({d})" for d, g in genre_seq[:5])
        pair = f"{u} ↔ {v}"
        print(f"  {pair:<50} {first_date:>5}  {third_date:>9}  {lag:>4}y  {path}")

# ============================================================
# 9. EXPORT — for visualisation
# ============================================================
print("\n" + "=" * 100)
print("9. GRAPH EXPORTED")
print("=" * 100)

# Export nodes
nodes_out = ROOT / "analysis" / "network_nodes.csv"
with open(nodes_out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["name", "birth", "death", "city", "role", "betweenness",
                "eigenvector", "disc_community", "corr_degree", "disc_degree"])
    ec_safe = ec if 'ec' in dir() else {}
    for name in sorted(G_analysis.nodes()):
        b, d, c = bio.get(name, (0, 0, "?"))
        r = roles.get(name, "?")
        w.writerow([name, b, d, c, r,
                    f"{bc.get(name, 0):.4f}",
                    f"{ec_safe.get(name, 0):.4f}",
                    comm_disc.get(name, -1),
                    f"{dc1.get(name, 0):.3f}",
                    f"{dc3.get(name, 0):.3f}"])

# Export edges
edges_out = ROOT / "analysis" / "network_edges.csv"
with open(edges_out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["source", "target", "layers", "corr_weight", "disc_weight", "disc_texts"])
    for u, v, d in G_analysis.edges(data=True):
        layers = "|".join(sorted(d.get("layers", set())))
        w.writerow([u, v, layers,
                    d.get("corr_weight", 0),
                    d.get("disc_weight", 0),
                    d.get("disc_texts", 0)])

print(f"  Nodes: {nodes_out}")
print(f"  Edges: {edges_out}")
