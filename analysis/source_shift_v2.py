"""Source-shift analysis v2 — with temporal and substring disambiguation."""
import re
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
    text = path.read_text(errors="replace")
    text = strip_gutenberg(text)
    return text.replace("ſ", "s")

# ============================================================
# AUTHORITIES WITH BIRTH DATES AND DISAMBIGUATION
# ============================================================
# (pattern, era, display_name, birth_year, exclude_patterns)
# exclude_patterns: if the match is a substring of any of these, skip it

authorities = [
    # ANCIENT — no birth-date filtering needed (all pre-period)
    ("aristotle", "ancient", "Aristotle", -384, []),
    ("plato", "ancient", "Plato", -428, []),
    ("cicero", "ancient", "Cicero", -106, []),
    ("seneca", "ancient", "Seneca", -4, []),
    ("plutarch", "ancient", "Plutarch", 46, []),
    ("homer", "ancient", "Homer", -800, []),
    ("virgil", "ancient", "Virgil", -70, []),
    ("horace", "ancient", "Horace", -65, []),  # also a first name — check
    ("ovid", "ancient", "Ovid", -43, []),
    ("juvenal", "ancient", "Juvenal", 55, []),
    ("tacitus", "ancient", "Tacitus", 56, []),
    ("livy", "ancient", "Livy", -59, []),
    ("thucydides", "ancient", "Thucydides", -460, []),
    ("hippocrates", "ancient", "Hippocrates", -460, []),
    ("galen", "ancient", "Galen", 129, []),
    ("epictetus", "ancient", "Epictetus", 50, []),
    ("lucretius", "ancient", "Lucretius", -99, []),
    ("demosthenes", "ancient", "Demosthenes", -384, []),

    # SCRIPTURE — no date filtering
    ("scripture", "scripture", "Scripture", -9999, []),
    ("moses", "scripture", "Moses", -9999, []),
    # Use multi-word to avoid matching Paul as a first name
    ("st paul", "scripture", "St Paul", -9999, []),
    ("saint paul", "scripture", "St Paul", -9999, []),
    ("st augustine", "patristic", "Augustine", 354, []),

    # RENAISSANCE
    ("machiavelli", "renaissance", "Machiavelli", 1469, []),
    ("montaigne", "renaissance", "Montaigne", 1533, ["montaigner"]),
    ("erasmus", "renaissance", "Erasmus", 1466, []),

    # REFORMATION
    ("luther", "reformation", "Luther", 1483, []),
    ("calvin", "reformation", "Calvin", 1509, []),

    # 17C — with birth dates for temporal filtering
    ("bacon", "17c", "Bacon", 1561, ["bacons"]),  # bacons = the meat plural, keep; but filter by date
    ("descartes", "17c", "Descartes", 1596, []),
    ("hobbes", "17c", "Hobbes", 1588, []),
    ("locke", "17c-18c", "Locke", 1632, ["locked", "locker", "locket", "locksmith"]),
    ("newton", "17c-18c", "Newton", 1643, []),
    ("boyle", "17c", "Boyle", 1627, []),
    ("milton", "17c", "Milton", 1608, []),
    ("shakespeare", "17c", "Shakespeare", 1564, []),
    ("dryden", "17c", "Dryden", 1631, []),

    # 18C — birth dates critical for filtering early texts
    ("addison", "18c", "Addison", 1672, []),
    ("steele", "18c", "Steele", 1672, ["steel", "steeled", "steely", "steelyard"]),
    ("voltaire", "18c", "Voltaire", 1694, []),
    ("montesquieu", "18c", "Montesquieu", 1689, []),
    ("rousseau", "18c", "Rousseau", 1712, []),
    ("hume", "18c", "Hume", 1711, ["humect", "humeral"]),
    ("shaftesbury", "18c", "Shaftesbury", 1671, []),
    ("hutcheson", "18c", "Hutcheson", 1694, []),
    ("burke", "18c", "Burke", 1729, []),
    ("gibbon", "18c", "Gibbon", 1737, ["gibbons"]),  # gibbons = apes
    ("richardson", "18c", "Richardson", 1689, []),
    ("fielding", "18c", "Fielding", 1707, []),
    ("johnson", "18c", "Johnson", 1709, []),

    # Pope — special handling: exclude papal contexts
    # We'll handle Pope separately below

    # Smith — special handling: exclude common surname/trade
    # We'll handle Smith separately below

    # Swift — exclude adjective 'swift'
    ("swift", "18c", "Swift", 1667, ["swiftly", "swiftness", "swifter", "swiftest"]),
]

def count_refs(text, text_date, text_lower):
    """Count authority references with temporal and substring filtering."""
    refs = {}

    for pattern, era, name, birth_year, excludes in authorities:
        # TEMPORAL FILTER: if text published before person born, skip
        if text_date < birth_year:
            continue

        # Find all matches
        matches = list(re.finditer(r'\b' + re.escape(pattern) + r'\b', text_lower))
        valid = 0
        for m in matches:
            start, end = m.start(), m.end()
            # SUBSTRING FILTER: check if match is part of an excluded word
            # Get the surrounding word (expand to word boundary)
            word_start = start
            while word_start > 0 and text_lower[word_start-1].isalpha():
                word_start -= 1
            word_end = end
            while word_end < len(text_lower) and text_lower[word_end].isalpha():
                word_end += 1
            full_word = text_lower[word_start:word_end]

            if full_word != pattern and full_word in excludes:
                continue
            # Also skip if the full word is longer than our pattern
            # (means it's a substring of something else like "locksmith")
            if len(full_word) > len(pattern) + 2:  # allow 's, 'd endings
                # Check if it's just a possessive or common suffix
                suffix = full_word[len(pattern):]
                if suffix not in ("s", "'s", "d", "'d", "n", "an", "ian"):
                    continue
            valid += 1

        if valid > 0:
            if name not in refs:
                refs[name] = (era, 0)
            refs[name] = (era, refs[name][1] + valid)

    # POPE — special handling
    if text_date >= 1688:  # Pope born 1688
        pope_matches = list(re.finditer(r'\bpope\b', text_lower))
        poet_pope = 0
        for m in pope_matches:
            start = max(0, m.start() - 80)
            end = min(len(text_lower), m.end() + 80)
            context = text_lower[start:end]
            # PAPAL indicators — skip these
            if any(w in context for w in ["papal", "papist", "papacy", "pontif",
                                           "the pope's", "a pope", "popes ",
                                           "pope's bull", "holy see", "holy father",
                                           "roman", "rome", "popery", "popish",
                                           "clement", "innocent", "urban",
                                           "pope gregory", "pope leo", "pope paul",
                                           "pope pius", "pope john", "pope benedict",
                                           "anti-pope", "antipope",
                                           "pope's head"]):  # Pope's Head = tavern
                continue
            poet_pope += 1
        if poet_pope > 0:
            refs["Pope"] = ("18c", poet_pope)

    # SMITH — special handling
    if text_date >= 1723:  # Adam Smith born 1723
        smith_matches = list(re.finditer(r'\bsmith\b', text_lower))
        adam_smith = 0
        for m in smith_matches:
            start = max(0, m.start() - 80)
            end = min(len(text_lower), m.end() + 80)
            context = text_lower[start:end]
            # CHARACTER/TRADE indicators — skip
            if any(w in context for w in ["mr smith", "mr. smith", "mrs smith",
                                           "mrs. smith", "miss smith",
                                           "blacksmith", "goldsmith", "silversmith",
                                           "locksmith", "gunsmith", "coppersmith",
                                           "tinsmith", "whitesmith", "smith's shop",
                                           "smithfield", "smithy",
                                           "captain smith", "colonel smith",
                                           "john smith", "smith a.b", "rev. john"]):
                continue
            # POSITIVE indicators for Adam Smith
            if any(w in context for w in ["adam smith", "wealth of nations",
                                           "moral sentiments", "smith's wealth",
                                           "smith's theory", "smith observes",
                                           "smith has shown", "dr. smith",
                                           "professor smith", "smith's inquiry"]):
                adam_smith += 1
            # Ambiguous — don't count
        if adam_smith > 0:
            refs["A. Smith"] = ("18c", adam_smith)

    return refs

# ============================================================
# LOAD TEXTS
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

# Non-fiction file dates (same as before, abbreviated)
nf_file_dates = {
    "Hobbes_Leviathan": 1651, "Spinoza_Ethics": 1675,
    "Locke_EssayVol1": 1690, "Locke_EssayConcerning": 1690,
    "Locke_TwoTreatises": 1689, "Berkeley_Principles": 1710,
    "Berkeley_ThreeDialogues": 1713, "Hume_TreatiseOfHuman": 1739,
    "Hume_Essays": 1741, "Hume_EnquiryConcerning": 1751,
    "Reid_Inquiry": 1764, "Hartley_Observations": 1749,
    "Condillac_Origin": 1746, "Leibniz_Monadology": 1714,
    "Paine_CommonSense": 1776, "Paine_RightsOfMan": 1791,
    "Godwin_PoliticalJustice": 1793, "Montesquieu_Spirit": 1748,
    "Rousseau_SocialContract": 1762,
    "Smith_TheoryOfMoral": 1759, "Smith_WealthOfNations": 1776,
    "Burke_Sublime": 1757, "Burke_Reflections": 1790, "Burke_Conciliation": 1775,
    "Newton_Opticks": 1704, "Newton_Principia": 1687,
    "Boyle_Sceptical": 1661, "Priestley_Electricity": 1767,
    "Priestley_Experiments": 1774, "Franklin_Electricity": 1751,
    "White_NaturalHistory": 1789, "Darwin_Zoonomia": 1794,
    "Euler_Elements": 1770, "Malthus_Population": 1798,
    "Buffon_NaturalHistory": 1749,
    "Bright_Treatise": 1586, "Burton_Anatomy": 1621,
    "Cheyne_English": 1733, "Cheyne_Health": 1724,
    "Mandeville_Hypochondriack": 1711, "Battie_Treatise": 1758,
    "Monro_Remarks": 1758, "Whytt_Nervous": 1765,
    "Cullen_Practice": 1777, "Arnold_Insanity": 1782,
    "Crichton_Mental": 1798, "Haslam_Observations": 1798,
    "Trotter_Nervous": 1807, "Howard_State": 1777,
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
    "Rapin_HistoryOfEngland": 1725, "Rollin_AncientHistory": 1730,
    "Voltaire_Age": 1751, "Voltaire_Essay": 1756,
    "Robertson_CharlesV": 1769, "Robertson_HistoryOfAmerica": 1777,
    "Hervey_Memoirs": 1848, "Wraxall_Historical": 1815,
    "Walpole_Memoirs": 1845, "Stow_Survey": 1598,
    "Camden_Britannia": 1695, "Pennant_Tour": 1769,
    "Price_Love": 1789, "Priestley_FirstPrinciples": 1768,
    "Blackstone_Commentaries": 1765,
    "Johnson_DictionaryPreface": 1755, "Lowth_English": 1762,
    "Harris_Hermes": 1751, "Blair_Rhetoric": 1783,
    "Tooke_Diversions": 1786,
    "Cook_Voyages": 1773, "Johnson_Western": 1775,
    "Boswell_TourToHebrides": 1785,
    "Franklin_Autobiography": 1791, "Gibbon_Memoirs": 1796,
    "Johnson_Rasselas": 1759, "Swift_Gullivers": 1726,
    "Dryden_Poems": 1693, "Pope_Poems": 1714, "Swift_Poems": 1711,
    "Johnson_Poems": 1749, "Thomson_TheSeasons": 1730,
    "EdwardYoung_NightThoughts": 1742, "Gray_Poems": 1751,
    "Collins_Odes": 1747, "Akenside_Pleasures": 1744,
    "Goldsmith_Poems": 1770, "Cowper_TheTask": 1785,
    "Crabbe_Poems": 1783, "Blake_Songs": 1789,
    "Burns_Poems": 1786, "Macpherson_Ossian": 1761,
    "Chatterton_Poems": 1770, "CharlotteSmith_Elegiac": 1784,
    "Barbauld_Poems": 1773, "More_Poems": 1786,
    "Robinson_Poems": 1800, "Seward_Poems": 1799,
    "Finch_Poems": 1713, "Wheatley_Poems": 1773,
    "Churchill_Poems": 1761, "PeterPindar_Poems": 1785,
}

nf_cat_map = {
    "conduct": "conduct", "state": "state", "medical": "medical",
    "philosophy": "philosophy", "science": "science", "philology": "philology",
    "history": "history", "travel": "travel", "misc": "misc", "poetry": "poetry",
}

for subdir in nf_cat_map:
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

# Also load conduct by author dir
conduct_dates = {
    "MarquessOfHalifax": 1688, "JohnLocke": 1693, "MaryAstell": 1694,
    "WilliamDarrell": 1704, "JonathanSwift": 1723, "Sophia": 1739,
    "JamesFordyce": 1740, "SamuelRichardson": 1741, "ElizaHaywood": 1744,
    "JamesNelson": 1753, "SarahPennington": 1761, "JeanJacquesRousseau": 1762,
    "LadyMaryWortleyMontagu": 1763, "JohnGregory": 1774, "HesterChapone": 1777,
    "LordChesterfield": 1774, "VicesimusKnox": 1781, "ThomasDay": 1783,
    "AdamPetrie": 1720, "JohnBennett": 1789, "MaryWollstonecraft": 1792,
    "CatharineMacaulay": 1790, "MariaEdgeworth": 1795, "ThomasGisborne": 1797,
    "PriscillaWakefield": 1798, "MaryHays": 1798, "HannahMore": 1799,
    "DanielDefoe": 1697,
}

print(f"Loaded {len(texts)} texts\n")

# ============================================================
# COUNT REFERENCES
# ============================================================
ref_data = []
for label, (date, cat, text) in sorted(texts.items(), key=lambda x: x[1][0]):
    tl = text.lower()
    total_words = len(re.findall(r'[a-z]+', tl))
    if total_words < 5000:
        continue
    refs = count_refs(text, date, tl)
    ref_data.append((label, date, cat, total_words, refs))

# ============================================================
# 1. AGGREGATE A/M RATIO BY PERIOD (cleaned)
# ============================================================
print("=" * 100)
print("1. AGGREGATE ANCIENT/MODERN RATIO BY PERIOD (v2 — disambiguated)")
print("=" * 100)

for p_start, p_end, plabel in [(1580, 1700, "1580-1700"), (1700, 1750, "1700-1750"),
                                 (1750, 1790, "1750-1790"), (1790, 1820, "1790-1820")]:
    ancient = modern = scripture = 0
    n = 0
    for label, date, cat, total, refs in ref_data:
        if p_start <= date < p_end:
            ancient += sum(c for name, (era, c) in refs.items() if era == "ancient")
            modern += sum(c for name, (era, c) in refs.items() if era in ("17c", "17c-18c", "18c"))
            scripture += sum(c for name, (era, c) in refs.items() if era in ("scripture", "patristic"))
            n += 1
    if n == 0:
        continue
    ratio = ancient / modern if modern > 0 else 999
    print(f"  {plabel}: {n} texts  ancient={ancient}  modern={modern}  "
          f"scripture={scripture}  A/M={ratio:.2f}")

# ============================================================
# 2. TOP AUTHORITIES BY PERIOD (cleaned)
# ============================================================
print("\n" + "=" * 100)
print("2. TOP CITED AUTHORITIES BY PERIOD (v2)")
print("=" * 100)

for p_start, p_end, plabel in [(1580, 1700, "1580-1700"), (1700, 1750, "1700-1750"),
                                 (1750, 1790, "1750-1790"), (1790, 1820, "1790-1820")]:
    name_counts = Counter()
    for label, date, cat, total, refs in ref_data:
        if p_start <= date < p_end:
            for name, (era, count) in refs.items():
                name_counts[name] += count
    if not name_counts:
        continue
    print(f"\n  {plabel}:")
    for name, count in name_counts.most_common(15):
        # Look up era
        era = "?"
        for _, e, n, _, _ in authorities:
            if n == name:
                era = e
                break
        if name == "Pope":
            era = "18c"
        elif name == "A. Smith":
            era = "18c"
        print(f"    {name:<20} {count:>6}  ({era})")

# ============================================================
# 3. BY GENRE (cleaned)
# ============================================================
print("\n" + "=" * 100)
print("3. CITATION PATTERNS BY GENRE (v2)")
print("=" * 100)

for cat in ["fiction", "conduct", "history", "philosophy", "science", "medical", "poetry", "philology"]:
    name_counts = Counter()
    ancient = modern = 0
    n = 0
    for label, date, cat2, total, refs in ref_data:
        if cat2 != cat:
            continue
        n += 1
        for name, (era, count) in refs.items():
            name_counts[name] += count
            if era == "ancient":
                ancient += count
            elif era in ("17c", "17c-18c", "18c"):
                modern += count
    if n == 0:
        continue
    ratio = ancient / modern if modern > 0 else 999
    print(f"\n  {cat.upper()} ({n} texts): A/M ratio = {ratio:.2f}")
    for name, count in name_counts.most_common(10):
        era = "?"
        for _, e, nm, _, _ in authorities:
            if nm == name:
                era = e
                break
        if name in ("Pope", "A. Smith"):
            era = "18c"
        print(f"    {name:<20} {count:>6}  ({era})")

# ============================================================
# 4. EPISTEMIC SHIFT (cleaned)
# ============================================================
print("\n" + "=" * 100)
print("4. EPISTEMIC AUTHORITY SHIFT — Scripture vs Classical vs Empirical (v2)")
print("=" * 100)

scripture_names = {"Scripture", "Moses", "St Paul", "Augustine"}
classical_names = {"Aristotle", "Plato", "Cicero", "Seneca", "Plutarch",
                    "Epictetus", "Lucretius", "Homer", "Virgil", "Horace"}
empirical_names = {"Locke", "Newton", "Boyle", "Bacon", "Descartes", "Hobbes"}

print(f"\n{'Decade':<8} {'Scripture':>10} {'Classical':>10} {'Empirical':>10} {'Texts':>6} {'Dominant':>12}")
print("-" * 60)
for dec_start in range(1580, 1820, 10):
    dec_end = dec_start + 10
    matching = [(refs, total) for label, date, cat, total, refs in ref_data
                if dec_start <= date < dec_end]
    if not matching:
        continue
    total_words = sum(t for _, t in matching)
    scrip = sum(refs.get(n, (None, 0))[1] for refs, _ in matching for n in scripture_names)
    class_ = sum(refs.get(n, (None, 0))[1] for refs, _ in matching for n in classical_names)
    empir = sum(refs.get(n, (None, 0))[1] for refs, _ in matching for n in empirical_names)
    s_rate = scrip / total_words * 100000
    c_rate = class_ / total_words * 100000
    e_rate = empir / total_words * 100000
    dominant = "scripture" if s_rate > c_rate and s_rate > e_rate else \
               "classical" if c_rate > e_rate else "empirical"
    n_texts = len(matching)
    flag = " *" if n_texts <= 2 else ""
    print(f"{dec_start}s   {s_rate:>10.1f} {c_rate:>10.1f} {e_rate:>10.1f} {n_texts:>6}{flag:>12}")

print("\n  * = 2 or fewer texts; treat as individual-text data, not decade trend")

# ============================================================
# 5. KEY TRAJECTORIES (cleaned)
# ============================================================
print("\n" + "=" * 100)
print("5. KEY AUTHORITY TRAJECTORIES (v2, per 100k words)")
print("=" * 100)

key_names = ["Aristotle", "Cicero", "Homer", "Locke", "Newton",
             "Bacon", "Milton", "Shakespeare", "Hume", "Rousseau",
             "Pope", "Voltaire", "Scripture"]

print(f"{'Decade':<8} {'#txt':>4}", end="")
for name in key_names:
    print(f" {name[:8]:>9}", end="")
print()
print("-" * (12 + 9 * len(key_names)))

for dec_start in range(1580, 1820, 10):
    dec_end = dec_start + 10
    matching = [(refs, total) for label, date, cat, total, refs in ref_data
                if dec_start <= date < dec_end]
    if not matching:
        continue
    total_words = sum(t for _, t in matching)
    n = len(matching)
    print(f"{dec_start}s   {n:>3} ", end="")
    for name in key_names:
        count = sum(refs.get(name, (None, 0))[1] for refs, _ in matching)
        rate = count / total_words * 100000
        if rate > 0.5:
            print(f" {rate:>9.1f}", end="")
        else:
            print(f" {'·':>9}", end="")
    print()

# ============================================================
# 6. COMPARISON: v1 vs v2 for worst offenders
# ============================================================
print("\n" + "=" * 100)
print("6. BEFORE/AFTER — impact of disambiguation on key counts")
print("=" * 100)

# Count 'pope' raw vs disambiguated across whole corpus
raw_pope = 0
clean_pope = 0
raw_smith = 0
clean_smith = 0
raw_johnson = 0
clean_johnson = 0

for label, date, cat, total, refs in ref_data:
    text = texts[label][2]
    tl = text.lower()
    raw_pope += len(re.findall(r'\bpope\b', tl))
    clean_pope += refs.get("Pope", (None, 0))[1]
    raw_smith += len(re.findall(r'\bsmith\b', tl))
    clean_smith += refs.get("A. Smith", (None, 0))[1]
    raw_johnson += len(re.findall(r'\bjohnson\b', tl))
    clean_johnson += refs.get("Johnson", (None, 0))[1]

print(f"""
  {'Name':<15} {'v1 (raw)':>10} {'v2 (clean)':>12} {'Reduction':>10}
  {'Pope':<15} {raw_pope:>10} {clean_pope:>12} {(1-clean_pope/raw_pope)*100 if raw_pope else 0:>9.0f}%
  {'Smith':<15} {raw_smith:>10} {clean_smith:>12} {(1-clean_smith/raw_smith)*100 if raw_smith else 0:>9.0f}%
  {'Johnson':<15} {raw_johnson:>10} {clean_johnson:>12} {(1-clean_johnson/raw_johnson)*100 if raw_johnson else 0:>9.0f}%
""")
