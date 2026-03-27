"""Track shifts in cited authorities, sources, and intellectual reference points across the corpus."""
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
# BUILD CORPUS WITH DATES
# ============================================================
texts = {}  # label -> (date, category, raw_text)

# Fiction
fiction_dirs = {
    "ArthurYoung": [(1768, "Lucy Watson"), (1784, "Julia Benson")],
    "FrancesBurney": [(1778, "Evelina"), (1782, "Cecilia"), (1796, "Camilla"), (1814, "Wanderer")],
    "SamuelRichardson": [(1740, "Pamela"), (1748, "Clarissa"), (1753, "Grandison")],
    "HenryFielding": [(1742, "Joseph Andrews"), (1749, "Tom Jones"), (1751, "Amelia")],
    "TobiasSmollett": [(1748, "Roderick Random"), (1751, "Peregrine Pickle"), (1753, "Ferdinand Fathom"), (1771, "Humphry Clinker")],
    "LaurenceSterne": [(1759, "Tristram Shandy"), (1768, "Sentimental Journey")],
    "JaneAusten": [(1811, "Sense Sensibility"), (1813, "Pride Prejudice"), (1814, "Mansfield Park"), (1816, "Emma"), (1817, "Persuasion"), (1818, "Northanger Abbey")],
    "AnnRadcliffe": [(1789, "Sicilian Romance"), (1790, "Romance Forest"), (1794, "Udolpho"), (1797, "Italian")],
    "CharlotteSmith": [(1788, "Emmeline"), (1793, "Old Manor House"), (1792, "Desmond")],
    "OliverGoldsmith": [(1766, "Vicar Wakefield")],
    "ElizabethInchbald": [(1791, "Simple Story"), (1796, "Nature Art")],
    "WilliamGodwin": [(1794, "Caleb Williams")],
    "MariaEdgeworth": [(1800, "Castle Rackrent"), (1801, "Belinda")],
    "ElizaHaywood": [(1719, "Love Excess"), (1751, "Betsy Thoughtless")],
    "HenryMackenzie": [(1771, "Man Feeling")],
    "CharlotteLennox": [(1752, "Female Quixote")],
    "HoraceWalpole": [(1764, "Castle Otranto")],
    "ClaraReeve": [(1778, "Old English Baron")],
    "MGLewis": [(1796, "Monk")],
    "WilliamBeckford": [(1786, "Vathek")],
}

for author, works in fiction_dirs.items():
    author_dir = ROOT / author
    if not author_dir.exists():
        continue
    # Load all txt files for this author as one block per work
    all_txts = sorted(author_dir.glob("*.txt"))
    if not all_txts:
        continue
    combined = "\n".join(load(f) for f in all_txts)
    # Use first work date as representative
    date = works[0][0]
    label = f"{author} ({date})"
    texts[label] = (date, "fiction", combined)

# Non-fiction — scan all subdirectories
nf_dates = {
    # Conduct
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

nf_file_dates = {
    # Individual file overrides where needed
    "Hobbes_Leviathan": 1651, "Spinoza_Ethics": 1675,
    "Locke_EssayVol1": 1690, "Locke_EssayConcerning": 1690,
    "Locke_TwoTreatises": 1689,
    "Berkeley_Principles": 1710, "Berkeley_ThreeDialogues": 1713,
    "Hume_TreatiseOfHumanNature": 1739, "Hume_Essays": 1741,
    "Hume_EnquiryConcerning": 1751,
    "Reid_Inquiry": 1764, "Hartley_Observations": 1749,
    "Condillac_Origin": 1746, "Leibniz_Monadology": 1714,
    "Paine_CommonSense": 1776, "Paine_RightsOfMan": 1791,
    "Godwin_PoliticalJustice": 1793, "Montesquieu_Spirit": 1748,
    "Rousseau_SocialContract": 1762,
    "Smith_TheoryOfMoral": 1759, "Smith_WealthOfNations": 1776,
    "Burke_SublimeAndBeautiful": 1757, "Burke_Reflections": 1790,
    "Burke_Conciliation": 1775,
    "Newton_Opticks": 1704, "Newton_Principia": 1687,
    "Boyle_Sceptical": 1661, "Priestley_Electricity": 1767,
    "Priestley_Experiments": 1774, "Franklin_Electricity": 1751,
    "White_NaturalHistory": 1789, "Darwin_Zoonomia": 1794,
    "Euler_Elements": 1770, "Malthus_Population": 1798,
    "Buffon_NaturalHistory": 1749, "Linnaeus_Systema": 1735,
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
    # Poetry
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

for subdir in ["conduct", "state", "medical", "philosophy", "science",
               "philology", "history", "travel", "misc", "poetry"]:
    base = ROOT / "nonfiction" / subdir
    if not base.exists():
        continue
    for f in base.rglob("*.txt"):
        text = load(f)
        if len(text) < 2000:
            continue
        stem = f.stem
        # Try to find date
        date = None
        for key, d in nf_file_dates.items():
            if key.lower() in stem.lower():
                date = d
                break
        if date is None:
            # Try parent dir name
            for key, d in nf_dates.items():
                if key.lower() in f.parent.name.lower():
                    date = d
                    break
        if date is None:
            date = 1750  # fallback
        cat = subdir
        label = f"{stem} ({date})"
        texts[label] = (date, cat, text)

print(f"Loaded {len(texts)} texts\n")

# ============================================================
# REFERENCE TRACKING — who cites whom?
# ============================================================

# Authorities to track — ancient, medieval, early modern, 17c, 18c
authorities = {
    # ANCIENT
    "aristotle": ("ancient", "Aristotle"),
    "plato": ("ancient", "Plato"),
    "cicero": ("ancient", "Cicero"),
    "seneca": ("ancient", "Seneca"),
    "plutarch": ("ancient", "Plutarch"),
    "homer": ("ancient", "Homer"),
    "virgil": ("ancient", "Virgil"),
    "horace": ("ancient", "Horace"),
    "ovid": ("ancient", "Ovid"),
    "juvenal": ("ancient", "Juvenal"),
    "tacitus": ("ancient", "Tacitus"),
    "livy": ("ancient", "Livy"),
    "thucydides": ("ancient", "Thucydides"),
    "hippocrates": ("ancient", "Hippocrates"),
    "galen": ("ancient", "Galen"),
    "epictetus": ("ancient", "Epictetus"),
    "lucretius": ("ancient", "Lucretius"),
    "demosthenes": ("ancient", "Demosthenes"),

    # SCRIPTURE / CHURCH FATHERS
    "scripture": ("scripture", "Scripture"),
    "moses": ("scripture", "Moses"),
    "st paul": ("scripture", "St Paul"),
    "saint paul": ("scripture", "St Paul"),
    "st augustine": ("patristic", "Augustine"),
    "augustine": ("patristic", "Augustine"),
    "aquinas": ("medieval", "Aquinas"),

    # RENAISSANCE / REFORMATION
    "machiavelli": ("renaissance", "Machiavelli"),
    "montaigne": ("renaissance", "Montaigne"),
    "montaigne's": ("renaissance", "Montaigne"),
    "montaignes": ("renaissance", "Montaigne"),
    "erasmus": ("renaissance", "Erasmus"),
    "luther": ("reformation", "Luther"),
    "calvin": ("reformation", "Calvin"),

    # 17C PHILOSOPHY / SCIENCE
    "bacon": ("17c", "Bacon"),
    "descartes": ("17c", "Descartes"),
    "hobbes": ("17c", "Hobbes"),
    "locke": ("17c-18c", "Locke"),
    "newton": ("17c-18c", "Newton"),
    "boyle": ("17c", "Boyle"),
    "milton": ("17c", "Milton"),
    "shakespeare": ("17c", "Shakespeare"),
    "dryden": ("17c", "Dryden"),

    # 18C FIGURES
    "pope": ("18c", "Pope"),
    "addison": ("18c", "Addison"),
    "steele": ("18c", "Steele"),
    "swift": ("18c", "Swift"),
    "voltaire": ("18c", "Voltaire"),
    "montesquieu": ("18c", "Montesquieu"),
    "rousseau": ("18c", "Rousseau"),
    "hume": ("18c", "Hume"),
    "shaftesbury": ("18c", "Shaftesbury"),
    "hutcheson": ("18c", "Hutcheson"),
    "smith": ("18c", "Smith"),
    "burke": ("18c", "Burke"),
    "johnson": ("18c", "Johnson"),
    "gibbon": ("18c", "Gibbon"),
    "richardson": ("18c", "Richardson"),
    "fielding": ("18c", "Fielding"),
}

# Count references per text
ref_data = []
for label, (date, cat, text) in sorted(texts.items(), key=lambda x: x[1][0]):
    tl = text.lower()
    total_words = len(re.findall(r'[a-z]+', tl))
    if total_words < 5000:
        continue
    refs = {}
    for pattern, (era, name) in authorities.items():
        count = len(re.findall(r'\b' + re.escape(pattern) + r'\b', tl))
        if count > 0:
            if name not in refs:
                refs[name] = (era, 0)
            refs[name] = (era, refs[name][1] + count)
    ref_data.append((label, date, cat, total_words, refs))

# ============================================================
# 1. ANCIENT vs MODERN REFERENCE RATIO OVER TIME
# ============================================================
print("=" * 100)
print("1. ANCIENT vs MODERN AUTHORITY RATIO — by text date")
print("=" * 100)

print(f"\n{'Text':<50} {'Date':>5} {'Cat':<10} {'Ancient':>8} {'17c':>6} {'18c':>6} {'A/M ratio':>10}")
print("-" * 100)
for label, date, cat, total, refs in ref_data:
    ancient = sum(c for name, (era, c) in refs.items() if era == "ancient")
    c17 = sum(c for name, (era, c) in refs.items() if era in ("17c", "17c-18c"))
    c18 = sum(c for name, (era, c) in refs.items() if era == "18c")
    modern = c17 + c18
    if ancient + modern < 3:
        continue
    ratio = ancient / modern if modern > 0 else 999
    print(f"{label[:49]:<50} {date:>5} {cat[:9]:<10} {ancient:>8} {c17:>6} {c18:>6} {ratio:>10.2f}")

# ============================================================
# 2. AGGREGATE BY PERIOD
# ============================================================
print("\n" + "=" * 100)
print("2. AGGREGATE ANCIENT/MODERN RATIO BY PERIOD")
print("=" * 100)

for p_start, p_end, plabel in [(1580, 1700, "1580-1700"), (1700, 1750, "1700-1750"),
                                 (1750, 1790, "1750-1790"), (1790, 1820, "1790-1820")]:
    ancient_total = 0
    modern_total = 0
    scripture_total = 0
    n = 0
    for label, date, cat, total, refs in ref_data:
        if p_start <= date < p_end:
            ancient_total += sum(c for name, (era, c) in refs.items() if era == "ancient")
            modern_total += sum(c for name, (era, c) in refs.items() if era in ("17c", "17c-18c", "18c"))
            scripture_total += sum(c for name, (era, c) in refs.items() if era in ("scripture", "patristic"))
            n += 1
    if n == 0:
        continue
    ratio = ancient_total / modern_total if modern_total > 0 else 999
    print(f"  {plabel}: {n} texts  ancient={ancient_total}  modern={modern_total}  "
          f"scripture={scripture_total}  A/M={ratio:.2f}")

# ============================================================
# 3. WHO GETS CITED MOST, BY PERIOD?
# ============================================================
print("\n" + "=" * 100)
print("3. TOP CITED AUTHORITIES BY PERIOD")
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
        era = next(e for n, (e, c) in list(refs.items()) if True)  # get era
        # look up era properly
        for pattern, (e, n) in authorities.items():
            if n == name:
                era = e
                break
        print(f"    {name:<20} {count:>6}  ({era})")

# ============================================================
# 4. BY CATEGORY — do different genres cite differently?
# ============================================================
print("\n" + "=" * 100)
print("4. CITATION PATTERNS BY GENRE")
print("=" * 100)

for cat in ["fiction", "conduct", "history", "philosophy", "science", "medical", "poetry", "philology"]:
    name_counts = Counter()
    ancient_total = 0
    modern_total = 0
    n = 0
    for label, date, cat2, total, refs in ref_data:
        if cat2 != cat:
            continue
        n += 1
        for name, (era, count) in refs.items():
            name_counts[name] += count
            if era == "ancient":
                ancient_total += count
            elif era in ("17c", "17c-18c", "18c"):
                modern_total += count
    if n == 0:
        continue
    ratio = ancient_total / modern_total if modern_total > 0 else 999
    print(f"\n  {cat.upper()} ({n} texts): A/M ratio = {ratio:.2f}")
    for name, count in name_counts.most_common(10):
        for pattern, (e, nm) in authorities.items():
            if nm == name:
                era = e
                break
        print(f"    {name:<20} {count:>6}  ({era})")

# ============================================================
# 5. SPECIFIC SHIFTS — Locke, Newton, Aristotle, Scripture
# ============================================================
print("\n" + "=" * 100)
print("5. KEY AUTHORITY TRAJECTORIES — per-decade density")
print("=" * 100)

key_names = ["Aristotle", "Cicero", "Homer", "Locke", "Newton",
             "Bacon", "Milton", "Shakespeare", "Hume", "Rousseau",
             "Pope", "Voltaire", "Scripture", "Augustine", "Hobbes"]

print(f"{'Decade':<8}", end="")
for name in key_names:
    print(f" {name[:8]:>9}", end="")
print()
print("-" * (8 + 9 * len(key_names)))

for dec_start in range(1580, 1820, 10):
    dec_end = dec_start + 10
    matching = [(refs, total) for label, date, cat, total, refs in ref_data
                if dec_start <= date < dec_end]
    if not matching:
        continue
    total_words = sum(t for _, t in matching)
    print(f"{dec_start}s  ", end="")
    for name in key_names:
        count = sum(refs.get(name, (None, 0))[1] for refs, _ in matching)
        rate = count / total_words * 100000  # per 100k words
        if rate > 0.5:
            print(f" {rate:>9.1f}", end="")
        else:
            print(f" {'·':>9}", end="")
    print()

# ============================================================
# 6. SCRIPTURE vs PHILOSOPHY vs EMPIRICISM
# ============================================================
print("\n" + "=" * 100)
print("6. EPISTEMIC AUTHORITY SHIFT — Scripture vs Classical Philosophy vs Empirical Science")
print("=" * 100)

scripture_names = {"Scripture", "Moses", "St Paul", "Augustine"}
classical_names = {"Aristotle", "Plato", "Cicero", "Seneca", "Plutarch",
                    "Epictetus", "Lucretius", "Homer", "Virgil", "Horace"}
empirical_names = {"Locke", "Newton", "Boyle", "Bacon", "Descartes", "Hobbes"}

print(f"\n{'Decade':<8} {'Scripture':>10} {'Classical':>10} {'Empirical':>10} {'Dominant':>12}")
print("-" * 55)
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
    print(f"{dec_start}s   {s_rate:>10.1f} {c_rate:>10.1f} {e_rate:>10.1f} {dominant:>12}")
