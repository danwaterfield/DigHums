"""Full corpus analysis: sociability, state, behavioral norms across fiction + non-fiction."""
import re
import math
from collections import Counter
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
    return text.replace("\xad", "").replace("ſ", "s")

def tokenize(text):
    return re.findall(r"[a-zA-Z']+", text.lower())

# ============================================================
# LOAD ALL TEXTS WITH DATES
# ============================================================
# Format: (date, label, category, paths)
catalog = []

# FICTION
fiction_entries = [
    (1768, "Young: Lucy Watson", "fiction", [ROOT/"ArthurYoung/AdventuresOfMissLucyWatson.txt"]),
    (1784, "Young: Julia Benson", "fiction", [ROOT/"ArthurYoung/HistoryOfJuliaBenson.txt"]),
    (1778, "Burney: Evelina", "fiction", [ROOT/"FrancesBurney/Evelina.txt"]),
    (1782, "Burney: Cecilia", "fiction", [ROOT/f"FrancesBurney/CeciliaVol{i}.txt" for i in range(1,4)]),
    (1796, "Burney: Camilla", "fiction", [ROOT/"FrancesBurney/Camilla.txt"]),
    (1814, "Burney: Wanderer", "fiction", [ROOT/f"FrancesBurney/TheWandererVol{i}.txt" for i in range(1,6)]),
]

# CONDUCT
conduct_dir = ROOT / "nonfiction/conduct"
conduct_entries = [
    (1688, "Halifax: Advice to Daughter", "conduct"),
    (1693, "JohnLocke: Education", "conduct"),
    (1694, "MaryAstell: Serious Proposal", "conduct"),
    (1700, "MaryAstell: Reflections Marriage", "conduct"),
    (1704, "WilliamDarrell: Gentleman Instructed", "conduct"),
    (1723, "JonathanSwift: Letter Young Lady", "conduct"),
    (1739, "Sophia: Woman Not Inferior", "conduct"),
    (1740, "JamesFordyce: Sermons", "conduct"),
    (1741, "SamuelRichardson: Familiar Letters", "conduct"),
    (1744, "ElizaHaywood: Female Spectator", "conduct"),
    (1753, "JamesNelson: Government Children", "conduct"),
    (1761, "SarahPennington: Mother's Advice", "conduct"),
    (1762, "JeanJacquesRousseau: Emile", "conduct"),
    (1763, "LadyMontagu: Embassy Letters", "conduct"),
    (1774, "JohnGregory: Father's Legacy", "conduct"),
    (1777, "HesterChapone: New Married Lady", "conduct"),
    (1781, "VicesimusKnox: Liberal Education", "conduct"),
    (1783, "ThomasDay: Sandford Merton", "conduct"),
    (1787, "Wollstonecraft: Education Daughters", "conduct"),
    (1788, "More: Manners of Great", "conduct"),
    (1789, "JohnBennett: Letters Young Lady", "conduct"),
    (1790, "Macaulay: Letters Education", "conduct"),
    (1792, "Wollstonecraft: Vindication", "conduct"),
    (1795, "Edgeworth: Literary Ladies", "conduct"),
    (1797, "Gisborne: Duties Female Sex", "conduct"),
    (1798, "Wakefield: Female Sex", "conduct"),
    (1798, "MaryHays: Appeal", "conduct"),
    (1798, "Edgeworth: Practical Ed Vol1", "conduct"),
    (1799, "More: Strictures", "conduct"),
    (1809, "More: Coelebs", "conduct"),
]

# STATE/THEOLOGY
state_entries = [
    (1651, "Hobbes: Leviathan", "philosophy", ROOT/"nonfiction/medical/passions/Hobbes_Leviathan.txt"),
    (1656, "More: Enthusiasmus", "anti-enthusiasm", None),  # skip if missing
    (1659, "Stillingfleet: Irenicum", "theology", ROOT/"nonfiction/state/theology/Stillingfleet_Irenicum.txt"),
    (1675, "Spinoza: Ethics", "philosophy", ROOT/"nonfiction/medical/passions/Spinoza_Ethics.txt"),
    (1690, "Locke: Essay Bk IV", "anti-enthusiasm", ROOT/"nonfiction/state/anti-enthusiasm/Locke_EssayConcerningHumanUnderstanding_Vol2.txt"),
    (1694, "Tillotson: Sermons", "theology", ROOT/"nonfiction/state/theology/Tillotson_Sermons_Vol1.txt"),
    (1698, "Ward: London Spy", "urban", ROOT/"nonfiction/state/urban/Ward_LondonSpy.txt"),
    (1699, "Burnet: 39 Articles", "theology", ROOT/"nonfiction/state/theology/Burnet_ThirtyNineArticles.txt"),
    (1699, "Woodward: Reformation Manners", "reformation", ROOT/"nonfiction/state/reformation/Woodward_ReformationOfManners.txt"),
    (1700, "Brown: Amusements", "urban", ROOT/"nonfiction/state/urban/Brown_AmusementsSeriousAndComical.txt"),
    (1704, "Swift: Tale of Tub", "anti-enthusiasm", ROOT/"nonfiction/state/anti-enthusiasm/Swift_TaleOfATub.txt"),
    (1704, "Clarke: Boyle Lectures", "theology", ROOT/"nonfiction/state/theology/Clarke_BoyleLectures.txt"),
    (1711, "Shaftesbury: Characteristics", "politeness", ROOT/"nonfiction/state/politeness/Shaftesbury_Characteristics_Vol1.txt"),
    (1725, "Hutcheson: Beauty Virtue", "politeness", ROOT/"nonfiction/state/politeness/Hutcheson_BeautyAndVirtue.txt"),
    (1729, "Law: Serious Call", "anti-enthusiasm", ROOT/"nonfiction/state/anti-enthusiasm/Law_SeriousCall.txt"),
    (1741, "Hume: Essays", "politeness", ROOT/"nonfiction/state/politeness/Hume_EssaysMoralAndPolitical.txt"),
    (1749, "Lavington: Enthusiasm Methodists", "anti-enthusiasm", ROOT/"nonfiction/state/anti-enthusiasm/Lavington_EnthusiasmOfMethodists.txt"),
    (1751, "Hume: Enquiry Morals", "politeness", ROOT/"nonfiction/state/politeness/Hume_EnquiryConcerningMorals.txt"),
    (1757, "Burke: Sublime Beautiful", "politeness", ROOT/"nonfiction/state/politeness/Burke_SublimeAndBeautiful.txt"),
    (1759, "Smith: Moral Sentiments", "politeness", ROOT/"nonfiction/state/politeness/Smith_TheoryOfMoralSentiments.txt"),
]

# MEDICAL
medical_entries = [
    (1586, "Bright: Melancholy", "medical", ROOT/"nonfiction/medical/melancholy/Bright_TreatiseOfMelancholy.txt"),
    (1621, "Burton: Anatomy Melancholy", "medical", ROOT/"nonfiction/medical/melancholy/Burton_AnatomyOfMelancholy.txt"),
    (1711, "Mandeville: Hypochondriack", "medical", ROOT/"nonfiction/medical/nervous/Mandeville_HypochondriackPassions.txt"),
    (1724, "Cheyne: Health Long Life", "medical", ROOT/"nonfiction/medical/nervous/Cheyne_HealthAndLongLife.txt"),
    (1733, "Cheyne: English Malady", "medical", ROOT/"nonfiction/medical/nervous/Cheyne_EnglishMalady.txt"),
    (1758, "Battie: Treatise Madness", "medical", ROOT/"nonfiction/medical/madness/Battie_TreatiseOnMadness.txt"),
    (1758, "Monro: Remarks Battie", "medical", ROOT/"nonfiction/medical/madness/Monro_RemarksOnBattie.txt"),
    (1765, "Whytt: Nervous Disorders", "medical", ROOT/"nonfiction/medical/nervous/Whytt_NervousDisorders.txt"),
    (1777, "Cullen: Practice Physic", "medical", ROOT/"nonfiction/medical/nervous/Cullen_PracticeOfPhysic.txt"),
    (1777, "Howard: State Prisons", "institutional", ROOT/"nonfiction/medical/institutional/Howard_StateOfPrisons.txt"),
    (1782, "Arnold: Insanity Vol1", "medical", ROOT/"nonfiction/medical/madness/Arnold_InsanityVol1.txt"),
    (1798, "Crichton: Mental Derangement", "medical", ROOT/"nonfiction/medical/madness/Crichton_MentalDerangement.txt"),
    (1798, "Haslam: Insanity", "medical", ROOT/"nonfiction/medical/madness/Haslam_ObservationsOnInsanity.txt"),
    (1807, "Trotter: Nervous Temperament", "medical", ROOT/"nonfiction/medical/nervous/Trotter_NervousTemperament.txt"),
]

# ============================================================
# LOAD TEXTS
# ============================================================
texts = {}  # label -> (date, category, words_list)

for date, label, cat, paths in fiction_entries:
    combined = "\n".join(load(p) for p in paths)
    texts[label] = (date, cat, tokenize(combined))

# Load conduct texts by scanning directories
conduct_loaded = {}
for author_dir in sorted(conduct_dir.iterdir()):
    if not author_dir.is_dir():
        continue
    for txt in sorted(author_dir.glob("*.txt")):
        key = f"{author_dir.name}: {txt.stem}"
        conduct_loaded[key] = tokenize(load(txt))

# Map conduct entries to loaded texts (fuzzy match by author dir name)
for date, label, cat in conduct_entries:
    author_part = label.split(":")[0].strip()
    for key, words in conduct_loaded.items():
        if author_part.lower().replace(" ", "") in key.lower().replace(" ", ""):
            if label not in texts:
                texts[label] = (date, cat, words)
            break

# Load state and medical
for date, label, cat, path in state_entries + medical_entries:
    if path and path.exists():
        texts[label] = (date, cat, tokenize(load(path)))

print(f"Loaded {len(texts)} texts total\n")

# ============================================================
# THEME LEXICONS
# ============================================================
themes = {
    "sociability": ["sociable", "sociability", "society", "social", "company", "companion",
                     "companions", "conversation", "converse", "commerce", "intercourse",
                     "civility", "civil", "polite", "politeness", "agreeable", "amiable",
                     "affable", "obliging", "complaisant", "complaisance"],
    "state_authority": ["state", "government", "law", "laws", "magistrate", "parliament",
                         "authority", "sovereign", "commonwealth", "constitution", "civil",
                         "public", "nation", "subjects", "obedience", "order", "regulation",
                         "police", "statute", "jurisdiction"],
    "conformity": ["conformity", "conform", "regular", "regularity", "proper", "propriety",
                    "decent", "decency", "decorum", "becoming", "seemly", "correct",
                    "standard", "normal", "common", "ordinary", "customary", "usual",
                    "moderate", "moderation", "temperance", "temperate"],
    "deviance": ["mad", "madness", "lunatic", "lunacy", "distracted", "distraction",
                  "insane", "insanity", "deranged", "derangement", "disorder", "disordered",
                  "frenzy", "frantic", "wild", "extravagant", "eccentric", "odd",
                  "singular", "strange", "peculiar", "irregular", "enthusiast", "enthusiasm",
                  "fanatic", "fanaticism", "melancholy", "spleen", "vapours", "hysteric"],
    "nervous_sensibility": ["nerve", "nerves", "nervous", "sensibility", "sensible",
                             "sensitive", "feeling", "feelings", "impression", "impressions",
                             "delicate", "delicacy", "refined", "refinement", "susceptible",
                             "tender", "tenderness", "exquisite", "acute", "irritable",
                             "irritability"],
    "reason_judgment": ["reason", "reasonable", "rational", "judgment", "understanding",
                         "intellect", "intelligence", "discernment", "prudence", "prudent",
                         "wisdom", "wise", "sense", "senses", "reflection", "thinking",
                         "thought", "mind", "comprehension"],
    "self_regulation": ["restraint", "self-command", "self-control", "composure", "command",
                         "govern", "government", "regulate", "regulation", "control",
                         "moderate", "moderation", "patience", "patient", "forbearance",
                         "endurance", "fortitude", "resolution", "firmness", "steady"],
    "overwhelm_excess": ["overwhelm", "overwhelmed", "excess", "excessive", "violent",
                          "violence", "tumult", "confusion", "disorder", "agitation",
                          "agitated", "disturbance", "disturbed", "uproar", "noise",
                          "crowd", "crowded", "throng", "bustle", "hurry", "fatigue",
                          "exhaustion", "oppression", "oppressed", "burden", "burdened"],
    "attention": ["attention", "attentive", "attentively", "notice", "observe", "observation",
                   "regard", "heed", "mindful", "watchful", "vigilant", "careful",
                   "carefulness", "negligent", "negligence", "inattention", "inattentive",
                   "distracted", "distraction", "absent", "absence", "abstracted"],
    "urban_environment": ["city", "town", "street", "streets", "london", "crowd", "noise",
                           "smoke", "air", "buildings", "houses", "shops", "coach", "carriages",
                           "traffic", "pavement", "walk", "public", "market", "exchange"],
}

# ============================================================
# 1. CHRONOLOGICAL THEME EVOLUTION
# ============================================================
print("=" * 100)
print("1. CHRONOLOGICAL THEME EVOLUTION (per 10k words, sorted by date)")
print("=" * 100)

sorted_texts = sorted(texts.items(), key=lambda x: x[1][0])

# Print header
key_themes = ["sociability", "conformity", "deviance", "nervous_sensibility",
              "self_regulation", "overwhelm_excess", "attention"]
header = f"{'Date':<6} {'Text':<40} {'Cat':<12}"
for t in key_themes:
    header += f" {t[:8]:>9}"
print(header)
print("-" * len(header))

for label, (date, cat, words) in sorted_texts:
    total = len(words)
    if total < 1000:
        continue
    freq = Counter(words)
    row = f"{date:<6} {label[:39]:<40} {cat[:11]:<12}"
    for theme in key_themes:
        count = sum(freq.get(t, 0) for t in themes[theme])
        rate = count / total * 10000
        row += f" {rate:>9.1f}"
    print(row)

# ============================================================
# 2. AGGREGATED BY CATEGORY AND PERIOD
# ============================================================
print("\n" + "=" * 100)
print("2. CATEGORY × PERIOD AGGREGATES (per 10k words)")
print("=" * 100)

periods = [(1580, 1700, "1580-1700"), (1700, 1760, "1700-1760"), (1760, 1820, "1760-1820")]
categories = ["theology", "anti-enthusiasm", "politeness", "conduct", "fiction", "medical", "urban"]

for theme in key_themes:
    print(f"\n--- {theme.upper()} ---")
    header = f"{'Category':<16}"
    for _, _, plabel in periods:
        header += f" {plabel:>12}"
    print(header)
    for cat in categories:
        row = f"{cat:<16}"
        for p_start, p_end, plabel in periods:
            matching = [(label, date, words) for label, (date, c, words) in texts.items()
                        if c == cat and p_start <= date < p_end and len(words) > 1000]
            if matching:
                total_words = sum(len(w) for _, _, w in matching)
                total_hits = 0
                for _, _, words in matching:
                    freq = Counter(words)
                    total_hits += sum(freq.get(t, 0) for t in themes[theme])
                rate = total_hits / total_words * 10000
                row += f" {rate:>12.1f}"
            else:
                row += f" {'—':>12}"
        print(row)

# ============================================================
# 3. FICTION vs NON-FICTION: DOES FICTION LEAD OR FOLLOW?
# ============================================================
print("\n" + "=" * 100)
print("3. FICTION vs NON-FICTION: WHO LEADS? (decade averages)")
print("=" * 100)

decades = range(1690, 1820, 10)

for theme in ["sociability", "nervous_sensibility", "deviance", "self_regulation",
              "attention", "overwhelm_excess", "conformity"]:
    print(f"\n--- {theme.upper()} ---")
    print(f"{'Decade':<8} {'Fiction':>10} {'Conduct':>10} {'Medical':>10} {'Theology':>10} {'Polite':>10}")
    for dec_start in decades:
        dec_end = dec_start + 10
        row = f"{dec_start}s   "
        for cat in ["fiction", "conduct", "medical", "theology", "politeness"]:
            matching = [(label, words) for label, (date, c, words) in texts.items()
                        if c == cat and dec_start <= date < dec_end and len(words) > 1000]
            if matching:
                total_w = sum(len(w) for _, w in matching)
                total_h = sum(sum(1 for w in words if w in themes[theme]) for _, words in matching)
                rate = total_h / total_w * 10000
                row += f" {rate:>9.1f}"
            else:
                row += f" {'—':>9}"
        print(row)

# ============================================================
# 4. SOCIABILITY-DEVIANCE CORRELATION
# ============================================================
print("\n" + "=" * 100)
print("4. SOCIABILITY vs DEVIANCE — per text (looking for inverse correlation)")
print("=" * 100)

soc_dev = []
for label, (date, cat, words) in sorted_texts:
    total = len(words)
    if total < 3000:
        continue
    freq = Counter(words)
    soc = sum(freq.get(t, 0) for t in themes["sociability"]) / total * 10000
    dev = sum(freq.get(t, 0) for t in themes["deviance"]) / total * 10000
    soc_dev.append((label, date, cat, soc, dev))

# Sort by sociability
soc_dev.sort(key=lambda x: -x[3])
print(f"\n{'Text':<45} {'Date':>5} {'Cat':<12} {'Social':>8} {'Devian':>8} {'Ratio':>8}")
print("-" * 90)
for label, date, cat, soc, dev in soc_dev[:15]:
    ratio = soc / dev if dev > 0 else 999
    print(f"{label[:44]:<45} {date:>5} {cat:<12} {soc:>8.1f} {dev:>8.1f} {ratio:>8.1f}")
print("...")
for label, date, cat, soc, dev in soc_dev[-15:]:
    ratio = soc / dev if dev > 0 else 999
    print(f"{label[:44]:<45} {date:>5} {cat:<12} {soc:>8.1f} {dev:>8.1f} {ratio:>8.1f}")

# ============================================================
# 5. ATTENTION VOCABULARY — crucial for neurodivergence argument
# ============================================================
print("\n" + "=" * 100)
print("5. ATTENTION & INATTENTION — the neurodivergence signal")
print("=" * 100)

attention_pos = ["attention", "attentive", "attentively", "heed", "mindful",
                  "watchful", "vigilant", "careful", "carefulness", "observe",
                  "observation", "notice", "regard"]
attention_neg = ["inattention", "inattentive", "negligent", "negligence",
                  "distracted", "distraction", "absent", "abstracted",
                  "heedless", "careless", "thoughtless", "unmindful",
                  "wandering", "rambling", "desultory"]

print(f"\n{'Text':<45} {'Date':>5} {'Attn+':>7} {'Attn-':>7} {'Ratio':>7}")
print("-" * 75)
for label, (date, cat, words) in sorted_texts:
    total = len(words)
    if total < 3000:
        continue
    freq = Counter(words)
    pos = sum(freq.get(t, 0) for t in attention_pos) / total * 10000
    neg = sum(freq.get(t, 0) for t in attention_neg) / total * 10000
    if pos + neg > 3.0:  # Only show texts where attention is a notable theme
        ratio = neg / pos if pos > 0 else 999
        print(f"{label[:44]:<45} {date:>5} {pos:>7.1f} {neg:>7.1f} {ratio:>7.2f}")

# ============================================================
# 6. THE RATCHET — sociability demand vs overwhelm across time
# ============================================================
print("\n" + "=" * 100)
print("6. THE RATCHET — sociability expectations vs overwhelm/excess (by decade)")
print("=" * 100)

print(f"{'Decade':<8} {'Sociab':>8} {'Overwh':>8} {'Gap':>8} {'Conform':>8} {'Devianc':>8}")
print("-" * 50)
for dec_start in range(1650, 1820, 10):
    dec_end = dec_start + 10
    matching = [(words, cat) for label, (date, cat, words) in texts.items()
                if dec_start <= date < dec_end and len(words) > 1000]
    if not matching:
        continue
    total_w = sum(len(w) for w, _ in matching)
    rates = {}
    for theme in ["sociability", "overwhelm_excess", "conformity", "deviance"]:
        total_h = 0
        for words, _ in matching:
            freq = Counter(words)
            total_h += sum(freq.get(t, 0) for t in themes[theme])
        rates[theme] = total_h / total_w * 10000

    gap = rates["sociability"] - rates["overwhelm_excess"]
    print(f"{dec_start}s   {rates['sociability']:>8.1f} {rates['overwhelm_excess']:>8.1f} "
          f"{gap:>8.1f} {rates['conformity']:>8.1f} {rates['deviance']:>8.1f}")

# ============================================================
# 7. KEY PASSAGES — "attention" in medical texts
# ============================================================
print("\n" + "=" * 100)
print("7. KEY PASSAGES — 'attention' in medical/psychological texts")
print("=" * 100)

attention_re = re.compile(r'(?:in)?attention|(?:in)?attentive|distract(?:ed|ion)', re.IGNORECASE)

for label, (date, cat, _) in sorted_texts:
    if cat not in ("medical", "anti-enthusiasm", "philosophy"):
        continue
    # Reload raw text for passage extraction
    path = None
    for d, l, c, p in state_entries + medical_entries:
        if l == label and p and p.exists():
            path = p
            break
    if not path:
        continue
    raw = load(path)
    paras = re.split(r'\n\s*\n', raw)
    hits = []
    for p in paras:
        if attention_re.search(p) and len(p) > 80:
            clean = " ".join(p.split())
            if len(clean) > 500:
                clean = clean[:500] + "..."
            hits.append(clean)
    if hits:
        print(f"\n{label} ({date}) — {len(hits)} passages mentioning attention:")
        for h in hits[:2]:
            print(f"  >>> {h}")
            print()
