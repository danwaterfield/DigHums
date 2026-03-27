"""Quick survey of the conduct book corpus against the Burney fiction."""
import re
from collections import Counter
from pathlib import Path

ROOT = Path("/Users/danielwaterfield/Documents/DigHums")
CONDUCT = ROOT / "nonfiction/conduct"

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

def tokenize(text):
    return re.findall(r"[a-zA-Z']+", text.lower())

# Load all conduct texts
conduct_texts = {}
for author_dir in sorted(CONDUCT.iterdir()):
    if not author_dir.is_dir():
        continue
    for txt in sorted(author_dir.glob("*.txt")):
        label = f"{author_dir.name}: {txt.stem}"
        conduct_texts[label] = load(txt)

# Load Burney fiction
burney_texts = {
    "BURNEY: Evelina": load(ROOT / "FrancesBurney/Evelina.txt"),
    "BURNEY: Cecilia": "\n".join(load(ROOT / f"FrancesBurney/CeciliaVol{i}.txt") for i in range(1,4)),
    "BURNEY: Camilla": load(ROOT / "FrancesBurney/Camilla.txt"),
    "BURNEY: Wanderer": "\n".join(load(ROOT / f"FrancesBurney/TheWandererVol{i}.txt") for i in range(1,6)),
}

# Also load Young for comparison
young_texts = {
    "YOUNG: Lucy Watson": load(ROOT / "ArthurYoung/AdventuresOfMissLucyWatson.txt"),
    "YOUNG: Julia Benson": load(ROOT / "ArthurYoung/HistoryOfJuliaBenson.txt"),
}

# ============================================================
# 1. KEY THEME CLUSTERS — which conduct books share Burney's concerns?
# ============================================================
print("=" * 90)
print("1. THEMATIC VOCABULARY PROFILES (per 10k words)")
print("=" * 90)

themes = {
    "female_education": ["education", "instruction", "learning", "knowledge", "study",
                          "reading", "books", "understanding", "reason", "mind", "improvement"],
    "modesty_delicacy": ["modesty", "delicacy", "reserve", "propriety", "decorum",
                          "bashful", "blush", "retiring", "diffidence", "timid"],
    "marriage_duty": ["marriage", "husband", "wife", "obedience", "submission",
                       "duty", "duties", "conjugal", "matrimony", "wedlock"],
    "independence": ["independence", "independent", "liberty", "freedom", "rights",
                      "equality", "equal", "oppression", "tyranny", "subjection"],
    "sensibility": ["sensibility", "feeling", "feelings", "sentiment", "tender",
                     "tenderness", "compassion", "sympathy", "pity", "heart"],
    "reputation": ["reputation", "character", "honour", "honor", "virtue",
                    "chastity", "innocence", "prudence", "discretion", "censure"],
}

all_texts = {**conduct_texts, **burney_texts, **young_texts}

# Calculate theme rates for all texts
results = {}
for label, text in all_texts.items():
    words = tokenize(text)
    total = len(words)
    if total < 1000:
        continue
    freq = Counter(words)
    rates = {}
    for theme, terms in themes.items():
        count = sum(freq.get(t, 0) for t in terms)
        rates[theme] = count / total * 10000
    results[label] = {"total": total, "rates": rates}

# Print sorted by each theme to find closest matches to Burney
for theme in themes:
    print(f"\n--- {theme.upper()} ---")
    ranked = sorted(results.items(), key=lambda x: -x[1]["rates"][theme])
    for label, data in ranked[:10]:
        marker = " ***" if "BURNEY" in label else " [Y]" if "YOUNG" in label else ""
        print(f"  {data['rates'][theme]:>6.1f}  {label}{marker}")

# ============================================================
# 2. VOCABULARY OVERLAP — which conduct books share most rare words with Burney?
# ============================================================
print("\n" + "=" * 90)
print("2. VOCABULARY OVERLAP — shared rare words with Burney's combined fiction")
print("=" * 90)

burney_combined = " ".join(burney_texts.values())
burney_words = tokenize(burney_combined)
burney_freq = Counter(burney_words)
burney_rare = {w for w, c in burney_freq.items() if len(w) >= 6 and 3 <= c <= 50}

overlaps = []
for label, text in conduct_texts.items():
    words = tokenize(text)
    freq = Counter(words)
    rare = {w for w, c in freq.items() if len(w) >= 6 and 2 <= c <= 30}
    shared = burney_rare & rare
    total = len(words)
    overlaps.append((label, len(shared), len(rare), total))

overlaps.sort(key=lambda x: -x[1])
print(f"\n{'Text':<55} {'Shared':>7} {'Own rare':>9} {'Words':>8}")
print("-" * 85)
for label, shared, own, total in overlaps:
    print(f"{label:<55} {shared:>7} {own:>9} {total:>8,}")

# ============================================================
# 3. DIRECT QUOTATION/REFERENCE — do Burney's novels mention these authors?
# ============================================================
print("\n" + "=" * 90)
print("3. DIRECT REFERENCES — conduct book authors/titles mentioned in Burney")
print("=" * 90)

search_terms = {
    "Gregory": r'\bgregory\b',
    "Fordyce": r'\bfordyce\b',
    "Chapone": r'\bchapone\b',
    "Chesterfield": r'\bchesterfield\b',
    "More (Hannah)": r'\bhannah more\b',
    "Wollstonecraft": r'\bwollstonecraft\b',
    "Locke": r'\blocke\b',
    "Rousseau": r'\brousseau\b',
    "Astell": r'\bastell\b',
    "Montagu": r'\bmontagu\b',
    "Richardson": r'\brichardson\b',
    "conduct": r'\bconduct\b',
    "education": r'\beducation\b',
    "Emile": r'\bemile\b',
    "delicacy": r'\bdelicacy\b',
    "propriety": r'\bpropriety\b',
    "female education": r'female education',
    "improvement of the mind": r'improvement of the mind',
    "duty of a": r'duty of a',
    "accomplishments": r'\baccomplishments?\b',
}

for title, text in burney_texts.items():
    print(f"\n{title}:")
    tl = text.lower()
    for term, pattern in search_terms.items():
        matches = re.findall(pattern, tl)
        if matches:
            print(f"  {term}: {len(matches)} occurrences")

# ============================================================
# 4. CONDUCT BOOK PHRASES IN BURNEY — shared distinctive bigrams
# ============================================================
print("\n" + "=" * 90)
print("4. CONDUCT BOOK PHRASES — distinctive bigrams shared with Burney")
print("=" * 90)

def get_bigrams(text, min_count=3):
    words = tokenize(text)
    bg = Counter(zip(words, words[1:]))
    return {(a, b): c for (a, b), c in bg.items()
            if c >= min_count and len(a) > 3 and len(b) > 3}

stopwords = {"the", "and", "to", "of", "a", "in", "that", "it", "is", "was",
             "i", "he", "she", "her", "his", "my", "for", "not", "but", "with",
             "you", "had", "have", "be", "which", "as", "at", "this", "from",
             "their", "they", "them", "been", "were", "are", "what", "when",
             "than", "will", "would", "could", "should", "may", "might",
             "shall", "upon", "into", "more", "most", "very", "such",
             "some", "only", "every", "your", "our", "said", "who", "whom"}

# Build conduct corpus bigrams
conduct_combined = " ".join(conduct_texts.values())
conduct_bg = get_bigrams(conduct_combined, 5)
burney_bg = get_bigrams(burney_combined, 3)

shared_bg = set(conduct_bg.keys()) & set(burney_bg.keys())
interesting = [(a, b) for a, b in shared_bg
               if a not in stopwords and b not in stopwords]

# Rank by product of rates (distinctive in both)
conduct_total = len(tokenize(conduct_combined))
burney_total = len(tokenize(burney_combined))
scored = []
for a, b in interesting:
    c_rate = conduct_bg[(a,b)] / conduct_total * 100000
    b_rate = burney_bg[(a,b)] / burney_total * 100000
    scored.append((a, b, conduct_bg[(a,b)], burney_bg[(a,b)], c_rate * b_rate))

scored.sort(key=lambda x: -x[4])
print(f"\n{'Bigram':<35} {'Conduct':>8} {'Burney':>8}")
print("-" * 55)
for a, b, cc, bc, _ in scored[:40]:
    print(f"{a + ' ' + b:<35} {cc:>8} {bc:>8}")

# ============================================================
# 5. CHRONOLOGICAL SHIFT — how does the conduct vocabulary evolve?
# ============================================================
print("\n" + "=" * 90)
print("5. CHRONOLOGICAL THEME EVOLUTION")
print("=" * 90)

dated_texts = [
    (1688, "Halifax: AdviceToADaughter"),
    (1693, "JohnLocke: SomeThoughtsConcerningEducation"),
    (1694, "MaryAstell: ASeriousProposalToTheLadies"),
    (1700, "MaryAstell: SomeReflectionsUponMarriage"),
    (1704, "WilliamDarrell: TheGentlemanInstructed"),
    (1714, "Sophia: WomanNotInferiorToMan"),
    (1720, "AdamPetrie: RulesOfGoodDeportment"),
    (1723, "JonathanSwift: LetterToAVeryYoungLady"),
    (1740, "JamesFordyce: SermonsToYoungWomen"),
    (1741, "SamuelRichardson: FamiliarLetters"),
    (1744, "ElizaHaywood: FemaleSpectatorVol1"),
    (1753, "JamesNelson: EssayOnTheGovernmentOfChildren"),
    (1761, "SarahPennington: UnfortunateMotherAdvice"),
    (1762, "JeanJacquesRousseau: Emile"),
    (1774, "JohnGregory: AFathersLegacy"),
    (1777, "HesterChapone: LetterToANewMarriedLady"),
    (1781, "VicesimusKnox: LiberalEducation"),
    (1787, "MaryWollstonecraft: ThoughtsOnTheEducationOfDaughters"),
    (1788, "HannahMore: ThoughtsOnMannersOfTheGreat"),
    (1789, "JohnBennett: LettersToAYoungLady"),
    (1790, "CatharineMacaulay: LettersOnEducation"),
    (1792, "MaryWollstonecraft: VindicationOfTheRightsOfWoman"),
    (1795, "MariaEdgeworth: LettersForLiteraryLadies"),
    (1797, "ThomasGisborne: DutiesOfTheFemaleSex"),
    (1798, "PriscillaWakefield: ReflectionsOnTheFemaleSex"),
    (1798, "MaryHays: AppealToTheMenOfGreatBritain"),
    (1799, "HannahMore: StricturesOnFemaleEducation"),
    (1809, "HannahMore: CoelebsInSearchOfAWife"),
]

key_themes = ["female_education", "independence", "modesty_delicacy", "sensibility"]

# Split into pre-1770 and post-1770 (roughly pre/post Burney)
pre = [(y, l) for y, l in dated_texts if y < 1770]
post = [(y, l) for y, l in dated_texts if y >= 1770]

for theme in key_themes:
    pre_rates = []
    post_rates = []
    for year, label in dated_texts:
        if label in results:
            rate = results[label]["rates"][theme]
            if year < 1770:
                pre_rates.append(rate)
            else:
                post_rates.append(rate)
    pre_avg = sum(pre_rates) / len(pre_rates) if pre_rates else 0
    post_avg = sum(post_rates) / len(post_rates) if post_rates else 0
    shift = ((post_avg - pre_avg) / pre_avg * 100) if pre_avg else 0
    print(f"  {theme:<22}  pre-1770: {pre_avg:>5.1f}  post-1770: {post_avg:>5.1f}  shift: {shift:>+6.1f}%")

# ============================================================
# 6. CLOSEST CONDUCT BOOK TO EACH BURNEY NOVEL
# ============================================================
print("\n" + "=" * 90)
print("6. NEAREST CONDUCT BOOK TO EACH BURNEY NOVEL (cosine similarity on theme rates)")
print("=" * 90)

import math

def cosine(a, b, keys):
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    mag_a = math.sqrt(sum(a.get(k, 0)**2 for k in keys))
    mag_b = math.sqrt(sum(b.get(k, 0)**2 for k in keys))
    if mag_a == 0 or mag_b == 0:
        return 0
    return dot / (mag_a * mag_b)

theme_keys = list(themes.keys())

for burney_label in burney_texts:
    bl = "BURNEY: " + burney_label.split(": ")[1] if ": " in burney_label else burney_label
    if bl not in results:
        continue
    b_rates = results[bl]["rates"]
    sims = []
    for c_label in conduct_texts:
        if c_label not in results:
            continue
        c_rates = results[c_label]["rates"]
        sim = cosine(b_rates, c_rates, theme_keys)
        sims.append((c_label, sim))
    sims.sort(key=lambda x: -x[1])
    print(f"\n{bl}:")
    for label, sim in sims[:5]:
        print(f"  {sim:.3f}  {label}")

# Also do Young
for young_label in young_texts:
    yl = "YOUNG: " + young_label.split(": ")[1] if ": " in young_label else young_label
    if yl not in results:
        continue
    y_rates = results[yl]["rates"]
    sims = []
    for c_label in conduct_texts:
        if c_label not in results:
            continue
        c_rates = results[c_label]["rates"]
        sim = cosine(y_rates, c_rates, theme_keys)
        sims.append((c_label, sim))
    sims.sort(key=lambda x: -x[1])
    print(f"\n{yl}:")
    for label, sim in sims[:5]:
        print(f"  {sim:.3f}  {label}")
