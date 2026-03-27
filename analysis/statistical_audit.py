"""Statistical audit: check for fallacies in the corpus analysis."""
import re
import math
from collections import Counter
from pathlib import Path
import random

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
    text = text.lower()
    phrases = {
        "self-command": "SELF_COMMAND", "self command": "SELF_COMMAND",
        "good breeding": "GOOD_BREEDING", "good-breeding": "GOOD_BREEDING",
        "fellow feeling": "FELLOW_FEELING", "fellow-feeling": "FELLOW_FEELING",
        "animal spirits": "ANIMAL_SPIRITS",
    }
    for p, t in phrases.items():
        text = text.replace(p, f" {t} ")
    return re.findall(r"[a-zA-Z_']+", text)

# Load texts with metadata
texts = {}

# Fiction
fiction_files = [
    (1768, "Young: Lucy Watson", "fiction", [ROOT/"ArthurYoung/AdventuresOfMissLucyWatson.txt"]),
    (1784, "Young: Julia Benson", "fiction", [ROOT/"ArthurYoung/HistoryOfJuliaBenson.txt"]),
    (1778, "Burney: Evelina", "fiction", [ROOT/"FrancesBurney/Evelina.txt"]),
    (1782, "Burney: Cecilia", "fiction", [ROOT/f"FrancesBurney/CeciliaVol{i}.txt" for i in range(1,4)]),
    (1796, "Burney: Camilla", "fiction", [ROOT/"FrancesBurney/Camilla.txt"]),
    (1814, "Burney: Wanderer", "fiction", [ROOT/f"FrancesBurney/TheWandererVol{i}.txt" for i in range(1,6)]),
]
for date, label, cat, paths in fiction_files:
    combined = "\n".join(load(p) for p in paths)
    texts[label] = (date, cat, tokenize(combined))

# Conduct
conduct_dir = ROOT / "nonfiction/conduct"
conduct_dates = {
    "MarquessOfHalifax": 1688, "JohnLocke": 1693, "MaryAstell": 1694,
    "WilliamDarrell": 1704, "JonathanSwift": 1723, "Sophia": 1739,
    "JamesFordyce": 1740, "SamuelRichardson": 1741, "ElizaHaywood": 1744,
    "JamesNelson": 1753, "SarahPennington": 1761, "JeanJacquesRousseau": 1762,
    "LadyMaryWortleyMontagu": 1763, "JohnGregory": 1774, "HesterChapone": 1777,
    "LordChesterfield": 1774, "VicesimusKnox": 1781, "ThomasDay": 1783,
    "AdamPetrie": 1720, "JohnBennett": 1789,
    "MaryWollstonecraft": 1792, "CatharineMacaulay": 1790,
    "MariaEdgeworth": 1795, "ThomasGisborne": 1797,
    "PriscillaWakefield": 1798, "MaryHays": 1798,
    "HannahMore": 1799, "DanielDefoe": 1697,
}
for author_dir in sorted(conduct_dir.iterdir()):
    if not author_dir.is_dir():
        continue
    for txt in sorted(author_dir.glob("*.txt")):
        date = conduct_dates.get(author_dir.name, 1750)
        label = f"{author_dir.name}: {txt.stem}"
        texts[label] = (date, "conduct", tokenize(load(txt)))

# State + Medical
state_med = [
    (1586, "Bright: Melancholy", "medical", ROOT/"nonfiction/medical/melancholy/Bright_TreatiseOfMelancholy.txt"),
    (1621, "Burton: Anatomy Melancholy", "medical", ROOT/"nonfiction/medical/melancholy/Burton_AnatomyOfMelancholy.txt"),
    (1651, "Hobbes: Leviathan", "philosophy", ROOT/"nonfiction/medical/passions/Hobbes_Leviathan.txt"),
    (1659, "Stillingfleet: Irenicum", "theology", ROOT/"nonfiction/state/theology/Stillingfleet_Irenicum.txt"),
    (1675, "Spinoza: Ethics", "philosophy", ROOT/"nonfiction/medical/passions/Spinoza_Ethics.txt"),
    (1690, "Locke: Essay Bk IV", "anti-enthusiasm", ROOT/"nonfiction/state/anti-enthusiasm/Locke_EssayConcerningHumanUnderstanding_Vol2.txt"),
    (1694, "Tillotson: Sermons", "theology", ROOT/"nonfiction/state/theology/Tillotson_Sermons_Vol1.txt"),
    (1698, "Ward: London Spy", "urban", ROOT/"nonfiction/state/urban/Ward_LondonSpy.txt"),
    (1699, "Burnet: 39 Articles", "theology", ROOT/"nonfiction/state/theology/Burnet_ThirtyNineArticles.txt"),
    (1699, "Woodward: Reformation", "reformation", ROOT/"nonfiction/state/reformation/Woodward_ReformationOfManners.txt"),
    (1700, "Brown: Amusements", "urban", ROOT/"nonfiction/state/urban/Brown_AmusementsSeriousAndComical.txt"),
    (1704, "Swift: Tale of Tub", "anti-enthusiasm", ROOT/"nonfiction/state/anti-enthusiasm/Swift_TaleOfATub.txt"),
    (1704, "Clarke: Boyle Lectures", "theology", ROOT/"nonfiction/state/theology/Clarke_BoyleLectures.txt"),
    (1711, "Shaftesbury: Characteristics", "politeness", ROOT/"nonfiction/state/politeness/Shaftesbury_Characteristics_Vol1.txt"),
    (1725, "Hutcheson: Beauty Virtue", "politeness", ROOT/"nonfiction/state/politeness/Hutcheson_BeautyAndVirtue.txt"),
    (1729, "Law: Serious Call", "anti-enthusiasm", ROOT/"nonfiction/state/anti-enthusiasm/Law_SeriousCall.txt"),
    (1733, "Cheyne: English Malady", "medical", ROOT/"nonfiction/medical/nervous/Cheyne_EnglishMalady.txt"),
    (1741, "Hume: Essays", "politeness", ROOT/"nonfiction/state/politeness/Hume_EssaysMoralAndPolitical.txt"),
    (1751, "Hume: Enquiry Morals", "politeness", ROOT/"nonfiction/state/politeness/Hume_EnquiryConcerningMorals.txt"),
    (1757, "Burke: Sublime Beautiful", "politeness", ROOT/"nonfiction/state/politeness/Burke_SublimeAndBeautiful.txt"),
    (1758, "Battie: Treatise Madness", "medical", ROOT/"nonfiction/medical/madness/Battie_TreatiseOnMadness.txt"),
    (1759, "Smith: Moral Sentiments", "politeness", ROOT/"nonfiction/state/politeness/Smith_TheoryOfMoralSentiments.txt"),
    (1765, "Whytt: Nervous Disorders", "medical", ROOT/"nonfiction/medical/nervous/Whytt_NervousDisorders.txt"),
    (1777, "Cullen: Practice Physic", "medical", ROOT/"nonfiction/medical/nervous/Cullen_PracticeOfPhysic.txt"),
    (1782, "Arnold: Insanity Vol1", "medical", ROOT/"nonfiction/medical/madness/Arnold_InsanityVol1.txt"),
    (1798, "Crichton: Mental Derangement", "medical", ROOT/"nonfiction/medical/madness/Crichton_MentalDerangement.txt"),
    (1798, "Haslam: Insanity", "medical", ROOT/"nonfiction/medical/madness/Haslam_ObservationsOnInsanity.txt"),
    (1807, "Trotter: Nervous Temperament", "medical", ROOT/"nonfiction/medical/nervous/Trotter_NervousTemperament.txt"),
]
for date, label, cat, path in state_med:
    if path.exists():
        texts[label] = (date, cat, tokenize(load(path)))

# Theme bins (v2)
nervous_terms = ["nerve", "nerves", "nervous", "sensibility", "fibres", "fibers",
                  "ANIMAL_SPIRITS", "constitution", "temperament", "irritable",
                  "irritability", "susceptible", "tenderness", "tender",
                  "exquisite", "delicacy", "delicate"]
humoral_terms = ["humour", "humours", "humor", "humors", "phlegmatic", "sanguine",
                  "choleric", "bilious", "blood", "bile", "constitution",
                  "temperament", "temper", "spirits", "complexion", "habit"]
sociability_terms = ["sociable", "society", "company", "conversation", "converse",
                      "civility", "polite", "politeness", "agreeable", "amiable",
                      "affable", "obliging", "complaisant", "complaisance",
                      "GOOD_BREEDING", "GOOD_MANNERS", "GOOD_HUMOUR",
                      "benevolence", "benevolent", "sympathy", "breeding",
                      "manners", "deportment", "genteel", "address"]

def rate(words, terms):
    freq = Counter(words)
    total = len(words)
    if total < 500:
        return 0.0
    return sum(freq.get(t, 0) for t in terms) / total * 10000

print("=" * 90)
print("STATISTICAL AUDIT")
print("=" * 90)

# ============================================================
# 1. SINGLE-TEXT DOMINANCE IN DECADE BINS
# ============================================================
print("\n" + "=" * 90)
print("1. SINGLE-TEXT DOMINANCE — are decade averages driven by one text?")
print("=" * 90)

for dec_start in range(1580, 1820, 10):
    dec_end = dec_start + 10
    matching = [(label, len(words)) for label, (date, cat, words) in texts.items()
                if dec_start <= date < dec_end and len(words) > 1000]
    if not matching:
        continue
    total_words = sum(w for _, w in matching)
    if len(matching) == 1:
        label, wc = matching[0]
        print(f"  {dec_start}s: SINGLE TEXT — {label} ({wc:,} words) — "
              f"ALL claims about this decade = claims about one text")
    else:
        largest = max(matching, key=lambda x: x[1])
        pct = largest[1] / total_words * 100
        if pct > 60:
            print(f"  {dec_start}s: DOMINATED — {largest[0]} is {pct:.0f}% of decade "
                  f"({largest[1]:,}/{total_words:,} words, {len(matching)} texts)")
        else:
            sizes = ", ".join(f"{l[:20]}({w:,})" for l, w in sorted(matching, key=lambda x: -x[1])[:3])
            print(f"  {dec_start}s: OK — {len(matching)} texts, largest={pct:.0f}%. Top: {sizes}")

# ============================================================
# 2. CORPUS SIZE ASYMMETRY
# ============================================================
print("\n" + "=" * 90)
print("2. CORPUS SIZE ASYMMETRY — are we comparing like with like?")
print("=" * 90)

cat_sizes = {}
for label, (date, cat, words) in texts.items():
    if len(words) < 1000:
        continue
    if cat not in cat_sizes:
        cat_sizes[cat] = {"count": 0, "words": 0, "min": 999999, "max": 0}
    cat_sizes[cat]["count"] += 1
    cat_sizes[cat]["words"] += len(words)
    cat_sizes[cat]["min"] = min(cat_sizes[cat]["min"], len(words))
    cat_sizes[cat]["max"] = max(cat_sizes[cat]["max"], len(words))

print(f"\n{'Category':<16} {'Texts':>6} {'Total words':>12} {'Min':>8} {'Max':>8} {'Ratio':>8}")
print("-" * 62)
for cat, data in sorted(cat_sizes.items(), key=lambda x: -x[1]["words"]):
    ratio = data["max"] / data["min"]
    print(f"{cat:<16} {data['count']:>6} {data['words']:>12,} {data['min']:>8,} {data['max']:>8,} {ratio:>8.0f}x")

# ============================================================
# 3. TERM OVERLAP BETWEEN BINS
# ============================================================
print("\n" + "=" * 90)
print("3. TERM OVERLAP — do bins share words? (collinearity)")
print("=" * 90)

all_bins = {
    "nervous_body": set(nervous_terms),
    "humoral_body": set(humoral_terms),
    "sociability": set(sociability_terms),
}
for name_a, terms_a in all_bins.items():
    for name_b, terms_b in all_bins.items():
        if name_a >= name_b:
            continue
        overlap = terms_a & terms_b
        if overlap:
            print(f"  {name_a} ∩ {name_b}: {overlap}")
            print(f"    → This means these bins are NOT independent. "
                  f"Changes in one mechanically affect the other.")

# ============================================================
# 4. THE NERVOUS/HUMORAL "PARADIGM SHIFT" — is it real?
# ============================================================
print("\n" + "=" * 90)
print("4. NERVOUS/HUMORAL SHIFT — decomposing the claim")
print("=" * 90)

print("\n4a. Is the 1730s spike just Cheyne?")
for dec_start in [1720, 1730, 1740]:
    dec_end = dec_start + 10
    matching = [(label, words) for label, (date, cat, words) in texts.items()
                if dec_start <= date < dec_end and len(words) > 1000]
    print(f"\n  {dec_start}s texts:")
    for label, words in matching:
        n = rate(words, nervous_terms)
        h = rate(words, humoral_terms)
        ratio = n / h if h > 0 else 999
        print(f"    {label[:45]:<46} nervous={n:>5.1f}  humoral={h:>5.1f}  ratio={ratio:.2f}  ({len(words):,} words)")

print("\n4b. 'constitution' and 'temperament' are in BOTH bins")
print("    Every hit for these words counts towards BOTH nervous and humoral scores.")
print("    This inflates correlation and makes the 'shift' look smoother than it is.")

# Count the shared terms' contribution
shared = {"constitution", "temperament"}
print(f"\n    Shared term frequencies across corpus:")
all_w = []
for label, (date, cat, words) in texts.items():
    all_w.extend(words)
corpus_freq = Counter(all_w)
for t in sorted(shared):
    print(f"      {t}: {corpus_freq.get(t, 0)} hits")

# What happens if we remove shared terms?
print("\n4c. Nervous/humoral ratio WITHOUT shared terms:")
nervous_clean = [t for t in nervous_terms if t not in shared]
humoral_clean = [t for t in humoral_terms if t not in shared]
for dec_start in range(1580, 1820, 20):
    dec_end = dec_start + 20
    matching = [words for label, (date, cat, words) in texts.items()
                if dec_start <= date < dec_end and len(words) > 1000]
    if not matching:
        continue
    all_dec = []
    for w in matching:
        all_dec.extend(w)
    n = rate(all_dec, nervous_clean)
    h = rate(all_dec, humoral_clean)
    ratio = n / h if h > 0 else 999
    n_shared = rate(all_dec, list(shared))
    print(f"    {dec_start}-{dec_end}: nervous={n:.1f}  humoral={h:.1f}  "
          f"ratio={ratio:.2f}  (shared terms={n_shared:.1f})")

# ============================================================
# 5. SURVIVORSHIP / SELECTION BIAS
# ============================================================
print("\n" + "=" * 90)
print("5. SURVIVORSHIP / SELECTION BIAS")
print("=" * 90)
print("""
  We selected texts BECAUSE they are relevant to the thesis:
  - Medical texts selected because they discuss nervous disorders
  - Conduct texts selected because they prescribe behavior
  - Fiction selected because it's Burney (the subject of study)

  This is not a random sample of 18c print culture. We CANNOT say:
  - "18c medical literature has higher conformity vocabulary than conduct lit"
    → We can only say: "THESE medical texts do"
  - "Sociability demand rises across the period"
    → We can only say: "In the texts we selected, it rises"

  The selection is defensible for the argument (these ARE the relevant texts),
  but we must not generalize to "18c discourse" without acknowledging the
  selection frame.

  Missing from the corpus that could complicate findings:
  - Novels by other authors (Richardson, Fielding, Smollett, Austen)
  - Popular periodicals (Gentleman's Magazine, London Magazine)
  - Parliamentary proceedings (Hansard)
  - Legal texts (statute books, court reports)
  - Religious texts beyond Anglicanism (Wesley, Dissenting sermons)
  - Commercial/trade literature
""")

# ============================================================
# 6. MULTIPLE COMPARISONS
# ============================================================
print("=" * 90)
print("6. MULTIPLE COMPARISONS — how many hypotheses are we testing?")
print("=" * 90)
print("""
  We have 8 theme bins × ~74 texts × 3 period slices × 6 categories.
  That's thousands of implicit comparisons. With no statistical test
  applied (no p-values, no confidence intervals, no effect sizes),
  ANY pattern we find could be noise.

  The specific claims that survive scrutiny are those with:
  - Large effect sizes (Cheyne nervous=54.7 vs corpus avg ~5)
  - Consistent direction across multiple texts (not just one outlier)
  - Theoretical motivation independent of the data

  Claims at risk:
  - "Conduct sociability rises 46%" — this is 20.4→29.7, within
    the range of individual text variation (Swift alone = 63.3)
  - "Fiction overwhelm is higher than conduct" — 4.1 vs 2.8,
    tiny absolute difference, no significance test
""")

# ============================================================
# 7. BOOTSTRAP CONFIDENCE INTERVALS for key claims
# ============================================================
print("=" * 90)
print("7. BOOTSTRAP TEST — are key findings robust to resampling?")
print("=" * 90)

random.seed(42)

def bootstrap_rate(words, terms, n_boot=1000):
    """Bootstrap confidence interval for rate per 10k."""
    total = len(words)
    freq = Counter(words)
    observed = sum(freq.get(t, 0) for t in terms) / total * 10000
    # Resample words (block bootstrap — sample 1000-word blocks)
    block_size = 1000
    n_blocks = total // block_size
    rates = []
    for _ in range(n_boot):
        sampled = []
        for _ in range(n_blocks):
            start = random.randint(0, total - block_size)
            sampled.extend(words[start:start+block_size])
        freq_b = Counter(sampled)
        r = sum(freq_b.get(t, 0) for t in terms) / len(sampled) * 10000
        rates.append(r)
    rates.sort()
    ci_lo = rates[int(0.025 * n_boot)]
    ci_hi = rates[int(0.975 * n_boot)]
    return observed, ci_lo, ci_hi

print("\n7a. Gregory's nervous-body score — is it really an outlier?")
# Compare Gregory to other conduct texts
conduct_texts_1760 = [(label, words) for label, (date, cat, words) in texts.items()
                       if cat == "conduct" and 1760 <= date < 1820 and len(words) > 3000]
for label, words in conduct_texts_1760:
    obs, lo, hi = bootstrap_rate(words, nervous_terms, 500)
    if obs > 15 or "Gregory" in label:
        print(f"  {label[:45]:<46} {obs:>5.1f} [{lo:>5.1f}, {hi:>5.1f}]")

print("\n7b. Burney's sensibility valence — is the sample big enough?")
approval = {"noble", "elegant", "beautiful", "amiable", "refined", "excellent",
            "virtue", "virtuous", "admirable", "worthy", "valuable", "true",
            "genuine", "natural", "pleasing", "charming"}
critique = {"excessive", "dangerous", "false", "affected", "weak", "weakness",
            "mischievous", "pernicious", "vain", "foolish", "ridiculous",
            "fatal", "destructive", "baneful", "pretended", "mere"}

for label, (date, cat, words) in sorted(texts.items(), key=lambda x: x[1][0]):
    positions = [i for i, w in enumerate(words) if w == "sensibility"]
    if len(positions) < 5:
        continue
    app = 0
    crit = 0
    for pos in positions:
        window = set(words[max(0, pos-6):pos] + words[pos+1:min(len(words), pos+7)])
        app += len(window & approval)
        crit += len(window & critique)
    total_c = app + crit
    if total_c == 0:
        continue
    # Bootstrap: resample positions
    boot_crit_pcts = []
    for _ in range(1000):
        samp = random.choices(positions, k=len(positions))
        ba, bc = 0, 0
        for pos in samp:
            window = set(words[max(0, pos-6):pos] + words[pos+1:min(len(words), pos+7)])
            ba += len(window & approval)
            bc += len(window & critique)
        bt = ba + bc
        if bt > 0:
            boot_crit_pcts.append(bc / bt * 100)
    if boot_crit_pcts:
        boot_crit_pcts.sort()
        lo = boot_crit_pcts[int(0.025 * len(boot_crit_pcts))]
        hi = boot_crit_pcts[int(0.975 * len(boot_crit_pcts))]
        obs_pct = crit / total_c * 100
        print(f"  {label[:40]:<41} {date}  n={len(positions):>3}  "
              f"collocates={total_c:>3}  critique={obs_pct:>5.1f}% [{lo:>5.1f}%, {hi:>5.1f}%]")

print("\n7c. Sociability 'rise' in conduct lit — confidence intervals by period")
for p_start, p_end, plabel in [(1580, 1700, "1580-1700"), (1700, 1760, "1700-1760"), (1760, 1820, "1760-1820")]:
    matching = [words for label, (date, cat, words) in texts.items()
                if cat == "conduct" and p_start <= date < p_end and len(words) > 1000]
    if not matching:
        print(f"  {plabel}: no data")
        continue
    all_w = []
    for w in matching:
        all_w.extend(w)
    obs, lo, hi = bootstrap_rate(all_w, sociability_terms, 500)
    print(f"  {plabel}: sociability = {obs:.1f} [{lo:.1f}, {hi:.1f}] ({len(matching)} texts, {len(all_w):,} words)")

# ============================================================
# 8. ECOLOGICAL FALLACY
# ============================================================
print("\n" + "=" * 90)
print("8. ECOLOGICAL FALLACY — do category averages mask individual variation?")
print("=" * 90)

print("\nConduct texts, 1760-1820 — sociability scores (showing within-group variance):")
conduct_soc = []
for label, (date, cat, words) in sorted(texts.items(), key=lambda x: x[1][0]):
    if cat == "conduct" and 1760 <= date < 1820 and len(words) > 3000:
        r = rate(words, sociability_terms)
        conduct_soc.append((label, date, r, len(words)))

for label, date, r, wc in conduct_soc:
    print(f"  {label[:45]:<46} {date}  sociab={r:>5.1f}  ({wc:,} words)")

if conduct_soc:
    rates = [r for _, _, r, _ in conduct_soc]
    mean = sum(rates) / len(rates)
    std = math.sqrt(sum((r - mean)**2 for r in rates) / len(rates))
    cv = std / mean * 100 if mean > 0 else 0
    print(f"\n  Mean={mean:.1f}  SD={std:.1f}  CV={cv:.0f}%")
    print(f"  Range: {min(rates):.1f} — {max(rates):.1f}")
    print(f"  → CV of {cv:.0f}% means massive within-group variation.")
    print(f"    The 'category average' hides as much as it reveals.")

# ============================================================
# 9. CONFOUNDS: GENRE vs TIME vs TOPIC
# ============================================================
print("\n" + "=" * 90)
print("9. CONFOUNDS — can we separate genre, time, and topic effects?")
print("=" * 90)
print("""
  The design is confounded:
  - Medical texts cluster 1720-1800; theology clusters 1650-1710
  - Fiction is ALL 1768-1814; politeness is ALL 1711-1759
  - We cannot separate "medical texts use more nervous vocabulary"
    from "later texts use more nervous vocabulary"

  To test: do CONDUCT texts (which span the full period) show
  the same nervous-body rise as the medical texts?
""")

print("  Conduct texts — nervous body by period:")
for p_start, p_end, plabel in [(1680, 1730, "1680-1730"), (1730, 1770, "1730-1770"), (1770, 1820, "1770-1820")]:
    matching = [(label, words) for label, (date, cat, words) in texts.items()
                if cat == "conduct" and p_start <= date < p_end and len(words) > 1000]
    if not matching:
        continue
    all_w = []
    for _, w in matching:
        all_w.extend(w)
    n = rate(all_w, nervous_terms)
    h = rate(all_w, humoral_terms)
    ratio = n / h if h > 0 else 999
    print(f"    {plabel}: nervous={n:.1f}  humoral={h:.1f}  ratio={ratio:.2f}  ({len(matching)} texts)")

# ============================================================
# 10. THE VALENCE ANALYSIS — window size sensitivity
# ============================================================
print("\n" + "=" * 90)
print("10. VALENCE SENSITIVITY — does window size change the story?")
print("=" * 90)

for label in ["Burney: Cecilia", "Burney: Wanderer", "MaryWollstonecraft: VindicationOfTheRightsOfWoman"]:
    if label not in texts:
        # Try partial match
        matches = [l for l in texts if label.split(":")[0] in l]
        if matches:
            label = matches[0]
        else:
            continue
    date, cat, words = texts[label]
    positions = [i for i, w in enumerate(words) if w == "sensibility"]
    if not positions:
        continue
    print(f"\n  {label} ({date}), sensibility × {len(positions)}:")
    for window_size in [4, 6, 8, 12, 20]:
        app = 0
        crit = 0
        for pos in positions:
            window = set(words[max(0, pos-window_size):pos] +
                        words[pos+1:min(len(words), pos+window_size+1)])
            app += len(window & approval)
            crit += len(window & critique)
        total_c = app + crit
        if total_c > 0:
            print(f"    window=±{window_size:>2}: approve={app:>3}  critique={crit:>3}  "
                  f"crit%={crit/total_c*100:>5.1f}%")
        else:
            print(f"    window=±{window_size:>2}: no collocates found")
