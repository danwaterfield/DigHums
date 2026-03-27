"""Rebuilt analysis with corpus-driven, epoch-validated lexicons."""
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

def tokenize_phrases(text):
    """Tokenize preserving key multi-word phrases as single tokens."""
    text = text.lower()
    # Replace key hyphenated/multi-word phrases with joined tokens
    phrase_map = {
        "self-command": "SELF_COMMAND", "self command": "SELF_COMMAND",
        "self-government": "SELF_GOVERNMENT", "self government": "SELF_GOVERNMENT",
        "self-denial": "SELF_DENIAL", "self denial": "SELF_DENIAL",
        "self-possession": "SELF_POSSESSION", "self possession": "SELF_POSSESSION",
        "good breeding": "GOOD_BREEDING", "good-breeding": "GOOD_BREEDING",
        "ill breeding": "ILL_BREEDING", "ill-breeding": "ILL_BREEDING",
        "fellow feeling": "FELLOW_FEELING", "fellow-feeling": "FELLOW_FEELING",
        "fellow creature": "FELLOW_CREATURE", "fellow-creature": "FELLOW_CREATURE",
        "fellow creatures": "FELLOW_CREATURES", "fellow-creatures": "FELLOW_CREATURES",
        "animal spirits": "ANIMAL_SPIRITS",
        "good manners": "GOOD_MANNERS", "good-manners": "GOOD_MANNERS",
        "ill manners": "ILL_MANNERS",
        "good humour": "GOOD_HUMOUR", "good-humour": "GOOD_HUMOUR",
        "good humor": "GOOD_HUMOUR",
        "good nature": "GOOD_NATURE", "good-nature": "GOOD_NATURE",
        "good natured": "GOOD_NATURED", "good-natured": "GOOD_NATURED",
        "presence of mind": "PRESENCE_OF_MIND",
        "want of attention": "WANT_OF_ATTENTION",
    }
    for phrase, token in phrase_map.items():
        text = text.replace(phrase, f" {token} ")
    return re.findall(r"[a-zA-Z_']+", text)

# ============================================================
# REBUILT LEXICONS — epoch-validated, polysemy-reduced
# ============================================================

themes = {
    "sociability": {
        "description": "Demands for social performance and mutual engagement",
        "terms": [
            # Core sociability (all attested pre-1750)
            "sociable", "society", "company", "conversation", "converse",
            "civility", "polite", "politeness", "agreeable", "amiable",
            "affable", "obliging", "complaisant", "complaisance",
            # Phrases (via phrase tokenizer)
            "GOOD_BREEDING", "GOOD_MANNERS", "GOOD_HUMOUR", "GOOD_NATURE",
            "GOOD_NATURED", "FELLOW_FEELING", "FELLOW_CREATURE", "FELLOW_CREATURES",
            # Period terms from collocate analysis
            "benevolence", "benevolent", "sympathy", "breeding", "manners",
            "deportment", "genteel", "address",
            # Removed: 'social' (anachronistic sense), 'commerce' (polysemy: trade),
            # 'civil' (polysemy: political), 'companion/s' (too generic),
            # 'intercourse' (polysemy: any exchange), 'sociability' (5 hits only)
        ],
    },
    "conformity_propriety": {
        "description": "Behavioral norms, propriety, the expected middle way",
        "terms": [
            # Core propriety (attested in conduct lit)
            "propriety", "decency", "decent", "decorum", "becoming",
            "conformity", "conform", "regularity",
            # Via media / moderation (theologically grounded)
            "moderate", "moderation", "temperance", "temperate",
            # Conduct-specific
            "customary", "seemly",
            # Removed: 'normal' (1828), 'standard' (19c behavioral sense),
            # 'common' (polysemy: shared/vulgar), 'proper' (polysemy: one's own),
            # 'ordinary' (polysemy: of no distinction), 'usual' (too generic),
            # 'regular' (polysemy: clergy/symmetry), 'correct' (polysemy: to amend)
        ],
    },
    "deviance_marked": {
        "description": "Behavioral difference marked as pathological or transgressive",
        "terms": [
            # Madness vocabulary (all period-appropriate)
            "mad", "madness", "lunatic", "lunacy", "frenzy", "frantic",
            "distracted",  # in 18c = driven mad, not merely diverted
            # Religious deviance (THE via media terms)
            "enthusiasm", "enthusiast", "enthusiasts", "enthusiastic",
            "fanatic", "fanaticism",
            # Nervous/humoral pathology
            "melancholy", "spleen", "vapours", "hysteric", "hysterical",
            "hypochondriac", "hypochondriacal",
            # Late 18c medical (borderline but attested)
            "insanity", "insane", "deranged", "derangement",
            "disordered",  # keep: specifically means 'of disordered mind'
            # Behavioral marking
            "odd", "peculiar",
            # Removed: 'wild' (landscape/animal), 'extravagant' (expensive),
            # 'singular' (positive: unique), 'strange' (foreign),
            # 'eccentric' (borderline 1780s), 'irregular' (too many senses),
            # 'disorder' alone (too medical/generic), 'distraction' (diversion)
        ],
    },
    "nervous_body": {
        "description": "The nervous body — medical and cultural, Cheyne through Gregory",
        "terms": [
            # The nervous system
            "nerve", "nerves", "nervous",
            # Sensibility (THE contested keyword)
            "sensibility",
            # The somatic vocabulary (from collocate analysis)
            "fibres", "fibers",
            "ANIMAL_SPIRITS",
            "constitution", "temperament",
            "irritable", "irritability",
            "susceptible",
            # Feeling/sensation (narrowed)
            "tenderness", "tender",
            "exquisite",  # in 18c = acutely sensitive
            # Delicacy as nervous quality
            "delicacy", "delicate",
            # Removed: 'sensible' (polysemy: rational), 'feeling/s' (too common),
            # 'impression/s' (polysemy: print/vague idea), 'refined/refinement'
            # (polysemy: chemistry/culture), 'acute' (polysemy: angles/disease),
            # 'sensitive' (rare before 1800 in modern sense)
        ],
    },
    "self_command": {
        "description": "Self-governance — the capacity the via media demands",
        "terms": [
            # Authentic 18c self-governance terms
            "SELF_COMMAND", "SELF_GOVERNMENT", "SELF_DENIAL", "SELF_POSSESSION",
            "PRESENCE_OF_MIND",
            "composure", "forbearance", "fortitude", "firmness", "steadiness",
            "restraint",
            # Patience (the adjective only — drop 'patient' noun)
            "patience",
            # Removed: 'government' (polysemy: state), 'command' (polysemy: military),
            # 'control' (18c = verify), 'regulate/regulation' (institutional),
            # 'patient' (medical noun), 'resolution' (decision), 'govern' (political),
            # 'moderate/moderation' (moved to conformity), 'steady' (too generic),
            # 'self-control' (1838), 'endurance' (rare)
        ],
    },
    "overwhelm_sensory": {
        "description": "Environmental sensory overload — urban, cognitive, bodily",
        "terms": [
            # Authentically sensory/urban (from collocate analysis)
            "tumult", "uproar", "bustle", "hurry", "throng",
            "noise", "din", "clamour", "clamor",
            "crowd", "crowded",
            "stench", "stink", "smoke",
            # Cognitive/emotional overwhelm (narrowed)
            "agitation", "agitated",
            "fatigue", "weariness",
            "overwhelm", "overwhelmed",
            # Removed: 'violent/violence' (physical/political), 'confusion' (intellectual),
            # 'disorder' (medical/political), 'disturbance' (political/riots),
            # 'oppression/oppressed' (political/social), 'burden' (financial/moral),
            # 'excessive/excess' (too generic), 'disturbed' (too generic),
            # 'exhaustion' (rare in 18c)
        ],
    },
    "attention_faculty": {
        "description": "Attention as cognitive faculty — the Lockean capacity",
        "terms": [
            # Positive attention (Lockean/Smithian)
            "attention", "attentive", "attentively",
            "heed", "heedful",
            "watchful", "vigilant",
            # Negative attention (failure/absence)
            "inattention", "inattentive",
            "heedless", "careless", "thoughtless",
            "abstracted",  # 18c: absent-minded
            "reverie",  # key 18c term for absorbed state
            "WANT_OF_ATTENTION",
            # Removed: 'notice' (written notice), 'observe' (obey/comply),
            # 'regard' (esteem/concerning), 'absent/absence' (physical),
            # 'distracted/distraction' (moved to deviance — dual meaning is the argument),
            # 'careful/carefulness' (too generic), 'mindful' (rare),
            # 'observation' (scientific), 'negligent/negligence' (legal)
        ],
    },
    "humoral_body": {
        "description": "Humoral/constitutional vocabulary — the older body model",
        "terms": [
            "humour", "humours", "humor", "humors",
            "phlegmatic", "sanguine", "choleric", "bilious",
            "blood", "bile",
            "constitution",
            "temperament", "temper",
            "spirits",  # animal spirits
            "complexion",  # 18c: bodily constitution
            "habit",  # 18c: bodily habit/constitution
        ],
    },
}

# ============================================================
# LOAD ALL TEXTS (same as before)
# ============================================================
fiction_entries = [
    (1768, "Young: Lucy Watson", "fiction", [ROOT/"ArthurYoung/AdventuresOfMissLucyWatson.txt"]),
    (1784, "Young: Julia Benson", "fiction", [ROOT/"ArthurYoung/HistoryOfJuliaBenson.txt"]),
    (1778, "Burney: Evelina", "fiction", [ROOT/"FrancesBurney/Evelina.txt"]),
    (1782, "Burney: Cecilia", "fiction", [ROOT/f"FrancesBurney/CeciliaVol{i}.txt" for i in range(1,4)]),
    (1796, "Burney: Camilla", "fiction", [ROOT/"FrancesBurney/Camilla.txt"]),
    (1814, "Burney: Wanderer", "fiction", [ROOT/f"FrancesBurney/TheWandererVol{i}.txt" for i in range(1,6)]),
]

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

state_entries = [
    (1651, "Hobbes: Leviathan", "philosophy", ROOT/"nonfiction/medical/passions/Hobbes_Leviathan.txt"),
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

texts = {}

for date, label, cat, paths in fiction_entries:
    combined = "\n".join(load(p) for p in paths)
    texts[label] = (date, cat, tokenize_phrases(combined))

for author_dir in sorted(conduct_dir.iterdir()):
    if not author_dir.is_dir():
        continue
    for txt in sorted(author_dir.glob("*.txt")):
        date = conduct_dates.get(author_dir.name, 1750)
        label = f"{author_dir.name}: {txt.stem}"
        texts[label] = (date, "conduct", tokenize_phrases(load(txt)))

for date, label, cat, path in state_entries + medical_entries:
    if path and path.exists():
        texts[label] = (date, cat, tokenize_phrases(load(path)))

print(f"Loaded {len(texts)} texts\n")

# ============================================================
# ANALYSIS
# ============================================================

def rate(words, term_list):
    freq = Counter(words)
    total = len(words)
    if total < 500:
        return 0.0
    return sum(freq.get(t, 0) for t in term_list) / total * 10000

sorted_texts = sorted(texts.items(), key=lambda x: x[1][0])

# 1. CHRONOLOGICAL TABLE
print("=" * 120)
print("1. CHRONOLOGICAL THEME EVOLUTION — rebuilt lexicons (per 10k words)")
print("=" * 120)

theme_keys = list(themes.keys())
short_names = {
    "sociability": "sociab",
    "conformity_propriety": "conform",
    "deviance_marked": "devianc",
    "nervous_body": "nervou",
    "self_command": "selfcmd",
    "overwhelm_sensory": "overwhl",
    "attention_faculty": "attentn",
    "humoral_body": "humoral",
}

header = f"{'Date':<6} {'Text':<42} {'Cat':<12}"
for t in theme_keys:
    header += f" {short_names.get(t, t[:7]):>8}"
print(header)
print("-" * len(header))

for label, (date, cat, words) in sorted_texts:
    if len(words) < 1000:
        continue
    row = f"{date:<6} {label[:41]:<42} {cat[:11]:<12}"
    for theme in theme_keys:
        r = rate(words, themes[theme]["terms"])
        row += f" {r:>8.1f}"
    print(row)

# 2. CATEGORY × PERIOD
print("\n" + "=" * 120)
print("2. CATEGORY × PERIOD AGGREGATES — rebuilt lexicons")
print("=" * 120)

periods = [(1580, 1700, "1580-1700"), (1700, 1760, "1700-1760"), (1760, 1820, "1760-1820")]
categories = ["theology", "anti-enthusiasm", "politeness", "conduct", "fiction", "medical"]

for theme in theme_keys:
    print(f"\n--- {theme.upper()} ---")
    header = f"{'Category':<16}"
    for _, _, plabel in periods:
        header += f" {plabel:>12}"
    print(header)
    for cat in categories:
        row = f"{cat:<16}"
        for p_start, p_end, _ in periods:
            matching = [(words, c) for label, (date, c, words) in texts.items()
                        if c == cat and p_start <= date < p_end and len(words) > 1000]
            if matching:
                all_w = []
                for w, _ in matching:
                    all_w.extend(w)
                r = rate(all_w, themes[theme]["terms"])
                row += f" {r:>12.1f}"
            else:
                row += f" {'—':>12}"
        print(row)

# 3. THE RATCHET — revised
print("\n" + "=" * 120)
print("3. THE RATCHET — sociability demand vs sensory overwhelm (revised, by decade)")
print("=" * 120)

print(f"{'Decade':<8} {'Sociab':>8} {'Overwhl':>8} {'Gap':>8} {'Conform':>8} {'Devianc':>8} {'NervBod':>8} {'Humoral':>8}")
print("-" * 72)
for dec_start in range(1580, 1820, 10):
    dec_end = dec_start + 10
    matching = [words for label, (date, cat, words) in texts.items()
                if dec_start <= date < dec_end and len(words) > 1000]
    if not matching:
        continue
    all_w = []
    for w in matching:
        all_w.extend(w)
    soc = rate(all_w, themes["sociability"]["terms"])
    ovr = rate(all_w, themes["overwhelm_sensory"]["terms"])
    gap = soc - ovr
    con = rate(all_w, themes["conformity_propriety"]["terms"])
    dev = rate(all_w, themes["deviance_marked"]["terms"])
    ner = rate(all_w, themes["nervous_body"]["terms"])
    hum = rate(all_w, themes["humoral_body"]["terms"])
    print(f"{dec_start}s   {soc:>8.1f} {ovr:>8.1f} {gap:>8.1f} {con:>8.1f} {dev:>8.1f} {ner:>8.1f} {hum:>8.1f}")

# 4. ATTENTION — positive vs negative, revised
print("\n" + "=" * 120)
print("4. ATTENTION FACULTY — positive vs negative (revised)")
print("=" * 120)

attn_pos = ["attention", "attentive", "attentively", "heed", "heedful", "watchful", "vigilant"]
attn_neg = ["inattention", "inattentive", "heedless", "careless", "thoughtless",
            "abstracted", "reverie", "WANT_OF_ATTENTION"]

print(f"\n{'Text':<45} {'Date':>5} {'Cat':<12} {'Attn+':>7} {'Attn-':>7} {'Ratio':>7}")
print("-" * 88)
for label, (date, cat, words) in sorted_texts:
    total = len(words)
    if total < 3000:
        continue
    freq = Counter(words)
    pos = sum(freq.get(t, 0) for t in attn_pos) / total * 10000
    neg = sum(freq.get(t, 0) for t in attn_neg) / total * 10000
    if pos + neg > 1.5:
        ratio = neg / pos if pos > 0 else 999
        print(f"{label[:44]:<45} {date:>5} {cat:<12} {pos:>7.1f} {neg:>7.1f} {ratio:>7.2f}")

# 5. PHRASE HITS — did phrase matching find anything?
print("\n" + "=" * 120)
print("5. PHRASE TOKEN HITS — multi-word terms captured by phrase tokenizer")
print("=" * 120)

phrase_tokens = ["SELF_COMMAND", "SELF_GOVERNMENT", "SELF_DENIAL", "SELF_POSSESSION",
                 "PRESENCE_OF_MIND", "GOOD_BREEDING", "ILL_BREEDING",
                 "FELLOW_FEELING", "FELLOW_CREATURE", "FELLOW_CREATURES",
                 "ANIMAL_SPIRITS", "GOOD_MANNERS", "ILL_MANNERS",
                 "GOOD_HUMOUR", "GOOD_NATURE", "GOOD_NATURED",
                 "WANT_OF_ATTENTION"]

all_corpus_words = []
for label, (date, cat, words) in texts.items():
    all_corpus_words.extend(words)
corpus_freq = Counter(all_corpus_words)

print(f"\n{'Phrase':<25} {'Corpus total':>12}")
print("-" * 40)
for pt in sorted(phrase_tokens):
    c = corpus_freq.get(pt, 0)
    if c > 0:
        print(f"{pt:<25} {c:>12}")
    else:
        print(f"{pt:<25} {'0 (not found)':>12}")

# Per-text breakdown for key phrases
print(f"\n{'Text':<45} {'SelfCmd':>8} {'GdBreed':>8} {'FellFel':>8} {'AnSpir':>8}")
print("-" * 75)
for label, (date, cat, words) in sorted_texts:
    if len(words) < 3000:
        continue
    freq = Counter(words)
    sc = freq.get("SELF_COMMAND", 0)
    gb = freq.get("GOOD_BREEDING", 0)
    ff = freq.get("FELLOW_FEELING", 0)
    asp = freq.get("ANIMAL_SPIRITS", 0)
    if sc + gb + ff + asp > 0:
        print(f"{label[:44]:<45} {sc:>8} {gb:>8} {ff:>8} {asp:>8}")

# 6. NERVOUS vs HUMORAL — the body model shift
print("\n" + "=" * 120)
print("6. NERVOUS vs HUMORAL BODY — the paradigm shift")
print("=" * 120)

print(f"\n{'Decade':<8} {'Nervous':>8} {'Humoral':>8} {'N/H ratio':>10}")
print("-" * 30)
for dec_start in range(1580, 1820, 10):
    dec_end = dec_start + 10
    matching = [words for label, (date, cat, words) in texts.items()
                if dec_start <= date < dec_end and len(words) > 1000]
    if not matching:
        continue
    all_w = []
    for w in matching:
        all_w.extend(w)
    ner = rate(all_w, themes["nervous_body"]["terms"])
    hum = rate(all_w, themes["humoral_body"]["terms"])
    ratio = ner / hum if hum > 0 else 999
    print(f"{dec_start}s   {ner:>8.1f} {hum:>8.1f} {ratio:>10.2f}")

# 7. VALENCE CHECK — does 'sensibility' appear with approval or critique words?
print("\n" + "=" * 120)
print("7. VALENCE CHECK — 'sensibility' collocates: approval vs critique")
print("=" * 120)

approval = {"noble", "elegant", "beautiful", "amiable", "refined", "excellent",
            "virtue", "virtuous", "admirable", "worthy", "valuable", "true",
            "genuine", "natural", "pleasing", "charming"}
critique = {"excessive", "dangerous", "false", "affected", "weak", "weakness",
            "mischievous", "pernicious", "vain", "foolish", "ridiculous",
            "fatal", "destructive", "baneful", "pretended", "mere"}

for label, (date, cat, words) in sorted_texts:
    if len(words) < 3000:
        continue
    # Find 'sensibility' positions
    positions = [i for i, w in enumerate(words) if w == "sensibility"]
    if len(positions) < 2:
        continue
    app_count = 0
    crit_count = 0
    for pos in positions:
        window = set(words[max(0, pos-6):pos] + words[pos+1:min(len(words), pos+7)])
        app_count += len(window & approval)
        crit_count += len(window & critique)
    total_colls = app_count + crit_count
    if total_colls > 0:
        app_pct = app_count / total_colls * 100
        crit_pct = crit_count / total_colls * 100
        print(f"{label[:44]:<45} {date:>5}  sensibility×{len(positions):<4} "
              f"approve:{app_count:>3} ({app_pct:>4.0f}%)  critique:{crit_count:>3} ({crit_pct:>4.0f}%)")
