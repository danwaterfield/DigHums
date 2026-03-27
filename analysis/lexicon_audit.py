"""Audit: are our lexicons epoch-appropriate? Check actual word frequencies in the corpus."""
import re
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

# Load everything into one big pot, but also keep per-text
all_words = []
all_files = []

for d in [ROOT/"nonfiction/conduct", ROOT/"nonfiction/state", ROOT/"nonfiction/medical",
          ROOT/"FrancesBurney", ROOT/"ArthurYoung"]:
    if not d.exists():
        continue
    for f in d.rglob("*.txt"):
        words = tokenize(load(f))
        if len(words) > 500:
            all_words.extend(words)
            all_files.append((f.name, words))

corpus_freq = Counter(all_words)
corpus_total = len(all_words)

print(f"Total corpus: {corpus_total:,} words across {len(all_files)} files\n")

# ============================================================
# THE LEXICONS AS USED — with honest assessment
# ============================================================

lexicons = {
    "sociability": {
        "terms": ["sociable", "sociability", "society", "social", "company", "companion",
                   "companions", "conversation", "converse", "commerce", "intercourse",
                   "civility", "civil", "polite", "politeness", "agreeable", "amiable",
                   "affable", "obliging", "complaisant", "complaisance"],
        "problems": [
            "ANACHRONISM: 'social' is rare before 1800 in modern sense (OED: 'social' as adj. of society widespread only from c.1830s)",
            "POLYSEMY: 'commerce' means trade, not sociability, in most 18c usage",
            "POLYSEMY: 'civil' means lawful/political as often as polite",
            "POLYSEMY: 'company' means military company, trading company, or mere physical presence",
            "POLYSEMY: 'intercourse' means any exchange, not specifically social",
            "MISSING: 'good-breeding' (hyphenated, won't tokenize), 'complaisance' variants",
            "MISSING: 'benevolence', 'fellow-feeling' (hyphenated), 'sympathy' (key Smithian term)",
        ]
    },
    "conformity": {
        "terms": ["conformity", "conform", "regular", "regularity", "proper", "propriety",
                   "decent", "decency", "decorum", "becoming", "seemly", "correct",
                   "standard", "normal", "common", "ordinary", "customary", "usual",
                   "moderate", "moderation", "temperance", "temperate"],
        "problems": [
            "ANACHRONISM: 'normal' is a 19c word (OED: 'normal' = conforming to standard, first 1828)",
            "ANACHRONISM: 'standard' in behavioral sense is 19c",
            "POLYSEMY: 'regular' means 'having rules', 'of regular clergy', or 'symmetrical' as often as 'conforming'",
            "POLYSEMY: 'common' means 'shared/public' or 'vulgar', not 'ordinary/conformist'",
            "POLYSEMY: 'proper' means 'one's own' (Latin proprius) as often as 'correct'",
            "POLYSEMY: 'moderate' in theology means via media position, which IS the argument — but in other contexts means merely 'not extreme'",
            "INFLATION: 'decent'/'decency' are very high-frequency in conduct lit regardless of conformity theme",
        ]
    },
    "deviance": {
        "terms": ["mad", "madness", "lunatic", "lunacy", "distracted", "distraction",
                   "insane", "insanity", "deranged", "derangement", "disorder", "disordered",
                   "frenzy", "frantic", "wild", "extravagant", "eccentric", "odd",
                   "singular", "strange", "peculiar", "irregular", "enthusiast", "enthusiasm",
                   "fanatic", "fanaticism", "melancholy", "spleen", "vapours", "hysteric"],
        "problems": [
            "ANACHRONISM: 'insane'/'insanity' are late 18c (OED: legal use from 1770s, medical from 1790s)",
            "ANACHRONISM: 'eccentric' as personality descriptor is late 18c (OED: 1780s)",
            "POLYSEMY: 'wild' is landscape/animal description far more often than behavioral",
            "POLYSEMY: 'extravagant' means 'wandering beyond bounds' or 'expensive' — only sometimes 'behaviorally deviant'",
            "POLYSEMY: 'singular' means 'unique/individual' (often positive) as often as 'odd'",
            "POLYSEMY: 'strange' means 'foreign' in earlier usage",
            "POLYSEMY: 'disorder' is medical (bodily disorder) far more than behavioral",
            "POLYSEMY: 'distraction' means 'diversion/entertainment' as well as 'madness'",
            "GOOD: 'enthusiasm' is a genuinely period-specific term for religious deviance",
            "GOOD: 'melancholy', 'spleen', 'vapours' are authentic 18c categories",
            "GOOD: 'lunatic', 'frenzy', 'frantic' are period-appropriate",
        ]
    },
    "nervous_sensibility": {
        "terms": ["nerve", "nerves", "nervous", "sensibility", "sensible",
                   "sensitive", "feeling", "feelings", "impression", "impressions",
                   "delicate", "delicacy", "refined", "refinement", "susceptible",
                   "tender", "tenderness", "exquisite", "acute", "irritable",
                   "irritability"],
        "problems": [
            "POLYSEMY: 'sensible' means 'having good sense/rational' as often as 'capable of feeling'",
            "POLYSEMY: 'feeling' is extremely common in non-technical usage",
            "POLYSEMY: 'impression' means 'printed impression' or 'vague idea' as well as sensory",
            "POLYSEMY: 'delicate'/'delicacy' means 'fine/choice' (of food, taste) as well as 'sensitive'",
            "POLYSEMY: 'refined' means 'purified' (chemistry/sugar) or 'cultured'",
            "POLYSEMY: 'acute' means 'sharp' (angles, diseases) as well as 'sensitive'",
            "GOOD: 'sensibility' is THE period keyword — semantically rich and contested",
            "GOOD: 'nervous' shifts meaning across exactly this period (vigorous → sensitive)",
            "GOOD: 'irritable'/'irritability' is a Hallerian medical term entering conduct lit",
            "MISSING: 'fibres' (as in 'delicate fibres' — key Cheyne/nervous body metaphor)",
        ]
    },
    "self_regulation": {
        "terms": ["restraint", "self-command", "self-control", "composure", "command",
                   "govern", "government", "regulate", "regulation", "control",
                   "moderate", "moderation", "patience", "patient", "forbearance",
                   "endurance", "fortitude", "resolution", "firmness", "steady"],
        "problems": [
            "ANACHRONISM: 'self-control' is 19c (OED: 1838). The 18c term is 'self-command'",
            "POLYSEMY: 'government' means state/political government far more often than self-governance",
            "POLYSEMY: 'command' means military/authority command as often as self-command",
            "POLYSEMY: 'regulate'/'regulation' means institutional regulation (Acts, rules)",
            "POLYSEMY: 'control' in 18c mostly means 'check/verify' not 'restrain'",
            "POLYSEMY: 'patient' means medical patient as well as adjective",
            "POLYSEMY: 'resolution' means 'decision' or 'legislative resolution'",
            "GOOD: 'forbearance', 'fortitude', 'composure' are genuine 18c self-governance terms",
            "GOOD: 'self-command' is the authentic 18c term (but hyphenated — tokenizes as two words)",
            "MISSING: 'self-government' (the actual Lockean term), 'self-denial'",
        ]
    },
    "overwhelm_excess": {
        "terms": ["overwhelm", "overwhelmed", "excess", "excessive", "violent",
                   "violence", "tumult", "confusion", "disorder", "agitation",
                   "agitated", "disturbance", "disturbed", "uproar", "noise",
                   "crowd", "crowded", "throng", "bustle", "hurry", "fatigue",
                   "exhaustion", "oppression", "oppressed", "burden", "burdened"],
        "problems": [
            "POLYSEMY: 'violent'/'violence' is mostly physical/political violence, not sensory",
            "POLYSEMY: 'confusion' means intellectual confusion or social confusion, not overwhelm",
            "POLYSEMY: 'disorder' is medical, political, military as well as environmental",
            "POLYSEMY: 'disturbance' is political (riots) as often as sensory",
            "POLYSEMY: 'oppression' is political/social as well as sensory",
            "POLYSEMY: 'burden' is financial/moral as well as cognitive",
            "GOOD: 'tumult', 'uproar', 'bustle', 'hurry', 'throng' are authentically sensory/urban",
            "GOOD: 'noise', 'crowd' are straightforward",
            "GOOD: 'fatigue' is period-appropriate for cognitive/sensory exhaustion",
            "MISSING: 'din' (common 18c noise word), 'stench', 'stink', 'smoke'",
        ]
    },
    "attention": {
        "terms": ["attention", "attentive", "attentively", "notice", "observe", "observation",
                   "regard", "heed", "mindful", "watchful", "vigilant", "careful",
                   "carefulness", "negligent", "negligence", "inattention", "inattentive",
                   "distracted", "distraction", "absent", "absence", "abstracted"],
        "problems": [
            "POLYSEMY: 'notice' means 'written notice/announcement' as well as cognitive",
            "POLYSEMY: 'observe' means 'obey/comply' (observe the law) as well as 'perceive'",
            "POLYSEMY: 'regard' means 'esteem' or 'concerning' as well as 'attend to'",
            "POLYSEMY: 'absent'/'absence' means physical absence far more than mental absence",
            "POLYSEMY: 'distracted'/'distraction' — dual meaning is PART OF THE ARGUMENT (diversion vs madness)",
            "GOOD: 'attention'/'attentive' are genuinely Lockean cognitive terms",
            "GOOD: 'abstracted' is authentic 18c for absent-mindedness",
            "GOOD: 'heed'/'heedless' are period-appropriate",
            "MISSING: 'reverie' (key 18c term for absorbed/absent state)",
        ]
    },
}

# ============================================================
# 1. PRINT THE AUDIT
# ============================================================
print("=" * 90)
print("LEXICON AUDIT — current bins, problems, actual corpus frequencies")
print("=" * 90)

for theme, data in lexicons.items():
    terms = data["terms"]
    problems = data["problems"]

    print(f"\n{'─' * 90}")
    print(f"THEME: {theme.upper()}")
    print(f"{'─' * 90}")
    print(f"Terms ({len(terms)}): {', '.join(terms)}")

    # Count actual occurrences
    print(f"\n  Actual frequencies in corpus ({corpus_total:,} words):")
    term_counts = []
    for t in sorted(terms):
        c = corpus_freq.get(t, 0)
        rate = c / corpus_total * 10000
        term_counts.append((t, c, rate))

    # Show sorted by frequency
    term_counts.sort(key=lambda x: -x[1])
    zero_terms = [t for t, c, _ in term_counts if c == 0]
    for t, c, rate in term_counts:
        if c > 0:
            flag = ""
            if rate > 20:
                flag = "  ← VERY HIGH FREQ (likely polysemous)"
            elif rate > 10:
                flag = "  ← HIGH FREQ (check polysemy)"
            elif c < 10:
                flag = "  ← RARE (may not be period-appropriate)"
            print(f"    {t:<20} {c:>8,}  ({rate:>5.1f}/10k){flag}")
    if zero_terms:
        print(f"    ZERO HITS: {', '.join(zero_terms)}")

    print(f"\n  Known problems:")
    for p in problems:
        print(f"    • {p}")

# ============================================================
# 2. WHAT WORDS DOES THE CORPUS ACTUALLY USE?
# ============================================================
print("\n\n" + "=" * 90)
print("CORPUS-DRIVEN DISCOVERY — what words do the TEXTS actually use?")
print("=" * 90)

# For each broad concept, search for actual period vocabulary by looking at
# words that co-occur with known anchor terms

def find_collocates(anchor_words, window=10, min_count=20):
    """Find words that frequently appear near anchor words."""
    collocates = Counter()
    for fname, words in all_files:
        for i, w in enumerate(words):
            if w in anchor_words:
                start = max(0, i - window)
                end = min(len(words), i + window + 1)
                for j in range(start, end):
                    if j != i:
                        collocates[words[j]] += 1
    # Filter to interesting words
    stopwords = {"the", "and", "to", "of", "a", "in", "that", "it", "is", "was",
                 "i", "he", "she", "her", "his", "my", "for", "not", "but", "with",
                 "you", "had", "have", "be", "which", "as", "at", "this", "from",
                 "their", "they", "them", "been", "were", "are", "what", "when",
                 "than", "will", "would", "could", "should", "may", "might",
                 "shall", "upon", "into", "more", "most", "very", "such", "an",
                 "some", "only", "every", "your", "our", "by", "so", "all", "no",
                 "or", "if", "on", "has", "there", "who", "we", "any", "how",
                 "one", "do", "can", "did", "its", "own", "other", "up", "out",
                 "about", "then", "these", "those", "being", "made", "make",
                 "much", "yet", "too", "many", "great", "first", "also", "let"}
    return [(w, c) for w, c in collocates.most_common(200)
            if w not in stopwords and len(w) > 3 and c >= min_count]

print("\n--- Words near NERVOUS/NERVES/NERVE ---")
colls = find_collocates({"nervous", "nerves", "nerve"}, window=8, min_count=10)
for w, c in colls[:30]:
    print(f"  {w:<20} {c:>6}")

print("\n--- Words near ATTENTION/ATTENTIVE ---")
colls = find_collocates({"attention", "attentive"}, window=8, min_count=10)
for w, c in colls[:30]:
    print(f"  {w:<20} {c:>6}")

print("\n--- Words near MAD/MADNESS/LUNATIC ---")
colls = find_collocates({"mad", "madness", "lunatic", "lunacy"}, window=8, min_count=10)
for w, c in colls[:30]:
    print(f"  {w:<20} {c:>6}")

print("\n--- Words near POLITE/POLITENESS/CIVILITY ---")
colls = find_collocates({"polite", "politeness", "civility", "civil"}, window=8, min_count=10)
for w, c in colls[:30]:
    print(f"  {w:<20} {c:>6}")

print("\n--- Words near ENTHUSIASM/ENTHUSIAST/FANATIC ---")
colls = find_collocates({"enthusiasm", "enthusiast", "enthusiasts", "fanatic", "fanaticism"}, window=8, min_count=5)
for w, c in colls[:30]:
    print(f"  {w:<20} {c:>6}")

print("\n--- Words near MELANCHOLY/SPLEEN/VAPOURS ---")
colls = find_collocates({"melancholy", "spleen", "vapours", "vapors", "hypochondriac"}, window=8, min_count=10)
for w, c in colls[:30]:
    print(f"  {w:<20} {c:>6}")

# ============================================================
# 3. TECHNIQUE ASSESSMENT
# ============================================================
print("\n\n" + "=" * 90)
print("TECHNIQUE ASSESSMENT — what are we actually doing and what should we be doing")
print("=" * 90)
print("""
CURRENT TECHNIQUE: Keyword-in-context frequency counting (bag of words)
- Simple word frequency per 10k tokens
- No word sense disambiguation
- No collocation analysis
- No sentiment/valence distinction
- No n-gram or phrase matching (hyphenated compounds lost)
- Tokenizer strips punctuation, lowercases, normalises long-s

PROBLEMS:
1. POLYSEMY: ~40-60% of terms in each bin have significant alternative meanings
   that inflate counts (e.g. 'civil' = political, 'patient' = medical noun,
   'absent' = physically away, 'command' = military order)

2. ANACHRONISM: Several terms are 19c coinages projected onto 18c texts:
   - 'normal' (1828), 'self-control' (1838), 'social' in modern sense (1830s+),
   - 'insane'/'insanity' (legal 1770s, medical 1790s — borderline),
   - 'eccentric' as personality (1780s — borderline),
   - 'standard' as behavioral benchmark (19c)

3. MISSING PERIOD VOCABULARY: Key 18c terms not captured:
   - Hyphenated compounds: 'self-command', 'good-breeding', 'fellow-feeling',
     'self-government', 'self-denial'
   - Period-specific: 'reverie', 'fibres' (nervous body), 'spirits' (animal spirits),
     'constitution', 'temperament', 'humour/humours', 'phlegmatic', 'sanguine',
     'choleric', 'bilious'
   - Conduct-specific: 'breeding', 'genteel', 'deportment', 'carriage', 'address'

4. NO SEMANTIC DIRECTION: 'sensibility' used approvingly (Gregory) counts the
   same as 'sensibility' used critically (Wollstonecraft). The valence distinction
   is central to the argument but invisible to word counting.

RECOMMENDED IMPROVEMENTS:
1. Corpus-driven lexicons: Build bins from actual collocate analysis (above)
2. Phrase matching: Match 'self command', 'good breeding', 'animal spirits' etc.
3. Period-validated vocabulary: Cross-reference with OED date-of-first-use
4. Separate polysemous terms: Split 'civil' into co-occurrence clusters
5. Valence analysis: Check whether anchor terms appear with approval/critique markers
6. Distributional semantics: Word2Vec or similar trained on the corpus to find
   semantic neighborhoods without pre-specifying lexicons
""")
