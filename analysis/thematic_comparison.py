"""Thematic comparison: father-daughter plots, sentiment arcs, key motifs."""
import re
from collections import Counter
from pathlib import Path

ROOT = Path("/Users/danielwaterfield/Documents/DigHums")

def load(path):
    text = path.read_text(errors="replace")
    # Strip Gutenberg
    for m in ["*** START OF", "***START OF"]:
        idx = text.find(m)
        if idx != -1:
            text = text[text.find("\n", idx)+1:]
            break
    for m in ["*** END OF", "***END OF", "End of the Project Gutenberg"]:
        idx = text.find(m)
        if idx != -1:
            text = text[:idx]
            break
    return text.replace("ſ", "s")

lucy = load(ROOT / "ArthurYoung/AdventuresOfMissLucyWatson.txt")
julia = load(ROOT / "ArthurYoung/HistoryOfJuliaBenson.txt")
evelina = load(ROOT / "FrancesBurney/Evelina.txt")
cecilia_parts = [load(ROOT / f"FrancesBurney/CeciliaVol{i}.txt") for i in range(1,4)]
cecilia = "\n".join(cecilia_parts)

texts = {
    "Young: Lucy Watson": lucy,
    "Young: Julia Benson": julia,
    "Burney: Evelina": evelina,
    "Burney: Cecilia": cecilia,
}

# ============================================================
# 1. FATHER-DAUGHTER THEME
# ============================================================
print("=" * 80)
print("1. FATHER-DAUGHTER THEME — keyword density")
print("=" * 80)

father_daughter_terms = {
    "paternal": ["father", "father's", "papa", "parent", "paternal", "parental"],
    "filial": ["daughter", "daughter's", "child", "filial", "obedience", "obedient", "dutiful", "dutifully"],
    "guardian": ["guardian", "guardian's", "protector", "protection", "ward", "charge", "care"],
    "authority": ["authority", "command", "consent", "permission", "forbid", "forbidden", "approve", "approval", "blessing"],
    "separation": ["absence", "absent", "separated", "separation", "parted", "parting", "reunited", "reunion"],
    "disclosure": ["secret", "concealment", "conceal", "reveal", "discover", "discovery", "confession", "confess", "acknowledge"],
}

for label, text in texts.items():
    words = re.findall(r"[a-z']+", text.lower())
    total = len(words)
    freq = Counter(words)
    print(f"\n{label} ({total:,} words):")
    for category, terms in father_daughter_terms.items():
        count = sum(freq.get(t, 0) for t in terms)
        rate = count / total * 10000
        top = sorted([(t, freq.get(t, 0)) for t in terms if freq.get(t, 0) > 0],
                     key=lambda x: -x[1])
        top_str = ", ".join(f"{t}({c})" for t, c in top[:5])
        print(f"  {category:<14} {rate:>6.1f}/10k  [{top_str}]")

# ============================================================
# 2. MARRIAGE PLOT VOCABULARY
# ============================================================
print("\n" + "=" * 80)
print("2. MARRIAGE PLOT — courtship & economic vocabulary")
print("=" * 80)

marriage_terms = {
    "courtship": ["love", "lover", "lovers", "beloved", "admirer", "admiration", "court", "courtship",
                   "passion", "affection", "affections", "attachment", "addresses", "suit", "suitor"],
    "virtue_test": ["virtue", "virtuous", "honour", "honor", "reputation", "character", "prudence",
                     "modesty", "delicacy", "propriety", "decorum", "discretion"],
    "economic": ["fortune", "fortune's", "fortunes", "estate", "estates", "income", "wealth",
                  "rich", "poor", "poverty", "independence", "dependent", "settlement", "portion",
                  "dowry", "inheritance", "heir", "heiress"],
    "social_rank": ["rank", "station", "birth", "family", "connections", "gentleman", "gentlewoman",
                     "lady", "lord", "noble", "quality", "distinction", "superior", "inferior"],
    "danger": ["ruin", "ruined", "seduction", "seduce", "libertine", "rake", "villain",
               "artifice", "design", "designs", "scheme", "trap", "snare"],
}

for label, text in texts.items():
    words = re.findall(r"[a-z']+", text.lower())
    total = len(words)
    freq = Counter(words)
    print(f"\n{label}:")
    for category, terms in marriage_terms.items():
        count = sum(freq.get(t, 0) for t in terms)
        rate = count / total * 10000
        top = sorted([(t, freq.get(t, 0)) for t in terms if freq.get(t, 0) > 0],
                     key=lambda x: -x[1])
        top_str = ", ".join(f"{t}({c})" for t, c in top[:5])
        print(f"  {category:<14} {rate:>6.1f}/10k  [{top_str}]")

# ============================================================
# 3. SENTIMENT ARC — sliding window emotion density
# ============================================================
print("\n" + "=" * 80)
print("3. SENTIMENT ARC — emotion density across narrative quarters")
print("=" * 80)

emotion_words = {"heart", "tears", "joy", "sorrow", "happiness", "misery",
                 "distress", "delight", "agony", "despair", "hope", "fear",
                 "love", "passion", "tenderness", "grief", "anxiety",
                 "pleasure", "pain", "shame", "pride", "honour", "honor",
                 "virtue", "duty", "sensibility", "affliction", "anguish",
                 "rapture", "ecstasy", "torment", "wretched", "wretchedness",
                 "melancholy", "consolation", "felicity", "gratitude"}

for label, text in texts.items():
    words = re.findall(r"[a-z']+", text.lower())
    n = len(words)
    quarters = [words[i*n//4:(i+1)*n//4] for i in range(4)]
    rates = []
    for q in quarters:
        ct = sum(1 for w in q if w in emotion_words)
        rates.append(ct / len(q) * 1000)
    trend = "RISING" if rates[3] > rates[0] * 1.2 else "FALLING" if rates[3] < rates[0] * 0.8 else "FLAT"
    peak = rates.index(max(rates)) + 1
    print(f"{label}:")
    print(f"  Q1={rates[0]:.1f}  Q2={rates[1]:.1f}  Q3={rates[2]:.1f}  Q4={rates[3]:.1f}  "
          f"peak=Q{peak}  trend={trend}")

# ============================================================
# 4. EPISTOLARY STRUCTURE — letter openings/closings
# ============================================================
print("\n" + "=" * 80)
print("4. EPISTOLARY STRUCTURE — letter markers")
print("=" * 80)

for label, text in [("Young: Lucy Watson", lucy), ("Young: Julia Benson", julia),
                     ("Burney: Evelina", evelina)]:
    # Count LETTER markers
    letter_heads = re.findall(r'LETTER\s+[IVXLC\d]+', text, re.IGNORECASE)
    # Common openings
    dear_patterns = re.findall(r'(?:dear|dearest|my dear)\s+\w+', text.lower())
    dear_counter = Counter(dear_patterns)
    # Closings
    closings = re.findall(r'(?:your\s+(?:obedient|affectionate|humble|faithful|devoted|loving|dutiful|ever)\s+\w+)',
                          text.lower())
    closing_counter = Counter(closings)
    # "I am" formula
    iam = re.findall(r'i am[,;]?\s+(?:dear|my|sir|madam|your)', text.lower())

    print(f"\n{label}:")
    print(f"  Letter markers found: {len(letter_heads)}")
    print(f"  'Dear X' openings (top 5): {dear_counter.most_common(5)}")
    print(f"  'Your X' closings (top 5): {closing_counter.most_common(5)}")
    print(f"  'I am, dear/your...' formulae: {len(iam)}")

# ============================================================
# 5. LONDON / PLACE REFERENCES
# ============================================================
print("\n" + "=" * 80)
print("5. LONDON & PLACE REFERENCES")
print("=" * 80)

london_places = [
    "london", "westminster", "covent garden", "drury lane", "vauxhall",
    "ranelagh", "pantheon", "hyde park", "st james", "the strand",
    "fleet street", "cheapside", "holborn", "marylebone", "soho",
    "bloomsbury", "the city", "tower", "temple", "opera", "theatre",
    "playhouse", "assembly", "masquerade", "ball", "park",
    "bath", "bristol", "oxford", "cambridge",
]

for label, text in texts.items():
    tl = text.lower()
    total_words = len(re.findall(r"[a-z']+", tl))
    found = []
    for place in london_places:
        ct = len(re.findall(r'\b' + re.escape(place) + r'\b', tl))
        if ct > 0:
            found.append((place, ct))
    found.sort(key=lambda x: -x[1])
    print(f"\n{label}:")
    for place, ct in found[:15]:
        print(f"  {place:<20} {ct:>4}")

# ============================================================
# 6. KEY PASSAGE EXTRACTION — father-daughter scenes
# ============================================================
print("\n" + "=" * 80)
print("6. KEY PASSAGES — father/daughter co-occurrence (sample)")
print("=" * 80)

father_re = re.compile(r'\b(?:father|papa|parent)\b', re.IGNORECASE)
daughter_re = re.compile(r'\b(?:daughter|child|girl)\b', re.IGNORECASE)

for label, text in texts.items():
    # Split into rough paragraphs
    paras = re.split(r'\n\s*\n', text)
    hits = []
    for p in paras:
        if father_re.search(p) and daughter_re.search(p) and len(p) > 100:
            # Clean up
            clean = " ".join(p.split())
            if len(clean) > 400:
                clean = clean[:400] + "..."
            hits.append(clean)
    print(f"\n{label}: {len(hits)} paragraphs with father+daughter co-occurrence")
    for h in hits[:3]:  # Show first 3
        print(f"  >>> {h}")
        print()

# ============================================================
# 7. NARRATIVE VOICE — direct address & interjection
# ============================================================
print("\n" + "=" * 80)
print("7. NARRATIVE VOICE — interjections & direct address")
print("=" * 80)

interjections = {
    "exclamatory": ["alas", "heavens", "good heavens", "good god", "oh",
                     "ah", "o", "gracious", "mercy", "lord"],
    "direct_address": ["dear sir", "dear madam", "my dear", "my lord",
                        "sir", "madam"],
    "self_deprecation": ["unworthy", "insignificant", "humble", "poor",
                          "wretched", "worthless"],
    "sensibility": ["sensibility", "feeling", "feelings", "sentiment",
                     "sentiments", "sympathize", "sympathy", "compassion",
                     "pity", "tender", "tenderness", "susceptible"],
}

for label, text in texts.items():
    tl = text.lower()
    total_words = len(re.findall(r"[a-z']+", tl))
    print(f"\n{label}:")
    for category, terms in interjections.items():
        count = sum(len(re.findall(r'\b' + re.escape(t) + r'\b', tl)) for t in terms)
        rate = count / total_words * 10000
        top = sorted([(t, len(re.findall(r'\b' + re.escape(t) + r'\b', tl))) for t in terms
                       if len(re.findall(r'\b' + re.escape(t) + r'\b', tl)) > 0],
                     key=lambda x: -x[1])
        top_str = ", ".join(f"{t}({c})" for t, c in top[:5])
        print(f"  {category:<18} {rate:>6.1f}/10k  [{top_str}]")
