"""Quick stylometric comparison: Arthur Young's fiction vs Frances Burney's fiction."""
import re
import os
from collections import Counter
from pathlib import Path

ROOT = Path("/Users/danielwaterfield/Documents/DigHums")

YOUNG = {
    "Young: Lucy Watson (1768)": [ROOT / "ArthurYoung/AdventuresOfMissLucyWatson.txt"],
    "Young: Julia Benson (1784)": [ROOT / "ArthurYoung/HistoryOfJuliaBenson.txt"],
}

BURNEY = {
    "Burney: Evelina (1778)": [ROOT / "FrancesBurney/Evelina.txt"],
    "Burney: Cecilia (1782)": [
        ROOT / "FrancesBurney/CeciliaVol1.txt",
        ROOT / "FrancesBurney/CeciliaVol2.txt",
        ROOT / "FrancesBurney/CeciliaVol3.txt",
    ],
    "Burney: Camilla (1796)": [ROOT / "FrancesBurney/Camilla.txt"],
    "Burney: Wanderer (1814)": [
        ROOT / f"FrancesBurney/TheWandererVol{i}.txt" for i in range(1, 6)
    ],
}

# Gutenberg boilerplate stripper
def strip_gutenberg(text):
    start_markers = ["*** START OF", "***START OF"]
    end_markers = ["*** END OF", "***END OF", "End of the Project Gutenberg"]
    for m in start_markers:
        idx = text.find(m)
        if idx != -1:
            nl = text.find("\n", idx)
            text = text[nl+1:]
            break
    for m in end_markers:
        idx = text.find(m)
        if idx != -1:
            text = text[:idx]
            break
    return text

def load_texts(file_map):
    results = {}
    for label, paths in file_map.items():
        combined = []
        for p in paths:
            raw = p.read_text(errors="replace")
            raw = strip_gutenberg(raw)
            combined.append(raw)
        results[label] = "\n".join(combined)
    return results

def tokenize(text):
    # Normalise long-s
    text = text.replace("ſ", "s")
    return re.findall(r"[a-zA-Z']+", text.lower())

def sentences(text):
    text = text.replace("ſ", "s")
    return [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]

def analyse(label, text):
    words = tokenize(text)
    sents = sentences(text)
    word_count = len(words)
    vocab = set(words)
    vocab_size = len(vocab)
    ttr = vocab_size / word_count if word_count else 0
    avg_word_len = sum(len(w) for w in words) / word_count if word_count else 0
    avg_sent_len = sum(len(tokenize(s)) for s in sents) / len(sents) if sents else 0

    # Top function words (style markers)
    function_words = [
        "the", "and", "to", "of", "a", "in", "that", "it", "is", "was",
        "i", "he", "she", "her", "his", "my", "for", "not", "but", "with",
        "you", "had", "have", "be", "which", "as", "at", "this", "from",
        "so", "very", "most", "could", "would", "should", "may", "might",
        "must", "shall", "will", "no", "all", "an", "if", "upon", "been",
        "one", "than", "every", "own", "such", "more", "too", "yet",
    ]
    freq = Counter(words)
    fw_profile = {w: freq[w] / word_count * 1000 for w in function_words if w in freq}

    # Hapax legomena ratio
    hapax = sum(1 for w, c in freq.items() if c == 1)
    hapax_ratio = hapax / vocab_size if vocab_size else 0

    # Contractions and dialogue markers
    contraction_count = sum(freq.get(w, 0) for w in [
        "don't", "can't", "won't", "didn't", "couldn't", "shouldn't",
        "i'm", "i'll", "i've", "he's", "she's", "it's", "we're", "they're",
        "you're", "you'll", "you've",
    ])
    contraction_rate = contraction_count / word_count * 1000 if word_count else 0

    # Exclamation density
    excl = text.count("!")
    excl_rate = excl / (word_count / 1000) if word_count else 0

    # Sentiment/emotion words
    emotion_words = ["heart", "tears", "joy", "sorrow", "happiness", "misery",
                     "distress", "delight", "agony", "despair", "hope", "fear",
                     "love", "passion", "tenderness", "grief", "anxiety",
                     "pleasure", "pain", "shame", "pride", "honour", "honor",
                     "virtue", "duty", "sensibility"]
    emotion_rate = sum(freq.get(w, 0) for w in emotion_words) / word_count * 1000

    return {
        "label": label,
        "word_count": word_count,
        "vocab_size": vocab_size,
        "ttr": ttr,
        "avg_word_len": avg_word_len,
        "avg_sent_len": avg_sent_len,
        "hapax_ratio": hapax_ratio,
        "contraction_rate": contraction_rate,
        "excl_rate": excl_rate,
        "emotion_rate": emotion_rate,
        "fw_profile": fw_profile,
    }

def shared_rare_vocab(texts_a, texts_b, min_len=6):
    """Find rare words shared between two text groups but uncommon overall."""
    words_a = set()
    words_b = set()
    for t in texts_a.values():
        toks = tokenize(t)
        freq = Counter(toks)
        # Words appearing but not super common
        words_a |= {w for w, c in freq.items() if len(w) >= min_len and 2 <= c <= 20}
    for t in texts_b.values():
        toks = tokenize(t)
        freq = Counter(toks)
        words_b |= {w for w, c in freq.items() if len(w) >= min_len and 2 <= c <= 20}
    return words_a & words_b

def top_distinctive(text, all_texts, n=20):
    """Words distinctively frequent in this text vs the rest."""
    words = tokenize(text)
    freq = Counter(words)
    total = len(words)

    other_words = []
    for t in all_texts.values():
        other_words.extend(tokenize(t))
    other_freq = Counter(other_words)
    other_total = len(other_words)

    scores = {}
    for w, c in freq.items():
        if c < 5 or len(w) < 4:
            continue
        rate_here = c / total
        rate_other = (other_freq.get(w, 0) + 1) / (other_total + 1)
        scores[w] = rate_here / rate_other

    return sorted(scores.items(), key=lambda x: -x[1])[:n]

# --- Main ---
print("=" * 80)
print("ARTHUR YOUNG vs FRANCES BURNEY — QUICK STYLOMETRIC COMPARISON")
print("=" * 80)

young_texts = load_texts(YOUNG)
burney_texts = load_texts(BURNEY)
all_texts = {**young_texts, **burney_texts}

results = []
for label, text in all_texts.items():
    r = analyse(label, text)
    results.append(r)

# Summary table
print(f"\n{'Text':<30} {'Words':>8} {'Vocab':>7} {'TTR':>6} {'AvgWd':>6} "
      f"{'AvgSnt':>7} {'Hapax':>6} {'Contr':>6} {'Excl!':>6} {'Emot':>6}")
print("-" * 100)
for r in results:
    print(f"{r['label']:<30} {r['word_count']:>8,} {r['vocab_size']:>7,} "
          f"{r['ttr']:>6.3f} {r['avg_word_len']:>6.2f} {r['avg_sent_len']:>7.1f} "
          f"{r['hapax_ratio']:>6.3f} {r['contraction_rate']:>6.2f} "
          f"{r['excl_rate']:>6.1f} {r['emotion_rate']:>6.2f}")

# Function word comparison
print("\n" + "=" * 80)
print("FUNCTION WORD FREQUENCIES (per 1000 words) — style fingerprint")
print("=" * 80)
key_fw = ["i", "she", "he", "her", "his", "my", "you", "not", "very",
          "would", "could", "upon", "which", "but", "yet", "most", "every",
          "such", "shall", "must", "may", "so"]
header = f"{'Word':<10}" + "".join(f"{r['label'][:20]:>22}" for r in results)
print(header)
print("-" * len(header))
for w in key_fw:
    row = f"{w:<10}"
    for r in results:
        val = r["fw_profile"].get(w, 0)
        row += f"{val:>22.2f}"
    print(row)

# Shared rare vocabulary
print("\n" + "=" * 80)
print("SHARED RARE VOCABULARY (6+ chars, used 2-20 times in each group)")
print("=" * 80)
shared = shared_rare_vocab(young_texts, burney_texts)
# Sort and show a sample
shared_sorted = sorted(shared)
print(f"Total shared rare words: {len(shared)}")
# Group by likely thematic interest
emotion = [w for w in shared_sorted if any(e in w for e in
           ["heart", "love", "fear", "joy", "grief", "hope", "shame",
            "pain", "pleas", "distress", "tender", "passion", "virtue",
            "honour", "honor", "duty", "sense", "spirit", "soul"])]
social = [w for w in shared_sorted if any(e in w for e in
          ["friend", "acquaint", "societ", "famil", "father", "mother",
           "sister", "brother", "husband", "wife", "marri", "fortune",
           "estate", "rank", "gentl", "lady", "lord"])]
print(f"\nEmotion/moral vocabulary ({len(emotion)}): {', '.join(emotion[:40])}")
print(f"\nSocial/family vocabulary ({len(social)}): {', '.join(social[:40])}")

# Distinctive words per text
print("\n" + "=" * 80)
print("DISTINCTIVE VOCABULARY — words disproportionately frequent per text")
print("=" * 80)
for label, text in all_texts.items():
    others = {k: v for k, v in all_texts.items() if k != label}
    dist = top_distinctive(text, others)
    words = [f"{w} ({s:.1f}x)" for w, s in dist[:15]]
    print(f"\n{label}:")
    print(f"  {', '.join(words)}")

# Key bigrams comparison
print("\n" + "=" * 80)
print("COMMON BIGRAMS — shared phrases")
print("=" * 80)
def bigrams(text, min_count=3):
    words = tokenize(text)
    bg = Counter(zip(words, words[1:]))
    return {(a, b): c for (a, b), c in bg.items() if c >= min_count and len(a) > 2 and len(b) > 2}

young_bg = Counter()
for t in young_texts.values():
    young_bg.update(bigrams(t, 2))
burney_bg = Counter()
for t in burney_texts.values():
    burney_bg.update(bigrams(t, 2))

shared_bg = set(young_bg.keys()) & set(burney_bg.keys())
# Rank by combined frequency
shared_ranked = sorted(shared_bg, key=lambda bg: young_bg[bg] + burney_bg[bg], reverse=True)

# Filter to interesting ones (not just "of the" etc)
stopwords = {"the", "and", "to", "of", "a", "in", "that", "it", "is", "was",
             "i", "he", "she", "her", "his", "my", "for", "not", "but", "with",
             "you", "had", "have", "be", "which", "as", "at", "this", "from"}
interesting = [(a, b) for a, b in shared_ranked
               if a not in stopwords or b not in stopwords][:30]
print(f"{'Bigram':<30} {'Young':>8} {'Burney':>8}")
print("-" * 50)
for a, b in interesting:
    print(f"{a + ' ' + b:<30} {young_bg[(a,b)]:>8} {burney_bg[(a,b)]:>8}")
