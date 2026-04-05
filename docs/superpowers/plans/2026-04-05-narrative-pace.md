# Narrative Pace Analysis — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a lexicon+spaCy sentence classifier and four-tab HTML visualisation that measures filler-vs-event ratios across 54 eighteenth-century novels (1719-1817), tracking the rise of bourgeois narrative regularity (Moretti) and cross-referencing with sensory map environmental data (Ghosh).

**Architecture:** Two scripts -- `analyse_narrative_pace.py` (classification engine -> SQLite + JSON) and `build_narrative_pace.py` (JSON -> self-contained HTML). Classification uses lexical markers + spaCy POS/tense features to score each sentence across six Genette-FID categories. Existing `strip_gutenberg_boilerplate()` from `burney-attribution/scripts/preprocess.py` is reused.

**Tech Stack:** Python 3, spaCy (`en_core_web_sm`), scipy (savgol_filter), sqlite3, json. HTML/CSS/JS (no external deps) for visualisation.

**Spec:** `docs/superpowers/specs/2026-04-05-narrative-pace-design.md`

---

### Task 1: Install dependencies and verify corpus access

**Files:**
- None created/modified -- environment setup only

- [ ] **Step 1: Install spaCy and download model**

Run:
```bash
pip install spacy scipy && python -m spacy download en_core_web_sm
```

- [ ] **Step 2: Verify corpus text accessibility**

Run:
```bash
python3 -c "
import csv
from pathlib import Path
root = Path('.')
with open('burney-attribution/data/metadata_v2.csv') as f:
    rows = list(csv.DictReader(f))
missing = [r['file_path'] for r in rows if not (root / r['file_path']).exists()]
print(f'{len(rows)} texts in metadata, {len(missing)} missing')
if missing: print('Missing:', missing[:5])
"
```
Expected: `54 texts in metadata, 0 missing`

- [ ] **Step 3: Verify spaCy model loads**

Run:
```bash
python3 -c "import spacy; nlp = spacy.load('en_core_web_sm'); doc = nlp('She seized the letter and fled.'); print([(t.text, t.tag_, t.pos_) for t in doc])"
```
Expected: output showing VBD tags for "seized" and "fled"

- [ ] **Step 4: Commit** (nothing to commit -- env setup only)

---

### Task 2: Gutenberg stripping and volume concatenation

**Files:**
- Create: `gazetteer/narrative_pace_corpus.py`
- Test: `gazetteer/tests/test_narrative_pace_corpus.py`

This module handles text loading: stripping Gutenberg boilerplate, concatenating multi-volume works, and recording volume boundary positions.

- [ ] **Step 1: Write the failing tests**

Create `gazetteer/tests/test_narrative_pace_corpus.py`:

```python
#!/usr/bin/env python3
"""Tests for corpus loading and volume concatenation."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from narrative_pace_corpus import (
    strip_boilerplate,
    load_novel_texts,
    concatenate_volumes,
)


class TestStripBoilerplate:
    def test_removes_gutenberg_header_and_footer(self):
        text = (
            "Blah blah preamble\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
            "Actual novel text here.\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
            "Blah blah license\n"
        )
        result = strip_boilerplate(text)
        assert result == "Actual novel text here."

    def test_returns_full_text_if_no_markers(self):
        text = "This text has no Gutenberg markers at all."
        result = strip_boilerplate(text)
        assert result == text


class TestConcatenateVolumes:
    def test_single_volume_returns_text_and_no_boundaries(self):
        texts = ["Volume one text."]
        combined, boundaries = concatenate_volumes(texts)
        assert combined == "Volume one text."
        assert boundaries == []

    def test_multi_volume_concatenation(self):
        texts = ["First volume.", "Second volume.", "Third volume."]
        combined, boundaries = concatenate_volumes(texts)
        assert "First volume." in combined
        assert "Second volume." in combined
        assert "Third volume." in combined
        assert len(boundaries) == 2  # 2 boundaries for 3 volumes

    def test_volume_boundaries_are_normalised_positions(self):
        # Two equal-length volumes
        texts = ["AAAA AAAA AAAA", "BBBB BBBB BBBB"]
        combined, boundaries = concatenate_volumes(texts)
        assert len(boundaries) == 1
        # Boundary should be near 0.5 (midpoint)
        assert 0.4 < boundaries[0] < 0.6


class TestLoadNovelTexts:
    def test_loads_single_volume_novel(self):
        """Evelina is a single-volume work -- should load without boundaries."""
        rows = [
            {"author": "burney", "title": "Evelina", "year": "1778",
             "genre": "epistolary", "volume": "",
             "file_path": "burney/Evelina.txt", "notes": ""}
        ]
        novels = load_novel_texts(rows, root=Path("."))
        assert len(novels) == 1
        assert novels[0]["title"] == "Evelina"
        assert novels[0]["volume_boundaries"] == []
        assert len(novels[0]["text"]) > 1000

    def test_loads_multi_volume_novel(self):
        """Cecilia has 3 volumes -- should concatenate."""
        rows = [
            {"author": "burney", "title": "Cecilia", "year": "1782",
             "genre": "domestic", "volume": str(v),
             "file_path": f"burney/CeciliaVol{v}.txt", "notes": ""}
            for v in [1, 2, 3]
        ]
        novels = load_novel_texts(rows, root=Path("."))
        assert len(novels) == 1
        assert len(novels[0]["volume_boundaries"]) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest gazetteer/tests/test_narrative_pace_corpus.py -v`
Expected: FAIL -- `ModuleNotFoundError: No module named 'narrative_pace_corpus'`

- [ ] **Step 3: Write minimal implementation**

Create `gazetteer/narrative_pace_corpus.py`:

```python
#!/usr/bin/env python3
"""Corpus loading for narrative pace analysis.

Handles Gutenberg boilerplate stripping, multi-volume concatenation,
and volume boundary position recording.
"""

import csv
import re
from pathlib import Path


def strip_boilerplate(text: str) -> str:
    """Remove Project Gutenberg header and footer."""
    start_patterns = [
        r'\*\*\* ?START OF THE PROJECT GUTENBERG EBOOK .+? \*\*\*',
        r'START OF THIS PROJECT GUTENBERG EBOOK',
    ]
    start_pos = 0
    for pat in start_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            start_pos = m.end()
            break

    end_patterns = [
        r'\*\*\* ?END OF THE PROJECT GUTENBERG EBOOK .+? \*\*\*',
        r'END OF THIS PROJECT GUTENBERG EBOOK',
    ]
    end_pos = len(text)
    for pat in end_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            end_pos = m.start()
            break

    return text[start_pos:end_pos].strip()


def concatenate_volumes(texts: list[str]) -> tuple[str, list[float]]:
    """Concatenate volume texts and return (combined, boundary_positions).

    boundary_positions are normalised 0.0-1.0 positions where volume
    breaks occur (len = num_volumes - 1).
    """
    if len(texts) == 1:
        return texts[0], []

    combined = "\n\n".join(texts)
    total_len = len(combined)
    boundaries = []
    running = 0
    for vol_text in texts[:-1]:
        running += len(vol_text) + 2  # +2 for "\n\n"
        boundaries.append(running / total_len)
    return combined, boundaries


def load_novel_texts(
    rows: list[dict], root: Path = Path(".")
) -> list[dict]:
    """Group metadata rows by (author, title), load and concatenate volumes.

    Returns list of novel dicts with keys:
        id, author, title, year, genre, text, word_count, volume_boundaries
    """
    # Group rows by (author, title)
    groups: dict[tuple, list] = {}
    for row in rows:
        key = (row["author"], row["title"])
        groups.setdefault(key, []).append(row)

    novels = []
    for (author, title), vol_rows in groups.items():
        # Sort by volume number (empty string = single volume)
        vol_rows.sort(key=lambda r: int(r["volume"]) if r["volume"] else 0)

        vol_texts = []
        for r in vol_rows:
            path = root / r["file_path"]
            raw = path.read_text(encoding="utf-8")
            vol_texts.append(strip_boilerplate(raw))

        combined, boundaries = concatenate_volumes(vol_texts)
        first = vol_rows[0]
        novel_id = f"{author}_{title.lower().replace(' ', '_')}_{first['year']}"

        novels.append({
            "id": novel_id,
            "author": author,
            "title": title,
            "year": int(first["year"]),
            "genre": first["genre"],
            "text": combined,
            "word_count": len(combined.split()),
            "volume_boundaries": boundaries,
        })

    # Sort by year then title
    novels.sort(key=lambda n: (n["year"], n["title"]))
    return novels


def load_corpus(
    metadata_path: Path = Path("burney-attribution/data/metadata_v2.csv"),
    root: Path = Path("."),
) -> list[dict]:
    """Load the full corpus from metadata CSV."""
    with open(metadata_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return load_novel_texts(rows, root=root)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest gazetteer/tests/test_narrative_pace_corpus.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add gazetteer/narrative_pace_corpus.py gazetteer/tests/test_narrative_pace_corpus.py
git commit -m "feat: add corpus loading module for narrative pace analysis"
```

---

### Task 3: Sentence classifier -- dialogue detection

**Files:**
- Create: `gazetteer/narrative_pace_classify.py`
- Test: `gazetteer/tests/test_narrative_pace_classify.py`

Build the dialogue classifier first since it is binary and gates the other five classifiers.

- [ ] **Step 1: Write the failing tests**

Create `gazetteer/tests/test_narrative_pace_classify.py`:

```python
#!/usr/bin/env python3
"""Tests for sentence classification engine."""

import sys
from pathlib import Path

import pytest
import spacy

sys.path.insert(0, str(Path(__file__).parent.parent))

from narrative_pace_classify import classify_sentence, SPEECH_VERBS

NLP = spacy.load("en_core_web_sm")


class TestDialogueDetection:
    def test_quoted_speech_is_dialogue(self):
        doc = NLP('"I am vastly pleased," said she.')
        result = classify_sentence(doc)
        assert result["is_dialogue"] is True

    def test_speech_verb_with_quotes(self):
        doc = NLP('"Let us go directly," cried Evelina.')
        result = classify_sentence(doc)
        assert result["is_dialogue"] is True

    def test_curly_quotes(self):
        doc = NLP('\u201cWhat a charming prospect!\u201d exclaimed Mrs. Selwyn.')
        result = classify_sentence(doc)
        assert result["is_dialogue"] is True

    def test_narration_without_quotes_is_not_dialogue(self):
        doc = NLP("She entered the room and sat down by the fire.")
        result = classify_sentence(doc)
        assert result["is_dialogue"] is False

    def test_dialogue_scores_are_zero_for_other_categories(self):
        doc = NLP('"Indeed!" replied he, with a sneer.')
        result = classify_sentence(doc)
        assert result["is_dialogue"] is True
        assert result["singulative"] == 0.0
        assert result["iterative"] == 0.0
        assert result["description"] == 0.0
        assert result["fid"] == 0.0
        assert result["commentary"] == 0.0

    def test_speech_verbs_list_is_comprehensive(self):
        expected = {"said", "cried", "answered", "returned", "replied",
                    "continued", "added", "repeated", "interrupted",
                    "declared", "exclaimed", "demanded", "pursued",
                    "resumed", "entreated", "whispered", "observed",
                    "inquired", "rejoined", "ejaculated"}
        assert expected.issubset(SPEECH_VERBS)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest gazetteer/tests/test_narrative_pace_classify.py::TestDialogueDetection -v`
Expected: FAIL -- `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `gazetteer/narrative_pace_classify.py`:

```python
#!/usr/bin/env python3
"""Sentence-level narrative mode classifier.

Classifies each sentence into six Genette-FID categories:
dialogue, singulative, iterative, description, FID, commentary.

Uses lexical markers + spaCy POS/tense features.
"""

import re
from spacy.tokens import Doc


SPEECH_VERBS = frozenset({
    "said", "cried", "answered", "returned", "replied",
    "continued", "added", "repeated", "interrupted",
    "declared", "exclaimed", "demanded", "pursued",
    "resumed", "entreated", "whispered", "observed",
    "inquired", "rejoined", "ejaculated",
})

_QUOTE_RE = re.compile(r'["\u201c\u201d\u201e\u201f\u2018\u2019\u00ab\u00bb]')


def _is_dialogue(doc: Doc) -> bool:
    """Detect dialogue: quoted speech or speech verb + quotation marks."""
    text = doc.text
    has_quotes = bool(_QUOTE_RE.search(text))
    if not has_quotes:
        return False
    # Quotes present -- treat as dialogue
    return True


def classify_sentence(doc: Doc, epistolary: bool = False) -> dict:
    """Classify a spaCy-processed sentence.

    Returns dict with keys:
        is_dialogue (bool), singulative, iterative, description, fid,
        commentary (all float, summing to 1.0 for non-dialogue).
        dominant_category (str).
    """
    result = {
        "is_dialogue": False,
        "singulative": 0.0,
        "iterative": 0.0,
        "description": 0.0,
        "fid": 0.0,
        "commentary": 0.0,
        "dominant_category": "dialogue",
    }

    if _is_dialogue(doc):
        result["is_dialogue"] = True
        return result

    # Placeholder -- other classifiers added in subsequent tasks
    for cat in ("singulative", "iterative", "description", "fid", "commentary"):
        result[cat] = 0.2
    result["dominant_category"] = "singulative"
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest gazetteer/tests/test_narrative_pace_classify.py::TestDialogueDetection -v`
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add gazetteer/narrative_pace_classify.py gazetteer/tests/test_narrative_pace_classify.py
git commit -m "feat: add dialogue classifier for narrative pace analysis"
```

---

### Task 4: Sentence classifier -- singulative, iterative, description

**Files:**
- Modify: `gazetteer/narrative_pace_classify.py`
- Modify: `gazetteer/tests/test_narrative_pace_classify.py`

- [ ] **Step 1: Write the failing tests**

Append to `gazetteer/tests/test_narrative_pace_classify.py`:

```python
class TestSingulativeDetection:
    def test_sudden_action(self):
        doc = NLP("Suddenly she seized the letter and fled the room.")
        result = classify_sentence(doc)
        assert result["dominant_category"] == "singulative"

    def test_temporal_rupture(self):
        doc = NLP("At that moment Lord Orville entered.")
        result = classify_sentence(doc)
        assert result["singulative"] > result["iterative"]

    def test_presently_as_immediacy(self):
        doc = NLP("He presently returned with a book.")
        result = classify_sentence(doc)
        assert result["singulative"] > result["iterative"]

    def test_directly_as_immediacy(self):
        doc = NLP("She directly quitted the room.")
        result = classify_sentence(doc)
        assert result["singulative"] > result["iterative"]


class TestIterativeDetection:
    def test_habitual_would(self):
        doc = NLP("She would often walk in the garden of a morning.")
        result = classify_sentence(doc)
        assert result["dominant_category"] == "iterative"

    def test_used_to(self):
        doc = NLP("He used to visit every Tuesday without fail.")
        result = classify_sentence(doc)
        assert result["iterative"] > result["singulative"]

    def test_frequency_adverbs(self):
        doc = NLP("Every evening the family assembled in the parlour.")
        result = classify_sentence(doc)
        assert result["iterative"] > result["singulative"]

    def test_was_wont_to(self):
        doc = NLP("She was wont to retire early each night.")
        result = classify_sentence(doc)
        assert result["iterative"] > result["singulative"]


class TestDescriptionDetection:
    def test_copular_adjective(self):
        doc = NLP("The room was large and handsomely furnished with velvet curtains.")
        result = classify_sentence(doc)
        assert result["dominant_category"] == "description"

    def test_spatial_prepositions(self):
        doc = NLP("Above the mantelpiece hung a portrait, and beneath it stood a marble table.")
        result = classify_sentence(doc)
        assert result["description"] > result["singulative"]

    def test_high_adjective_density(self):
        doc = NLP("The ancient grey stone walls were cold and damp and dark.")
        result = classify_sentence(doc)
        assert result["description"] > result["singulative"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest gazetteer/tests/test_narrative_pace_classify.py -k "Singulative or Iterative or Description" -v`
Expected: FAIL -- assertions fail because classifier returns uniform 0.2 scores

- [ ] **Step 3: Write the three classifiers**

Add lexicon constants and scorer functions to `gazetteer/narrative_pace_classify.py`, and update `classify_sentence` to call them instead of the uniform 0.2 placeholder. The full lexicon constants, scorer functions (`_score_singulative`, `_score_iterative`, `_score_description`), and updated `classify_sentence` are specified in the spec section "Category Definitions and Signals" and "Tense and POS Analysis". Each scorer:

1. Counts lexical marker hits (temporal adverbs, frequency markers, spatial prepositions, etc.)
2. Adds spaCy-derived features (VBD counts for singulative, copular+ADJ for description, AUX(would)+VERB for iterative)
3. Returns raw score normalised by sentence length

The three scorers follow this pattern:

```python
SINGULATIVE_TEMPORAL = frozenset({
    "suddenly", "presently", "directly", "instantly", "immediately",
})
SINGULATIVE_TEMPORAL_PHRASES = [
    "at once", "at that moment", "in a moment", "without warning",
    "on a sudden", "in a trice", "ere long", "by and by",
]
SINGULATIVE_ACTION_VERBS = frozenset({
    "seize", "run", "throw", "strike", "flee", "enter", "discover",
    "rush", "leap", "snatch", "burst", "plunge", "dash", "spring",
    "start", "faint", "scream", "shriek",
})

ITERATIVE_MARKERS = frozenset({
    "every", "often", "always", "never", "usually", "generally",
    "commonly", "frequently", "daily", "nightly",
})
ITERATIVE_PHRASES = [
    "used to", "was accustomed to", "was wont to", "were wont to",
    "each morning", "each evening", "each day", "ever and anon",
]

SPATIAL_PREPOSITIONS = frozenset({
    "above", "beneath", "beside", "beyond", "within", "around",
    "between", "behind", "below", "underneath", "amidst",
})


def _score_singulative(doc: Doc) -> float:
    text_lower = doc.text.lower()
    score = 0.0
    n_tokens = len(doc)
    if n_tokens == 0:
        return 0.0
    for t in doc:
        if t.lemma_.lower() in SINGULATIVE_TEMPORAL:
            score += 2.0
    for phrase in SINGULATIVE_TEMPORAL_PHRASES:
        if phrase in text_lower:
            score += 2.0
    for t in doc:
        if t.lemma_.lower() in SINGULATIVE_ACTION_VERBS and t.tag_ == "VBD":
            score += 1.5
    propn_count = sum(1 for t in doc if t.pos_ == "PROPN")
    score += propn_count * 0.3
    vbd_count = sum(1 for t in doc if t.tag_ == "VBD")
    score += vbd_count * 0.2
    return score / n_tokens


def _score_iterative(doc: Doc) -> float:
    text_lower = doc.text.lower()
    score = 0.0
    n_tokens = len(doc)
    if n_tokens == 0:
        return 0.0
    for t in doc:
        if t.text.lower() in ITERATIVE_MARKERS:
            score += 2.0
    for phrase in ITERATIVE_PHRASES:
        if phrase in text_lower:
            score += 3.0
    has_if = any(t.text.lower() == "if" for t in doc)
    for t in doc:
        if t.text.lower() == "would" and t.pos_ == "AUX" and not has_if:
            if t.i + 1 < len(doc) and doc[t.i + 1].pos_ == "VERB":
                score += 2.5
    for t in doc:
        if t.lemma_ == "be" and t.tag_ == "VBD" and t.text.lower() in ("was", "were"):
            if t.i + 1 < len(doc) and doc[t.i + 1].tag_ == "VBG":
                score += 0.5
    return score / n_tokens


def _score_description(doc: Doc) -> float:
    score = 0.0
    n_tokens = len(doc)
    if n_tokens == 0:
        return 0.0
    for t in doc:
        if t.lemma_ == "be" and t.dep_ == "ROOT":
            for child in t.children:
                if child.dep_ in ("acomp", "attr") and child.pos_ == "ADJ":
                    score += 2.0
                elif child.dep_ == "prep":
                    score += 1.0
    for t in doc:
        if t.text.lower() in SPATIAL_PREPOSITIONS:
            score += 1.5
    adj_count = sum(1 for t in doc if t.pos_ == "ADJ")
    adj_ratio = adj_count / n_tokens
    score += adj_ratio * 8.0
    verb_count = sum(1 for t in doc if t.pos_ == "VERB")
    if verb_count > 0 and adj_count > verb_count:
        score += 1.0
    return score / n_tokens
```

Update `classify_sentence` to use the scorers:

```python
def classify_sentence(doc: Doc, epistolary: bool = False) -> dict:
    result = {
        "is_dialogue": False, "singulative": 0.0, "iterative": 0.0,
        "description": 0.0, "fid": 0.0, "commentary": 0.0,
        "dominant_category": "dialogue",
    }
    if _is_dialogue(doc):
        result["is_dialogue"] = True
        return result

    scores = {
        "singulative": _score_singulative(doc),
        "iterative": _score_iterative(doc),
        "description": _score_description(doc),
        "fid": 0.0,        # placeholder -- Task 5
        "commentary": 0.0,  # placeholder -- Task 5
    }
    total = sum(scores.values())
    if total > 0:
        for cat in scores:
            scores[cat] /= total
    else:
        for cat in scores:
            scores[cat] = 0.2
    result.update(scores)
    result["dominant_category"] = max(scores, key=scores.get)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest gazetteer/tests/test_narrative_pace_classify.py -v`
Expected: all tests PASS (dialogue + singulative + iterative + description)

- [ ] **Step 5: Commit**

```bash
git add gazetteer/narrative_pace_classify.py gazetteer/tests/test_narrative_pace_classify.py
git commit -m "feat: add singulative, iterative, and description classifiers"
```

---

### Task 5: Sentence classifier -- FID, commentary, epistolary flag

**Files:**
- Modify: `gazetteer/narrative_pace_classify.py`
- Modify: `gazetteer/tests/test_narrative_pace_classify.py`

- [ ] **Step 1: Write the failing tests**

Append to `gazetteer/tests/test_narrative_pace_classify.py`:

```python
class TestFIDDetection:
    def test_exclamatory_without_quotes(self):
        doc = NLP("How delightful was the prospect before her!")
        result = classify_sentence(doc)
        assert result["dominant_category"] == "fid"

    def test_interrogative_without_quotes(self):
        doc = NLP("Was she then to endure this insupportable treatment?")
        result = classify_sentence(doc)
        assert result["fid"] > result["commentary"]

    def test_evaluative_adjectives(self):
        doc = NLP("The scene was indeed charming, and the company agreeable beyond measure.")
        result = classify_sentence(doc)
        assert result["fid"] > 0.1

    def test_deictic_shift(self):
        doc = NLP("She would go there tomorrow, and nothing could prevent her now.")
        result = classify_sentence(doc)
        assert result["fid"] > result["singulative"]

    def test_prodigious_as_intensifier(self):
        doc = NLP("It was a prodigious fine evening and she was monstrous pleased.")
        result = classify_sentence(doc)
        assert result["fid"] > result["description"]


class TestCommentaryDetection:
    def test_reader_address(self):
        doc = NLP("The reader will not be surprised to learn that she was disappointed.")
        result = classify_sentence(doc)
        assert result["dominant_category"] == "commentary"

    def test_first_person_plural(self):
        doc = NLP("We must leave our heroine for a moment to explain the circumstances.")
        result = classify_sentence(doc)
        assert result["commentary"] > result["singulative"]

    def test_moral_generalisation(self):
        doc = NLP("Virtue is the only sure foundation of honour and esteem in this world.")
        result = classify_sentence(doc)
        assert result["commentary"] > result["description"]

    def test_present_tense_maxim(self):
        doc = NLP("A woman of delicacy never forgives an affront to her sensibility.")
        result = classify_sentence(doc)
        assert result["commentary"] > result["fid"]


class TestEpistolaryFlag:
    def test_present_tense_without_flag_is_commentary(self):
        doc = NLP("I sit now by the fire and write to you of all that has passed.")
        result = classify_sentence(doc, epistolary=False)
        assert result["commentary"] >= result["singulative"]

    def test_present_tense_with_flag_is_singulative(self):
        doc = NLP("I sit now by the fire and write to you of all that has passed.")
        result = classify_sentence(doc, epistolary=True)
        assert result["singulative"] >= result["commentary"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest gazetteer/tests/test_narrative_pace_classify.py -k "FID or Commentary or Epistolary" -v`
Expected: FAIL -- FID and commentary return 0.0 (placeholders)

- [ ] **Step 3: Implement FID and commentary classifiers**

Add lexicon constants and scorer functions to `gazetteer/narrative_pace_classify.py`:

```python
EVALUATIVE_ADJ = frozenset({
    "cruel", "dreadful", "charming", "agreeable", "disagreeable",
    "amiable", "wretched", "shocking", "odious", "delightful",
    "barbarous", "horrid", "insufferable", "divine", "unaccountable",
    "insupportable", "prodigious", "monstrous", "excellent", "vile",
    "admirable", "exquisite",
})

EPISTEMIC_HEDGES = frozenset({
    "indeed", "perhaps", "surely", "certainly", "truly",
    "doubtless", "undoubtedly", "assuredly",
})
EPISTEMIC_PHRASES = ["no doubt", "i dare say", "it seemed"]

FID_DEICTICS = frozenset({
    "now", "here", "tomorrow", "yesterday", "tonight", "this",
})

COMMENTARY_PRONOUNS = frozenset({"we", "our", "us"})
COMMENTARY_NOUNS = frozenset({"mankind", "reader", "world"})
MORAL_VOCABULARY = frozenset({
    "honour", "honor", "virtue", "duty", "tenderness", "gratitude",
    "esteem", "prudence", "delicacy", "sensibility", "propriety",
    "fortitude", "benevolence", "compassion", "modesty", "discretion",
    "condescension",
})


def _score_fid(doc: Doc, epistolary: bool = False) -> float:
    text_lower = doc.text.lower()
    score = 0.0
    n_tokens = len(doc)
    if n_tokens == 0:
        return 0.0
    if re.match(r'^(how|what a)\b', text_lower) and not _QUOTE_RE.search(doc.text):
        score += 4.0
    if doc.text.rstrip().endswith("?") and not _QUOTE_RE.search(doc.text):
        score += 2.5
    for t in doc:
        if t.lemma_.lower() in EVALUATIVE_ADJ:
            score += 1.5
    for t in doc:
        if t.text.lower() in EPISTEMIC_HEDGES:
            score += 1.0
    for phrase in EPISTEMIC_PHRASES:
        if phrase in text_lower:
            score += 1.5
    has_past = any(t.tag_ == "VBD" for t in doc)
    if has_past and not epistolary:
        for t in doc:
            if t.text.lower() in FID_DEICTICS:
                score += 2.0
    for t in doc:
        if t.text.lower() in ("prodigious", "monstrous") and t.pos_ == "ADV":
            score += 2.0
        elif t.text.lower() in ("prodigious", "monstrous") and t.pos_ == "ADJ":
            score += 0.8
    return score / n_tokens


def _score_commentary(doc: Doc, epistolary: bool = False) -> float:
    text_lower = doc.text.lower()
    score = 0.0
    n_tokens = len(doc)
    if n_tokens == 0:
        return 0.0
    for t in doc:
        if t.text.lower() in COMMENTARY_PRONOUNS:
            score += 2.0
    if "reader" in text_lower:
        score += 4.0
    for t in doc:
        if t.lemma_.lower() in COMMENTARY_NOUNS:
            score += 1.5
    for t in doc:
        if t.lemma_.lower() in MORAL_VOCABULARY:
            score += 1.0
    if not epistolary:
        present_verbs = sum(
            1 for t in doc if t.tag_ in ("VBP", "VBZ") and t.pos_ == "VERB"
        )
        score += present_verbs * 1.5
    else:
        has_generalising = any(t.lemma_.lower() in COMMENTARY_NOUNS for t in doc)
        if has_generalising:
            present_verbs = sum(
                1 for t in doc if t.tag_ in ("VBP", "VBZ") and t.pos_ == "VERB"
            )
            score += present_verbs * 1.0
    return score / n_tokens
```

Update `classify_sentence` to call the new scorers and handle epistolary flag:

```python
def classify_sentence(doc: Doc, epistolary: bool = False) -> dict:
    result = {
        "is_dialogue": False, "singulative": 0.0, "iterative": 0.0,
        "description": 0.0, "fid": 0.0, "commentary": 0.0,
        "dominant_category": "dialogue",
    }
    if _is_dialogue(doc):
        result["is_dialogue"] = True
        return result

    scores = {
        "singulative": _score_singulative(doc),
        "iterative": _score_iterative(doc),
        "description": _score_description(doc),
        "fid": _score_fid(doc, epistolary=epistolary),
        "commentary": _score_commentary(doc, epistolary=epistolary),
    }
    # Epistolary boost: present tense -> singulative
    if epistolary:
        present_verbs = sum(
            1 for t in doc if t.tag_ in ("VBP", "VBZ") and t.pos_ == "VERB"
        )
        if present_verbs > 0:
            scores["singulative"] += (present_verbs * 1.5) / len(doc)

    total = sum(scores.values())
    if total > 0:
        for cat in scores:
            scores[cat] /= total
    else:
        for cat in scores:
            scores[cat] = 0.2
    result.update(scores)
    result["dominant_category"] = max(scores, key=scores.get)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest gazetteer/tests/test_narrative_pace_classify.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add gazetteer/narrative_pace_classify.py gazetteer/tests/test_narrative_pace_classify.py
git commit -m "feat: add FID, commentary classifiers and epistolary flag"
```

---

### Task 6: Analysis engine -- full pipeline

**Files:**
- Create: `gazetteer/analyse_narrative_pace.py`
- Test: `gazetteer/tests/test_analyse_narrative_pace.py`

Orchestrates: load corpus -> spaCy process -> classify -> SQLite -> smooth -> JSON.

- [ ] **Step 1: Write the failing tests**

Create `gazetteer/tests/test_analyse_narrative_pace.py`:

```python
#!/usr/bin/env python3
"""Tests for the narrative pace analysis pipeline."""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

REPO_ROOT = Path(__file__).parent.parent.parent
DB_PATH = REPO_ROOT / "gazetteer" / "narrative_pace.db"
JSON_PATH = REPO_ROOT / "gazetteer" / "narrative_pace_data.json"


@pytest.fixture(scope="module")
def run_analysis():
    """Run the analysis pipeline on a small subset."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "gazetteer/analyse_narrative_pace.py", "--limit", "3"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"Analysis failed: {result.stderr}"
    return result.stdout


@pytest.fixture(scope="module")
def db(run_analysis):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def json_data(run_analysis):
    return json.loads(JSON_PATH.read_text(encoding="utf-8"))


class TestSQLiteOutput:
    def test_novels_table_populated(self, db):
        count = db.execute("SELECT COUNT(*) FROM novels").fetchone()[0]
        assert count == 3  # --limit 3

    def test_sentences_table_populated(self, db):
        count = db.execute("SELECT COUNT(*) FROM sentences").fetchone()[0]
        assert count > 100

    def test_sentence_scores_normalise(self, db):
        rows = db.execute(
            "SELECT singulative, iterative, description, fid, commentary "
            "FROM sentences WHERE is_dialogue = 0 LIMIT 50"
        ).fetchall()
        for row in rows:
            total = sum(row)
            assert abs(total - 1.0) < 0.01, f"Scores sum to {total}, not 1.0"

    def test_environmental_table_populated(self, db):
        count = db.execute("SELECT COUNT(*) FROM environmental").fetchone()[0]
        assert count > 0


class TestJSONOutput:
    def test_novels_present(self, json_data):
        assert len(json_data["novels"]) == 3

    def test_arc_has_200_points(self, json_data):
        for novel in json_data["novels"]:
            assert len(novel["arc"]["positions"]) == 200
            assert len(novel["arc"]["dialogue"]) == 200
            assert len(novel["arc"]["singulative"]) == 200

    def test_summary_proportions_sum_to_one(self, json_data):
        for novel in json_data["novels"]:
            total = sum(novel["summary"].values())
            assert abs(total - 1.0) < 0.01

    def test_environmental_data_present(self, json_data):
        assert len(json_data["environmental"]) > 0

    def test_lexicon_version(self, json_data):
        assert json_data["lexicon_version"] == "1.0"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest gazetteer/tests/test_analyse_narrative_pace.py -v`
Expected: FAIL -- `analyse_narrative_pace.py` does not exist

- [ ] **Step 3: Write the analysis engine**

Create `gazetteer/analyse_narrative_pace.py`. This script:

1. Loads corpus via `narrative_pace_corpus.load_corpus()`
2. Processes each novel through spaCy (with `nlp.max_length = 2_000_000` for Clarissa)
3. Classifies each sentence via `narrative_pace_classify.classify_sentence()`
4. Stores sentence-level results in SQLite (`narrative_pace.db`)
5. Applies Savitzky-Golay smoothing and resamples to 200 points
6. Queries `sensory.db` for environmental evidence by decade
7. Writes `narrative_pace_data.json`

Accepts `--limit N` flag for testing with subset.

Key functions:
- `_init_db(path)` -- creates schema (novels, sentences, environmental tables)
- `_process_novel(nlp, novel)` -- runs spaCy + classify on all sentences
- `_smooth_and_resample(scores, n_out=200)` -- Savitzky-Golay with 5% window, resample via `np.interp`
- `_compute_summary(scores)` -- whole-novel category proportions
- `_load_environmental(sensory_db)` -- SQL query grouping by decade for olfactory+thermal modalities
- `main()` -- orchestrates everything

See the spec sections "Processing pipeline" and "Data Structures" for the complete schema and JSON format. The environmental query is:

```sql
SELECT (date_min / 10) * 10 AS decade,
       SUM(CASE WHEN modality = 'olfactory' THEN 1 ELSE 0 END) AS smell,
       SUM(CASE WHEN modality = 'thermal' THEN 1 ELSE 0 END) AS smoke,
       SUM(CASE WHEN modality IN ('olfactory', 'thermal') THEN 1 ELSE 0 END) AS pollution,
       COUNT(*) AS total
FROM sensory_evidence
WHERE source_type != 'fiction' AND date_min IS NOT NULL
GROUP BY decade ORDER BY decade
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest gazetteer/tests/test_analyse_narrative_pace.py -v --timeout=300`
Expected: all tests PASS (takes 1-2 minutes for 3 novels)

- [ ] **Step 5: Commit**

```bash
git add gazetteer/analyse_narrative_pace.py gazetteer/tests/test_analyse_narrative_pace.py
git commit -m "feat: add analysis pipeline -- spaCy + classify + smooth + SQLite/JSON"
```

---

### Task 7: HTML builder -- all four tabs

**Files:**
- Create: `gazetteer/build_narrative_pace.py`
- Test: `gazetteer/tests/test_build_narrative_pace.py`

Single self-contained HTML with four tab views: Century, Arcs, Grid, Ecology. All charts rendered client-side on `<canvas>` elements from embedded JSON data. No external dependencies.

- [ ] **Step 1: Write the failing tests**

Create `gazetteer/tests/test_build_narrative_pace.py`:

```python
#!/usr/bin/env python3
"""Tests for the narrative pace HTML builder."""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

REPO_ROOT = Path(__file__).parent.parent.parent
HTML_PATH = REPO_ROOT / "gazetteer" / "narrative_pace.html"
JSON_PATH = REPO_ROOT / "gazetteer" / "narrative_pace_data.json"


@pytest.fixture(scope="module")
def html():
    """Build the HTML and return content."""
    assert JSON_PATH.exists(), "Run analyse_narrative_pace.py first"
    result = subprocess.run(
        [sys.executable, "gazetteer/build_narrative_pace.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return HTML_PATH.read_text(encoding="utf-8")


def test_html_generated(html):
    assert "<!DOCTYPE html>" in html
    assert "Narrative Pace" in html


def test_tab_buttons_present(html):
    for tab in ["Century", "Arcs", "Grid", "Ecology"]:
        assert tab in html


def test_data_injected(html):
    assert "__NOVELS_DATA__" not in html
    assert "novels" in html


def test_genre_colours_defined(html):
    for genre in ["domestic", "gothic", "picaresque",
                  "epistolary", "amatory", "satirical"]:
        assert genre in html


def test_category_colours_defined(html):
    for cat in ["dialogue", "singulative", "iterative",
                "description", "fid", "commentary"]:
        assert cat in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest gazetteer/tests/test_build_narrative_pace.py -v`
Expected: FAIL -- `build_narrative_pace.py` does not exist

- [ ] **Step 3: Write the HTML builder**

Create `gazetteer/build_narrative_pace.py`. This follows the `build_comparison.py` pattern:

```python
#!/usr/bin/env python3
"""
Build the self-contained narrative pace HTML.

Reads narrative_pace_data.json, writes narrative_pace.html.

Usage:
    python3 gazetteer/build_narrative_pace.py
    open gazetteer/narrative_pace.html
"""

import json
from pathlib import Path

JSON_PATH = Path(__file__).parent / "narrative_pace_data.json"
OUT_PATH = Path(__file__).parent / "narrative_pace.html"


def build(json_path: Path = JSON_PATH, out_path: Path = OUT_PATH) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    data_js = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    html = HTML_TEMPLATE.replace("__NOVELS_DATA__", data_js, 1)
    out_path.write_text(html, encoding="utf-8")
    n_env = len(data.get("environmental", []))
    print(f"Narrative pace -> {out_path}")
    print(f"  {len(data['novels'])} novels, {n_env} env decades")


HTML_TEMPLATE = """<!DOCTYPE html>
...
"""


if __name__ == "__main__":
    build()
```

The `HTML_TEMPLATE` string is a complete self-contained HTML document (~800 lines) containing:

**CSS:** Tab styling, chart containers, grid cards, legend items, tooltip, bar rows, eco panels, callout box. Uses CSS custom properties for category colours (`--dialogue: #3498db`, `--singulative: #2ecc71`, `--iterative: #e67e22`, `--description: #9b59b6`, `--fid: #e74c3c`, `--commentary: #95a5a6`) and genre colours (`--domestic: #2ecc71`, `--gothic: #9b59b6`, `--picaresque: #e74c3c`, `--epistolary: #3498db`, `--amatory: #e67e22`, `--satirical: #95a5a6`).

**HTML structure:**
- Header with title and subtitle
- Tab bar: Century | Arcs | Grid | Ecology
- Four `<div class="view">` containers, each with appropriate canvas/control elements
- Tooltip div

**JavaScript (~500 lines):**
- `DATA = __NOVELS_DATA__` -- injected at build time
- Tab switching via click handlers
- `drawCentury()`: scatter plot (year vs filler%) with genre colours, linear regression trend line, hover tooltips, click-to-navigate-to-arcs. Below: stacked horizontal bars showing six-category breakdown per novel.
- `drawArc()`: stacked area chart for selected novel. Controls: novel selector dropdown, overlay multi-select, smoothing range slider, volume boundaries checkbox, blended/dominant toggle. Volume boundaries as dashed vertical lines.
- `drawGrid()`: filter pills (genre + author), grid of cards each containing a miniature stacked-bar sparkline. Click navigates to Arcs view.
- `drawEcology()`: dual-panel canvas. Left = filler+FID scatter with trend. Right = environmental evidence bar chart by decade from `DATA.environmental`. Argument callout box below.
- Shared: `showTip(e, text)` / `hideTip()` tooltip helpers, `buildLegend(id, items)` legend builder.

All DOM manipulation uses safe methods: `document.createElement`, `textContent` for labels, canvas `getContext('2d')` for charts. No raw HTML string injection for user-facing data.

The full template follows the visualisation spec sections (Century, Arcs, Grid, Ecology) exactly. Use `__NOVELS_DATA__` as the single placeholder, replaced via `.replace()`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest gazetteer/tests/test_build_narrative_pace.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add gazetteer/build_narrative_pace.py gazetteer/tests/test_build_narrative_pace.py
git commit -m "feat: add HTML builder with Century, Arcs, Grid, and Ecology tabs"
```

---

### Task 8: Run full pipeline and verify end-to-end

**Files:**
- None created -- integration run

- [ ] **Step 1: Run analysis on full corpus**

Run:
```bash
python3 gazetteer/analyse_narrative_pace.py
```
Expected: processes all novels (may take 10-20 minutes), creates `narrative_pace.db` and `narrative_pace_data.json`

- [ ] **Step 2: Build HTML**

Run:
```bash
python3 gazetteer/build_narrative_pace.py
```
Expected: creates `narrative_pace.html`

- [ ] **Step 3: Run all tests**

Run:
```bash
pytest gazetteer/tests/test_narrative_pace_corpus.py gazetteer/tests/test_narrative_pace_classify.py gazetteer/tests/test_build_narrative_pace.py -v
```
Expected: all tests PASS

- [ ] **Step 4: Verify HTML opens and check all four tabs**

Run:
```bash
open gazetteer/narrative_pace.html
```
Expected: four-tab page loads with data visible in all tabs. Verify:
- Century: scatter plot with dots, trend line, stacked bars below
- Arcs: stacked area chart, dropdown works, volume boundaries visible on multi-volume works
- Grid: cards render with sparklines, filter pills work
- Ecology: dual panels with data, callout text visible

- [ ] **Step 5: Add generated files to gitignore and commit**

Add `narrative_pace.db` to `.gitignore` (large binary), then commit the JSON and HTML:

```bash
echo "gazetteer/narrative_pace.db" >> .gitignore
git add .gitignore gazetteer/narrative_pace_data.json gazetteer/narrative_pace.html
git commit -m "feat: generate narrative pace data and visualisation for full corpus"
```
