"""Sentence-level narrative mode classifier.

Implements dialogue detection plus singulative, iterative, and description
scorers.  FID and commentary remain placeholders (0.0) for a later task.
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

_NON_DIALOGUE_CATEGORIES = ("singulative", "iterative", "description", "fid", "commentary")

# ---------------------------------------------------------------------------
# Singulative lexicon
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Iterative lexicon
# ---------------------------------------------------------------------------

ITERATIVE_MARKERS = frozenset({
    "every", "often", "always", "never", "usually", "generally",
    "commonly", "frequently", "daily", "nightly",
})
ITERATIVE_PHRASES = [
    "used to", "was accustomed to", "was wont to", "were wont to",
    "each morning", "each evening", "each day", "ever and anon",
]

# ---------------------------------------------------------------------------
# Description lexicon
# ---------------------------------------------------------------------------

SPATIAL_PREPOSITIONS = frozenset({
    "above", "beneath", "beside", "beyond", "within", "around",
    "between", "behind", "below", "underneath", "amidst",
})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_dialogue(doc: Doc) -> bool:
    """Detect dialogue: presence of quotation marks."""
    return bool(_QUOTE_RE.search(doc.text))


def _lowered_text(doc: Doc) -> str:
    """Return the full lowercased text of the doc."""
    return doc.text.lower()


def _score_singulative(doc: Doc) -> float:
    """Score a sentence for singulative (punctual event) narrative mode."""
    if not doc:
        return 0.0
    text_lower = _lowered_text(doc)
    score = 0.0

    # Temporal rupture adverbs
    for token in doc:
        if token.lower_ in SINGULATIVE_TEMPORAL:
            score += 2.0

    # Temporal phrases
    for phrase in SINGULATIVE_TEMPORAL_PHRASES:
        if phrase in text_lower:
            score += 2.0

    # Past-tense action verbs and general VBD count
    for token in doc:
        if token.tag_ == "VBD":
            if token.lemma_.lower() in SINGULATIVE_ACTION_VERBS:
                score += 1.5
            else:
                score += 0.2

    # Proper nouns (named participants signal scene-level specificity)
    for token in doc:
        if token.pos_ == "PROPN":
            score += 0.3

    return score / len(doc)


def _score_iterative(doc: Doc) -> float:
    """Score a sentence for iterative (habitual/repeated) narrative mode."""
    if not doc:
        return 0.0
    text_lower = _lowered_text(doc)
    score = 0.0

    # Frequency adverbs / determiners
    for token in doc:
        if token.lower_ in ITERATIVE_MARKERS:
            score += 2.0

    # Iterative phrases
    for phrase in ITERATIVE_PHRASES:
        if phrase in text_lower:
            score += 3.0

    # Habitual "would": AUX, not preceded by "if", followed by VERB in same subtree
    tokens = list(doc)
    for i, token in enumerate(tokens):
        if token.lower_ == "would" and token.pos_ == "AUX":
            # Check not conditional ("if ... would")
            preceding_text = " ".join(t.lower_ for t in tokens[max(0, i - 5):i])
            if "if" not in preceding_text:
                # Check that a VERB follows somewhere in the sentence
                if any(t.pos_ == "VERB" for t in tokens[i + 1:]):
                    score += 2.5

    # Past progressive: was/were + VBG
    for i, token in enumerate(tokens):
        if token.lower_ in {"was", "were"} and token.tag_ in {"VBD", "VBP"}:
            if i + 1 < len(tokens) and tokens[i + 1].tag_ == "VBG":
                score += 0.5

    return score / len(doc)


def _score_description(doc: Doc) -> float:
    """Score a sentence for descriptive narrative mode."""
    if not doc:
        return 0.0
    score = 0.0
    tokens = list(doc)

    # Copular "be" as ROOT with ADJ child
    for token in tokens:
        if token.lemma_ == "be" and token.dep_ == "ROOT":
            for child in token.children:
                if child.pos_ == "ADJ" and child.dep_ in {"acomp", "attr"}:
                    score += 2.0
                    break
            else:
                # prep child indicates locative description
                for child in token.children:
                    if child.dep_ == "prep":
                        score += 1.0
                        break

    # Spatial prepositions
    for token in tokens:
        if token.lower_ in SPATIAL_PREPOSITIONS:
            score += 1.5

    # Adjective density
    adj_count = sum(1 for t in tokens if t.pos_ == "ADJ")
    verb_count = sum(1 for t in tokens if t.pos_ == "VERB")
    adj_ratio = adj_count / len(doc)
    score += adj_ratio * 8.0

    # Bonus when adjectives dominate verbs
    if verb_count < adj_count:
        score += 1.0

    return score / len(doc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_sentence(doc: Doc, epistolary: bool = False) -> dict:
    """Classify a spaCy-processed sentence.

    Returns a dict with:
      is_dialogue (bool)
      singulative, iterative, description, fid, commentary (float)
      dominant_category (str)

    If dialogue: all 5 non-dialogue scores = 0.0.
    If not dialogue: scores are computed by the three scorers; fid and
    commentary remain 0.0 placeholders.
    """
    dialogue = _is_dialogue(doc)

    result = {"is_dialogue": dialogue}

    if dialogue:
        scores = {cat: 0.0 for cat in _NON_DIALOGUE_CATEGORIES}
        result.update(scores)
        result["dominant_category"] = "dialogue"
        return result

    scores = {
        "singulative": _score_singulative(doc),
        "iterative": _score_iterative(doc),
        "description": _score_description(doc),
        "fid": 0.0,        # placeholder
        "commentary": 0.0,  # placeholder
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
