"""Sentence-level narrative mode classifier.

Implements dialogue detection plus singulative, iterative, description,
free indirect discourse (FID), and commentary scorers, with an epistolary
genre flag that adjusts tense logic for letter-novels.
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
# FID lexicon
# ---------------------------------------------------------------------------

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
FID_DEICTICS = frozenset({"now", "here", "tomorrow", "yesterday", "tonight", "this"})

# ---------------------------------------------------------------------------
# Commentary lexicon
# ---------------------------------------------------------------------------

COMMENTARY_PRONOUNS = frozenset({"we", "our", "us"})
COMMENTARY_NOUNS = frozenset({"mankind", "reader", "world"})
MORAL_VOCABULARY = frozenset({
    "honour", "honor", "virtue", "duty", "tenderness", "gratitude",
    "esteem", "prudence", "delicacy", "sensibility", "propriety",
    "fortitude", "benevolence", "compassion", "modesty", "discretion",
    "condescension",
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


def _score_fid(doc: Doc, epistolary: bool = False) -> float:
    """Score a sentence for free indirect discourse."""
    if not doc:
        return 0.0
    text = doc.text
    text_lower = text.lower()
    score = 0.0
    tokens = list(doc)

    # Exclamatory "How/What a" without quotes
    if not _QUOTE_RE.search(text):
        if re.match(r"(?i)^(how|what\s+a)\b", text.strip()):
            score += 4.0

        # Interrogative (ends with ?)
        if text.strip().endswith("?"):
            score += 2.5

    # Evaluative adjectives (lemma in EVALUATIVE_ADJ)
    for token in tokens:
        if token.lemma_.lower() in EVALUATIVE_ADJ:
            # handle "prodigious"/"monstrous" as ADV vs ADJ separately below
            if token.lemma_.lower() not in {"prodigious", "monstrous"}:
                score += 1.5

    # Epistemic hedges (exact token text)
    for token in tokens:
        if token.lower_ in EPISTEMIC_HEDGES:
            score += 1.0

    # Epistemic phrases
    for phrase in EPISTEMIC_PHRASES:
        if phrase in text_lower:
            score += 1.5

    # Deictic shift: deictic words in past-tense context, not epistolary
    # "past-tense context" includes VBD tokens or narrative modal auxiliaries (would/could)
    if not epistolary:
        has_past = any(t.tag_ == "VBD" for t in tokens) or any(
            t.lower_ in {"would", "could"} and t.pos_ == "AUX" for t in tokens
        )
        if has_past:
            for token in tokens:
                if token.lower_ in FID_DEICTICS:
                    score += 2.0

    # "prodigious"/"monstrous" as ADV (+2.0) or ADJ (+0.8)
    # Also treat as intensifier (ADV, +2.0) if immediately followed by another ADJ
    for i, token in enumerate(tokens):
        if token.lemma_.lower() in {"prodigious", "monstrous"}:
            if token.pos_ == "ADV":
                score += 2.0
            elif token.dep_ == "amod" and token.head.pos_ == "ADJ":
                # modifying an adjective directly — intensifier
                score += 2.0
            elif i + 1 < len(tokens) and tokens[i + 1].pos_ == "ADJ":
                # immediately precedes another adjective — intensifier position
                score += 2.0
            else:
                score += 0.8

    return score / len(doc)


def _score_commentary(doc: Doc, epistolary: bool = False) -> float:
    """Score a sentence for authorial commentary."""
    if not doc:
        return 0.0
    text_lower = _lowered_text(doc)
    score = 0.0
    tokens = list(doc)

    # First-person plural pronouns
    for token in tokens:
        if token.lower_ in COMMENTARY_PRONOUNS:
            score += 2.0

    # "reader" anywhere in text (high-signal address)
    if "reader" in text_lower:
        score += 4.0

    # Generalising nouns (lemma in COMMENTARY_NOUNS)
    has_generalising = False
    for token in tokens:
        if token.lemma_.lower() in COMMENTARY_NOUNS:
            score += 1.5
            has_generalising = True

    # Moral vocabulary
    for token in tokens:
        if token.lemma_.lower() in MORAL_VOCABULARY:
            score += 1.0

    # Present-tense verbs
    present_verbs = [
        t for t in tokens
        if t.tag_ in {"VBP", "VBZ"} and t.pos_ == "VERB"
    ]
    if not epistolary:
        score += len(present_verbs) * 1.5
    else:
        # In epistolary mode, present tense only contributes if generalising nouns present
        if has_generalising:
            score += len(present_verbs) * 1.0

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
    If not dialogue: all five scorers are applied and scores are normalised.
    """
    dialogue = _is_dialogue(doc)

    result = {"is_dialogue": dialogue}

    if dialogue:
        scores = {cat: 0.0 for cat in _NON_DIALOGUE_CATEGORIES}
        result.update(scores)
        result["dominant_category"] = "dialogue"
        return result

    sing_raw = _score_singulative(doc)
    iter_raw = _score_iterative(doc)
    desc_raw = _score_description(doc)
    fid_raw = _score_fid(doc, epistolary=epistolary)
    comm_raw = _score_commentary(doc, epistolary=epistolary)

    # Epistolary boost: present-tense verbs boost singulative ("writing to the moment")
    if epistolary:
        present_verb_count = sum(
            1 for t in doc if t.tag_ in {"VBP", "VBZ"} and t.pos_ == "VERB"
        )
        sing_raw += (present_verb_count * 1.5) / len(doc)

    scores = {
        "singulative": sing_raw,
        "iterative": iter_raw,
        "description": desc_raw,
        "fid": fid_raw,
        "commentary": comm_raw,
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
