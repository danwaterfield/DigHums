"""Sentence-level narrative mode classifier.

Currently implements dialogue detection only.  The five remaining classifiers
(singulative, iterative, description, FID, commentary) are placeholders that
return uniform 0.2 scores and will be filled in by later tasks.
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


def _is_dialogue(doc: Doc) -> bool:
    """Detect dialogue: presence of quotation marks."""
    return bool(_QUOTE_RE.search(doc.text))


def classify_sentence(doc: Doc, epistolary: bool = False) -> dict:
    """Classify a spaCy-processed sentence.

    Returns a dict with:
      is_dialogue (bool)
      singulative, iterative, description, fid, commentary (float)
      dominant_category (str)

    If dialogue: all 5 non-dialogue scores = 0.0.
    If not dialogue: all 5 scores = 0.2 (placeholder).
    """
    dialogue = _is_dialogue(doc)

    if dialogue:
        scores = {cat: 0.0 for cat in _NON_DIALOGUE_CATEGORIES}
        dominant = "dialogue"
    else:
        scores = {cat: 0.2 for cat in _NON_DIALOGUE_CATEGORIES}
        # All equal — pick the first one as the nominal dominant
        dominant = _NON_DIALOGUE_CATEGORIES[0]

    return {
        "is_dialogue": dialogue,
        **scores,
        "dominant_category": dominant,
    }
