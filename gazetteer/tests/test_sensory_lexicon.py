import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from sensory_lexicon import tag_modalities, MODALITY_PATTERNS

def test_auditory_match():
    results = tag_modalities("The din of the carriages was insupportable.")
    assert any(m == "auditory" for _, m in results)

def test_olfactory_match():
    results = tag_modalities("A most offensive stench arose from the kennel.")
    assert any(m == "olfactory" for _, m in results)

def test_visual_match():
    results = tag_modalities("The street was narrow and the buildings lofty.")
    assert any(m == "visual" for _, m in results)

def test_no_false_positive():
    results = tag_modalities("She had a feeling of relief.")
    assert results == []

def test_returns_matched_term():
    results = tag_modalities("The clatter of hooves echoed down the street.")
    terms = [t for t, _ in results]
    assert "clatter" in terms

def test_multiple_modalities():
    text = "The smoke was thick and the din of the mob overwhelming."
    results = tag_modalities(text)
    modalities = {m for _, m in results}
    assert "auditory" in modalities
    assert "olfactory" in modalities
    assert "crowd" in modalities
