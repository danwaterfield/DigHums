"""Tests for narrative_pace_classify.py — dialogue classifier (Task 3)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import spacy
import pytest
from narrative_pace_classify import classify_sentence, SPEECH_VERBS

NLP = spacy.load("en_core_web_sm")


class TestDialogueDetection:
    def test_quoted_speech_is_dialogue(self):
        doc = NLP('"I am vastly pleased," said she.')
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
        assert result["fid"] == 0.0

    def test_speech_verbs_list_is_comprehensive(self):
        expected = {"said", "cried", "answered", "returned", "replied",
                    "continued", "added", "repeated", "interrupted",
                    "declared", "exclaimed", "demanded", "pursued",
                    "resumed", "entreated", "whispered", "observed",
                    "inquired", "rejoined", "ejaculated"}
        assert expected.issubset(SPEECH_VERBS)


class TestNonDialoguePlaceholders:
    def test_non_dialogue_all_scores_0_2(self):
        doc = NLP("She walked slowly toward the window.")
        result = classify_sentence(doc)
        assert result["is_dialogue"] is False
        assert result["singulative"] == pytest.approx(0.2)
        assert result["iterative"] == pytest.approx(0.2)
        assert result["description"] == pytest.approx(0.2)
        assert result["fid"] == pytest.approx(0.2)
        assert result["commentary"] == pytest.approx(0.2)

    def test_non_dialogue_scores_sum_to_1(self):
        doc = NLP("He opened the letter and read it twice.")
        result = classify_sentence(doc)
        total = (result["singulative"] + result["iterative"] +
                 result["description"] + result["fid"] + result["commentary"])
        assert total == pytest.approx(1.0)

    def test_dialogue_non_dialogue_scores_sum_to_0(self):
        doc = NLP('"You are mistaken," said Cecilia.')
        result = classify_sentence(doc)
        total = (result["singulative"] + result["iterative"] +
                 result["description"] + result["fid"] + result["commentary"])
        assert total == pytest.approx(0.0)


class TestReturnShape:
    def test_all_keys_present(self):
        doc = NLP("The fire crackled in the grate.")
        result = classify_sentence(doc)
        expected_keys = {"is_dialogue", "singulative", "iterative",
                         "description", "fid", "commentary", "dominant_category"}
        assert expected_keys == set(result.keys())

    def test_dominant_category_dialogue(self):
        doc = NLP('"Come here," she whispered.')
        result = classify_sentence(doc)
        assert result["dominant_category"] == "dialogue"

    def test_dominant_category_non_dialogue_placeholder(self):
        doc = NLP("The sky was perfectly clear that morning.")
        result = classify_sentence(doc)
        # With uniform 0.2 scores the dominant category should be one of the five
        valid = {"singulative", "iterative", "description", "fid", "commentary"}
        assert result["dominant_category"] in valid

    def test_epistolary_flag_accepted(self):
        doc = NLP("I write to you from Bath.")
        result = classify_sentence(doc, epistolary=True)
        assert "is_dialogue" in result
