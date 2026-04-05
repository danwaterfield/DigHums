"""Tests for narrative_pace_classify.py — dialogue classifier (Task 3),
singulative/iterative/description classifiers (Task 4), and FID/commentary
classifiers and epistolary flag (Task 5)."""

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
    def test_non_dialogue_is_not_dialogue(self):
        doc = NLP("She walked slowly toward the window.")
        result = classify_sentence(doc)
        assert result["is_dialogue"] is False

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
        # dominant category should be one of the five
        valid = {"singulative", "iterative", "description", "fid", "commentary"}
        assert result["dominant_category"] in valid

    def test_epistolary_flag_accepted(self):
        doc = NLP("I write to you from Bath.")
        result = classify_sentence(doc, epistolary=True)
        assert "is_dialogue" in result


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
