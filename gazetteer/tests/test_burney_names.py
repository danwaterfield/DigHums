"""Tests for gazetteer/burney_names.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from burney_names import normalise, community, is_artefact, CANONICAL_NAMES, COMMUNITIES


# ── Catalogue abbreviation mapping ────────────────────────────────

def test_sbp():
    assert normalise("SBP") == "Susanna Burney Phillips"


def test_seb():
    assert normalise("SEB") == "Susanna Burney Phillips"


def test_hltp():
    assert normalise("HLTP") == "Hester Thrale Piozzi"


def test_cb():
    assert normalise("CB") == "Charles Burney"


def test_cb_jr():
    assert normalise("CB Jr") == "Charles Burney Jr"


def test_cbfb():
    assert normalise("CBFB") == "Charlotte Broome"


def test_crfb():
    assert normalise("CRFB") == "Charlotte Broome"


def test_ebb():
    assert normalise("EBB") == "Esther Burney"


def test_jb():
    assert normalise("JB") == "James Burney"


def test_aa():
    assert normalise("AA") == "Alexander d'Arblay"


def test_m_da():
    assert normalise("M d'A") == "Alexandre d'Arblay"


def test_fba():
    assert normalise("FBA") == "Frances Burney"


def test_fb():
    assert normalise("FB") == "Frances Burney"


def test_cpb():
    assert normalise("CPB") == "Charles Parr Burney"


def test_shb():
    assert normalise("SHB") == "Sarah Harriet Burney"


def test_cfbt():
    assert normalise("CFBt") == "Charlotte Barrett"


def test_cfpb():
    assert normalise("CFPB") == "Charlotte Barrett"


def test_waddington_variants():
    expected = "Georgiana Waddington"
    assert normalise("GMAPWaddington") == expected
    assert normalise("GMAFWaddington") == expected
    assert normalise("CMAPWaddington") == expected


def test_lb():
    assert normalise("LB") == "Lucy Burney"


# ── Existing OUP aliases preserved ────────────────────────────────

def test_susanna_burney_alias():
    assert normalise("Susanna Burney") == "Susanna Burney Phillips"


def test_susanna_phillips_alias():
    assert normalise("Susanna Phillips") == "Susanna Burney Phillips"


def test_dr_burney_alias():
    assert normalise("Dr Burney") == "Dr Charles Burney"


def test_charlotte_cambridge_alias():
    assert normalise("Charlotte Cambridge") == "Charlotte Broome"


def test_charlotte_francis_alias():
    assert normalise("Charlotte Francis") == "Charlotte Broome"


def test_hester_lynch_thrale_alias():
    assert normalise("Hester Lynch Thrale") == "Hester Thrale Piozzi"


def test_publisher_alias():
    assert normalise("Longman, Hurst, Rees, Orme and Brown") == "Longman & Co"
    assert normalise("Messrs Longman and Company") == "Longman & Co"


def test_longmans_alias():
    assert normalise("Longmans & Co") == "Longman & Co"


# ── Passthrough for unknown names ─────────────────────────────────

def test_passthrough_unknown():
    assert normalise("Unknown Person") == "Unknown Person"


def test_passthrough_full_name():
    assert normalise("Samuel Johnson") == "Samuel Johnson"


# ── Community assignment ──────────────────────────────────────────

def test_community_family():
    assert community("SBP") == "Family"
    assert community("Charles Burney") == "Family"
    assert community("Charlotte Broome") == "Family"


def test_community_literary():
    assert community("Samuel Crisp") == "Literary"
    assert community("HLTP") == "Literary"
    assert community("William Bewley") == "Literary"
    assert community("Mrs Crewe") == "Literary"


def test_community_court():
    assert community("Queen Charlotte") == "Court"
    assert community("Princess Elizabeth") == "Court"


def test_community_royal():
    assert community("Princess Sophia") == "Royal"
    assert community("Princess Mary") == "Royal"
    assert community("Princess Augusta") == "Royal"


def test_community_publishers():
    assert community("Thomas Lowndes") == "Publishers"
    assert community("Longman & Co") == "Publishers"


def test_community_intimate():
    assert community("Frederica Locke") == "Intimate circle"
    assert community("Viscountess Keith") == "Intimate circle"


def test_community_french():
    assert community("Marie de Maisonneuve") == "French circle"
    assert community("M d'A") == "Family"  # husband is Family, not French


def test_community_musical():
    assert community("Thomas Twining") == "Musical circle"
    assert community("Padre Martini") == "Musical circle"
    assert community("David Garrick") == "Musical circle"


def test_community_scholarly():
    assert community("Samuel Parr") == "Scholarly/Church"


def test_community_unknown():
    assert community("Totally Unknown Person") == "Unknown"


def test_community_via_abbreviation():
    """Community lookup works through abbreviation -> canonical -> community."""
    assert community("EBB") == "Family"
    assert community("CB Jr") == "Family"
    assert community("FBA") == "Family"


# ── Artefact filtering ────────────────────────────────────────────

def test_artefact_question_mark():
    assert is_artefact("[?]")
    assert is_artefact("?")


def test_artefact_repository_codes():
    assert is_artefact("NYPL(B)")
    assert is_artefact("BM(Bar)")
    assert is_artefact("PML")
    assert is_artefact("Comyn")
    assert is_artefact("Osb")
    assert is_artefact("Hyde")


def test_artefact_direction_words():
    assert is_artefact("to")
    assert is_artefact("from")


def test_artefact_empty():
    assert is_artefact("")


def test_artefact_copy_draft():
    assert is_artefact("copy")
    assert is_artefact("draft")


def test_not_artefact_real_name():
    assert not is_artefact("Samuel Crisp")
    assert not is_artefact("SBP")
    assert not is_artefact("Thomas Twining")


# ── Parenthetical name normalisation ──────────────────────────────

def test_maria_rishton():
    assert normalise("Maria (Allen) Rishton") == "Maria Rishton"


def test_amelia_angerstein():
    assert normalise("Amelia (Locke) Angerstein") == "Amelia Angerstein"


def test_ann_agnew():
    assert normalise("Ann (Astley) Agnew") == "Ann Agnew"


def test_uncertainty_marker_cleanup():
    assert normalise("William Cutler [?]") == "William Cutler"
    assert normalise("[George Henry Law?]") == "George Henry Law"


def test_trailing_note_cleanup():
    assert normalise("Samuel Crisp (with PS by CB)") == "Samuel Crisp"
    assert normalise("George Hay (on verso of NYPL(B))") == "George Hay"


def test_title_and_suffix_cleanup():
    assert normalise("Mr. George Crabbe") == "George Crabbe"
    assert normalise("James Boswell, Esq") == "James Boswell"


def test_parenthetical_and_married_name_cleanup():
    assert normalise("Elizabeth (Allen) Meeke") == "Elizabeth Meeke"
    assert normalise("Esther (Sleepe) Burney") == "Esther Burney"


def test_caroline_princess_variant_cleanup():
    assert normalise("Caroline, Princess of Wales") == "Princess Caroline"
    assert (
        normalise("Caroline, Princess of Wales (per Lady Charlotte Campbell)")
        == "Princess Caroline"
    )


def test_chapter_prefix_cleanup():
    assert normalise("ch 22. Rev. Dr. Leland") == "Rev. Dr. Leland"


def test_final_ocr_cleanup_labels():
    assert normalise("('ait.  James Ceib") == "Capt. James Keir"
    assert normalise("Brigit[a] Piez[e] [Frontoni]") == "Brigita Piez Frontoni"
