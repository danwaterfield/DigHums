"""Tests for gazetteer/build_correspondent_network.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from build_correspondent_network import parse_headers


# --- Fixture: a minimal subset of real headers ---

SAMPLE_TEXT = """\
1. Journal 27 March 1768

To have some account of my thoughts...

9. Verse Letter to Dr Charles Burney 23 June 1769

To Doctor Last...

17. From Journal Letters to Susanna Burney
(Teignmouth Journal) August–September 1773

We set out from Teignmouth...

44. From Letter to Susanna Burney 5 July 1778

My dearest Susy...

79. From Journal Letter to Susanna Burney and Charlotte Ann Burney June 1781

My dearest girls...

103. From Letter to Charlotte Cambridge 12June 1786

My dearest Charlotte...

209. Waterloo Journal 27 April and 13 May 1815

My best friend left me...

26. From Letter to Samuel Crisp 2 March 1775 and Journal 1775

My dear Daddy Crisp...

36. Letters to Thomas Lowndes 25 and 26 December 1776

Sir, I take the liberty...

214. Journal for 22 July 1815

The day began...
"""


def test_parse_headers_counts():
    entries = parse_headers(SAMPLE_TEXT)
    assert len(entries) == 10


def test_pure_journal_has_no_correspondent():
    entries = parse_headers(SAMPLE_TEXT)
    journal = [e for e in entries if e["number"] == 1][0]
    assert journal["correspondents"] == []
    assert journal["type"] == "journal"
    assert journal["year"] == 1768


def test_letter_extracts_correspondent():
    entries = parse_headers(SAMPLE_TEXT)
    letter = [e for e in entries if e["number"] == 44][0]
    assert letter["correspondents"] == ["Susanna Burney"]
    assert letter["type"] == "letter"
    assert letter["year"] == 1778


def test_verse_letter_extracts_correspondent():
    entries = parse_headers(SAMPLE_TEXT)
    verse = [e for e in entries if e["number"] == 9][0]
    assert verse["correspondents"] == ["Dr Charles Burney"]
    assert verse["type"] == "verse letter"


def test_multi_recipient_splits():
    entries = parse_headers(SAMPLE_TEXT)
    multi = [e for e in entries if e["number"] == 79][0]
    assert set(multi["correspondents"]) == {
        "Susanna Burney", "Charlotte Ann Burney"
    }


def test_location_journal_excluded():
    entries = parse_headers(SAMPLE_TEXT)
    waterloo = [e for e in entries if e["number"] == 209][0]
    assert waterloo["correspondents"] == []
    assert waterloo["type"] == "journal"


def test_compound_entry_extracts_letter_only():
    entries = parse_headers(SAMPLE_TEXT)
    compound = [e for e in entries if e["number"] == 26][0]
    assert compound["correspondents"] == ["Samuel Crisp"]
    assert compound["year"] == 1775


def test_plural_letters():
    entries = parse_headers(SAMPLE_TEXT)
    plural = [e for e in entries if e["number"] == 36][0]
    assert plural["correspondents"] == ["Thomas Lowndes"]
    assert plural["type"] == "letter"


def test_journal_for_excluded():
    entries = parse_headers(SAMPLE_TEXT)
    jf = [e for e in entries if e["number"] == 214][0]
    assert jf["correspondents"] == []
    assert jf["type"] == "journal"


def test_multi_line_header():
    entries = parse_headers(SAMPLE_TEXT)
    e17 = [e for e in entries if e["number"] == 17][0]
    assert e17["correspondents"] == ["Susanna Burney"]
    assert e17["year"] == 1773


def test_month_range_date():
    entries = parse_headers(SAMPLE_TEXT)
    e17 = [e for e in entries if e["number"] == 17][0]
    assert e17["month"] == 8


def test_no_space_date():
    entries = parse_headers(SAMPLE_TEXT)
    e103 = [e for e in entries if e["number"] == 103][0]
    assert e103["year"] == 1786
    assert e103["correspondents"] == ["Charlotte Cambridge"]


# ── Task 2: Name normalisation & community assignment ───────────────

from build_correspondent_network import (
    normalise_name, assign_community, assign_phase,
    NAME_ALIASES, COMMUNITIES,
)


def test_normalise_susanna():
    assert normalise_name("Susanna Burney") == "Susanna Burney Phillips"
    assert normalise_name("Susanna Phillips") == "Susanna Burney Phillips"


def test_normalise_thrale_piozzi():
    assert normalise_name("Hester Lynch Thrale") == "Hester Thrale Piozzi"
    assert normalise_name("Hester Lynch Piozzi") == "Hester Thrale Piozzi"


def test_normalise_dr_burney():
    assert normalise_name("Dr Burney") == "Dr Charles Burney"
    assert normalise_name("Dr Charles Burney") == "Dr Charles Burney"


def test_normalise_charlotte_broome():
    assert normalise_name("Charlotte Cambridge") == "Charlotte Broome"


def test_alexandre_vs_alexander_distinct():
    assert normalise_name("Alexandre d'Arblay") != normalise_name("Alexander d'Arblay")


def test_publisher_normalisation():
    assert normalise_name("Longman, Hurst, Rees, Orme and Brown") == "Longman & Co"
    assert normalise_name("Messrs Longman and Company") == "Longman & Co"


def test_community_family():
    assert assign_community("Dr Charles Burney") == "Family"
    assert assign_community("Susanna Burney Phillips") == "Family"
    assert assign_community("Alexandre d'Arblay") == "Family"
    assert assign_community("Alexander d'Arblay") == "Family"


def test_community_literary():
    assert assign_community("Samuel Crisp") == "Literary"
    assert assign_community("Hester Thrale Piozzi") == "Literary"


def test_community_court():
    assert assign_community("Queen Charlotte") == "Court"


def test_community_publishers():
    assert assign_community("Thomas Lowndes") == "Publishers"
    assert assign_community("Longman & Co") == "Publishers"


def test_community_intimate():
    assert assign_community("Frederica Locke") == "Intimate circle"


def test_community_unknown_flagged(capsys):
    result = assign_community("Unknown Person")
    assert result == "Unknown"
    assert "Unknown Person" in capsys.readouterr().err


def test_phase_apprentice():
    assert assign_phase(1770, None) == "Apprentice Years"


def test_phase_court():
    assert assign_phase(1786, None) == "Court Years"


def test_phase_boundary_1786():
    assert assign_phase(1786, None) == "Court Years"


def test_phase_france_mid_1802():
    assert assign_phase(1802, 7) == "France"
    assert assign_phase(1802, 3) == "Camilla & Camilla Cottage"
    assert assign_phase(1802, None) == "Camilla & Camilla Cottage"


def test_phase_widowhood():
    assert assign_phase(1820, None) == "Widowhood"
    assert assign_phase(1839, None) == "Widowhood"


# ── Task 3: JSON data assembly ──────────────────────────────────────

from build_correspondent_network import build_network_data


def test_build_network_data_structure():
    data = build_network_data(SAMPLE_TEXT)
    assert "nodes" in data
    assert "edges" in data
    assert "letters" in data
    burney = [n for n in data["nodes"] if n["id"] == "Frances Burney"]
    assert len(burney) == 1


def test_build_network_data_edges():
    data = build_network_data(SAMPLE_TEXT)
    correspondent_ids = {e["target"] for e in data["edges"]}
    assert "Dr Charles Burney" in correspondent_ids
    assert "Susanna Burney Phillips" in correspondent_ids


def test_build_network_data_no_journal_edges():
    data = build_network_data(SAMPLE_TEXT)
    for edge in data["edges"]:
        assert edge["target"] != "Frances Burney"


def test_letters_list_has_required_fields():
    data = build_network_data(SAMPLE_TEXT)
    for letter in data["letters"]:
        assert "number" in letter
        assert "correspondent" in letter
        assert "year" in letter
        assert "type" in letter
        assert "phase" in letter


# ── Task 4: Build function & HTML output ──────────────────────────

import re
import subprocess
import tempfile

from build_correspondent_network import build


def test_build_produces_html():
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        out = Path(f.name)
    try:
        build(out_path=out)
        html = out.read_text(encoding="utf-8")
        assert "<!doctype html>" in html.lower()
        assert "Frances Burney" in html
        assert "d3" in html.lower()
    finally:
        out.unlink(missing_ok=True)


def test_build_html_has_no_unsubstituted_placeholders():
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        out = Path(f.name)
    try:
        build(out_path=out)
        html = out.read_text(encoding="utf-8")
        unsubst = re.findall(r"(?<!\{)\{[A-Z_]+\}(?!\})", html)
        assert unsubst == [], f"Unsubstituted placeholders: {unsubst}"
    finally:
        out.unlink(missing_ok=True)


def test_build_cli(tmp_path):
    out = tmp_path / "network.html"
    result = subprocess.run(
        [sys.executable, "-c",
         f"from pathlib import Path; import sys; sys.path.insert(0,'gazetteer');"
         f"from build_correspondent_network import build; build(out_path=Path('{out}'))"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out.exists()


# ── Task 5: Integration tests against real data ──────────────────

from build_correspondent_network import TEXT_PATH


def test_real_data_parse_count():
    """Verify we parse all headers from the real OUP text."""
    text = TEXT_PATH.read_text(encoding="utf-8")
    entries = parse_headers(text)
    assert len(entries) >= 240, f"Only parsed {len(entries)} entries (expected ~243)"
    for e in entries:
        assert e["year"] is not None, f"Entry {e['number']} has no year"


def test_real_data_all_correspondents_have_communities():
    """Every correspondent in the real data should be assigned a community."""
    text = TEXT_PATH.read_text(encoding="utf-8")
    data = build_network_data(text)
    unknowns = [
        n["id"] for n in data["nodes"]
        if n["community"] == "Unknown"
    ]
    assert unknowns == [], f"Unassigned correspondents: {unknowns}"


def test_real_data_susanna_is_top_correspondent():
    """Susanna should have the most letters — sanity check."""
    text = TEXT_PATH.read_text(encoding="utf-8")
    data = build_network_data(text)
    nodes = sorted(data["nodes"], key=lambda n: -n["count"])
    assert nodes[0]["id"] == "Frances Burney"
    assert "Susanna" in nodes[1]["id"]


# ── Task 6: Catalogue-based pipeline tests ────────────────────────

from build_correspondent_network import load_catalogue, build_catalogue_network


def test_load_catalogue_frances_count():
    rows = load_catalogue("frances")
    assert len(rows) >= 6000, f"Expected ~6057 rows, got {len(rows)}"


def test_load_catalogue_cb_sr_count():
    rows = load_catalogue("cb_sr")
    assert len(rows) >= 950, f"Expected ~991 rows, got {len(rows)}"


def test_load_catalogue_cb_jr_count():
    rows = load_catalogue("cb_jr")
    assert len(rows) >= 900, f"Expected ~908 rows, got {len(rows)}"


def test_load_catalogue_invalid_key():
    import pytest
    with pytest.raises(ValueError, match="Unknown catalogue key"):
        load_catalogue("invalid")


def test_load_catalogue_has_required_fields():
    rows = load_catalogue("frances")
    for key in ("date", "direction", "correspondent"):
        assert key in rows[0], f"Missing field: {key}"


def test_build_catalogue_network_structure_frances():
    data = build_catalogue_network("frances")
    assert "subject" in data
    assert "nodes" in data
    assert "edges" in data
    assert "letters" in data
    assert "phases" in data
    assert data["subject"] == "Frances Burney d'Arblay"


def test_build_catalogue_network_structure_cb_sr():
    data = build_catalogue_network("cb_sr")
    assert data["subject"] == "Dr Charles Burney"


def test_build_catalogue_network_structure_cb_jr():
    data = build_catalogue_network("cb_jr")
    assert data["subject"] == "Charles Burney Jr"


def test_catalogue_nodes_have_direction_counts():
    data = build_catalogue_network("frances")
    for node in data["nodes"]:
        assert "count" in node
        assert "to_count" in node
        assert "from_count" in node


def test_catalogue_edges_have_direction_weights():
    data = build_catalogue_network("frances")
    for edge in data["edges"]:
        assert "weight" in edge
        assert "to_weight" in edge
        assert "from_weight" in edge


def test_catalogue_direction_sum():
    """to_weight + from_weight == weight for every edge."""
    data = build_catalogue_network("frances")
    for edge in data["edges"]:
        assert edge["to_weight"] + edge["from_weight"] == edge["weight"], (
            f"Direction mismatch for {edge['target']}: "
            f"{edge['to_weight']} + {edge['from_weight']} != {edge['weight']}"
        )


def test_catalogue_artefacts_filtered():
    """Artefacts like [?], NYPL(B) etc. should not appear as nodes."""
    data = build_catalogue_network("frances")
    node_ids = {n["id"] for n in data["nodes"]}
    assert "[?]" not in node_ids
    assert "NYPL(B)" not in node_ids
    assert "BM(Bar)" not in node_ids
    assert "to" not in node_ids
    assert "from" not in node_ids
    assert "" not in node_ids


def test_catalogue_artefacts_filtered_cb_sr():
    data = build_catalogue_network("cb_sr")
    node_ids = {n["id"] for n in data["nodes"]}
    assert "[?]" not in node_ids


def test_catalogue_frances_susanna_top():
    """Susanna should be the top correspondent in the Frances catalogue."""
    data = build_catalogue_network("frances")
    # Skip the subject node (index 0)
    correspondents = data["nodes"][1:]
    top = correspondents[0]
    assert "Susanna" in top["id"], f"Expected Susanna at top, got {top['id']}"


def test_catalogue_cb_sr_twining_top():
    """Thomas Twining should be the top correspondent in CB Sr catalogue."""
    data = build_catalogue_network("cb_sr")
    correspondents = data["nodes"][1:]
    # Twining should be in the top 3 (after FBA which is huge)
    top_names = [n["id"] for n in correspondents[:3]]
    assert any("Twining" in n for n in top_names), (
        f"Expected Twining in top 3, got {top_names}"
    )


def test_catalogue_frances_phases():
    data = build_catalogue_network("frances")
    labels = [p["label"] for p in data["phases"]]
    assert "Apprentice Years" in labels
    assert "Widowhood" in labels


def test_catalogue_cb_sr_phases():
    data = build_catalogue_network("cb_sr")
    labels = [p["label"] for p in data["phases"]]
    assert "Lynn & Early Career" in labels
    assert "Final Years" in labels


def test_catalogue_cb_jr_phases():
    data = build_catalogue_network("cb_jr")
    labels = [p["label"] for p in data["phases"]]
    assert "Early Life & Cambridge" in labels
    assert "Late Career & DD" in labels


def test_catalogue_no_self_loops_frances():
    """The subject should not appear as a correspondent edge target."""
    data = build_catalogue_network("frances")
    targets = {e["target"] for e in data["edges"]}
    assert "Frances Burney d'Arblay" not in targets
    assert "Frances Burney" not in targets


def test_catalogue_no_self_loops_cb_sr():
    data = build_catalogue_network("cb_sr")
    targets = {e["target"] for e in data["edges"]}
    assert "Dr Charles Burney" not in targets
    assert "Charles Burney" not in targets


def test_catalogue_letters_have_direction():
    data = build_catalogue_network("frances")
    for letter in data["letters"][:20]:
        assert "direction" in letter
        assert "correspondent" in letter
        assert "phase" in letter


# ── Task 7: Three-network HTML build ─────────────────────────────

def test_build_produces_html_with_three_networks():
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        out = Path(f.name)
    try:
        build(out_path=out)
        html = out.read_text(encoding="utf-8")
        assert "Frances Burney" in html
        assert "Dr Charles Burney" in html or "Charles Burney Mus Doc" in html
        assert "Charles Burney Jr" in html or "Charles Burney DD" in html
    finally:
        out.unlink(missing_ok=True)


def test_build_html_has_catalogue_json():
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        out = Path(f.name)
    try:
        build(out_path=out)
        html = out.read_text(encoding="utf-8")
        # The catalogue data should embed all three keys
        assert '"frances"' in html
        assert '"cb_sr"' in html
        assert '"cb_jr"' in html
    finally:
        out.unlink(missing_ok=True)


def test_build_html_has_tab_buttons():
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        out = Path(f.name)
    try:
        build(out_path=out)
        html = out.read_text(encoding="utf-8")
        assert "tab-btn" in html
        assert "Charles Sr" in html
        assert "Charles Jr" in html
    finally:
        out.unlink(missing_ok=True)
