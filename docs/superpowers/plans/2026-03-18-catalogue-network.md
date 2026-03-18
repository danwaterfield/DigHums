# Catalogue-Powered Correspondent Network Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a three-network correspondent visualisation from 7,956 Hemlow catalogue entries, replacing the 243-entry OUP-only version.

**Architecture:** A shared name dictionary module (`burney_names.py`) normalises catalogue abbreviations. The existing build script is extended to load three CSVs, produce three network datasets, and embed them in a single HTML file with a tab switcher. The HTML template gains directional edges, cross-network badges, a threshold slider, and per-network phase presets.

**Tech Stack:** Python 3 (stdlib only), D3.js v7 (inlined), pytest

**Spec:** `docs/superpowers/specs/2026-03-18-catalogue-network-design.md`

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `gazetteer/burney_names.py` | Canonical names, aliases, communities, normalise/community functions |
| Create | `gazetteer/tests/test_burney_names.py` | Name dictionary tests |
| Modify | `gazetteer/build_correspondent_network.py` | Extended: load 3 CSVs, use burney_names, build 3 networks, enhanced HTML |
| Modify | `gazetteer/tests/test_build_correspondent_network.py` | Extended with catalogue-loading and multi-network tests |
| Output | `gazetteer/correspondent_network.html` | Generated (not committed) |

---

## Task 1: Name Dictionary Module

**Files:**
- Create: `gazetteer/burney_names.py`
- Create: `gazetteer/tests/test_burney_names.py`

- [ ] **Step 1: Write failing tests for name normalisation**

```python
"""Tests for gazetteer/burney_names.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from burney_names import normalise, community, CANONICAL_NAMES, COMMUNITIES


def test_catalogue_abbreviations():
    assert normalise("SBP") == "Susanna Burney Phillips"
    assert normalise("HLTP") == "Hester Thrale Piozzi"
    assert normalise("CB") == "Charles Burney"
    assert normalise("CB Jr") == "Charles Burney Jr"
    assert normalise("CBFB") == "Charlotte Broome"
    assert normalise("CRFB") == "Charlotte Broome"
    assert normalise("SEB") == "Susanna Burney Phillips"
    assert normalise("EBB") == "Esther Burney"
    assert normalise("JB") == "James Burney"
    assert normalise("AA") == "Alexander d'Arblay"
    assert normalise("M d'A") == "Alexandre d'Arblay"
    assert normalise("FBA") == "Frances Burney"
    assert normalise("FB") == "Frances Burney"
    assert normalise("CPB") == "Charles Parr Burney"
    assert normalise("SHB") == "Sarah Harriet Burney"
    assert normalise("CFBt") == "Charlotte Barrett"


def test_existing_aliases_preserved():
    """Aliases from v1 build script still work."""
    assert normalise("Susanna Burney") == "Susanna Burney Phillips"
    assert normalise("Susanna Phillips") == "Susanna Burney Phillips"
    assert normalise("Dr Burney") == "Charles Burney"
    assert normalise("Dr Charles Burney") == "Charles Burney"
    assert normalise("Hester Lynch Thrale") == "Hester Thrale Piozzi"
    assert normalise("Charlotte Cambridge") == "Charlotte Broome"
    assert normalise("Charlotte Francis") == "Charlotte Broome"


def test_passthrough_for_unknown():
    assert normalise("Thomas Twining") == "Thomas Twining"
    assert normalise("Padre Martini") == "Padre Martini"


def test_community_assignment():
    assert community("Susanna Burney Phillips") == "Family"
    assert community("Hester Thrale Piozzi") == "Literary"
    assert community("Queen Charlotte") == "Court"
    assert community("Thomas Twining") == "Musical circle"
    assert community("Marie de Maisonneuve") == "French circle"
    assert community("Princess Sophia") == "Royal"
    assert community("Samuel Parr") == "Scholarly/Church"
    assert community("Longman & Co") == "Publishers"


def test_community_unknown():
    result = community("Completely Unknown Person")
    assert result == "Unknown"


def test_filtering_artefacts():
    """Parsing artefacts should not be in CANONICAL_NAMES."""
    for artefact in ["[?]", "NYPL(B)", "BM(Bar)", "to", ""]:
        assert artefact not in CANONICAL_NAMES
```

- [ ] **Step 2: Run tests — expect ImportError**

Run: `pytest gazetteer/tests/test_burney_names.py -v`

- [ ] **Step 3: Implement `burney_names.py`**

Create `gazetteer/burney_names.py` with:
- `CANONICAL_NAMES` dict: map every catalogue abbreviation and name variant to a canonical form. Source from the existing `NAME_ALIASES` dict in `build_correspondent_network.py` plus all abbreviations found in the catalogue CSVs. Include at minimum: SBP, SEB, HLTP, CB, CB Jr, CBFB, CRFB, EBB, JB, AA, M d'A, FBA, FB, CPB, SHB, CFBt, GMAPWaddington, GMAFWaddington, CMAPWaddington, LB, and all full-name variants.
- `COMMUNITIES` dict: map canonical names to community categories. Populate from the existing `COMMUNITIES` dict plus new entries for Charles Sr's and Jr's networks (Thomas Twining → Musical circle, Padre Martini → Musical circle, Samuel Parr → Scholarly/Church, Marie de Maisonneuve → French circle, Princess Sophia/Mary/Augusta → Royal, etc.)
- `normalise(name)` function: look up in CANONICAL_NAMES, return canonical form or passthrough
- `community(name)` function: look up in COMMUNITIES, return category or "Unknown"

Run against real data to populate: load all three catalogue CSVs, extract unique correspondent names, assign communities. Print any unassigned names to stderr so they can be added.

- [ ] **Step 4: Run tests — expect all PASS**

Run: `pytest gazetteer/tests/test_burney_names.py -v`

- [ ] **Step 5: Validate against real catalogue data**

Run: `python3 -c "import sys; sys.path.insert(0,'gazetteer'); from burney_names import normalise, community; ..."`

Load all three CSVs, normalise all names, check for any that return "Unknown" community. Add missing entries to `COMMUNITIES` until all correspondents with >3 letters are assigned.

- [ ] **Step 6: Commit**

```bash
git add gazetteer/burney_names.py gazetteer/tests/test_burney_names.py
git commit -m "feat: add shared burney_names module for catalogue normalisation"
```

---

## Task 2: Extended Data Pipeline

**Files:**
- Modify: `gazetteer/build_correspondent_network.py`
- Modify: `gazetteer/tests/test_build_correspondent_network.py`

- [ ] **Step 1: Write failing tests for CSV loading and multi-network data**

Add to test file:

```python
from build_correspondent_network import load_catalogue, build_catalogue_network


def test_load_catalogue_frances():
    entries = load_catalogue("frances")
    assert len(entries) > 5000
    for e in entries:
        assert "date" in e
        assert "direction" in e
        assert "correspondent" in e


def test_load_catalogue_cb_sr():
    entries = load_catalogue("cb_sr")
    assert len(entries) > 500


def test_load_catalogue_cb_jr():
    entries = load_catalogue("cb_jr")
    assert len(entries) > 500


def test_build_catalogue_network_structure():
    data = build_catalogue_network("frances")
    assert "nodes" in data
    assert "edges" in data
    assert "letters" in data
    assert "subject" in data
    assert data["subject"] == "Frances Burney d'Arblay"


def test_catalogue_network_has_directions():
    data = build_catalogue_network("frances")
    for edge in data["edges"]:
        assert "to_weight" in edge
        assert "from_weight" in edge
        assert edge["weight"] == edge["to_weight"] + edge["from_weight"]


def test_catalogue_network_filters_artefacts():
    data = build_catalogue_network("frances")
    node_ids = {n["id"] for n in data["nodes"]}
    assert "[?]" not in node_ids
    assert "NYPL(B)" not in node_ids
    assert "BM(Bar)" not in node_ids


def test_catalogue_susanna_is_top_frances():
    data = build_catalogue_network("frances")
    nodes = sorted(data["nodes"], key=lambda n: -n["count"])
    # Frances herself is node 0, Susanna should be top correspondent
    non_self = [n for n in nodes if n["id"] != "Frances Burney"]
    assert "Susanna" in non_self[0]["id"]


def test_catalogue_twining_is_top_cb_sr():
    data = build_catalogue_network("cb_sr")
    nodes = sorted(data["nodes"], key=lambda n: -n["count"])
    non_self = [n for n in nodes if n["id"] != "Charles Burney"]
    assert "Twining" in non_self[0]["id"]
```

- [ ] **Step 2: Run tests — expect ImportError**

- [ ] **Step 3: Implement `load_catalogue()` and `build_catalogue_network()`**

Add to `build_correspondent_network.py`:

```python
from burney_names import normalise, community

CATALOGUE_PATHS = {
    "frances": Path(__file__).resolve().parent.parent / "nonfiction/FrancesBurney/catalogue_frances_darblay.csv",
    "cb_sr": Path(__file__).resolve().parent.parent / "nonfiction/FrancesBurney/catalogue_cb_sr.csv",
    "cb_jr": Path(__file__).resolve().parent.parent / "nonfiction/FrancesBurney/catalogue_cb_jr.csv",
}

SUBJECTS = {
    "frances": "Frances Burney d'Arblay",
    "cb_sr": "Charles Burney Mus Doc",
    "cb_jr": "Charles Burney DD",
}

ARTEFACTS = {"[?]", "NYPL(B)", "BM(Bar)", "to", "", "NYPL(B) ", "copy", "draft"}
```

`load_catalogue(key)`: read CSV, return list of dicts with `date, direction, correspondent, repository, year` (year extracted from date field).

`build_catalogue_network(key)`: normalise names, filter artefacts, compute nodes/edges with direction counts, return data dict matching the spec structure.

The existing `build_network_data()` (OUP headers) is preserved but the `build()` function is updated to also call `build_catalogue_network()` for all three keys and embed the results.

- [ ] **Step 4: Run tests — expect all PASS**

- [ ] **Step 5: Commit**

```bash
git add gazetteer/build_correspondent_network.py gazetteer/tests/test_build_correspondent_network.py
git commit -m "feat: load catalogue CSVs and build multi-network data"
```

---

## Task 3: Enhanced HTML Template

**Files:**
- Modify: `gazetteer/build_correspondent_network.py` (HTML_TEMPLATE)
- Modify: `gazetteer/tests/test_build_correspondent_network.py`

- [ ] **Step 1: Write failing tests for enhanced build**

```python
def test_build_produces_html_with_three_networks():
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        out = Path(f.name)
    try:
        build(out_path=out)
        html = out.read_text(encoding="utf-8")
        assert "Frances Burney" in html
        assert "Charles Burney Mus Doc" in html
        assert "Charles Burney DD" in html
        assert "CATALOGUE_DATA" in html or "catalogue" in html.lower()
    finally:
        out.unlink(missing_ok=True)


def test_build_html_has_tab_switcher():
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        out = Path(f.name)
    try:
        build(out_path=out)
        html = out.read_text(encoding="utf-8")
        assert "network-tab" in html or "tab-frances" in html
    finally:
        out.unlink(missing_ok=True)
```

- [ ] **Step 2: Run tests — expect FAIL**

- [ ] **Step 3: Update HTML_TEMPLATE**

This is the bulk of the work. The template needs:

1. **Tab switcher** in the header — three buttons: Frances / Charles Sr / Charles Jr
2. **Data embedding** — `{CATALOGUE_JSON}` placeholder containing all three networks
3. **Tab switching JS** — on click, swap the active network data, rebuild the force simulation, update phases, update timeline range
4. **Threshold slider** — for Frances's network (500+ correspondents), a slider to control minimum letter count for visibility. Default 5. Hidden for smaller networks.
5. **Directional edges** — arrow markers on SVG lines, thickness by direction
6. **Cross-network badges** — small secondary indicator on nodes that appear in other networks
7. **Direction breakdown in detail panel** — "42 sent, 38 received" bar
8. **Repository list in detail panel** — small list of where letters are held
9. **Cross-network links in detail panel** — "Also in Charles Sr: 40 letters"
10. **Updated colour palette** for 9 community categories

The `build()` function is updated:
- Call `build_catalogue_network()` for each of the 3 keys
- Also call existing `build_network_data()` for the OUP data (kept as subset within Frances)
- JSON-encode all four datasets
- Substitute into template

**Important:** Template convention — all JS `{` `}` doubled. Un-double *before* inserting data (learned from v1 bug).

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Build and visually inspect**

Run: `python3 gazetteer/build_correspondent_network.py && open gazetteer/correspondent_network.html`

Check:
- [ ] Three tabs visible, switching works
- [ ] Frances network shows ~500 correspondents (with threshold slider)
- [ ] Charles Sr network shows Twining, Padre Martini, Garrick
- [ ] Charles Jr network shows CB, CPB, Samuel Parr
- [ ] Directional arrows on edges
- [ ] Detail panel shows direction breakdown
- [ ] Cross-network badges visible
- [ ] Timeline slider and phase presets update per tab
- [ ] Professional aesthetic maintained

- [ ] **Step 6: Commit**

```bash
git add gazetteer/build_correspondent_network.py gazetteer/tests/test_build_correspondent_network.py
git commit -m "feat: catalogue-powered three-network visualisation (Phase 1)"
```

---

## Task 4: Integration Tests & Polish

**Files:**
- Modify: `gazetteer/tests/test_build_correspondent_network.py`

- [ ] **Step 1: Write integration tests**

```python
def test_all_three_networks_have_entries():
    from build_correspondent_network import build_catalogue_network
    for key in ("frances", "cb_sr", "cb_jr"):
        data = build_catalogue_network(key)
        assert len(data["nodes"]) > 10, f"{key} has too few nodes"
        assert len(data["edges"]) > 10, f"{key} has too few edges"


def test_cross_network_correspondents():
    from build_correspondent_network import build_catalogue_network
    frances = build_catalogue_network("frances")
    cb_sr = build_catalogue_network("cb_sr")
    f_ids = {n["id"] for n in frances["nodes"]}
    c_ids = {n["id"] for n in cb_sr["nodes"]}
    shared = f_ids & c_ids
    assert len(shared) > 5, "Should have shared correspondents"
    assert "Hester Thrale Piozzi" in shared


def test_no_parsing_artefacts_in_any_network():
    from build_correspondent_network import build_catalogue_network
    for key in ("frances", "cb_sr", "cb_jr"):
        data = build_catalogue_network(key)
        for node in data["nodes"]:
            assert node["id"] not in ("[?]", "NYPL(B)", "BM(Bar)", "to", "")
```

- [ ] **Step 2: Run full test suite**

Run: `pytest gazetteer/tests/ -v -k "correspondent or burney_names"`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add gazetteer/tests/test_build_correspondent_network.py
git commit -m "test: integration tests for catalogue-powered network"
```
