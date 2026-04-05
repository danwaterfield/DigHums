"""Tests for build_narrative_pace.py — HTML builder for narrative pace visualisation."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
JSON_PATH = REPO_ROOT / "gazetteer" / "narrative_pace_data.json"
HTML_PATH = REPO_ROOT / "gazetteer" / "narrative_pace.html"


@pytest.fixture(scope="module")
def html():
    assert JSON_PATH.exists(), "narrative_pace_data.json not found — run analyse first or create fixture"
    result = subprocess.run(
        [sys.executable, "gazetteer/build_narrative_pace.py"],
        cwd=REPO_ROOT, capture_output=True, text=True
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
    for genre in ["domestic", "gothic", "picaresque", "epistolary", "amatory", "satirical"]:
        assert genre in html


def test_category_colours_defined(html):
    for cat in ["dialogue", "singulative", "iterative", "description", "fid", "commentary"]:
        assert cat in html
