"""Tests for build_sensory_timeline.py."""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import build_sensory_timeline as bst

REPO_ROOT = Path(__file__).parent.parent.parent
HTML_PATH = REPO_ROOT / "gazetteer" / "sensory_timeline.html"


# ── Subprocess fixture ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def html():
    """Build the sensory timeline and return HTML content."""
    result = subprocess.run(
        [sys.executable, "gazetteer/build_sensory_timeline.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return HTML_PATH.read_text(encoding="utf-8")


# ── HTML content tests ──────────────────────────────────────────────────────

def test_html_generated(html):
    assert "<html" in html


def test_venues_json_present(html):
    assert "VENUES_JSON" not in html, "Placeholder was not replaced"
    assert "const VENUES" in html
    assert "Vauxhall" in html


def test_evidence_json_present(html):
    assert "EVIDENCE_JSON" not in html, "Placeholder was not replaced"
    assert "const EVIDENCE" in html


def test_decade_summaries_present(html):
    assert "DECADE_SUMMARIES_JSON" not in html, "Placeholder was not replaced"
    assert "const DECADE_SUMMARIES" in html


def test_highlights_present(html):
    assert "HIGHLIGHTS_JSON" not in html, "Placeholder was not replaced"
    assert "const HIGHLIGHTS" in html


def test_only_london_venues(html):
    assert '"LON' in html
    assert '"RUR' not in html


# ── Unit tests for pure functions ───────────────────────────────────────────

def test_decade_assignment():
    """Decade computation: floor to nearest 10."""
    assert bst.assign_decade(1720) == 1720
    assert bst.assign_decade(1725) == 1720
    assert bst.assign_decade(1729) == 1720
    assert bst.assign_decade(1730) == 1730
    assert bst.assign_decade(1660) == 1660
    assert bst.assign_decade(1819) == 1810


def test_threshold_filtering():
    """Only venues with 20+ passages and 3+ distinct source_ids qualify."""
    passages = [
        {"source_id": "s1", "text": "a"},
        {"source_id": "s2", "text": "b"},
        {"source_id": "s3", "text": "c"},
    ]
    # Exactly 3 sources, 3 passages — below passage threshold
    assert bst.venue_qualifies(passages, min_passages=20, min_sources=3) is False

    # Enough passages but only 2 distinct sources
    two_source = [{"source_id": "s1", "text": "x"}, {"source_id": "s2", "text": "y"}] * 12
    assert bst.venue_qualifies(two_source, min_passages=20, min_sources=3) is False

    # Enough of both
    good = [{"source_id": f"s{i % 4}", "text": "x" * 10} for i in range(25)]
    assert bst.venue_qualifies(good, min_passages=20, min_sources=3) is True


def test_highlight_prefers_longer():
    """Select highlight prefers the passage with longer text."""
    passages = [
        {"source_id": "s1", "source_type": "topography", "text": "short text"},
        {"source_id": "s2", "source_type": "topography", "text": "a much longer passage of evidence here"},
    ]
    result = bst.select_highlight(passages)
    assert result["text"] == "a much longer passage of evidence here"


def test_highlight_prefers_fiction():
    """select_highlight prefers fiction over non-fiction when lengths are comparable."""
    passages = [
        {"source_id": "s1", "source_type": "topography", "text": "x" * 50},
        {"source_id": "s2", "source_type": "fiction",    "text": "x" * 40},
    ]
    result = bst.select_highlight(passages)
    assert result["source_type"] == "fiction"


# ── Tasks 3-6: HTML structure tests ─────────────────────────────────────────

def test_decade_sidebar_buttons(html):
    assert 'data-decade="1660"' in html
    assert 'data-decade="1750"' in html
    assert 'data-decade="1810"' in html


def test_timeline_strip_present(html):
    assert "timeline-strip" in html


def test_sense_legend(html):
    for label in ["Smell", "Noise", "Crowd", "Visual", "Thermal"]:
        assert label in html


def test_venue_selector(html):
    assert "<select" in html
    assert "venue-selector" in html


def test_story_explore_toggle(html):
    assert "Story" in html
    assert "Explore" in html


def test_view_nav_links(html):
    assert "sensory_time_map.html" in html
    assert "venue_explorer.html" in html
    assert "comparison.html" in html


def test_no_border_radius(html):
    import re
    matches = re.findall(r'border-radius:\s*[^0;][^;]*;', html)
    real_radii = [m for m in matches if not re.match(r'border-radius:\s*0', m)]
    assert len(real_radii) == 0, f"Found border-radius: {real_radii}"


def test_source_type_labels(html):
    assert "source_type" in html


def test_modality_tags_in_template(html):
    assert "modality" in html


def test_georgia_font(html):
    assert "Georgia" in html


def test_story_mode_keyboard(html):
    assert "ArrowRight" in html
    assert "ArrowLeft" in html
    assert "Escape" in html


def test_story_bar_present(html):
    assert "story-bar" in html


# ── Navigation integration tests ──────────────────────────────────────────

def test_sensory_map_has_timeline_link():
    """Sensory time map must link to timeline."""
    path = REPO_ROOT / "gazetteer" / "sensory_time_map.html"
    if path.exists():
        content = path.read_text(encoding="utf-8")
        assert "sensory_timeline.html" in content


def test_venue_explorer_has_timeline_link():
    """Venue explorer must link to timeline."""
    path = REPO_ROOT / "gazetteer" / "venue_explorer.html"
    if path.exists():
        content = path.read_text(encoding="utf-8")
        assert "sensory_timeline.html" in content
