"""Integration tests for analyse_narrative_pace.py (Task 6).

Runs the analysis pipeline with --limit 3 via subprocess, then checks the
resulting SQLite database and JSON output file for correctness.
"""

import json
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
SCRIPT = REPO_ROOT / "gazetteer" / "analyse_narrative_pace.py"


# ---------------------------------------------------------------------------
# Module-scoped fixture: run the pipeline once, share outputs across tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def run_analysis(tmp_path_factory):
    """Run the pipeline with --limit 3 into a temporary directory."""
    tmp = tmp_path_factory.mktemp("pace_test")
    db_path = tmp / "narrative_pace.db"
    json_path = tmp / "narrative_pace_data.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--limit", "3",
            "--db", str(db_path),
            "--json", str(json_path),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"Pipeline failed (exit {result.returncode}).\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    return {"stdout": result.stdout, "db": db_path, "json": json_path}


@pytest.fixture(scope="module")
def db(run_analysis):
    conn = sqlite3.connect(run_analysis["db"])
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def json_data(run_analysis):
    return json.loads(run_analysis["json"].read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# SQLite — novels table
# ---------------------------------------------------------------------------

class TestNovelsTable:
    def test_novels_table_has_three_entries(self, db):
        count = db.execute("SELECT COUNT(*) FROM novels").fetchone()[0]
        assert count == 3

    def test_novels_have_required_columns(self, db):
        row = db.execute("SELECT * FROM novels LIMIT 1").fetchone()
        expected = {"id", "author", "title", "year", "genre", "word_count",
                    "sentence_count", "volume_boundaries"}
        assert expected.issubset(set(row.keys()))

    def test_novels_word_count_positive(self, db):
        rows = db.execute("SELECT word_count FROM novels").fetchall()
        for row in rows:
            assert row["word_count"] > 0

    def test_novels_sentence_count_positive(self, db):
        rows = db.execute("SELECT sentence_count FROM novels").fetchall()
        for row in rows:
            assert row["sentence_count"] > 0

    def test_novels_year_is_integer(self, db):
        rows = db.execute("SELECT year FROM novels").fetchall()
        for row in rows:
            assert isinstance(row["year"], int)
            assert 1600 < row["year"] < 1900


# ---------------------------------------------------------------------------
# SQLite — sentences table
# ---------------------------------------------------------------------------

class TestSentencesTable:
    def test_sentences_table_has_more_than_100_entries(self, db):
        count = db.execute("SELECT COUNT(*) FROM sentences").fetchone()[0]
        assert count > 100

    def test_sentences_have_required_columns(self, db):
        row = db.execute("SELECT * FROM sentences LIMIT 1").fetchone()
        expected = {"id", "novel_id", "sentence_index", "position", "text",
                    "is_dialogue", "singulative", "iterative", "description",
                    "fid", "commentary", "dominant_category"}
        assert expected.issubset(set(row.keys()))

    def test_sentence_position_between_zero_and_one(self, db):
        rows = db.execute("SELECT MIN(position), MAX(position) FROM sentences").fetchone()
        assert rows[0] >= 0.0
        assert rows[1] <= 1.0

    def test_non_dialogue_scores_sum_to_approx_one(self, db):
        """For non-dialogue sentences, the five category scores should sum to ~1."""
        rows = db.execute(
            "SELECT singulative, iterative, description, fid, commentary "
            "FROM sentences WHERE is_dialogue=0 LIMIT 100"
        ).fetchall()
        assert len(rows) > 0
        for row in rows:
            total = row[0] + row[1] + row[2] + row[3] + row[4]
            assert abs(total - 1.0) < 1e-4, f"Scores sum to {total} for row {dict(row)}"

    def test_dialogue_sentences_non_dialogue_scores_zero(self, db):
        rows = db.execute(
            "SELECT singulative, iterative, description, fid, commentary "
            "FROM sentences WHERE is_dialogue=1 LIMIT 50"
        ).fetchall()
        # There may be no dialogue at all in some tiny test novels; skip if so
        if not rows:
            pytest.skip("No dialogue sentences found in --limit 3 subset")
        for row in rows:
            total = row[0] + row[1] + row[2] + row[3] + row[4]
            assert total == pytest.approx(0.0)

    def test_dominant_category_valid(self, db):
        valid = {"dialogue", "singulative", "iterative", "description", "fid", "commentary"}
        rows = db.execute("SELECT DISTINCT dominant_category FROM sentences").fetchall()
        for row in rows:
            assert row[0] in valid

    def test_sentences_linked_to_novels(self, db):
        """Every sentence novel_id must appear in the novels table."""
        orphans = db.execute(
            "SELECT COUNT(*) FROM sentences s "
            "LEFT JOIN novels n ON s.novel_id=n.id WHERE n.id IS NULL"
        ).fetchone()[0]
        assert orphans == 0


# ---------------------------------------------------------------------------
# SQLite — environmental table
# ---------------------------------------------------------------------------

class TestEnvironmentalTable:
    def test_environmental_table_populated(self, db):
        count = db.execute("SELECT COUNT(*) FROM environmental").fetchone()[0]
        assert count > 0

    def test_environmental_has_required_columns(self, db):
        row = db.execute("SELECT * FROM environmental LIMIT 1").fetchone()
        expected = {"decade", "smell", "smoke", "pollution", "total"}
        assert expected.issubset(set(row.keys()))

    def test_environmental_decade_values_plausible(self, db):
        rows = db.execute("SELECT decade FROM environmental").fetchall()
        for row in rows:
            assert 1600 <= row[0] <= 1900

    def test_environmental_pollution_le_total(self, db):
        rows = db.execute("SELECT pollution, total FROM environmental").fetchall()
        for row in rows:
            assert row[0] <= row[1]


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

class TestJsonOutput:
    def test_json_has_novels_key(self, json_data):
        assert "novels" in json_data

    def test_json_has_three_novels(self, json_data):
        assert len(json_data["novels"]) == 3

    def test_json_has_environmental_key(self, json_data):
        assert "environmental" in json_data

    def test_json_environmental_non_empty(self, json_data):
        assert len(json_data["environmental"]) > 0

    def test_json_lexicon_version(self, json_data):
        assert json_data["lexicon_version"] == "1.0"

    def test_json_generated_is_iso8601(self, json_data):
        from datetime import datetime
        ts = json_data["generated"]
        # Should parse without error
        datetime.fromisoformat(ts.replace("Z", "+00:00"))

    def test_json_novel_arc_has_200_points(self, json_data):
        for novel in json_data["novels"]:
            arc = novel["arc"]
            assert len(arc["positions"]) == 200, \
                f"Expected 200 positions for {novel['id']}, got {len(arc['positions'])}"

    def test_json_novel_arc_has_all_categories(self, json_data):
        categories = {"dialogue", "singulative", "iterative", "description", "fid", "commentary"}
        for novel in json_data["novels"]:
            arc = novel["arc"]
            assert categories.issubset(arc.keys()), \
                f"Missing arc categories for {novel['id']}: {categories - set(arc.keys())}"

    def test_json_novel_arc_category_length_200(self, json_data):
        for novel in json_data["novels"]:
            for cat in ("dialogue", "singulative", "iterative", "description", "fid", "commentary"):
                assert len(novel["arc"][cat]) == 200, \
                    f"{novel['id']} arc[{cat}] has {len(novel['arc'][cat])} points"

    def test_json_summary_proportions_sum_to_one(self, json_data):
        for novel in json_data["novels"]:
            total = sum(novel["summary"].values())
            assert abs(total - 1.0) < 1e-4, \
                f"Summary for {novel['id']} sums to {total}"

    def test_json_summary_has_all_categories(self, json_data):
        categories = {"dialogue", "singulative", "iterative", "description", "fid", "commentary"}
        for novel in json_data["novels"]:
            assert categories.issubset(novel["summary"].keys())

    def test_json_novel_required_fields(self, json_data):
        required = {"id", "author", "title", "year", "genre",
                    "word_count", "sentence_count", "volume_boundaries", "arc", "summary"}
        for novel in json_data["novels"]:
            assert required.issubset(novel.keys()), \
                f"Missing fields for {novel.get('id')}: {required - set(novel.keys())}"

    def test_json_novel_sentence_count_matches_db(self, db, json_data):
        for novel in json_data["novels"]:
            db_count = db.execute(
                "SELECT sentence_count FROM novels WHERE id=?", (novel["id"],)
            ).fetchone()
            assert db_count is not None, f"Novel {novel['id']} not in DB"
            assert db_count[0] == novel["sentence_count"]

    def test_json_arc_positions_ascending(self, json_data):
        for novel in json_data["novels"]:
            positions = novel["arc"]["positions"]
            for i in range(len(positions) - 1):
                assert positions[i] <= positions[i + 1], \
                    f"Positions not ascending for {novel['id']} at index {i}"

    def test_json_arc_values_in_range(self, json_data):
        for novel in json_data["novels"]:
            for cat in ("dialogue", "singulative", "iterative",
                        "description", "fid", "commentary"):
                values = novel["arc"][cat]
                for v in values:
                    assert 0.0 <= v <= 1.0, \
                        f"{novel['id']} arc[{cat}] value {v} out of [0,1]"
