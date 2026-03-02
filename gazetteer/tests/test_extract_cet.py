"""Tests for extract_cet.py"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from sensory_db import init_db
from extract_cet import parse_cet, load, CORPUS_START, CORPUS_END


# ── Sample CET data fixture ────────────────────────────────────────────────────

SAMPLE_CET = """\
Mean Central England Temperature (Degrees Celsius)
1659-1973 Manley (Q.J.R.METEOROL.SOC., 1974)
1974 on Parker et al. (INT.J.CLIM., 1992)
Parker and Horton (INT.J.CLIM., 2005)
  Year    Jan    Feb    Mar    Apr    May    Jun    Jul    Aug    Sep    Oct    Nov    Dec  Annual
  1659    3.0    4.0    6.0    7.0   11.0   13.0   16.0   16.0   13.0   10.0    5.0    2.0   8.87
  1660    0.0    4.0    6.0    9.0   11.0   14.0   15.0   16.0   13.0   10.0    6.0    5.0   9.10
  1684   -3.0   -1.0    3.0    6.5   13.0   15.0   16.0   15.5   12.0   11.0    3.0    4.0   7.95
  1820    5.0    6.0    8.0    9.0   12.0   15.0   17.0   17.0   14.0   10.0    7.0    4.0  10.33
  1821    5.0    6.0    8.0    9.0   12.0   15.0   17.0   17.0   14.0   10.0    7.0    4.0  10.33
  2026    4.4    7.2  -99.9  -99.9  -99.9  -99.9  -99.9  -99.9  -99.9  -99.9  -99.9  -99.9 -99.90
"""


@pytest.fixture
def sample_cet_file(tmp_path):
    p = tmp_path / "cet_sample.txt"
    p.write_text(SAMPLE_CET, encoding="utf-8")
    return p


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_parse_skips_before_corpus_start(sample_cet_file):
    records = parse_cet(sample_cet_file)
    years = {r[0] for r in records}
    assert 1659 not in years   # before CORPUS_START (1660)


def test_parse_includes_corpus_start_and_end(sample_cet_file):
    records = parse_cet(sample_cet_file)
    years = {r[0] for r in records}
    assert 1660 in years
    assert 1820 in years


def test_parse_skips_after_corpus_end(sample_cet_file):
    records = parse_cet(sample_cet_file)
    years = {r[0] for r in records}
    assert 1821 not in years


def test_parse_skips_missing_values(sample_cet_file):
    records = parse_cet(sample_cet_file)
    # 2026 is outside corpus range but also has -99.9 values — test the filter
    # The 1660 row has no -99.9 so all 13 records (12 monthly + 1 annual)
    year_1660 = [r for r in records if r[0] == 1660]
    assert len(year_1660) == 13  # 12 months + annual mean


def test_frost_year_has_negative_january(sample_cet_file):
    records = parse_cet(sample_cet_file)
    jan_1684 = next((r for r in records if r[0] == 1684 and r[1] == 1), None)
    assert jan_1684 is not None
    assert jan_1684[2] == -3.0


def test_annual_mean_stored_as_month_zero(sample_cet_file):
    records = parse_cet(sample_cet_file)
    annual_1660 = next((r for r in records if r[0] == 1660 and r[1] == 0), None)
    assert annual_1660 is not None
    assert abs(annual_1660[2] - 9.10) < 0.01


def test_load_is_idempotent(tmp_path, sample_cet_file):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    stats1 = load(write=True,  db_path=db_path, cet_path=sample_cet_file)
    stats2 = load(write=True,  db_path=db_path, cet_path=sample_cet_file)
    # Second run should skip all rows already present
    assert stats2["inserted"] == 0
    assert stats2["skipped"]  == stats1["inserted"]


def test_decade_average_1684_is_cold(sample_cet_file):
    """1684 annual mean (7.95°C) is well below the 1660 annual mean (9.10°C)."""
    records = parse_cet(sample_cet_file)
    annual_1684 = next((r for r in records if r[0] == 1684 and r[1] == 0), None)
    annual_1660 = next((r for r in records if r[0] == 1660 and r[1] == 0), None)
    assert annual_1684 is not None
    assert annual_1660 is not None
    assert annual_1684[2] < annual_1660[2]
