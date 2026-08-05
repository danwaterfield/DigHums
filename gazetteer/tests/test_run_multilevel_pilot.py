import sys
from pathlib import Path

import pytest

pytest.importorskip("statsmodels")

ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

import run_multilevel_pilot as pilot


def test_merge_manifest_overrides_modeling_and_filters():
    base = {
        "filters": {
            "city_scope": ["London"],
            "min_confidence": 0.0,
        },
        "modeling": {
            "source_type_mode": "detailed",
            "min_source_type_n": 3,
        },
    }
    overrides = {
        "name": "fiction_vs_documentary_only",
        "source_type_mode": "fiction_vs_documentary",
        "min_confidence": 0.8,
    }

    merged = pilot.merge_manifest(base, overrides)

    assert merged["filters"]["city_scope"] == ["London"]
    assert merged["filters"]["min_confidence"] == 0.8
    assert merged["modeling"]["source_type_mode"] == "fiction_vs_documentary"
    assert merged["modeling"]["min_source_type_n"] == 3


def test_source_type_group_collapses_rare_levels():
    source_types = pilot.pd.Series(["fiction", "fiction", "topography", "diary", "rare"])
    source_families = pilot.pd.Series(["fiction", "fiction", "descriptive", "life_writing", "other"])

    grouped = pilot.source_type_group(source_types, source_families, mode="detailed", min_n=2)

    assert list(grouped.astype(str)) == ["fiction", "fiction", "other", "other", "other"]


def test_source_type_group_can_switch_to_binary_mode():
    source_types = pilot.pd.Series(["fiction", "topography", "diary"])
    source_families = pilot.pd.Series(["fiction", "descriptive", "life_writing"])

    grouped = pilot.source_type_group(source_types, source_families, mode="fiction_vs_documentary", min_n=99)

    assert list(grouped.astype(str)) == ["fiction", "documentary", "documentary"]


def test_source_type_group_can_use_source_family_mode():
    source_types = pilot.pd.Series(["fiction", "topography", "diary", "letters", "legal"])
    source_families = pilot.pd.Series(["fiction", "descriptive", "life_writing", "life_writing", "legal"])

    grouped = pilot.source_type_group(source_types, source_families, mode="source_family", min_n=2)

    assert list(grouped.astype(str)) == ["fiction", "other", "life_writing", "life_writing", "other"]
