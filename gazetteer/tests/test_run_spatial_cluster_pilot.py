import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("esda")
pytest.importorskip("libpysal")

ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

import run_spatial_cluster_pilot as spatial


def test_merge_manifest_overrides_spatial_and_filters():
    base = {
        "filters": {
            "city_scope": ["London"],
            "min_total_presence": 1,
        },
        "spatial": {
            "k_neighbors": 5,
            "alpha": 0.05,
        },
    }
    overrides = {
        "name": "min_total_presence_3",
        "min_total_presence": 3,
        "k_neighbors": 3,
    }

    merged = spatial.merge_manifest(base, overrides)

    assert merged["filters"]["city_scope"] == ["London"]
    assert merged["filters"]["min_total_presence"] == 3
    assert merged["spatial"]["k_neighbors"] == 3
    assert merged["spatial"]["alpha"] == 0.05


def test_prepare_venue_panel_aggregates_work_venue_presence(tmp_path):
    derived_dir = tmp_path / "derived"
    derived_dir.mkdir()

    evidence = pd.DataFrame(
        [
            {
                "source_id": "fiction_a",
                "source_type": "fiction",
                "source_family": "fiction",
                "city": "London",
                "venue_id": "LON001",
                "venue_name": "Vauxhall",
                "x_m": 0.0,
                "y_m": 0.0,
                "has_geocode": True,
                "confidence": 0.9,
                "pub_year": 1770,
                "date_min": 1770,
                "date_max": 1770,
            },
            {
                "source_id": "fiction_a",
                "source_type": "fiction",
                "source_family": "fiction",
                "city": "London",
                "venue_id": "LON001",
                "venue_name": "Vauxhall",
                "x_m": 0.0,
                "y_m": 0.0,
                "has_geocode": True,
                "confidence": 0.9,
                "pub_year": 1770,
                "date_min": 1770,
                "date_max": 1770,
            },
            {
                "source_id": "doc_a",
                "source_type": "topography",
                "source_family": "descriptive",
                "city": "London",
                "venue_id": "LON001",
                "venue_name": "Vauxhall",
                "x_m": 0.0,
                "y_m": 0.0,
                "has_geocode": True,
                "confidence": 0.95,
                "pub_year": 1770,
                "date_min": 1770,
                "date_max": 1770,
            },
            {
                "source_id": "doc_b",
                "source_type": "letters",
                "source_family": "life_writing",
                "city": "London",
                "venue_id": "LON002",
                "venue_name": "Tyburn",
                "x_m": 1.0,
                "y_m": 1.0,
                "has_geocode": True,
                "confidence": 0.95,
                "pub_year": 1780,
                "date_min": 1780,
                "date_max": 1780,
            },
        ]
    )
    evidence.to_parquet(derived_dir / "evidence_points.parquet", index=False)

    manifest = {
        "filters": {
            "city_scope": ["London"],
            "source_types": ["fiction", "topography", "letters"],
            "date_range": [1700, 1820],
            "min_confidence": 0.0,
            "min_total_presence": 1,
        }
    }

    venue_panel, diagnostics = spatial.prepare_venue_panel(derived_dir, manifest)

    assert diagnostics["venues"] == 2
    assert diagnostics["source_families"] == {"fiction": 2, "descriptive": 1, "life_writing": 1}
    assert diagnostics["presence_source_families"] == {"fiction": 1, "descriptive": 1, "life_writing": 1}
    row = venue_panel.loc[venue_panel["venue_id"] == "LON001"].iloc[0]
    assert row["fiction"] == 1
    assert row["documentary"] == 1
    assert row["total_presence"] == 2
    assert row["dominant_class"] == "balanced"
