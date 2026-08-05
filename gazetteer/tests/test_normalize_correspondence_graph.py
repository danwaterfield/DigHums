import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from normalize_correspondence_graph import normalize_rows


def test_burke_arrow_label_expands_into_real_edge():
    rows, stats = normalize_rows(
        [
            {
                "person_a": "Edmund Burke",
                "person_b": "Rt. Hon. Edrn. Burke -> Philip Francis",
                "weight": "1",
                "year_min": "1785",
                "year_max": "1785",
                "sources": "burke_1844",
            }
        ]
    )

    assert stats["expanded_rows"] == 1
    assert rows == [
        {
            "person_a": "Edmund Burke",
            "person_b": "Philip Francis",
            "weight": "1",
            "year_min": "1785",
            "year_max": "1785",
            "sources": "burke_1844",
        }
    ]


def test_burke_joint_recipient_label_splits_into_multiple_edges():
    rows, stats = normalize_rows(
        [
            {
                "person_a": "Edmund Burke",
                "person_b": "Mr. Rich. Burke, Jun., and Mr. T. King",
                "weight": "1",
                "year_min": "1773",
                "year_max": "1773",
                "sources": "burke_1844",
            }
        ]
    )

    assert stats["expanded_rows"] == 1
    pairs = {(row["person_a"], row["person_b"]) for row in rows}
    assert pairs == {
        ("Edmund Burke", "Mr. T. King"),
        ("Edmund Burke", "Richard Burke Jr"),
    }
