import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import build_full_network
from export_correspondence_person_seed import build_seed_rows


def test_load_graph_merges_canonicalised_duplicates(tmp_path, monkeypatch):
    graph_path = tmp_path / "graph.csv"
    graph_path.write_text(
        "\n".join(
            [
                "person_a,person_b,weight,year_min,year_max,sources",
                "Elizabeth (Allen) Meeke,Frances Burney,2,1776,1776,hemlow_catalogue",
                "Elizabeth Meeke,Charles Burney,1,1791,1791,hemlow_catalogue",
                "Mr. George Crabbe,Edmund Burke,1,1781,1781,burke_1844",
                "George Crabbe,Edmund Burke,1,1781,1781,burke_1844",
                "\"\"\"\",Charles Burney Jr,1,1808,1808,coulombeau_waterfield",
            ]
        ),
        encoding="utf-8",
    )
    locations_path = tmp_path / "locations.csv"
    locations_path.write_text(
        "\n".join(
            [
                "name,lat,lon,place_name",
                "Elizabeth (Allen) Meeke,51.5,-0.1,London",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(build_full_network, "GRAPH_CSV", graph_path)
    monkeypatch.setattr(build_full_network, "LOCATIONS_CSV", locations_path)

    data = build_full_network.load_graph()

    node_ids = {node["id"] for node in data["nodes"]}
    assert "Elizabeth Meeke" in node_ids
    assert "George Crabbe" in node_ids
    assert '"' not in node_ids

    meeke = next(node for node in data["nodes"] if node["id"] == "Elizabeth Meeke")
    assert meeke["weight"] == 3
    assert meeke["place"] == "London"

    crabbe_edges = [
        edge for edge in data["edges"]
        if {edge["source"], edge["target"]} == {"Edmund Burke", "George Crabbe"}
    ]
    assert len(crabbe_edges) == 1
    assert crabbe_edges[0]["weight"] == 2


def test_build_embeds_optional_enrichment_without_requiring_it(tmp_path, monkeypatch):
    graph_path = tmp_path / "graph.csv"
    graph_path.write_text(
        "\n".join(
            [
                "person_a,person_b,weight,year_min,year_max,sources",
                "George Crabbe,Edmund Burke,2,1781,1785,burke_1844",
            ]
        ),
        encoding="utf-8",
    )
    locations_path = tmp_path / "locations.csv"
    locations_path.write_text("name,lat,lon,place_name\n", encoding="utf-8")
    person_info_path = tmp_path / "person_info.json"
    person_info_path.write_text("{}", encoding="utf-8")
    out_path = tmp_path / "network.html"

    monkeypatch.setattr(build_full_network, "GRAPH_CSV", graph_path)
    monkeypatch.setattr(build_full_network, "LOCATIONS_CSV", locations_path)
    monkeypatch.setattr(build_full_network, "PERSON_INFO", person_info_path)
    monkeypatch.setattr(build_full_network, "OUT_PATH", out_path)
    monkeypatch.setattr(build_full_network, "_get_d3_source", lambda: "")
    monkeypatch.setattr(
        build_full_network,
        "load_optional_enrichment",
        lambda: {
            "people": {
                "George Crabbe": {
                    "external_ids": [
                        {
                            "authority": "wikidata",
                            "identifier": "Q123",
                            "label": "",
                            "url": "https://www.wikidata.org/wiki/Q123",
                            "source_id": "wikidata_live",
                            "source_label": "Wikidata",
                            "confidence": "",
                            "notes": "",
                        }
                    ],
                    "relationships": [],
                    "addresses": [],
                }
            },
            "source_registry": {},
            "stats": {
                "people": 1,
                "external_ids": 1,
                "relationships": 0,
                "addresses": 0,
            },
        },
    )

    build_full_network.build()

    html = out_path.read_text(encoding="utf-8")
    assert "Q123" in html
    assert "External IDs" in html


def test_build_seed_rows_reflect_optional_enrichment_flags(tmp_path, monkeypatch):
    graph_path = tmp_path / "graph.csv"
    graph_path.write_text(
        "\n".join(
            [
                "person_a,person_b,weight,year_min,year_max,sources",
                "George Crabbe,Edmund Burke,2,1781,1785,burke_1844",
                "George Crabbe,Frances Burney,1,1791,1791,hemlow_catalogue",
            ]
        ),
        encoding="utf-8",
    )
    locations_path = tmp_path / "locations.csv"
    locations_path.write_text(
        "\n".join(
            [
                "name,lat,lon,place_name",
                "George Crabbe,51.5,-0.1,London",
            ]
        ),
        encoding="utf-8",
    )
    person_info_path = tmp_path / "person_info.json"
    person_info_path.write_text(
        '{"George Crabbe": {"dates": "1754-1832", "bio": "Poet."}}',
        encoding="utf-8",
    )

    monkeypatch.setattr(build_full_network, "GRAPH_CSV", graph_path)
    monkeypatch.setattr(build_full_network, "LOCATIONS_CSV", locations_path)
    monkeypatch.setattr(build_full_network, "PERSON_INFO", person_info_path)
    monkeypatch.setattr(
        "export_correspondence_person_seed.PERSON_INFO",
        person_info_path,
    )
    monkeypatch.setattr(
        "export_correspondence_person_seed.load_optional_enrichment",
        lambda: {
            "people": {
                "George Crabbe": {
                    "external_ids": [{"authority": "wikidata", "identifier": "Q123"}],
                    "relationships": [{"relationship_type": "correspondent", "related_person": "Edmund Burke"}],
                    "addresses": [{"place_name": "London"}],
                }
            }
        },
    )

    rows = build_seed_rows()
    crabbe = next(row for row in rows if row["person"] == "George Crabbe")
    assert crabbe["place"] == "London"
    assert crabbe["correspondent_count"] == "2"
    assert crabbe["has_person_info"] == "yes"
    assert crabbe["has_external_ids"] == "yes"
    assert crabbe["has_relationships"] == "yes"
    assert crabbe["has_addresses"] == "yes"
