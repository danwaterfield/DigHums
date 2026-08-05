import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from correspondence_enrichment import load_optional_enrichment


def test_load_optional_enrichment_is_empty_when_workspace_missing(tmp_path):
    data = load_optional_enrichment(tmp_path)
    assert data["people"] == {}
    assert data["stats"] == {
        "people": 0,
        "external_ids": 0,
        "relationships": 0,
        "addresses": 0,
    }


def test_load_optional_enrichment_canonicalises_people_and_resolves_sources(tmp_path):
    (tmp_path / "source_registry.csv").write_text(
        "\n".join(
            [
                "source_id,label,domain",
                "wikidata_live,Wikidata,person_authority",
                "familysearch_tree,FamilySearch Tree,genealogy_api",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "person_external_ids.csv").write_text(
        "\n".join(
            [
                "person,authority,identifier,url,source_id",
                "Mr. George Crabbe,wikidata,Q123,https://www.wikidata.org/wiki/Q123,wikidata_live",
                "George Crabbe,wikidata,Q123,https://www.wikidata.org/wiki/Q123,wikidata_live",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "person_relationships.csv").write_text(
        "\n".join(
            [
                "person,related_person,relationship_type,date_from,date_to,source_id",
                "Mr. George Crabbe,Edmund Burke,correspondent,1781,1810,wikidata_live",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "address_assertions.csv").write_text(
        "\n".join(
            [
                "person,label,street,place_name,date_from,date_to,source_id",
                "Mr. George Crabbe,Home,Great Queen Street,London,1792,1795,familysearch_tree",
            ]
        ),
        encoding="utf-8",
    )

    data = load_optional_enrichment(tmp_path)

    assert set(data["people"]) == {"George Crabbe"}
    crabbe = data["people"]["George Crabbe"]
    assert crabbe["external_ids"] == [
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
    ]
    assert crabbe["relationships"][0]["related_person"] == "Edmund Burke"
    assert crabbe["relationships"][0]["source_label"] == "Wikidata"
    assert crabbe["addresses"][0]["place_name"] == "London"
    assert crabbe["addresses"][0]["source_label"] == "FamilySearch Tree"
    assert data["stats"] == {
        "people": 1,
        "external_ids": 1,
        "relationships": 1,
        "addresses": 1,
    }
