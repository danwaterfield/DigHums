import sys
from pathlib import Path

BURKE_DIR = Path(__file__).resolve().parents[2] / "nonfiction" / "EdmundBurke"
sys.path.insert(0, str(BURKE_DIR))

from extract_catalogue import try_parse_entry


def test_try_parse_entry_records_sender_recipient_for_third_party_letter():
    entry = try_parse_entry(
        "June 24 Chief Justice Aston to Mr. Secretary Hamilton 12",
        current_year=1762,
        vol_num=1,
        letter_num=12,
    )

    assert entry["direction"] == "from"
    assert entry["correspondent"] == "Chief Justice Aston -> Mr. Secretary Hamilton"
    assert entry["sender"] == "Chief Justice Aston"
    assert entry["recipient"] == "Mr. Secretary Hamilton"


def test_try_parse_entry_records_sender_recipient_for_letter_to_burke():
    entry = try_parse_entry(
        "July 27 Rev. Dr. Leland to Edmund Burke 25",
        current_year=1765,
        vol_num=1,
        letter_num=25,
    )

    assert entry["direction"] == "from"
    assert entry["correspondent"] == "Rev. Dr. Leland"
    assert entry["sender"] == "Rev. Dr. Leland"
    assert entry["recipient"] == "Edmund Burke"
