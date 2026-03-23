#!/usr/bin/env python3
"""Seed bookseller entity and location CSVs from hardcoded data.

Idempotent: overwrites existing files on re-run.

Usage:
    python3 gazetteer/seed_booksellers.py
"""

import csv
from pathlib import Path

HERE = Path(__file__).parent

BOOKSELLERS_COLS = ("bookseller_id", "name", "sign", "type", "active_min", "active_max", "notes")
LOCATIONS_COLS = ("bookseller_id", "venue_id", "date_min", "date_max", "address_detail", "source")

BOOKSELLERS = [
    {
        "bookseller_id": "BS_NOON",
        "name": "John Noon",
        "sign": "White Hart",
        "type": "bookseller",
        "active_min": 1735,
        "active_max": 1745,
        "notes": "Cheapside bookseller near Mercer's Chapel",
    },
    {
        "bookseller_id": "BS_BORBET",
        "name": "C. Borbet",
        "sign": "Addison's Head",
        "type": "bookseller",
        "active_min": 1738,
        "active_max": 1742,
        "notes": "Fleet Street bookseller opposite St Dunstan's Church",
    },
    {
        "bookseller_id": "BS_LONGMAN",
        "name": "Thomas Longman I",
        "sign": "Ship",
        "type": "bookseller|publisher",
        "active_min": 1724,
        "active_max": 1755,
        "notes": "Founded the Longman publishing dynasty at Paternoster Row",
    },
    {
        "bookseller_id": "BS_MILLAR",
        "name": "Andrew Millar",
        "sign": "Buchanan's Head",
        "type": "bookseller|publisher",
        "active_min": 1728,
        "active_max": 1768,
        "notes": "Publisher of Hume, Fielding, and Johnson's Dictionary",
    },
    {
        "bookseller_id": "BS_CADELL",
        "name": "Thomas Cadell I",
        "sign": "",
        "type": "bookseller|publisher",
        "active_min": 1767,
        "active_max": 1802,
        "notes": "Succeeded Millar; published Gibbon, Smith, and Burney",
    },
    {
        "bookseller_id": "BS_COOPER",
        "name": "Mary Cooper",
        "sign": "Globe",
        "type": "bookseller",
        "active_min": 1743,
        "active_max": 1761,
        "notes": "One of the most prominent women in the 18c book trade",
    },
    {
        "bookseller_id": "BS_DODSLEY",
        "name": "Robert Dodsley",
        "sign": "Tully's Head",
        "type": "bookseller|publisher",
        "active_min": 1735,
        "active_max": 1764,
        "notes": "Poet and publisher; The Annual Register with Burke",
    },
    {
        "bookseller_id": "BS_TONSON_SR",
        "name": "Jacob Tonson I",
        "sign": "Shakespeare's Head",
        "type": "bookseller|publisher",
        "active_min": 1678,
        "active_max": 1720,
        "notes": "Publisher of Dryden and Pope; Kit-Cat Club secretary",
    },
    {
        "bookseller_id": "BS_TONSON_JR",
        "name": "Jacob Tonson II",
        "sign": "Shakespeare's Head",
        "type": "bookseller|publisher",
        "active_min": 1720,
        "active_max": 1735,
        "notes": "Nephew and successor to Jacob Tonson I",
    },
    {
        "bookseller_id": "BS_RIVINGTON",
        "name": "Charles Rivington",
        "sign": "Bible and Crown",
        "type": "bookseller|publisher",
        "active_min": 1711,
        "active_max": 1742,
        "notes": "St Paul's Churchyard publisher; Pamela co-publisher",
    },
    {
        "bookseller_id": "BS_STRAHAN",
        "name": "William Strahan",
        "sign": "",
        "type": "printer",
        "active_min": 1738,
        "active_max": 1785,
        "notes": "King's Printer; printed Johnson's Dictionary and Gibbon",
    },
    {
        "bookseller_id": "BS_MURRAY_I",
        "name": "John Murray I",
        "sign": "",
        "type": "bookseller|publisher",
        "active_min": 1768,
        "active_max": 1793,
        "notes": "Founded the Murray publishing house at 32 Fleet Street",
    },
    {
        "bookseller_id": "BS_LACKINGTON",
        "name": "James Lackington",
        "sign": "Temple of the Muses",
        "type": "bookseller",
        "active_min": 1794,
        "active_max": 1804,
        "notes": "Pioneered cheap book sales; 'Temple of the Muses' was London's largest bookshop",
    },
    {
        "bookseller_id": "BS_FLEMING_ALISON",
        "name": "R. Fleming and A. Alison",
        "sign": "",
        "type": "printer",
        "active_min": 1741,
        "active_max": 1742,
        "notes": "Edinburgh printers",
    },
    {
        "bookseller_id": "BS_KINCAID",
        "name": "Alexander Kincaid",
        "sign": "above the Cross",
        "type": "bookseller|publisher",
        "active_min": 1741,
        "active_max": 1770,
        "notes": "Edinburgh bookseller and King's Printer for Scotland",
    },
    {
        "bookseller_id": "BS_DONALDSON",
        "name": "Alexander Donaldson",
        "sign": "",
        "type": "bookseller|publisher",
        "active_min": 1752,
        "active_max": 1794,
        "notes": "Edinburgh bookseller; won landmark copyright case Donaldson v Becket (1774)",
    },
]

# Edinburgh booksellers are excluded from locations (no London venue).
LOCATIONS = [
    {
        "bookseller_id": "BS_NOON",
        "venue_id": "LON098",
        "date_min": 1735,
        "date_max": 1745,
        "address_detail": "White Hart, near Mercer's Chapel, Cheapside",
        "source": "Harris",
    },
    {
        "bookseller_id": "BS_BORBET",
        "venue_id": "LON031",
        "date_min": 1738,
        "date_max": 1742,
        "address_detail": "Addison's Head, over against St Dunstan's Church, Fleet Street",
        "source": "Harris",
    },
    {
        "bookseller_id": "BS_LONGMAN",
        "venue_id": "LON099",
        "date_min": 1724,
        "date_max": 1755,
        "address_detail": "Ship, Paternoster Row",
        "source": "Harris",
    },
    {
        "bookseller_id": "BS_MILLAR",
        "venue_id": "LON100",
        "date_min": 1728,
        "date_max": 1768,
        "address_detail": "opposite Catherine Street in the Strand",
        "source": "Harris",
    },
    {
        "bookseller_id": "BS_CADELL",
        "venue_id": "LON100",
        "date_min": 1767,
        "date_max": 1802,
        "address_detail": "Strand (successor to Millar)",
        "source": "Harris",
    },
    {
        "bookseller_id": "BS_COOPER",
        "venue_id": "LON099",
        "date_min": 1743,
        "date_max": 1761,
        "address_detail": "Globe, Paternoster Row",
        "source": "Harris",
    },
    {
        "bookseller_id": "BS_DODSLEY",
        "venue_id": "LON101",
        "date_min": 1735,
        "date_max": 1764,
        "address_detail": "Tully's Head, Pall Mall",
        "source": "General knowledge",
    },
    {
        "bookseller_id": "BS_TONSON_SR",
        "venue_id": "LON031",
        "date_min": 1678,
        "date_max": 1720,
        "address_detail": "Shakespeare's Head, Fleet Street",
        "source": "General knowledge",
    },
    {
        "bookseller_id": "BS_TONSON_JR",
        "venue_id": "LON031",
        "date_min": 1720,
        "date_max": 1735,
        "address_detail": "Shakespeare's Head, Fleet Street",
        "source": "General knowledge",
    },
    {
        "bookseller_id": "BS_RIVINGTON",
        "venue_id": "LON102",
        "date_min": 1711,
        "date_max": 1742,
        "address_detail": "Bible and Crown, St Paul's Churchyard",
        "source": "General knowledge",
    },
    {
        "bookseller_id": "BS_STRAHAN",
        "venue_id": "LON031",
        "date_min": 1738,
        "date_max": 1785,
        "address_detail": "New Street, Shoe Lane (off Fleet Street)",
        "source": "General knowledge",
    },
    {
        "bookseller_id": "BS_MURRAY_I",
        "venue_id": "LON103",
        "date_min": 1768,
        "date_max": 1793,
        "address_detail": "32 Fleet Street",
        "source": "General knowledge",
    },
    {
        "bookseller_id": "BS_LACKINGTON",
        "venue_id": "LON104",
        "date_min": 1794,
        "date_max": 1804,
        "address_detail": "Finsbury Square",
        "source": "General knowledge",
    },
]


def write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def main():
    write_csv(HERE / "booksellers.csv", BOOKSELLERS_COLS, BOOKSELLERS)
    write_csv(HERE / "bookseller_locations.csv", LOCATIONS_COLS, LOCATIONS)


if __name__ == "__main__":
    main()
