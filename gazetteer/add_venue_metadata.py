"""
Add architectural metadata columns to venues.csv.

New columns:
  enclosure     open | semi_open | enclosed
  building_type garden | theatre | assembly | church | park | square |
                street | market | prison | court | landmark | district |
                execution | misc
  material      stone | brick | timber | mixed | outdoor
  capacity      vast | large | medium | small | intimate | n_a

Run: python3 gazetteer/add_venue_metadata.py
"""

import csv
from pathlib import Path

VENUES_PATH = Path(__file__).parent / "venues.csv"

# (enclosure, building_type, material, capacity)
METADATA: dict[str, tuple[str, str, str, str]] = {
    # ── London pleasure gardens ──────────────────────────────────────────
    "LON001": ("open",      "garden",    "outdoor", "vast"),    # Vauxhall
    "LON002": ("semi_open", "garden",    "mixed",   "large"),   # Ranelagh (Rotunda + gardens)
    "LON003": ("open",      "garden",    "outdoor", "large"),   # Marylebone Gardens
    "LON017": ("open",      "garden",    "outdoor", "medium"),  # Spring Gardens Charing Cross
    "LON055": ("open",      "garden",    "outdoor", "medium"),  # White-Conduit House
    "LON056": ("open",      "garden",    "outdoor", "medium"),  # Bagnigge Wells

    # ── Theatres ─────────────────────────────────────────────────────────
    "LON004": ("enclosed",  "assembly",  "brick",   "large"),   # The Pantheon
    "LON006": ("enclosed",  "theatre",   "brick",   "large"),   # Drury Lane
    "LON007": ("enclosed",  "theatre",   "brick",   "large"),   # Covent Garden Theatre
    "LON008": ("enclosed",  "theatre",   "brick",   "large"),   # King's Theatre Haymarket

    # ── Assembly rooms & clubs ───────────────────────────────────────────
    "LON005": ("enclosed",  "assembly",  "brick",   "medium"),  # Almack's
    "LON018": ("enclosed",  "assembly",  "brick",   "small"),   # White's Club
    "LON019": ("enclosed",  "assembly",  "brick",   "small"),   # Boodle's
    "LON020": ("enclosed",  "assembly",  "brick",   "small"),   # Brooks's

    # ── Parks ────────────────────────────────────────────────────────────
    "LON009": ("open",      "park",      "outdoor", "vast"),    # St James's Park
    "LON010": ("open",      "park",      "outdoor", "vast"),    # Hyde Park
    "LON096": ("open",      "park",      "outdoor", "vast"),    # Greenwich Park

    # ── Residential squares ──────────────────────────────────────────────
    "LON011": ("semi_open", "square",    "brick",   "n_a"),     # Grosvenor Square
    "LON012": ("semi_open", "square",    "brick",   "n_a"),     # Hanover Square
    "LON013": ("semi_open", "square",    "brick",   "n_a"),     # Berkeley Square
    "LON014": ("semi_open", "square",    "brick",   "n_a"),     # St James's Square
    "LON024": ("semi_open", "square",    "brick",   "n_a"),     # Portman Square
    "LON026": ("semi_open", "square",    "brick",   "n_a"),     # Bedford Square
    "LON037": ("semi_open", "square",    "mixed",   "n_a"),     # Leicester Fields
    "LON038": ("semi_open", "square",    "brick",   "n_a"),     # Cavendish Square

    # ── Streets & thoroughfares ──────────────────────────────────────────
    "LON015": ("open",      "street",    "mixed",   "n_a"),     # The Strand
    "LON025": ("open",      "street",    "brick",   "n_a"),     # Portland Place
    "LON027": ("open",      "street",    "mixed",   "n_a"),     # Bond Street
    "LON028": ("open",      "street",    "mixed",   "n_a"),     # Pall Mall
    "LON029": ("open",      "street",    "mixed",   "n_a"),     # Piccadilly
    "LON030": ("open",      "street",    "mixed",   "n_a"),     # St James's Street
    "LON031": ("open",      "street",    "mixed",   "n_a"),     # Fleet Street
    "LON032": ("open",      "street",    "mixed",   "n_a"),     # Cheapside (early)
    "LON033": ("open",      "street",    "mixed",   "n_a"),     # Oxford Street
    "LON034": ("open",      "street",    "mixed",   "n_a"),     # Ludgate Hill
    "LON035": ("open",      "street",    "mixed",   "n_a"),     # Cornhill
    "LON036": ("open",      "street",    "mixed",   "n_a"),     # Charing Cross
    "LON039": ("open",      "street",    "brick",   "n_a"),     # Harley Street
    "LON040": ("open",      "street",    "brick",   "n_a"),     # Brook Street
    "LON090": ("open",      "street",    "mixed",   "n_a"),     # Cheapside (dup)
    "LON092": ("open",      "street",    "mixed",   "n_a"),     # Fleet Street (dup)

    # ── Markets ──────────────────────────────────────────────────────────
    "LON016": ("semi_open", "market",    "stone",   "large"),   # Covent Garden Piazza
    "LON082": ("open",      "market",    "outdoor", "vast"),    # Smithfield
    "LON083": ("open",      "market",    "outdoor", "large"),   # Billingsgate
    "LON084": ("semi_open", "market",    "stone",   "large"),   # Leadenhall (covered)
    "LON085": ("open",      "market",    "outdoor", "large"),   # Covent Garden Market

    # ── Churches ─────────────────────────────────────────────────────────
    "LON022": ("enclosed",  "church",    "stone",   "large"),   # St Paul's Cathedral
    "LON023": ("enclosed",  "church",    "stone",   "large"),   # Westminster Abbey

    # ── Civic landmark ───────────────────────────────────────────────────
    "LON021": ("semi_open", "landmark",  "stone",   "large"),   # Tower of London

    # ── Coffee houses ────────────────────────────────────────────────────
    "LON041": ("enclosed",  "assembly",  "timber",  "small"),   # Will's Coffee House
    "LON042": ("enclosed",  "assembly",  "timber",  "small"),   # Button's Coffee House
    "LON043": ("enclosed",  "assembly",  "timber",  "small"),   # Tom's Coffee House
    "LON044": ("enclosed",  "assembly",  "timber",  "small"),   # Slaughter's Coffee House
    "LON045": ("enclosed",  "assembly",  "timber",  "small"),   # Lloyd's Coffee House
    "LON046": ("enclosed",  "assembly",  "timber",  "small"),   # Chapter Coffee House
    "LON047": ("enclosed",  "assembly",  "timber",  "small"),   # Garraway's Coffee House
    "LON054": ("enclosed",  "assembly",  "timber",  "small"),   # Don Saltero's

    # ── Taverns ──────────────────────────────────────────────────────────
    "LON057": ("enclosed",  "assembly",  "timber",  "small"),   # Mother Red Cap's

    # ── Urban districts ──────────────────────────────────────────────────
    "LON048": ("open",      "district",  "mixed",   "n_a"),     # City of London
    "LON049": ("open",      "district",  "brick",   "n_a"),     # Bloomsbury
    "LON050": ("open",      "district",  "mixed",   "n_a"),     # Holborn
    "LON051": ("open",      "district",  "mixed",   "n_a"),     # Soho
    "LON052": ("open",      "district",  "mixed",   "n_a"),     # Southwark
    "LON053": ("open",      "district",  "brick",   "n_a"),     # Mayfair
    "LON086": ("open",      "district",  "timber",  "n_a"),     # St Giles/Seven Dials
    "LON087": ("open",      "district",  "mixed",   "n_a"),     # Whitechapel
    "LON088": ("open",      "district",  "mixed",   "n_a"),     # Spitalfields
    "LON089": ("open",      "district",  "mixed",   "n_a"),     # Wapping
    "LON091": ("open",      "district",  "mixed",   "n_a"),     # Holborn (dup)
    "LON093": ("open",      "district",  "mixed",   "n_a"),     # Southwark/Borough (dup)

    # ── Prisons ──────────────────────────────────────────────────────────
    "LON074": ("enclosed",  "prison",    "stone",   "large"),   # Newgate (Dance rusticated stone)
    "LON075": ("enclosed",  "prison",    "brick",   "medium"),  # Fleet Prison
    "LON076": ("semi_open", "prison",    "brick",   "medium"),  # Marshalsea (yard)
    "LON077": ("enclosed",  "prison",    "stone",   "medium"),  # Bridewell
    "LON078": ("semi_open", "prison",    "brick",   "medium"),  # King's Bench (Rules area)
    "LON079": ("semi_open", "prison",    "brick",   "small"),   # Tothill Fields Bridewell

    # ── Legal courts ─────────────────────────────────────────────────────
    "LON080": ("enclosed",  "court",     "stone",   "medium"),  # Old Bailey
    "LON081": ("enclosed",  "court",     "brick",   "small"),   # Bow Street

    # ── Execution sites ──────────────────────────────────────────────────
    "LON094": ("open",      "execution", "outdoor", "vast"),    # Tyburn Gallows
    "LON095": ("semi_open", "execution", "stone",   "large"),   # Newgate Gallows (in front of prison)

    # ── Bath ─────────────────────────────────────────────────────────────
    "BAT001": ("enclosed",  "assembly",  "stone",   "medium"),  # Lower Assembly Rooms
    "BAT002": ("enclosed",  "assembly",  "stone",   "medium"),  # Upper Assembly Rooms
    "BAT003": ("enclosed",  "assembly",  "stone",   "medium"),  # Pump Room
    "BAT004": ("semi_open", "square",    "stone",   "n_a"),     # The Circus
    "BAT005": ("semi_open", "square",    "stone",   "n_a"),     # Royal Crescent
    "BAT006": ("semi_open", "misc",      "stone",   "n_a"),     # Pulteney Bridge
    "BAT007": ("open",      "garden",    "outdoor", "medium"),  # Sydney Gardens
    "BAT008": ("semi_open", "square",    "stone",   "n_a"),     # Queen Square
    "BAT009": ("open",      "street",    "stone",   "n_a"),     # Milsom Street
    "BAT010": ("enclosed",  "church",    "stone",   "medium"),  # Bath Abbey
    "BAT011": ("open",      "street",    "stone",   "n_a"),     # North Parade
    "BAT012": ("open",      "street",    "stone",   "n_a"),     # South Parade
    "BAT013": ("open",      "street",    "stone",   "n_a"),     # Gay Street
    "BAT014": ("open",      "street",    "stone",   "n_a"),     # Cheap Street
    "BAT015": ("open",      "street",    "stone",   "n_a"),     # Argyle Street
    "BAT016": ("open",      "street",    "stone",   "n_a"),     # Pulteney Street

    # ── Pseudo-venues ─────────────────────────────────────────────────────
    "RUR001": ("open",      "misc",      "outdoor", "n_a"),
    "RUR002": ("open",      "misc",      "mixed",   "n_a"),
}

def main():
    rows = list(csv.DictReader(VENUES_PATH.open(newline="", encoding="utf-8")))
    existing_cols = list(rows[0].keys())

    new_cols = ["enclosure", "building_type", "material", "capacity"]
    # Only add if not already present
    for col in new_cols:
        if col not in existing_cols:
            existing_cols.append(col)

    covered = 0
    missing = []
    for row in rows:
        vid = row["id"]
        if vid in METADATA:
            enc, btype, mat, cap = METADATA[vid]
            row["enclosure"]     = enc
            row["building_type"] = btype
            row["material"]      = mat
            row["capacity"]      = cap
            covered += 1
        else:
            row.setdefault("enclosure",     "")
            row.setdefault("building_type", "")
            row.setdefault("material",      "")
            row.setdefault("capacity",      "")
            missing.append(vid)

    with VENUES_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=existing_cols)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {VENUES_PATH}")
    print(f"  {covered} venues annotated")
    if missing:
        print(f"  {len(missing)} without metadata: {missing}")

if __name__ == "__main__":
    main()
