#!/usr/bin/env python3
"""
Download non-fiction texts for corpus expansion.

Organized by research cluster:
  1. Politeness / Periodicals  — the normative apparatus (Spectator, Tatler)
  2. Conduct Literature        — gendered conformity (Chesterfield, Fordyce, Gregory, More)
  3. Philosophy                — sociability, sympathy, commerce (Hume, Smith, Mandeville)
  4. Politics                  — the 1790s rupture (Burke, Paine, Wollstonecraft, Godwin)
  5. Medical / Nervous         — proto-neurodivergence (Cheyne, Trotter, Whytt)
  6. Religious                 — Anglican foundation and dissent (Law, Tillotson)

Sources:
  - Project Gutenberg: clean UTF-8 text, directly downloadable
  - Internet Archive: OCR'd djvu.txt, noisier but adequate for corpus analysis

NOT YET AVAILABLE PROGRAMMATICALLY:
  - Shaftesbury, Characteristics of Men, Manners, Opinions, Times (1711)
    → The founding text of 18c politeness theory. Seek via ECCO or Liberty Fund
      (oll.libertyfund.org) and add manually as:
      philosophy/Shaftesbury/Characteristics.txt
  - Francis Hutcheson, Inquiry into the Original of our Ideas of Beauty and Virtue (1725)
    → Available through ECCO; foundational for moral sense theory
"""

import urllib.request
import os
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# GUTENBERG DOWNLOADS
# Clean UTF-8 text. Format: (subdir, filename, gutenberg_id)
# ---------------------------------------------------------------------------
GUTENBERG = [

    # 1. POLITENESS / PERIODICALS
    # The Spectator (1711-12): primary vehicle for polite culture's dissemination
    ("periodicals/AddisonSteele", "SpectatorVol1.txt",  "9334"),
    ("periodicals/AddisonSteele", "SpectatorVol2.txt",  "11010"),
    # The Tatler (1709-11): precursor, more overtly political
    ("periodicals/AddisonSteele", "TatlerVol1.txt",     "13645"),
    ("periodicals/AddisonSteele", "TatlerVol2.txt",     "45769"),
    ("periodicals/AddisonSteele", "TatlerVol3.txt",     "31645"),
    ("periodicals/AddisonSteele", "TatlerVol4.txt",     "49009"),

    # 2. CONDUCT LITERATURE
    # Chesterfield: politeness as explicit instruction (the gap between natural
    # social cognition and its teachability is central here)
    ("conduct/LordChesterfield",  "LettersToHisSon.txt", "3361"),
    # Gregory: Anglican father instructing daughters — the theological valence
    # of female conduct
    ("conduct/JohnGregory",       "AFathersLegacy.txt",  "50108"),

    # 3. PHILOSOPHY
    # Hume: Essays include "Of Essay Writing", "Of Refinement in the Arts",
    # "Of the Delicacy of Taste and Passion" — key for politeness as aesthetic
    ("philosophy/DavidHume",      "EssaysMoralPolitical.txt",    "36120"),
    # Smith: sympathy and sociability — the imaginative mechanism underlying
    # polite social performance
    ("philosophy/AdamSmith",      "TheoryOfMoralSentiments.txt", "67363"),
    # Mandeville: commerce and sociability, vices as social glue —
    # the dark underside of the polite consensus
    ("philosophy/Mandeville",     "FableOfTheBees.txt",          "57260"),

    # 4. POLITICS
    # Burke: politeness as political theology; the Revolution as rupture
    # of the polite order; also key for his latent Catholic sympathies
    ("politics/EdmundBurke",      "ReflectionsOnTheRevolution.txt", "15679"),
    # Paine: bluntness as a deliberate counter-register to Burke's politeness
    ("politics/ThomasPaine",      "RightsOfMan.txt",               "3742"),
    # Wollstonecraft: both texts — the Rights of Men is the immediate
    # response to Burke; Rights of Woman extends the argument to conduct
    # literature directly (she reads Fordyce and Gregory closely)
    ("politics/Wollstonecraft",   "VindicationOfRightsOfWoman.txt", "3420"),
    ("politics/Wollstonecraft",   "VindicationOfRightsOfMen.txt",   "62757"),
]


# ---------------------------------------------------------------------------
# INTERNET ARCHIVE DOWNLOADS
# OCR'd djvu.txt — noisier than Gutenberg but adequate for:
#   collocation analysis, topic modelling, frequency analysis
# Less suitable for: neural models expecting clean text
# Format: (subdir, filename, ia_identifier)
# ---------------------------------------------------------------------------
INTERNET_ARCHIVE = [

    # 2. CONDUCT LITERATURE (continued)
    # Fordyce: Wollstonecraft's primary target; constructs the feminine ideal
    # as affective, deferent, sociable — Anglican womanhood as conduct norm
    ("conduct/JamesFordyce",  "SermonsToYoungWomen.txt",
     "bub_gb_5gjjojYJiogC"),
    # More: evangelical Anglican conduct; politeness meets moral reform
    ("conduct/HannahMore",    "StricturesOnFemaleEducation.txt",
     "bim_eighteenth-century_strictures-on-the-modern_more-hannah_1799_1_0"),

    # 4. POLITICS (continued)
    # Godwin: the radical pole; associationist psychology applied to politics;
    # useful contrast for the nervous/sensibility cluster
    ("politics/WilliamGodwin", "PoliticalJustice.txt",
     "enquiryconcernin01godw"),

    # 5. MEDICAL / NERVOUS TEMPERAMENT
    # Cheyne: "The English Malady" (1733) — foundational; urban luxury causes
    # nervous disorder; proto-neurodivergence as class disease of refinement
    ("medical/GeorgeCheyne",   "TheEnglishMalady.txt",
     "englishmaladyort00cheyuoft"),
    # Trotter: extends Cheyne into the early 19c; more systematic nosology
    ("medical/ThomasTrotter",  "ViewOfTheNervousTemperament.txt",
     "viewofnervoustem00trot"),
    # Whytt: mid-century; nervous disorders as constitutional, not moral —
    # the "can't" vs "won't" shift that underpins the neurodivergence argument
    ("medical/RobertWhytt",    "ObservationsOnNervousDisorders.txt",
     "observationsonna00whytuoft"),

    # 6. RELIGIOUS
    # Law: High Church counterpoint to Latitudinarian politeness; useful
    # for mapping the Anglican internal dispute
    ("religious/WilliamLaw",   "ASeriousCall.txt",
     "seriouscallt00laww"),
    # Tillotson: Archbishop of Canterbury; Latitudinarian politeness theology
    # at its source — sermons that circulated as widely as novels
    ("religious/JohnTillotson", "SermonsVol1.txt",
     "worksofmostrever01till"),
]


def download_gutenberg(subdir: str, filename: str, gutenberg_id: str,
                       corpus_root: Path) -> bool:
    url = f"https://www.gutenberg.org/ebooks/{gutenberg_id}.txt.utf-8"
    dest = corpus_root / "nonfiction" / subdir / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  SKIP  nonfiction/{subdir}/{filename} ({dest.stat().st_size:,} bytes)")
        return True
    print(f"  GET   nonfiction/{subdir}/{filename} (Gutenberg {gutenberg_id}) ...",
          end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"{dest.stat().st_size:,} bytes")
        time.sleep(1)
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        if dest.exists():
            dest.unlink()
        return False


def download_ia(subdir: str, filename: str, ia_id: str,
                corpus_root: Path) -> bool:
    url = f"https://archive.org/download/{ia_id}/{ia_id}_djvu.txt"
    dest = corpus_root / "nonfiction" / subdir / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  SKIP  nonfiction/{subdir}/{filename} ({dest.stat().st_size:,} bytes)")
        return True
    print(f"  GET   nonfiction/{subdir}/{filename} (IA: {ia_id}) ...",
          end=" ", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as response:
            dest.write_bytes(response.read())
        print(f"{dest.stat().st_size:,} bytes")
        time.sleep(2)
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        if dest.exists():
            dest.unlink()
        return False


def main():
    corpus_root = Path(__file__).parent.parent.parent

    print("=" * 60)
    print("Non-Fiction Corpus Download")
    print(f"Target: {corpus_root / 'nonfiction'}")
    print("=" * 60)

    gutenberg_ok = gutenberg_fail = 0
    ia_ok = ia_fail = 0

    print("\n--- Project Gutenberg (clean text) ---")
    for subdir, filename, gid in GUTENBERG:
        if download_gutenberg(subdir, filename, gid, corpus_root):
            gutenberg_ok += 1
        else:
            gutenberg_fail += 1

    print("\n--- Internet Archive (OCR text) ---")
    for subdir, filename, ia_id in INTERNET_ARCHIVE:
        if download_ia(subdir, filename, ia_id, corpus_root):
            ia_ok += 1
        else:
            ia_fail += 1

    print("\n" + "=" * 60)
    print(f"Gutenberg: {gutenberg_ok} ok, {gutenberg_fail} failed")
    print(f"Internet Archive: {ia_ok} ok, {ia_fail} failed")
    print(f"Total: {gutenberg_ok + ia_ok} texts downloaded")
    print()
    print("NOTE: Shaftesbury, Characteristics (1711) requires manual")
    print("acquisition — seek via ECCO or libertyfund.org and save as:")
    print("  nonfiction/philosophy/Shaftesbury/Characteristics.txt")
    print()
    print("Internet Archive texts are OCR'd and noisier than Gutenberg.")
    print("Run a cleaning pass before using in neural models.")
    print("=" * 60)


if __name__ == "__main__":
    main()
