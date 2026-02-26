#!/usr/bin/env python3
"""
Second corpus expansion: sectarian/Catholic, philosophical foundations,
Burney's own journals, and missing fiction.

New additions by cluster:

  POLITICS / SECTARIAN
    Burke, Writings Vol IV          — letters to Ireland & Langrishe on Catholic relief
    Priestley, First Principles     — Dissenter political philosophy (OLL PDF)
    Priestley, Letters to Burke     — direct response to Reflections (OLL PDF)
    Price, Love of Our Country      — the sermon Burke attacked (OLL PDF)
    Price, Observations on America  — religious liberty + political reform (OLL PDF)

  PHILOSOPHY (foundations)
    Hume, Treatise of Human Nature  — Books 2-3: passions and morals
    Hutcheson, Inquiry              — moral sense theory; Shaftesbury's heir (OLL PDF)

  CONDUCT
    Chapone, Letters on Improvement — the one woman in the conduct cluster; Burney knew her

  RELIGIOUS
    Butler, Fifteen Sermons         — Anglican ethical centre (CCEL plain text)
    Butler, Analogy of Religion     — Anglican apologetics

  FICTION (missing)
    Richardson, Sir Charles Grandison Vol 4  — only vol available on Gutenberg
    Mackenzie, The Man of Feeling            — sentimental masculinity; social incapacity
    Sheridan, Memoirs of Miss Sidney Bidulph — female precursor to Burney

  BURNEY NON-FICTION
    Diary and Letters of Madame d'Arblay, Vols 1-3  — biographical core

NOTE: Richardson's Grandison Vols 1-3, 5-7 are not on Gutenberg.
      Seek via ECCO and save to SamuelRichardson/ alongside Vol 4.
"""

import urllib.request
import time
from pathlib import Path

import pdfplumber


# ---------------------------------------------------------------------------
# GUTENBERG — direct UTF-8 text
# (subdir_from_corpus_root, filename, gutenberg_id)
# ---------------------------------------------------------------------------
GUTENBERG = [

    # POLITICS: Burke's Catholic letters (Vol IV of his Writings)
    ("politics/EdmundBurke",    "WritingsVol4_IrelandAndLangrishe.txt", "15700"),

    # PHILOSOPHY: Hume's Treatise — the philosophical engine behind the Essays
    ("philosophy/DavidHume",    "TreatiseOfHumanNature.txt",            "4705"),

    # CONDUCT: Chapone — the female conduct writer; Burney knew her personally
    ("conduct/HesterChapone",   "LettersOnImprovementOfMind.txt",       "35890"),

    # RELIGIOUS: Butler's Analogy
    ("religious/JosephButler",  "AnalogyOfReligion.txt",                "53346"),

    # BURNEY NON-FICTION: her own self-record — biographical evidence for
    # the neurodivergence argument, the Cambridge episode, the court years
    ("FrancesBurney",           "DiaryAndLettersVol1.txt",              "5826"),
    ("FrancesBurney",           "DiaryAndLettersVol2.txt",              "6042"),
    ("FrancesBurney",           "DiaryAndLettersVol3.txt",              "6457"),
]

# Fiction additions go into the main corpus root (alongside other novel dirs)
GUTENBERG_FICTION = [
    # Richardson: only vol 4 available on Gutenberg
    ("SamuelRichardson",        "SirCharlesGrandison_Vol4.txt",         "13884"),
    # Mackenzie: sentimental masculinity, social-cognitive incapacity in a man
    ("HenryMackenzie",          "TheManOfFeeling.txt",                  "5083"),
    # Sheridan: female precursor to Burney, epistolary, published anonymously
    ("FrancesSheridan",         "MemoirsOfMissSidneyBidulph.txt",       "43437"),
]


# ---------------------------------------------------------------------------
# OLL PDFs — text-based, extract with pdfplumber (same method as Shaftesbury)
# (subdir_from_nonfiction, filename, pdf_url)
# ---------------------------------------------------------------------------
OLL_PDFS = [
    # POLITICS / SECTARIAN
    ("politics/JosephPriestley",
     "FirstPrinciplesOfGovernment.txt",
     "https://oll-resources.s3.us-east-2.amazonaws.com/oll3/store/titles/1767/Priestley_0893_EBk_v6.0.pdf"),

    ("politics/JosephPriestley",
     "LettersToBurke.txt",
     "https://oll-resources.s3.us-east-2.amazonaws.com/oll3/store/titles/1790/Priestley_1289_EBk_v6.0.pdf"),

    ("politics/RichardPrice",
     "DiscourseOnLoveOfOurCountry.txt",
     "https://oll-resources.s3.us-east-2.amazonaws.com/oll3/store/titles/368/Price_1290_EBk_v6.0.pdf"),

    ("politics/RichardPrice",
     "ObservationsOnAmericanRevolution.txt",
     "https://oll-resources.s3.us-east-2.amazonaws.com/oll3/store/titles/1788/Price_0894_EBk_v6.0.pdf"),

    # PHILOSOPHY: Hutcheson — moral sense theory; the missing link between
    # Shaftesbury and Hume, foundational for the politeness-as-ethics argument
    ("philosophy/FrancisHutcheson",
     "InquiryIntoBeautyAndVirtue.txt",
     "https://oll-resources.s3.us-east-2.amazonaws.com/oll3/store/titles/2462/Hutcheson_1458_EBk_v7.0.pdf"),
]


# ---------------------------------------------------------------------------
# CCEL — Christian Classics Ethereal Library plain text
# (subdir_from_nonfiction, filename, url)
# ---------------------------------------------------------------------------
CCEL = [
    ("religious/JosephButler",
     "FifteenSermons.txt",
     "https://ccel.org/ccel/b/butler/sermons/cache/sermons.txt"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_text(url: str, dest: Path, label: str, delay: int = 1) -> bool:
    if dest.exists():
        print(f"  SKIP  {label} ({dest.stat().st_size:,} bytes)")
        return True
    print(f"  GET   {label} ...", end=" ", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            dest.write_bytes(r.read())
        print(f"{dest.stat().st_size:,} bytes")
        time.sleep(delay)
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        if dest.exists():
            dest.unlink()
        return False


def pdf_to_text(pdf_path: Path) -> str:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def download_oll_pdf(subdir: str, filename: str, pdf_url: str,
                     nonfiction_root: Path) -> bool:
    dest = nonfiction_root / subdir / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  SKIP  nonfiction/{subdir}/{filename} ({dest.stat().st_size:,} bytes)")
        return True

    tmp_dir = dest.parent / "_pdf_tmp"
    tmp_dir.mkdir(exist_ok=True)
    pdf_path = tmp_dir / filename.replace(".txt", ".pdf")

    print(f"  GET   nonfiction/{subdir}/{filename} (OLL PDF) ...", end=" ", flush=True)
    try:
        req = urllib.request.Request(pdf_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            pdf_path.write_bytes(r.read())
        text = pdf_to_text(pdf_path)
        dest.write_text(text, encoding="utf-8")
        print(f"{len(text):,} chars")
        pdf_path.unlink()
        try:
            tmp_dir.rmdir()
        except OSError:
            pass
        time.sleep(1)
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        for p in [pdf_path, dest]:
            if p.exists():
                p.unlink()
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    corpus_root = Path(__file__).parent.parent.parent
    nonfiction_root = corpus_root / "nonfiction"

    print("=" * 60)
    print("Corpus Expansion 2: Sectarian, Philosophical, Biographical")
    print("=" * 60)

    ok = fail = 0

    print("\n--- Gutenberg: non-fiction ---")
    for subdir, filename, gid in GUTENBERG:
        dest = nonfiction_root / subdir / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://www.gutenberg.org/ebooks/{gid}.txt.utf-8"
        if download_text(url, dest, f"nonfiction/{subdir}/{filename}"):
            ok += 1
        else:
            fail += 1

    print("\n--- Gutenberg: fiction ---")
    for author_dir, filename, gid in GUTENBERG_FICTION:
        dest = corpus_root / author_dir / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://www.gutenberg.org/ebooks/{gid}.txt.utf-8"
        if download_text(url, dest, f"{author_dir}/{filename}"):
            ok += 1
        else:
            fail += 1

    print("\n--- Online Library of Liberty (PDF extraction) ---")
    for subdir, filename, pdf_url in OLL_PDFS:
        if download_oll_pdf(subdir, filename, pdf_url, nonfiction_root):
            ok += 1
        else:
            fail += 1

    print("\n--- Christian Classics Ethereal Library ---")
    for subdir, filename, url in CCEL:
        dest = nonfiction_root / subdir / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        if download_text(url, dest, f"nonfiction/{subdir}/{filename}"):
            ok += 1
        else:
            fail += 1

    print(f"\n{'=' * 60}")
    print(f"Total: {ok} ok, {fail} failed")
    print()
    print("Still to source manually:")
    print("  Richardson, Sir Charles Grandison Vols 1-3, 5-7")
    print("  → ECCO; save to SamuelRichardson/ alongside Vol 4")
    print("  Burney, Early Journals and Letters (pre-Evelina)")
    print("  → Yale Burney edition; not freely available online")
    print("=" * 60)


if __name__ == "__main__":
    main()
