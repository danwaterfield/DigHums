#!/usr/bin/env python3
"""
Replace noisy Internet Archive OCR files with clean alternatives where available.

Replacements:
  Godwin, Political Justice    IA djvu.txt → OLL text-based PDF (2 vols)
  Law, A Serious Call          IA djvu.txt → CCEL plain UTF-8 text
  Cheyne, The English Malady   original IA → BIM edition (Tesseract 5.3.0)
                                NOTE: still noisy, but better long-s handling

No clean alternative found for:
  Whytt, Observations on Nervous Disorders   → keep IA, note noise
  Trotter, View of the Nervous Temperament   → keep IA, note noise
  Tillotson, Sermons Vol 1                   → keep IA, note noise
  Fordyce, Sermons to Young Women            → keep IA, note noise
"""

import urllib.request
import shutil
import time
from pathlib import Path

import pdfplumber


NONFICTION = Path(__file__).parent.parent.parent / "nonfiction"


def download_text(url: str, dest: Path, label: str) -> bool:
    print(f"  GET  {label} ...", end=" ", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            dest.write_bytes(r.read())
        print(f"{dest.stat().st_size:,} bytes")
        time.sleep(1)
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def download_oll_pdf_to_text(pdf_url: str, dest: Path, label: str) -> bool:
    print(f"  GET  {label} (OLL PDF) ...", end=" ", flush=True)
    tmp = dest.parent / ("_tmp_" + dest.name.replace(".txt", ".pdf"))
    try:
        req = urllib.request.Request(pdf_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            tmp.write_bytes(r.read())
        pages = []
        with pdfplumber.open(tmp) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    pages.append(t)
        text = "\n\n".join(pages)
        dest.write_text(text, encoding="utf-8")
        tmp.unlink()
        print(f"{len(text):,} chars")
        time.sleep(1)
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        for p in [tmp, dest]:
            if p.exists():
                p.unlink()
        return False


def backup_and_replace(dest: Path) -> None:
    """Move existing file to .bak before replacing."""
    bak = dest.with_suffix(".txt.bak")
    if dest.exists() and not bak.exists():
        shutil.move(str(dest), str(bak))
        print(f"  BAK  {dest.name} → {bak.name}")


def main():
    print("=" * 60)
    print("Fixing noisy OCR texts")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. GODWIN — replace single noisy IA file with two clean OLL vols
    # ------------------------------------------------------------------
    print("\n--- Godwin, Political Justice (IA → OLL, 2 vols) ---")

    godwin_dir = NONFICTION / "politics" / "WilliamGodwin"

    # Back up and remove old single-file version
    old = godwin_dir / "PoliticalJustice.txt"
    backup_and_replace(old)

    download_oll_pdf_to_text(
        "https://oll-resources.s3.us-east-2.amazonaws.com/oll3/store/titles/90/Godwin_0164-01_EBk_v6.0.pdf",
        godwin_dir / "PoliticalJusticeVol1.txt",
        "PoliticalJusticeVol1.txt",
    )
    download_oll_pdf_to_text(
        "https://oll-resources.s3.us-east-2.amazonaws.com/oll3/store/titles/236/Godwin_0164-02_EBk_v6.0.pdf",
        godwin_dir / "PoliticalJusticeVol2.txt",
        "PoliticalJusticeVol2.txt",
    )

    # ------------------------------------------------------------------
    # 2. LAW — replace noisy IA file with CCEL clean text
    # ------------------------------------------------------------------
    print("\n--- Law, A Serious Call (IA → CCEL) ---")

    law_dest = NONFICTION / "religious" / "WilliamLaw" / "ASeriousCall.txt"
    backup_and_replace(law_dest)
    download_text(
        "https://ccel.org/ccel/l/law/serious_call/cache/serious_call.txt",
        law_dest,
        "ASeriousCall.txt",
    )

    # ------------------------------------------------------------------
    # 3. CHEYNE — swap to BIM edition (Tesseract 5.3.0, better long-s)
    # ------------------------------------------------------------------
    print("\n--- Cheyne, English Malady (original IA → BIM edition) ---")

    cheyne_dest = NONFICTION / "medical" / "GeorgeCheyne" / "TheEnglishMalady.txt"
    backup_and_replace(cheyne_dest)
    download_text(
        "https://archive.org/download/bim_eighteenth-century_the-english-malady-or-_cheyne-george_1733/bim_eighteenth-century_the-english-malady-or-_cheyne-george_1733_djvu.txt",
        cheyne_dest,
        "TheEnglishMalady.txt (BIM/Tesseract 5.3.0)",
    )

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Done. Backups saved as .txt.bak alongside originals.")
    print()
    print("Remaining noisy files (no clean alternative available):")
    noisy = [
        "medical/RobertWhytt/ObservationsOnNervousDisorders.txt",
        "medical/ThomasTrotter/ViewOfTheNervousTemperament.txt",
        "religious/JohnTillotson/SermonsVol1.txt",
        "conduct/JamesFordyce/SermonsToYoungWomen.txt",
    ]
    for f in noisy:
        size = (NONFICTION / f).stat().st_size if (NONFICTION / f).exists() else 0
        print(f"  {f}  ({size:,} bytes)")
    print()
    print("These are usable for frequency/collocation analysis but")
    print("should be excluded from neural models.")
    print("=" * 60)


if __name__ == "__main__":
    main()
