#!/usr/bin/env python3
"""
Download Shaftesbury's Characteristicks of Men, Manners, Opinions, Times (1711)
from Liberty Fund's Online Library of Liberty (text-based PDFs).

Extracts clean text and saves to:
  nonfiction/philosophy/Shaftesbury/CharacteristicsVol1.txt
  nonfiction/philosophy/Shaftesbury/CharacteristicsVol2.txt
  nonfiction/philosophy/Shaftesbury/CharacteristicsVol3.txt
"""

import urllib.request
import time
from pathlib import Path

import pdfplumber


VOLUMES = [
    (
        "CharacteristicsVol1.txt",
        "https://oll-resources.s3.us-east-2.amazonaws.com/oll3/store/titles/811/Shaftesbury_5987_EBk_v6.0.pdf",
    ),
    (
        "CharacteristicsVol2.txt",
        "https://oll-resources.s3.us-east-2.amazonaws.com/oll3/store/titles/812/Shaftesbury_6666_EBk_v6.0.pdf",
    ),
    (
        "CharacteristicsVol3.txt",
        "https://oll-resources.s3.us-east-2.amazonaws.com/oll3/store/titles/813/Shaftesbury_5989_EBk_v6.0.pdf",
    ),
]


def download_pdf(url: str, dest: Path) -> bool:
    print(f"  GET  {url.split('/')[-1]} ...", end=" ", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            dest.write_bytes(r.read())
        print(f"{dest.stat().st_size:,} bytes")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def pdf_to_text(pdf_path: Path) -> str:
    """Extract text from a text-based PDF using pdfplumber."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def main():
    corpus_root = Path(__file__).parent.parent.parent
    out_dir = corpus_root / "nonfiction" / "philosophy" / "Shaftesbury"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_pdf_tmp"
    tmp_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Shaftesbury - Characteristicks (Liberty Fund PDFs)")
    print("=" * 60)

    for filename, url in VOLUMES:
        txt_dest = out_dir / filename
        if txt_dest.exists():
            print(f"  SKIP {filename} (already exists)")
            continue

        pdf_path = tmp_dir / filename.replace(".txt", ".pdf")

        # Download PDF
        if not pdf_path.exists():
            if not download_pdf(url, pdf_path):
                continue
            time.sleep(1)

        # Extract text
        print(f"  EXTRACT {filename} ...", end=" ", flush=True)
        try:
            text = pdf_to_text(pdf_path)
            txt_dest.write_text(text, encoding="utf-8")
            print(f"{len(text):,} chars → {txt_dest.name}")
            # Clean up PDF after successful extraction
            pdf_path.unlink()
        except Exception as e:
            print(f"FAILED: {e}")

    # Remove temp dir if empty
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    print("\nDone. Files saved to:", out_dir)


if __name__ == "__main__":
    main()
