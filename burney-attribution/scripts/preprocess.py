#!/usr/bin/env python3
"""
Preprocessing pipeline for 18th-century novels from Project Gutenberg.

Handles:
- Stripping Project Gutenberg headers/footers
- Normalizing quotation marks and whitespace
- Preserving sentence/paragraph boundaries
- Cleaning OCR artifacts
"""

import re
import os
from pathlib import Path
from typing import Tuple


def strip_gutenberg_boilerplate(text: str) -> str:
    """
    Remove Project Gutenberg header and footer text.

    Args:
        text: Raw text from Project Gutenberg file

    Returns:
        Text with boilerplate removed
    """
    # Find the start marker
    start_patterns = [
        r'\*\*\* START OF THE PROJECT GUTENBERG EBOOK .+? \*\*\*',
        r'\*\*\*START OF THE PROJECT GUTENBERG EBOOK .+?\*\*\*',
        r'START OF THIS PROJECT GUTENBERG EBOOK',
    ]

    start_pos = 0
    for pattern in start_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start_pos = match.end()
            break

    # Find the end marker
    end_patterns = [
        r'\*\*\* END OF THE PROJECT GUTENBERG EBOOK .+? \*\*\*',
        r'\*\*\*END OF THE PROJECT GUTENBERG EBOOK .+?\*\*\*',
        r'END OF THIS PROJECT GUTENBERG EBOOK',
    ]

    end_pos = len(text)
    for pattern in end_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            end_pos = match.start()
            break

    return text[start_pos:end_pos].strip()


def normalize_quotes(text: str) -> str:
    """
    Normalize various quotation mark styles to standard ASCII.

    Args:
        text: Input text

    Returns:
        Text with normalized quotes
    """
    # Smart quotes to straight quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    text = text.replace('`', "'").replace('´', "'")

    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace while preserving paragraph structure.

    Args:
        text: Input text

    Returns:
        Text with normalized whitespace
    """
    # Remove carriage returns
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Collapse multiple spaces (but not newlines)
    text = re.sub(r'[ \t]+', ' ', text)

    # Remove trailing whitespace from lines
    text = re.sub(r' +\n', '\n', text)

    # Normalize paragraph breaks (3+ newlines -> 2 newlines)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def clean_ocr_artifacts(text: str) -> str:
    """
    Remove common OCR artifacts and formatting issues.

    Args:
        text: Input text

    Returns:
        Cleaned text
    """
    # Remove page numbers (common pattern: standalone numbers on lines)
    text = re.sub(r'\n\d+\n', '\n', text)

    # Remove standalone Roman numerals (chapter markers often get separated)
    text = re.sub(r'\n[IVXLCDM]+\n', '\n', text)

    # Remove excessive underscores (used for emphasis in some versions)
    text = text.replace('_', '')

    return text


def clean_text(text: str) -> str:
    """
    Full cleaning pipeline for a single text.

    Args:
        text: Raw text from Project Gutenberg

    Returns:
        Cleaned text ready for analysis
    """
    text = strip_gutenberg_boilerplate(text)
    text = normalize_quotes(text)
    text = clean_ocr_artifacts(text)
    text = normalize_whitespace(text)

    return text


def process_file(input_path: Path, output_path: Path) -> Tuple[int, int]:
    """
    Process a single file through the cleaning pipeline.

    Args:
        input_path: Path to raw text file
        output_path: Path for cleaned output

    Returns:
        Tuple of (original_length, cleaned_length)
    """
    # Read with UTF-8 encoding, handling BOM if present
    with open(input_path, 'r', encoding='utf-8-sig') as f:
        raw_text = f.read()

    original_length = len(raw_text)

    # Clean the text
    cleaned_text = clean_text(raw_text)

    cleaned_length = len(cleaned_text)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)

    return original_length, cleaned_length


def process_corpus(raw_dir: Path, processed_dir: Path):
    """
    Process entire corpus of texts.

    Args:
        raw_dir: Directory containing raw text files
        processed_dir: Directory for processed output
    """
    print(f"Processing corpus from {raw_dir} to {processed_dir}\n")

    total_files = 0
    total_original = 0
    total_cleaned = 0

    # Process all .txt files in subdirectories
    for txt_file in raw_dir.rglob('*.txt'):
        # Get relative path to maintain directory structure
        rel_path = txt_file.relative_to(raw_dir)
        output_path = processed_dir / rel_path

        try:
            orig_len, clean_len = process_file(txt_file, output_path)
            reduction = ((orig_len - clean_len) / orig_len * 100) if orig_len > 0 else 0

            print(f"✓ {rel_path}")
            print(f"  {orig_len:,} → {clean_len:,} chars ({reduction:.1f}% reduction)")

            total_files += 1
            total_original += orig_len
            total_cleaned += clean_len

        except Exception as e:
            print(f"✗ {rel_path}: {e}")

    print(f"\n{'='*60}")
    print(f"Processed {total_files} files")
    print(f"Total: {total_original:,} → {total_cleaned:,} chars")
    total_reduction = ((total_original - total_cleaned) / total_original * 100) if total_original > 0 else 0
    print(f"Overall reduction: {total_reduction:.1f}%")


if __name__ == "__main__":
    # Set up paths relative to script location
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    raw_dir = project_dir / "data" / "raw"
    processed_dir = project_dir / "data" / "processed"

    process_corpus(raw_dir, processed_dir)
