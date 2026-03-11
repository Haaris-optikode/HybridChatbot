"""
pdf_parser.py — Universal PDF → Structured Markdown converter.

Two extraction strategies (auto-fallback):
  1. pymupdf4llm — Analyzes font sizes for headers, preserves tables as
     markdown, maintains lists. Uses ONNX layout detection.
  2. pymupdf (fitz) — Direct text extraction with heuristic markdown formatting.
     Used as fallback when pymupdf4llm fails (e.g., ONNX compatibility issues).

Both paths produce per-page markdown with headers, tables, and lists.
"""

import re
import logging
from typing import List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_pdf_to_markdown(pdf_path: str) -> List[Dict]:
    """Convert a PDF to structured markdown, one entry per page.

    Tries pymupdf4llm first (better quality), falls back to pymupdf direct
    extraction if ONNX or other runtime errors occur.

    Args:
        pdf_path: Full path to the PDF file.

    Returns:
        List of dicts: [{"page_number": int, "markdown": str}, ...]
    """
    pdf_path = str(Path(pdf_path).resolve())
    logger.info("[PDF Parser] Parsing: %s", Path(pdf_path).name)

    # Strategy 1: pymupdf4llm (font-aware markdown with tables)
    try:
        import pymupdf4llm
        pages = pymupdf4llm.to_markdown(
            pdf_path,
            page_chunks=True,
            show_progress=False,
        )

        result = []
        for page_data in pages:
            page_num = page_data.get("metadata", {}).get("page", 0) + 1
            md_text = _clean_markdown(page_data.get("text", ""))
            if md_text.strip():
                result.append({"page_number": page_num, "markdown": md_text})

        logger.info("[PDF Parser] pymupdf4llm: %d pages extracted", len(result))
        return result

    except Exception as e:
        logger.warning("[PDF Parser] pymupdf4llm failed (%s), using pymupdf fallback", str(e)[:120])

    # Strategy 2: pymupdf direct text extraction (robust fallback)
    return _extract_with_pymupdf(pdf_path)


def _extract_with_pymupdf(pdf_path: str) -> List[Dict]:
    """Extract text from PDF using pymupdf (fitz) with heuristic markdown formatting.

    Detects headers by uppercase lines and numbered patterns.
    Detects tables by consistent column alignment.
    """
    import pymupdf

    doc = pymupdf.open(pdf_path)
    result = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text("text")
        if not text.strip():
            continue

        md_text = _format_as_markdown(text)
        md_text = _clean_markdown(md_text)

        if md_text.strip():
            result.append({
                "page_number": page_idx + 1,
                "markdown": md_text,
            })

    doc.close()
    logger.info("[PDF Parser] pymupdf fallback: %d pages extracted", len(result))
    return result


def _format_as_markdown(text: str) -> str:
    """Apply heuristic markdown formatting to plain extracted text.

    Detects:
      - Section headings (numbered or all-caps lines)
      - Table-like structures (tab/multi-space separated columns)
    """
    lines = text.split("\n")
    output = []
    # Pattern for numbered headings: "1 TITLE", "18a. Title", etc.
    heading_re = re.compile(r"^(\d{1,2}(?:[a-d])?(?:\.\d{1,2})*)\.?\s+([A-Z][A-Z &/\-]+(?:\s+[A-Z &/\-]+)*)$")

    for line in lines:
        stripped = line.strip()
        if not stripped:
            output.append("")
            continue

        # Detect numbered section headings
        m = heading_re.match(stripped)
        if m:
            output.append(f"## {stripped}")
            continue

        # Detect all-caps short lines as headings (e.g., "PATIENT MEDICAL RECORD")
        if (stripped.isupper() and len(stripped) > 3 and len(stripped) < 80
                and not re.search(r"\d{3,}", stripped)):
            output.append(f"## {stripped}")
            continue

        # Detect table rows (Field\tValue or multi-space alignment)
        if "\t" in line or re.search(r"  {2,}", stripped):
            # Convert tabs/multi-spaces to markdown table separators
            cells = re.split(r"\t|  {2,}", stripped)
            cells = [c.strip() for c in cells if c.strip()]
            if len(cells) >= 2:
                output.append("| " + " | ".join(cells) + " |")
                continue

        output.append(stripped)

    return "\n".join(output)


def get_full_markdown(pdf_path: str) -> str:
    """Convert entire PDF to a single markdown string.

    Args:
        pdf_path: Full path to the PDF file.

    Returns:
        Full markdown text of the document.
    """
    pages = parse_pdf_to_markdown(pdf_path)
    return "\n\n".join(p["markdown"] for p in pages)


def _clean_markdown(text: str) -> str:
    """Clean up markdown output from pymupdf4llm.

    - Collapses runs of 3+ blank lines to 2
    - Strips trailing whitespace per line
    - Removes leading/trailing whitespace from the whole text
    """
    import re
    # Collapse excessive blank lines (3+ → 2)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    # Strip trailing whitespace per line
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()
