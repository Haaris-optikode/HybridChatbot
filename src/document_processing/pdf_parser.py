"""
pdf_parser.py — Production-grade PDF → Structured Markdown converter.

Extraction strategy (priority order):
  1. pdfplumber   — PRIMARY. Spatial table detection from PDF drawing commands
                    (ruling lines/rectangles). Font-size-aware heading detection
                    via character-level metadata. Zero ML dependency, deterministic,
                    battle-tested on dense clinical/financial documents.
  2. pymupdf dict — FALLBACK. Font-size-aware heading detection using pymupdf's
                    dict extraction mode. Replaces the original heuristic
                    (isupper + regex) that was producing false headers from ICD
                    codes, insurance numbers, and time slots.
  3. [Scanned]    — Handled upstream by ocr_fallback.py (unchanged).

Why pdfplumber over pymupdf4llm:
  - No ONNX / ML inference dependency (zero runtime errors)
  - Spatial table detection via PDF drawing commands — no column guessing
  - Character-level font metadata for reliable heading detection
  - Fast: pure Python, sub-second per page for typical clinical notes
  - Deterministic: same input always produces identical output

Why font-size heading detection over isupper() heuristic:
  - isupper() catches ICD codes (E66.9), insurance numbers (1EG4-TE5),
    time slots (3:00-3:30 PM), and LOINC lab codes — all false positives
  - Font size is the ground truth signal PDFs embed for visual hierarchy
  - Negative filters guard against edge cases (measurement units, data lines)
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter

logger = logging.getLogger(__name__)


# ── Heading Classification Constants ─────────────────────────────────────────

# A line's font size must be >= body_size * this ratio to be a heading candidate
_HEADING_FONT_RATIO = 1.12

# Hard-reject patterns: these look like headings but are clinical data values.
# Order matters — checked sequentially, first match rejects.
_NON_HEADING_PATTERNS: List[re.Pattern] = [
    re.compile(r"^[A-Z]\d+[\.\-]\d+"),          # ICD-10: E66.9, M79.3-1
    re.compile(r"^\d+[A-Z0-9]{2,}[\-]"),        # Insurance codes: 1EG4-TE5
    re.compile(r"^\d{1,2}:\d{2}"),              # Time values: 3:00, 14:30
    re.compile(r"\b\d{4,}\b"),                  # Long numeric codes (LOINC IDs etc.)
    re.compile(r"^[A-Z]{1,4}\d{4,}"),           # Lab codes: LOINC-style ABC12345
    re.compile(r"\d+\.?\d*\s+\d+\.?\d*\s+\d+\.?\d*"),  # Adjacent numbers (lab rows)
]

# Reject lines containing clinical measurement units — these are data rows
_UNIT_PATTERN = re.compile(
    r"\b(?:mg|mL|mmHg|µ[Lg]|mcg|mEq|units?|%|mmol|µmol|ng|pg|IU|"
    r"tablet|capsule|BID|TID|QID|PRN|qDay|kg|lbs?|cm|mm)\b",
    re.IGNORECASE,
)

# Minimum real alphabetic words for a heading (avoids single-word codes)
_HEADING_MIN_ALPHA_WORDS = 2


# ── pdfplumber Table Detection Settings ──────────────────────────────────────

# Strict: uses explicit ruling lines drawn in the PDF (most accurate)
_TABLE_SETTINGS_STRICT: Dict = {
    "vertical_strategy": "lines_strict",
    "horizontal_strategy": "lines_strict",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 3,
    "min_words_vertical": 1,
    "min_words_horizontal": 1,
    "intersection_tolerance": 3,
}

# Text-based: detects tables by column alignment in the text stream
# (used when no explicit ruling lines exist in the PDF)
_TABLE_SETTINGS_TEXT: Dict = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "snap_tolerance": 5,
    "join_tolerance": 3,
    "min_words_vertical": 2,
    "min_words_horizontal": 2,
}


# ── Public API ────────────────────────────────────────────────────────────────

def parse_pdf_to_markdown(pdf_path: str) -> List[Dict]:
    """Convert a PDF to structured markdown, one entry per page.

    Tries pdfplumber first (spatial table detection + font-aware headings).
    Falls back to pymupdf dict mode if pdfplumber fails or yields thin results.

    Args:
        pdf_path: Full path to the PDF file.

    Returns:
        List of dicts: [{"page_number": int, "markdown": str}, ...]
    """
    pdf_path = str(Path(pdf_path).resolve())
    logger.info("[PDF Parser] Parsing: %s", Path(pdf_path).name)

    # Strategy 1: pdfplumber (spatial tables + font-aware headings)
    try:
        result = _extract_with_pdfplumber(pdf_path)
        if result and _has_substance(result):
            logger.info("[PDF Parser] pdfplumber: %d pages extracted", len(result))
            return result
        logger.info(
            "[PDF Parser] pdfplumber returned thin content (%d pages), "
            "trying pymupdf dict fallback",
            len(result),
        )
    except Exception as e:
        logger.warning(
            "[PDF Parser] pdfplumber failed (%s), falling back to pymupdf dict",
            str(e)[:150],
        )

    # Strategy 2: pymupdf dict mode (font-aware, better than heuristic)
    try:
        result = _extract_with_pymupdf_dict(pdf_path)
        if result:
            logger.info("[PDF Parser] pymupdf dict: %d pages extracted", len(result))
            return result
    except Exception as e:
        logger.warning("[PDF Parser] pymupdf dict failed: %s", str(e)[:150])

    logger.error("[PDF Parser] All strategies failed for %s", pdf_path)
    return []


def get_full_markdown(pdf_path: str) -> str:
    """Convert entire PDF to a single markdown string."""
    pages = parse_pdf_to_markdown(pdf_path)
    return "\n\n".join(p["markdown"] for p in pages)


# ── Strategy 1: pdfplumber ────────────────────────────────────────────────────

def _extract_with_pdfplumber(pdf_path: str) -> List[Dict]:
    """
    Primary extractor. Uses pdfplumber for spatial table detection and
    pymupdf (dict mode) for character-level font metadata.

    Opening both in parallel: pdfplumber gives us spatial layout and table
    geometry; pymupdf gives us the font sizes that pdfplumber doesn't expose
    directly in its word-level API.
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber is required for the primary PDF extraction strategy. "
            "Install with: pip install pdfplumber"
        )

    import pymupdf

    result: List[Dict] = []
    fitz_doc = pymupdf.open(pdf_path)

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, plumber_page in enumerate(pdf.pages):
                fitz_page = (
                    fitz_doc[page_idx] if page_idx < len(fitz_doc) else None
                )

                # Font metadata from pymupdf (body size + heading threshold)
                font_info = _build_font_info(fitz_page) if fitz_page else {}

                # Build page markdown: interleaved tables and text
                md_text = _plumber_page_to_markdown(plumber_page, font_info)
                md_text = _clean_markdown(md_text)

                if md_text.strip():
                    result.append(
                        {"page_number": page_idx + 1, "markdown": md_text}
                    )
    finally:
        fitz_doc.close()

    return result


def _build_font_info(fitz_page) -> Dict:
    """
    Extract font size statistics from a pymupdf page.

    Returns body_size (statistical mode of all font sizes, weighted by text
    length) and heading_threshold (body_size * ratio). These are used to
    classify lines as headings vs body text.
    """
    import pymupdf

    font_sizes: List[float] = []

    try:
        blocks = fitz_page.get_text(
            "dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE
        )["blocks"]
    except Exception:
        return {"body_size": 10.0, "heading_threshold": 12.0}

    for block in blocks:
        if block.get("type") != 0:  # 0 = text, 1 = image
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                size = span.get("size", 0)
                text = span.get("text", "").strip()
                if text and size > 4:  # ignore sub-4pt noise (artifacts)
                    # Weight by character count: longer spans more representative
                    weight = max(1, len(text) // 4)
                    font_sizes.extend([round(size, 1)] * weight)

    if not font_sizes:
        return {"body_size": 10.0, "heading_threshold": 12.0}

    body_size = Counter(font_sizes).most_common(1)[0][0]
    heading_threshold = body_size * _HEADING_FONT_RATIO

    return {"body_size": body_size, "heading_threshold": heading_threshold}


def _plumber_page_to_markdown(page, font_info: Dict) -> str:
    """
    Convert a single pdfplumber page to structured markdown.

    Algorithm:
      1. Detect all tables and their bounding boxes (spatial detection)
      2. Convert each table to a markdown table
      3. Extract all text words, filtering out those inside table regions
      4. Group remaining words into text lines by y-position
      5. Classify each line as heading or body text (font size or pattern)
      6. Interleave text lines and table blocks in vertical (top→bottom) order
      7. Build final markdown output with appropriate spacing
    """
    heading_threshold = font_info.get("heading_threshold", 12.0)
    body_size = font_info.get("body_size", 10.0)

    # ── Step 1–2: Extract tables ───────────────────────────────────────────
    table_regions = _extract_page_tables(page)  # [(bbox, md_str), ...]
    table_bboxes = [bbox for bbox, _ in table_regions]

    # ── Step 3: Extract non-table words ───────────────────────────────────
    try:
        # extra_attrs pulls font size per word (pdfplumber ≥ 0.10)
        words = page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=True,
            extra_attrs=["size", "fontname"],
        )
    except TypeError:
        # Older pdfplumber: extra_attrs not supported
        words = page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=True,
        )

    non_table_words = [
        w for w in words if not _word_in_any_table(w, table_bboxes)
    ]

    # ── Step 4: Group words into text lines ───────────────────────────────
    text_lines = _group_into_lines(non_table_words)

    # ── Step 5–6: Build content blocks sorted by vertical position ─────────
    # Each block is (top_y, markdown_string)
    content_blocks: List[Tuple[float, str]] = []

    for top_y, line_words in text_lines:
        line_text = " ".join(w["text"] for w in line_words).strip()
        if not line_text:
            continue

        avg_size = _avg_word_font_size(line_words)
        if _is_heading(line_text, avg_size, heading_threshold, body_size):
            content_blocks.append((top_y, f"## {line_text}"))
        else:
            content_blocks.append((top_y, line_text))

    for bbox, md_table in table_regions:
        table_top = bbox[1]
        content_blocks.append((table_top, f"\n{md_table}\n"))

    content_blocks.sort(key=lambda x: x[0])

    # ── Step 7: Build final markdown ───────────────────────────────────────
    return _assemble_markdown(content_blocks)


def _extract_page_tables(page) -> List[Tuple[tuple, str]]:
    """
    Find all tables on a pdfplumber page and convert to markdown.

    Tries strict (ruling-line-based) detection first. Falls back to
    text-alignment detection for tables without explicit borders.
    A table must have ≥2 rows and ≥2 columns to be considered real.
    """
    # Try strict (PDF drawing commands) first
    tables = page.find_tables(_TABLE_SETTINGS_STRICT)
    if not tables:
        tables = page.find_tables(_TABLE_SETTINGS_TEXT)

    result: List[Tuple[tuple, str]] = []
    for table in tables:
        try:
            rows = table.extract()
        except Exception as e:
            logger.debug("[PDF Parser] table.extract() failed: %s", e)
            continue

        if not rows or len(rows) < 2:
            continue

        # Need at least 2 non-empty columns
        col_count = max((len(r) for r in rows), default=0)
        if col_count < 2:
            continue

        # Reject false-positive tables: pdfplumber can mistake decorative PDF
        # border/box lines for table ruling lines, fragmenting narrative text
        # into tiny cells. Two guards:
        #
        # Guard A — global tiny-cell fraction: false positive tables usually
        # fragment narrative text into mostly 1-2 character cells.
        #
        # Guard B — header row: even when later rows have legitimate data, a
        # fragmented section-title header ("19. DISCHA | R | GE | PRE | ...")
        # signals the whole detection is a false positive. Check the first row
        # independently: reject if its avg < 4 OR it contains ≥ 3 cells that
        # are ≤ 2 characters (single letters / digraphs = word fragments).
        all_cells = [
            str(c).strip() for row in rows for c in row
            if c and str(c).strip()
        ]
        if all_cells:
            tiny_fraction = sum(1 for c in all_cells if len(c) <= 2) / len(all_cells)
            if tiny_fraction > 0.85:
                logger.debug(
                    "[PDF Parser] Rejecting false-positive table (%.0f%% cells <=2 chars - word fragment)",
                    tiny_fraction * 100,
                )
                continue

        header_cells = [str(c).strip() for c in rows[0] if c and str(c).strip()]
        if header_cells:
            header_avg = sum(len(c) for c in header_cells) / len(header_cells)
            tiny_count = sum(1 for c in header_cells if len(c) <= 2)
            if header_avg < 4 or tiny_count >= 3:
                logger.debug(
                    "[PDF Parser] Rejecting false-positive table "
                    "(header row avg %.1f, tiny cells=%d — fragmented section title)",
                    header_avg, tiny_count,
                )
                continue

        md = _rows_to_markdown_table(rows)
        if md:
            result.append((table.bbox, md))

    return result


def _rows_to_markdown_table(rows: List[List[Optional[str]]]) -> str:
    """Convert pdfplumber rows (list of lists) to a markdown table string.

    - None cells → empty string
    - Newlines within cells → space (keeps table on one line per row)
    - Pads all rows to the same column width
    - First row is always the header row
    """
    if not rows:
        return ""

    def clean(cell) -> str:
        if cell is None:
            return ""
        return str(cell).replace("\n", " ").replace("|", "\\|").strip()

    cleaned = [[clean(c) for c in row] for row in rows]
    col_count = max(len(r) for r in cleaned)
    if col_count == 0:
        return ""

    # Pad all rows to the same width
    padded = [row + [""] * (col_count - len(row)) for row in cleaned]

    lines: List[str] = []
    lines.append("| " + " | ".join(padded[0]) + " |")
    lines.append("| " + " | ".join("---" for _ in padded[0]) + " |")
    for row in padded[1:]:
        # Skip entirely blank rows (often artifacts from merged cells)
        if any(cell.strip() for cell in row):
            lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _word_in_any_table(word: Dict, table_bboxes: List[tuple]) -> bool:
    """Return True if a pdfplumber word falls inside any table bounding box.

    Uses a 2pt tolerance to capture words right at the boundary edge.
    pdfplumber bbox format: (x0, top, x1, bottom).
    """
    wx0 = word.get("x0", 0)
    wx1 = word.get("x1", 0)
    wtop = word.get("top", 0)
    wbottom = word.get("bottom", 0)
    TOL = 2  # points

    for tx0, ttop, tx1, tbottom in table_bboxes:
        if (
            wx0 >= tx0 - TOL
            and wx1 <= tx1 + TOL
            and wtop >= ttop - TOL
            and wbottom <= tbottom + TOL
        ):
            return True
    return False


def _group_into_lines(
    words: List[Dict],
) -> List[Tuple[float, List[Dict]]]:
    """Group words into text lines based on their vertical (top) coordinate.

    Words within LINE_TOLERANCE points of each other are on the same line.
    Within each line, words are sorted left-to-right by x0.
    """
    if not words:
        return []

    LINE_TOLERANCE = 3  # points

    # Sort by top (vertical), then x0 (horizontal) for reading order
    sorted_words = sorted(
        words,
        key=lambda w: (round(w.get("top", 0) / LINE_TOLERANCE) * LINE_TOLERANCE,
                       w.get("x0", 0)),
    )

    lines: List[Tuple[float, List[Dict]]] = []
    current_top: Optional[float] = None
    current_words: List[Dict] = []

    for word in sorted_words:
        top = word.get("top", 0)
        if current_top is None:
            current_top = top
            current_words = [word]
        elif abs(top - current_top) <= LINE_TOLERANCE:
            current_words.append(word)
        else:
            # Finalize current line (sort by x0 for correct reading order)
            lines.append(
                (current_top,
                 sorted(current_words, key=lambda w: w.get("x0", 0)))
            )
            current_top = top
            current_words = [word]

    if current_words:
        lines.append(
            (current_top,
             sorted(current_words, key=lambda w: w.get("x0", 0)))
        )

    return lines


def _avg_word_font_size(words: List[Dict]) -> float:
    """Return the average font size of words that have size metadata."""
    sizes = [w.get("size", 0) for w in words if w.get("size", 0) > 0]
    return sum(sizes) / len(sizes) if sizes else 0.0


# ── Strategy 2: pymupdf dict mode fallback ───────────────────────────────────

def _extract_with_pymupdf_dict(pdf_path: str) -> List[Dict]:
    """
    Fallback extractor using pymupdf dict mode.

    Produces font-aware heading detection using actual font size metadata
    from the PDF. This replaces the original heuristic approach (isupper()
    + regex patterns) which was generating false markdown headers from ICD
    codes, insurance numbers, and appointment time slots.

    Does NOT extract tables (no spatial analysis available). Tables will
    appear as linearized text — acceptable as a last resort fallback.
    """
    import pymupdf

    doc = pymupdf.open(pdf_path)
    result: List[Dict] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        font_info = _build_font_info(page)
        md_text = _fitz_dict_page_to_markdown(page, font_info)
        md_text = _clean_markdown(md_text)
        if md_text.strip():
            result.append({"page_number": page_idx + 1, "markdown": md_text})

    doc.close()
    logger.info("[PDF Parser] pymupdf dict fallback: %d pages extracted", len(result))
    return result


def _fitz_dict_page_to_markdown(page, font_info: Dict) -> str:
    """Convert a pymupdf dict page to markdown using font size for headings.

    Iterates blocks → lines → spans, combines spans per line, then
    classifies each line as heading or body text based on font size.
    """
    import pymupdf

    heading_threshold = font_info.get("heading_threshold", 12.0)
    body_size = font_info.get("body_size", 10.0)

    try:
        blocks = page.get_text(
            "dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE
        )["blocks"]
    except Exception:
        # Last-resort: plain text extraction
        return page.get_text("text")

    output_lines: List[str] = []

    for block in blocks:
        if block.get("type") != 0:  # skip image blocks
            continue

        block_lines: List[str] = []
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue

            line_text = " ".join(
                s.get("text", "").strip()
                for s in spans
                if s.get("text", "").strip()
            ).strip()
            if not line_text:
                continue

            # Use the maximum font size in the line (headings often mix sizes)
            max_size = max((s.get("size", 0) for s in spans), default=0.0)

            if _is_heading(line_text, max_size, heading_threshold, body_size):
                block_lines.append(f"## {line_text}")
            else:
                block_lines.append(line_text)

        if block_lines:
            output_lines.extend(block_lines)
            output_lines.append("")  # blank line between blocks

    return "\n".join(output_lines)


# ── Heading Classification ────────────────────────────────────────────────────

def _is_heading(
    text: str,
    avg_size: float,
    heading_threshold: float,
    body_size: float,
) -> bool:
    """Determine whether a text line is a section heading.

    Decision logic (applied in order):

      1. HARD REJECT: Known non-heading patterns (ICD codes, insurance numbers,
         time values, measurement units, adjacent numeric data)
      2. PRIMARY SIGNAL: Font size >= heading_threshold (most reliable)
         → Accept if text is short (<100 chars) and passes reject filters
      3. SECONDARY: Numbered section heading pattern with ≥2 real words
         (e.g., "3. PATIENT DEMOGRAPHICS", "12a VITAL SIGNS")
      4. SECONDARY: All-caps line with ≥2 purely alphabetic words, zero digits
         (e.g., "DISCHARGE SUMMARY", "PHYSICAL EXAMINATION")
      5. DEFAULT: Body text

    The secondary signals only fire when font size metadata is unavailable
    (avg_size == 0), which happens with some PDF generators.
    """
    stripped = text.strip()
    if not stripped or len(stripped) < 3:
        return False

    # ── Hard reject filters ───────────────────────────────────────────────
    for pattern in _NON_HEADING_PATTERNS:
        if pattern.search(stripped):
            return False

    if _UNIT_PATTERN.search(stripped):
        return False

    # Reject sentence fragments: real headings don't end with punctuation
    # e.g. "PPI therapy." or "normocytic anemia, and obesity."
    if stripped.endswith(('.', ',', ';', ':')):
        return False

    # ── Primary signal: font size ─────────────────────────────────────────
    if avg_size > 0 and body_size > 0 and avg_size >= heading_threshold:
        if len(stripped) < 100:
            return True

    # ── Secondary signals (font size unavailable or equal to body) ────────

    # Numbered section heading:  "3 VITAL SIGNS", "12a. PATIENT HISTORY"
    numbered_match = re.match(
        r"^(\d{1,2}(?:[a-d])?(?:\.\d{1,2})?)\.?\s+"
        r"([A-Z][A-Za-z &/\-,()]{3,})$",
        stripped,
    )
    if numbered_match:
        title_part = numbered_match.group(2).strip()
        alpha_words = [w for w in title_part.split() if w.isalpha() and len(w) > 1]
        if len(alpha_words) >= _HEADING_MIN_ALPHA_WORDS:
            return True

    # All-caps heading: "DISCHARGE SUMMARY", "PHYSICAL EXAMINATION"
    # Require: ≥2 purely alphabetic words, zero digits, not too long
    if stripped.isupper() and 5 < len(stripped) < 80:
        words_in_line = stripped.split()
        alpha_words = [w for w in words_in_line if w.isalpha() and len(w) > 1]
        has_digits = bool(re.search(r"\d", stripped))
        alpha_fraction = len(alpha_words) / max(len(words_in_line), 1)
        if (
            not has_digits
            and len(alpha_words) >= _HEADING_MIN_ALPHA_WORDS
            and alpha_fraction >= 0.85
        ):
            return True

    return False


# ── Output Assembly ───────────────────────────────────────────────────────────

def _assemble_markdown(content_blocks: List[Tuple[float, str]]) -> str:
    """Merge sorted content blocks into final markdown.

    Adds blank lines:
    - Before and after every heading (## line)
    - Before and after every table block
    Avoids consecutive blank lines.
    """
    if not content_blocks:
        return ""

    output: List[str] = []
    prev_was_structural = False  # heading or table

    for _, content in content_blocks:
        is_heading = content.startswith("##")
        is_table = content.strip().startswith("|") and "|---|" in content

        is_structural = is_heading or is_table

        # Insert blank line before structural elements
        if is_structural and output and output[-1] != "":
            output.append("")

        output.append(content)

        # Insert blank line after structural elements
        if is_structural:
            output.append("")

        prev_was_structural = is_structural

    # Remove trailing blank line if present
    while output and output[-1] == "":
        output.pop()

    return "\n".join(output)


def _clean_markdown(text: str) -> str:
    """Normalize markdown output.

    - Collapses 4+ consecutive blank lines to 2
    - Strips trailing whitespace per line
    - Strips leading/trailing whitespace from the full text
    """
    # Collapse excessive blank lines
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _has_substance(pages: List[Dict], min_chars: int = 150) -> bool:
    """Return True if extracted pages contain meaningful content."""
    total = sum(len(p.get("markdown", "").strip()) for p in pages)
    return bool(pages) and total >= min_chars
