"""
quality_gate.py — Text quality validation for the document ingestion pipeline.

Prevents indexing of documents that produce unusable text: corrupted PDFs,
password-protected files, image-only documents where OCR also fails, or
documents with garbled/binary content.

Every document must pass this gate before chunking and indexing.
"""

import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────

# Minimum total characters for the entire document
MIN_TOTAL_CHARS = 100

# Minimum ratio of alphabetic characters to total characters.
# Normal English text is ~80%. Below 30% usually means binary garbage or
# OCR artifacts on blank pages.
MIN_ALPHA_RATIO = 0.30

# Maximum ratio of non-printable / control characters (excluding newlines/tabs).
# Normal text should be near 0. Garbled PDFs often have high control char ratios.
MAX_CONTROL_CHAR_RATIO = 0.05

# Minimum average characters per page (if page count is known).
MIN_CHARS_PER_PAGE_QUALITY = 20


def validate_extracted_text(
    text: str,
    source_path: str,
    page_count: int = 0,
) -> Tuple[bool, str]:
    """Validate extracted text quality before indexing.

    Args:
        text: The full extracted text from the document.
        source_path: Path to the source file (for logging).
        page_count: Number of pages in the document (0 if unknown).

    Returns:
        (is_valid, reason) — True if text passes quality checks, with reason string.
    """
    if not text or not text.strip():
        logger.warning("[QualityGate] REJECTED %s: empty text", source_path)
        return False, "empty_text"

    stripped = text.strip()
    total_chars = len(stripped)

    # Check 1: Minimum total length
    if total_chars < MIN_TOTAL_CHARS:
        logger.warning(
            "[QualityGate] REJECTED %s: too short (%d chars, min=%d)",
            source_path, total_chars, MIN_TOTAL_CHARS,
        )
        return False, f"too_short ({total_chars} chars)"

    # Check 2: Alphabetic ratio — filters binary/garbled content
    alpha_count = sum(1 for c in stripped if c.isalpha())
    alpha_ratio = alpha_count / total_chars
    if alpha_ratio < MIN_ALPHA_RATIO:
        logger.warning(
            "[QualityGate] REJECTED %s: low alpha ratio (%.1f%%, min=%.0f%%)",
            source_path, alpha_ratio * 100, MIN_ALPHA_RATIO * 100,
        )
        return False, f"low_alpha_ratio ({alpha_ratio:.1%})"

    # Check 3: Control character ratio — filters corrupted content
    # Allow newlines (\n), carriage returns (\r), and tabs (\t)
    control_count = sum(
        1 for c in stripped
        if ord(c) < 32 and c not in ('\n', '\r', '\t')
    )
    if total_chars > 0:
        control_ratio = control_count / total_chars
        if control_ratio > MAX_CONTROL_CHAR_RATIO:
            logger.warning(
                "[QualityGate] REJECTED %s: high control char ratio (%.1f%%, max=%.0f%%)",
                source_path, control_ratio * 100, MAX_CONTROL_CHAR_RATIO * 100,
            )
            return False, f"high_control_chars ({control_ratio:.1%})"

    # Check 4: Per-page density (if page count is known)
    if page_count > 0:
        chars_per_page = total_chars / page_count
        if chars_per_page < MIN_CHARS_PER_PAGE_QUALITY:
            logger.warning(
                "[QualityGate] REJECTED %s: low text density (%.0f chars/page, min=%d)",
                source_path, chars_per_page, MIN_CHARS_PER_PAGE_QUALITY,
            )
            return False, f"low_density ({chars_per_page:.0f} chars/page)"

    # Check 5: Repetition detection — OCR artifacts sometimes produce repeated
    # characters or words (e.g., "aaaaaaa" or "the the the the")
    # Look for any single character repeated 20+ times consecutively
    if re.search(r'(.)\1{19,}', stripped):
        logger.warning(
            "[QualityGate] REJECTED %s: excessive character repetition detected",
            source_path,
        )
        return False, "excessive_repetition"

    logger.debug(
        "[QualityGate] PASSED %s: %d chars, %.0f%% alpha",
        source_path, total_chars, alpha_ratio * 100,
    )
    return True, "ok"
