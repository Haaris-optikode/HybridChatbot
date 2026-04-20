"""
ocr_fallback.py — OCR fallback for scanned PDFs and image-based documents.

When PyMuPDF text extraction yields low-quality results (empty or garbled text),
this module extracts text from rendered page images.

Three-tier extraction cascade:
  1. Native text extraction (handled by pdf_parser / format_handlers)
  2. Tesseract OCR (local, fast, free) — requires binary installed on host OS
  3. LLM Vision API (remote, high-quality) — last resort for:
       - Tesseract unavailable or failed
       - OCR confidence is poor
       - Complex tables/layouts, handwriting
       - Critical pages where extraction produced garbage

Requirements:
    - Pillow (pip install Pillow) — already in requirements.txt
    - For Tesseract: pytesseract + Tesseract binary on PATH
    - For Vision fallback: openai (already in requirements.txt) + OPENAI_API_KEY
"""

import base64
import io
import logging
import os
import shutil
import sys
from typing import List, Dict

logger = logging.getLogger(__name__)

# ── Auto-detect Tesseract binary path ─────────────────────────────────────────
# On Windows, winget / UB-Mannheim installs to Program Files but does NOT add
# it to PATH.  We probe common locations and configure pytesseract once at
# import time so every downstream call just works.

_TESSERACT_SEARCH_DIRS = [
    r"C:\Program Files\Tesseract-OCR",
    r"C:\Program Files (x86)\Tesseract-OCR",
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "Tesseract-OCR"),
]


def _configure_tesseract_path() -> None:
    """Set pytesseract.tesseract_cmd if the binary is found outside PATH."""
    # Already on PATH?
    if shutil.which("tesseract"):
        return
    for d in _TESSERACT_SEARCH_DIRS:
        candidate = os.path.join(d, "tesseract.exe") if sys.platform == "win32" else os.path.join(d, "tesseract")
        if os.path.isfile(candidate):
            try:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = candidate
                logger.info("[OCR] Tesseract binary auto-detected at %s", candidate)
            except ImportError:
                pass
            return


_configure_tesseract_path()

# Minimum average characters per page to consider native extraction usable.
MIN_CHARS_PER_PAGE = 50

# Vision model used when Tesseract is unavailable or produces poor results
_VISION_MODEL = "gpt-4.1-mini"

# Below this alpha-to-total ratio, OCR output is considered garbage / low-confidence
_MIN_OCR_ALPHA_RATIO = 0.25

# Below this many chars, a single-page OCR result is considered a failure
_MIN_OCR_PAGE_CHARS = 30


def _tesseract_available() -> bool:
    """Check whether Tesseract binary is reachable."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _ocr_quality_acceptable(text: str) -> bool:
    """Return True if OCR text looks reasonable (not garbled)."""
    text = text.strip()
    if len(text) < _MIN_OCR_PAGE_CHARS:
        return False
    alpha = sum(1 for c in text if c.isalpha())
    ratio = alpha / len(text) if text else 0
    return ratio >= _MIN_OCR_ALPHA_RATIO


def needs_ocr(pages: List[Dict], min_chars_per_page: int = MIN_CHARS_PER_PAGE) -> bool:
    """Determine whether extracted pages need OCR fallback.

    Args:
        pages: List of {"page_number": int, "markdown": str} from pdf_parser.
        min_chars_per_page: Threshold below which OCR is triggered.

    Returns:
        True if OCR is needed (native extraction produced insufficient text).
    """
    if not pages:
        return True

    total_chars = sum(len(p.get("markdown", "").strip()) for p in pages)
    avg_chars = total_chars / len(pages) if pages else 0

    if avg_chars < min_chars_per_page:
        logger.info(
            "[OCR] Native extraction yielded %.0f chars/page (threshold=%d) — OCR needed",
            avg_chars, min_chars_per_page,
        )
        return True
    return False


def ocr_pdf(pdf_path: str) -> List[Dict]:
    """Extract text from a PDF using Tesseract OCR, with LLM Vision fallback.

    Cascade: Tesseract → Vision LLM (only for failed/poor-quality pages).

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of {"page_number": int, "markdown": str} — same schema as pdf_parser.
    """
    try:
        import pymupdf
    except ImportError:
        logger.error("[OCR] PyMuPDF is required for page rendering")
        return []

    doc = pymupdf.open(pdf_path)
    result = []
    vision_needed_pages = []  # (page_idx, png_bytes) for pages needing vision

    use_tesseract = _tesseract_available()
    if use_tesseract:
        logger.info("[OCR] Running Tesseract OCR on %s", pdf_path)
    else:
        logger.info("[OCR] Tesseract not available — will use Vision LLM on %s", pdf_path)

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")

        text = ""
        if use_tesseract:
            text = _tesseract_extract_from_bytes(img_bytes, page_idx + 1)

        if _ocr_quality_acceptable(text):
            result.append({"page_number": page_idx + 1, "markdown": text.strip()})
        else:
            # Queue for vision fallback
            vision_needed_pages.append((page_idx + 1, img_bytes))

    doc.close()

    # Vision LLM fallback for pages that failed Tesseract or had poor quality
    if vision_needed_pages:
        logger.info(
            "[OCR] %d page(s) need Vision LLM fallback", len(vision_needed_pages),
        )
        for page_num, img_bytes in vision_needed_pages:
            vision_text = _vision_extract_from_bytes(img_bytes, page_num)
            if vision_text:
                result.append({"page_number": page_num, "markdown": vision_text})

        # Re-sort by page number
        result.sort(key=lambda p: p["page_number"])

    logger.info("[OCR] Extracted text from %d pages of %s", len(result), pdf_path)
    return result


def ocr_image(image_path: str) -> str:
    """Extract text from a standalone image file.

    Cascade: Tesseract → Vision LLM (only if Tesseract fails or produces garbage).

    Args:
        image_path: Path to an image file (PNG, JPG, TIFF, BMP).

    Returns:
        Extracted text string.
    """
    # Tier 2 — Tesseract
    if _tesseract_available():
        logger.info("[OCR] Running Tesseract on image: %s", image_path)
        try:
            from PIL import Image
            import pytesseract

            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, lang="eng").strip()
            if _ocr_quality_acceptable(text):
                return text
            logger.info("[OCR] Tesseract output poor quality (%d chars) — falling back to Vision LLM", len(text))
        except Exception as e:
            logger.warning("[OCR] Tesseract failed on image %s: %s", image_path, e)
    else:
        logger.info("[OCR] Tesseract not available for image: %s", image_path)

    # Tier 3 — Vision LLM
    logger.info("[OCR] Using Vision LLM fallback on image: %s", image_path)
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        return _vision_extract_from_bytes(img_bytes, page_num=1)
    except Exception as e:
        logger.error("[OCR] Vision LLM failed on image %s: %s", image_path, e)
        return ""


# ── Internal helpers ──────────────────────────────────────────────────────────


def _tesseract_extract_from_bytes(png_bytes: bytes, page_num: int) -> str:
    """Run Tesseract on raw PNG bytes.  Returns extracted text or ''."""
    try:
        from PIL import Image
        import pytesseract

        img = Image.open(io.BytesIO(png_bytes))
        text = pytesseract.image_to_string(img, lang="eng")
        return text.strip()
    except Exception as e:
        logger.warning("[OCR] Tesseract failed on page %d: %s", page_num, e)
        return ""


def _vision_extract_from_bytes(png_bytes: bytes, page_num: int) -> str:
    """Send a PNG image to OpenAI Vision API for text extraction."""
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("[OCR-Vision] openai package not installed")
        return ""

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("[OCR-Vision] OPENAI_API_KEY not set — cannot use Vision fallback")
        return ""

    b64 = base64.b64encode(png_bytes).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64}"

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=_VISION_MODEL,
            max_tokens=4096,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a clinical document OCR assistant. "
                        "Extract ALL text from the image exactly as written. "
                        "Preserve the original structure: headings, lists, tables (as markdown), "
                        "and paragraph breaks. Do NOT summarise or interpret — output the raw text only."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri, "detail": "high"},
                        },
                        {
                            "type": "text",
                            "text": "Extract all text from this clinical document image.",
                        },
                    ],
                },
            ],
        )
        text = response.choices[0].message.content.strip()
        logger.info("[OCR-Vision] Extracted %d chars from page %d", len(text), page_num)
        return text
    except Exception as e:
        logger.error("[OCR-Vision] API call failed for page %d: %s", page_num, e)
        return ""
