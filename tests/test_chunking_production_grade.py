"""
Production-grade chunking validation for medical document ingestion.

Focus areas:
  - Stable parser/chunker quality on the two target clinical PDFs
  - Semantic-quality guardrails (duplicates, heading noise, table fragmentation)
  - Multi-format ingestion robustness (PDF, DOCX, image OCR)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest


# Ensure src/ imports resolve when running from repository root.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from document_processing import process_document
from document_processing.format_handlers import SUPPORTED_EXTENSIONS
from document_processing.ocr_fallback import _tesseract_available
from document_processing.pdf_parser import parse_pdf_to_markdown


TARGET_PDFS = {
    "inpatient": ROOT / "data" / "unstructured_docs" / "clinical_notes" / "patient_inpatient_record.pdf",
    "patient1": ROOT / "data" / "unstructured_docs" / "clinical_notes" / "patient1_clinical_note.pdf",
}

# Calibrated from current parser/chunker outputs; thresholds are intentionally
# robust to small model/parser drift while still catching real regressions.
EXPECTED_PAGE_COUNTS = {
    "inpatient": 28,
    "patient1": 11,
}

CHUNK_BOUNDS = {
    "inpatient": (45, 80),
    "patient1": (18, 40),
}

_BAD_HEADING_RE = re.compile(r"^#{1,4}\s+[A-Za-z].*[\.!?,;:]$")
_FRAGMENTED_ROW_RE = re.compile(r"^\|\s*[^|]{0,3}(?:\|\s*[^|]{0,3}){2,}\|$")


def _chunk_body(text: str) -> str:
    """Remove chunk context prefix to inspect raw extracted content."""
    parts = (text or "").split("\n", 1)
    return parts[1] if len(parts) == 2 else (text or "")


def _quality_signals(documents):
    chunks = [d.page_content for d in documents]
    duplicate_count = len(chunks) - len(set(chunks))
    bad_heading_count = 0
    fragmented_table_row_count = 0

    for chunk in chunks:
        for raw in _chunk_body(chunk).splitlines():
            line = raw.strip()
            if _BAD_HEADING_RE.match(line):
                bad_heading_count += 1
            if _FRAGMENTED_ROW_RE.match(line):
                fragmented_table_row_count += 1

    return {
        "duplicate_count": duplicate_count,
        "bad_heading_count": bad_heading_count,
        "fragmented_table_row_count": fragmented_table_row_count,
    }


@pytest.mark.parametrize("doc_key", ["inpatient", "patient1"])
def test_target_pdfs_parse_full_page_count(doc_key: str):
    pdf_path = TARGET_PDFS[doc_key]
    assert pdf_path.exists(), f"Missing fixture PDF: {pdf_path}"

    pages = parse_pdf_to_markdown(str(pdf_path))
    assert len(pages) == EXPECTED_PAGE_COUNTS[doc_key]

    empty_like = [
        p["page_number"]
        for p in pages
        if len((p.get("markdown") or "").strip()) < 20
    ]
    assert not empty_like, f"Unexpected near-empty parsed pages: {empty_like}"


@pytest.mark.parametrize("doc_key", ["inpatient", "patient1"])
def test_target_pdfs_chunking_quality_guardrails(doc_key: str):
    pdf_path = TARGET_PDFS[doc_key]
    assert pdf_path.exists(), f"Missing fixture PDF: {pdf_path}"

    docs = process_document(
        file_path=str(pdf_path),
        chunk_size=1000,
        chunk_overlap=80,
    )
    assert docs, "process_document returned no chunks"

    lo, hi = CHUNK_BOUNDS[doc_key]
    assert lo <= len(docs) <= hi, f"Chunk count {len(docs)} outside expected range [{lo}, {hi}]"

    signals = _quality_signals(docs)
    assert signals["duplicate_count"] == 0
    assert signals["bad_heading_count"] <= 1
    assert signals["fragmented_table_row_count"] <= 2


def test_inpatient_pdf_semantic_distribution_signals_present():
    pdf_path = TARGET_PDFS["inpatient"]
    docs = process_document(
        file_path=str(pdf_path),
        chunk_size=1000,
        chunk_overlap=80,
    )
    corpus = "\n".join(d.page_content for d in docs).lower()

    # Distribution checks stay semantic and resilient (not exact-string brittle).
    assert corpus.count("discharge") >= 8
    assert corpus.count("summary") >= 3
    assert corpus.count("referral") >= 1
    assert corpus.count("allerg") >= 1


def test_supported_format_set_covers_pdf_docx_and_images():
    expected = {".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
    assert expected.issubset(SUPPORTED_EXTENSIONS)


def test_docx_medical_note_chunks_with_quality_metadata(tmp_path: Path):
    docx = pytest.importorskip("docx")
    document = docx.Document()

    document.add_heading("Discharge Summary", level=1)
    document.add_paragraph("Patient Name: Maria Lewis")
    document.add_paragraph("MRN: MRN-TEST-900001")
    document.add_paragraph(
        "History of Present Illness: 66-year-old female admitted with acute decompensated "
        "heart failure, dyspnea, orthopnea, and bilateral lower extremity edema. "
        "Improved with IV diuresis and guideline-directed medication optimization."
    )
    document.add_heading("Discharge Medications", level=2)
    document.add_paragraph("Sacubitril/Valsartan 24/26 mg BID; Carvedilol 12.5 mg BID; Spironolactone 25 mg daily.")
    document.add_paragraph("Follow-up: cardiology within 7 days; PCP within 2 weeks.")

    table = document.add_table(rows=2, cols=3)
    table.rows[0].cells[0].text = "Lab"
    table.rows[0].cells[1].text = "Value"
    table.rows[0].cells[2].text = "Date"
    table.rows[1].cells[0].text = "BNP"
    table.rows[1].cells[1].text = "1800"
    table.rows[1].cells[2].text = "2026-04-21"

    fpath = tmp_path / "medical_note.docx"
    document.save(fpath)

    docs = process_document(file_path=str(fpath), chunk_size=900, chunk_overlap=80)
    assert docs, "DOCX ingestion produced no chunks"

    combined = "\n".join(d.page_content.lower() for d in docs)
    assert "maria lewis" in combined
    assert "mrn-test-900001" in combined
    assert "sacubitril" in combined
    assert "cardiology" in combined


def test_image_medical_note_ocr_ingestion(tmp_path: Path):
    if not _tesseract_available():
        pytest.skip("Tesseract is not available; image OCR ingestion cannot be validated on this host.")

    pil = pytest.importorskip("PIL.Image")
    draw_mod = pytest.importorskip("PIL.ImageDraw")

    image = pil.new("RGB", (1800, 1200), color="white")
    draw = draw_mod.Draw(image)
    lines = [
        "Patient Name: Ethan Morris",
        "MRN: MRN-TEST-900777",
        "Assessment: Type 2 diabetes with hypertension and volume overload.",
        "Plan: Increase diuresis, continue metformin, monitor BMP daily.",
        "Allergies: Penicillin rash.",
        "Follow-up: Cardiology and primary care in one week.",
    ]
    y = 40
    for line in lines:
        draw.text((40, y), line, fill="black")
        y += 55

    image_path = tmp_path / "medical_note.png"
    image.save(image_path)

    docs = process_document(file_path=str(image_path), chunk_size=900, chunk_overlap=80)
    assert docs, "Image OCR ingestion produced no chunks"

    combined = "\n".join(d.page_content.lower() for d in docs)
    assert len(combined) > 120
    assert any(token in combined for token in ["mrn", "allerg", "cardio", "metformin", "patient"])
