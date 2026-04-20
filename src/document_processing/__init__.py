"""
document_processing — Adaptive, format-agnostic document processing pipeline.

Supports multiple document formats with quality validation and deduplication:
  PDF  → pymupdf4llm → pymupdf fallback → OCR fallback
  DOCX → python-docx
  Images → Tesseract OCR

Public API:
    process_document(file_path, chunk_size, chunk_overlap, document_id) -> list[Document]
    compute_content_hash(text) -> str
    validate_document_text(text, source_path, page_count) -> (bool, str)

Usage:
    from document_processing import process_document

    docs = process_document("path/to/file.pdf")
    # Returns list of LangChain Document objects ready for Qdrant indexing
"""

import hashlib
import os
import uuid
import logging
from datetime import datetime
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from document_processing.format_handlers import extract_text
from document_processing.quality_gate import validate_extracted_text
from document_processing.metadata_extractor import (
    extract_mrn,
    extract_patient_name,
    classify_document_type,
    extract_encounter_date,
    extract_doc_level_diagnoses_and_meds,
    extract_hba1c_from_chunk_text,
)
from document_processing.adaptive_chunker import AdaptiveDocumentChunker

logger = logging.getLogger(__name__)

PAYLOAD_SCHEMA_VERSION = "clinical_values_v1"


def compute_content_hash(text: str) -> str:
    """Compute SHA-256 hash of normalized document text for deduplication.

    Normalization: lowercased, whitespace-collapsed, stripped.
    Two documents with the same clinical content but different formatting
    will produce the same hash.
    """
    import re
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validate_document_text(
    text: str,
    source_path: str,
    page_count: int = 0,
) -> Tuple[bool, str]:
    """Public wrapper for the quality gate validation.

    Returns:
        (is_valid, reason) — True if text passes quality checks.
    """
    return validate_extracted_text(text, source_path, page_count)


def _normalize_date_to_iso(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip()
    if not s:
        return None
    s_norm = s.replace("-", "/")
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            dt = datetime.strptime(s_norm, fmt)
            return dt.isoformat()
        except Exception:
            continue
    return None


def process_document(
    pdf_path: str = "",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    document_id: Optional[str] = None,
    min_chunk_size: int = 150,
    *,
    file_path: Optional[str] = None,
) -> List[Document]:
    """Process a document into chunked Documents with rich metadata.

    Supports PDF, DOCX, and image files. Includes OCR fallback for scanned
    PDFs, quality gate validation, and content hashing for deduplication.

    Full pipeline:
      1. Format-aware text extraction (PDF/DOCX/image with OCR fallback)
      2. Quality gate — rejects empty, corrupted, or garbled documents
      3. Extract metadata (MRN, patient name, document type, encounter date)
      4. Adaptive element-aware chunking (tables kept whole, SOAP per-day, etc.)
      5. Context prefixing, metadata attachment, and content hash

    Args:
        pdf_path: Path to the document file (kept for backward compatibility).
        chunk_size: Target max characters per chunk (default 1000).
        chunk_overlap: Character overlap for narrative text splits (default 200).
        document_id: Optional unique ID for this document. Auto-generated if None.
        min_chunk_size: Minimum chunk size before merging with neighbors (default 150).
        file_path: Path to the document file. Takes precedence over pdf_path.

    Returns:
        List of LangChain Document objects ready for vector DB indexing.
        Empty list if the document fails quality validation.
    """
    # Support both old (pdf_path) and new (file_path) parameter names
    actual_path = file_path or pdf_path
    if not actual_path:
        raise ValueError("Either file_path or pdf_path must be provided")

    source_file = os.path.basename(actual_path)
    doc_id = document_id or str(uuid.uuid4())
    logger.info("[DocProcessor] Processing: %s (id=%s)", source_file, doc_id[:8])

    # Step 1: Format-aware text extraction (with OCR fallback for PDFs)
    pages = extract_text(actual_path)
    if not pages:
        logger.warning("[DocProcessor] No content extracted from %s", source_file)
        return []

    # Build full markdown text and page offset map
    full_markdown = ""
    page_map = []  # (char_start, char_end, page_number)
    for page_data in pages:
        start = len(full_markdown)
        if full_markdown:
            full_markdown += "\n\n"
            start = len(full_markdown)
        full_markdown += page_data["markdown"]
        end = len(full_markdown)
        page_map.append((start, end, page_data["page_number"]))

    # Step 1.5: Quality gate — reject documents with unusable text
    is_valid, reason = validate_extracted_text(
        full_markdown, actual_path, page_count=len(pages)
    )
    if not is_valid:
        logger.warning(
            "[DocProcessor] Document rejected by quality gate: %s (reason=%s)",
            source_file, reason,
        )
        return []

    # Compute content hash for deduplication
    content_hash = compute_content_hash(full_markdown)

    # Step 2: Extract metadata
    patient_mrn = extract_mrn(full_markdown)
    patient_name = extract_patient_name(full_markdown)
    doc_type = classify_document_type(full_markdown)
    encounter_date = extract_encounter_date(full_markdown)

    # Document-scoped clinical tags used for cohort filtering.
    # (Chunk-scoped lab numeric values are extracted per-chunk below.)
    doc_level_values = extract_doc_level_diagnoses_and_meds(full_markdown) or {}

    logger.info("  MRN      : %s", patient_mrn or "not found")
    logger.info("  Patient  : %s", patient_name or "not found")
    logger.info("  Doc Type : %s", doc_type)
    logger.info("  Date     : %s", encounter_date or "not found")
    logger.info("  Hash     : %s", content_hash[:16])

    # Step 3: Adaptive chunking
    chunker = AdaptiveDocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
    )

    base_metadata = {
        "source": source_file,
        "filename": source_file,
        "patient_mrn": patient_mrn,
        "patient_name": patient_name,
        "document_type": doc_type,
        "document_id": doc_id,
        "content_hash": content_hash,
    }
    if encounter_date:
        iso = _normalize_date_to_iso(encounter_date)
        if iso:
            base_metadata["encounter_date"] = iso
        else:
            base_metadata["encounter_date"] = encounter_date
    # Propagate doc-level tags into every chunk so cohort filters can match
    # without re-extracting for each chunk.
    base_metadata.update(doc_level_values)
    base_metadata["payload_schema_version"] = PAYLOAD_SCHEMA_VERSION

    documents = chunker.chunk_document(
        markdown_text=full_markdown,
        metadata=base_metadata,
        page_map=page_map,
    )

    # Chunk-scoped numeric lab extraction (HbA1c + nearest date).
    # IMPORTANT: we do NOT propagate lab values to all chunks; we only store
    # them on the chunk(s) where the evidence appears.
    for d in documents:
        # Chunker prepends a prefix like "[Patient: ... | MRN: ... | Section: ...]\n"
        # Strip it so lab regexes run on the actual clinical content.
        page_content = d.page_content or ""
        parts = page_content.split("\n", 1)
        chunk_text = parts[1] if len(parts) == 2 else page_content
        hba1c_payload = extract_hba1c_from_chunk_text(chunk_text)
        if hba1c_payload:
            d.metadata.update(hba1c_payload)

    logger.info("  Chunks   : %d", len(documents))

    # Log a sample chunk for verification
    if documents:
        sample = documents[len(documents) // 2]
        logger.info("  [Sample Chunk #%d] Section: %s | Page: %s",
                    sample.metadata.get("chunk_index", 0),
                    sample.metadata.get("section_title", "?"),
                    sample.metadata.get("page_number", "?"))
        logger.info("  Content preview: %s...", sample.page_content[:150])

    return documents
