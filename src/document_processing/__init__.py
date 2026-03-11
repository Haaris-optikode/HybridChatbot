"""
document_processing — Adaptive, format-agnostic document processing pipeline.

Replaces the rigid clinical-note-specific chunking with a universal pipeline:
  PDF → Structured Markdown (pymupdf4llm) → Element-aware chunking

Public API:
    process_document(pdf_path, chunk_size, chunk_overlap, document_id) -> list[Document]

Usage:
    from document_processing import process_document

    docs = process_document("path/to/file.pdf")
    # Returns list of LangChain Document objects ready for Qdrant indexing
"""

import os
import uuid
import logging
from typing import List, Optional

from langchain_core.documents import Document

from document_processing.pdf_parser import parse_pdf_to_markdown
from document_processing.metadata_extractor import (
    extract_mrn,
    extract_patient_name,
    classify_document_type,
    extract_encounter_date,
)
from document_processing.adaptive_chunker import AdaptiveDocumentChunker

logger = logging.getLogger(__name__)


def process_document(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    document_id: Optional[str] = None,
    min_chunk_size: int = 150,
) -> List[Document]:
    """Process a PDF into chunked Documents with rich metadata.

    Full pipeline:
      1. PDF → structured markdown (preserves headers, tables, lists)
      2. Extract metadata (MRN, patient name, document type, encounter date)
      3. Adaptive element-aware chunking (tables kept whole, SOAP per-day, etc.)
      4. Context prefixing and metadata attachment

    Args:
        pdf_path: Full path to the PDF file.
        chunk_size: Target max characters per chunk (default 1000).
        chunk_overlap: Character overlap for narrative text splits (default 200).
        document_id: Optional unique ID for this document. Auto-generated if None.
        min_chunk_size: Minimum chunk size before merging with neighbors (default 150).

    Returns:
        List of LangChain Document objects ready for vector DB indexing.
    """
    pdf_file = os.path.basename(pdf_path)
    doc_id = document_id or str(uuid.uuid4())
    logger.info("[DocProcessor] Processing: %s (id=%s)", pdf_file, doc_id[:8])

    # Step 1: PDF → Structured Markdown
    pages = parse_pdf_to_markdown(pdf_path)
    if not pages:
        logger.warning("[DocProcessor] No content extracted from %s", pdf_file)
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

    # Step 2: Extract metadata
    patient_mrn = extract_mrn(full_markdown)
    patient_name = extract_patient_name(full_markdown)
    doc_type = classify_document_type(full_markdown)
    encounter_date = extract_encounter_date(full_markdown)

    logger.info("  MRN      : %s", patient_mrn or "not found")
    logger.info("  Patient  : %s", patient_name or "not found")
    logger.info("  Doc Type : %s", doc_type)
    logger.info("  Date     : %s", encounter_date or "not found")

    # Step 3: Adaptive chunking
    chunker = AdaptiveDocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
    )

    base_metadata = {
        "source": pdf_file,
        "filename": pdf_file,
        "patient_mrn": patient_mrn,
        "patient_name": patient_name,
        "document_type": doc_type,
        "document_id": doc_id,
    }
    if encounter_date:
        base_metadata["encounter_date"] = encounter_date

    documents = chunker.chunk_document(
        markdown_text=full_markdown,
        metadata=base_metadata,
        page_map=page_map,
    )

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
