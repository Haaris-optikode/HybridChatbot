#!/usr/bin/env python3
"""
prepare_vector_db.py — Standalone, production-grade clinical document vectorization.

This module is fully decoupled from the chatbot/agent graph and can be reused
independently for any PDF-to-vector pipeline.

Architecture decisions:
  1. EMBEDDING: NeuML/pubmedbert-base-embeddings (768-dim, local CPU, HIPAA-safe)
     - Fine-tuned from Microsoft PubMedBERT on PubMed title-abstract pairs
     - Outperforms all general-purpose models on medical text (Pearson 95.62)
     - Runs on CPU (~50 sent/s on i5), no GPU required
     - Data never leaves your server — full HIPAA compliance

  2. CHUNKING: Structure-aware hybrid chunking (content-zone adaptive)
     - Two-pass heading detection: regex candidate matching + multi-heuristic
       validation (rejects CPT/LOINC/SNOMED/RxNorm codes, table data rows,
       ICD-10 lines, medication dosages, and long data lines)
     - Content-zone strategies:
         • SOAP daily notes → split on "Day N —" boundaries (one chunk per day)
         • Short sections (≤ 1.5× chunk_size) → kept whole (tables stay intact)
         • Long sections → RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
     - Tiny fragments (< 100 chars) are merged with neighbors
     - Each chunk prepended with "[Patient: X | MRN: Y | Section: Z]" for
       contextual grounding in both embeddings and LLM retrieval

  3. INDEXING: Qdrant embedded mode (in-process, on-disk, no Docker needed)
     - Cosine distance for normalized sentence-transformer embeddings
     - HNSW m=16, ef_construct=128 (optimal for <100K vectors)
     - Payload indexes created BEFORE insert (builds extra HNSW edges)
     - INT8 scalar quantization with always_ram=True for fast retrieval
     - Can also connect to a Qdrant server via URL if configured

  4. METADATA: Rich payload for filtered search
     - patient_mrn, patient_name, section_title, page_number, source
     - Enables Qdrant payload-filtered search (e.g., filter by patient MRN)

Usage:
    # CLI (reads config from tools_config.yml):
    python src/prepare_vector_db.py

    # Programmatic — embedded mode (no Docker, no server):
    from prepare_vector_db import ClinicalNotesVectorizer
    v = ClinicalNotesVectorizer(qdrant_path="./qdrant_storage")
    v.vectorize_folder("path/to/pdfs", collection_name="PatientData")
    v.vectorize_single_pdf("path/to/file.pdf", collection_name="PatientData", append=True)

    # Programmatic — server mode (Docker / cloud):
    v = ClinicalNotesVectorizer(qdrant_url="http://localhost:6333")
    v.vectorize_folder("path/to/pdfs")
"""

import os
import re
import sys
from typing import List, Dict, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PayloadSchemaType,
    ScalarQuantizationConfig, ScalarType, ScalarQuantization,
    OptimizersConfigDiff, HnswConfigDiff,
)


# ──────────────────────────────────────────────────────────────────────────────
# Section Detection — Structure-Aware Clinical Heading Parser
# ──────────────────────────────────────────────────────────────────────────────

# Matches clinical document section headings in two common formats:
#   "N TITLE"   → "1 Patient Demographics", "6.1.2 Subjective"
#   "N. TITLE"  → "1. PATIENT DEMOGRAPHICS", "18a. Laboratory Orders"
#
# The first numeric component is limited to 1–2 digits (\d{1,2}) which
# automatically excludes CPT codes (85025), LOINC (6690-2), SNOMED (91936005),
# and RxNorm (314076) codes that appear at the start of table data rows.
# An optional single lowercase letter [a-d] handles sub-section labels (18a, 18b).
HEADING_RE = re.compile(
    r"^(\d{1,2}(?:[a-d])?(?:\.\d{1,2})*)\.?\s+(.+)$",
    re.MULTILINE,
)

# SOAP daily progress note boundary: "Day 1 — ...", "Day 30 — ..."
SOAP_DAY_RE = re.compile(r"^(Day\s+\d+\s*[—\-–].*)$", re.MULTILINE)

# Clinical measurement / medication units — indicate table data, not headings
_DATA_UNIT_RE = re.compile(
    r"\b(?:mg|mL|mmHg|tablet|capsule|injection|oral|daily|once|twice|"
    r"BID|TID|QID|mcg|mEq|units?|puffs?|inject|inhale[ds]?|"
    r"kg/m|solution|spray|subcutaneous|intravenous|intramuscular)\b",
    re.IGNORECASE,
)

# ICD-10 code at the very start of title text (e.g., I50.23, E11.9, N17.9)
_ICD10_START_RE = re.compile(r"^[A-Z]\d+\.\d+")


def _is_valid_section_heading(title_text: str) -> bool:
    """Multi-heuristic validation to distinguish real section headings from
    table data rows, prescription items, procedure descriptions, etc.

    Rejects:
      - Lines that don't start with an uppercase letter
      - Overly long lines (>80 chars — clinical headings are concise)
      - Lines without at least 2 alphabetic words
      - Lines that are predominantly numeric (codes, measurements)
      - Lines starting with an ICD-10 code (e.g., "I50.23 Acute on chronic…")
      - Lines containing clinical measurement/medication units
        (mg, mL, mmHg, tablet, BID, etc.)
    """
    title = title_text.strip()
    if not title:
        return False

    # Must start with an uppercase letter (rejects dates "01/15…", numbers "158 94…")
    if not title[0].isupper():
        return False

    # Clinical section headings are concise — rarely exceed 80 characters
    if len(title) > 80:
        return False

    # Must contain at least 2 alphabetic words (2+ letters each)
    alpha_words = re.findall(r"[A-Za-z]{2,}", title)
    if len(alpha_words) < 2:
        return False

    # Title should be predominantly text (≥60% alpha + space)
    alpha_space = sum(1 for c in title if c.isalpha() or c.isspace())
    if alpha_space / max(len(title), 1) < 0.6:
        return False

    # Reject if starts with ICD-10 code (e.g., "I50.23 Acute on chronic…")
    if _ICD10_START_RE.match(title):
        return False

    # Reject if contains clinical measurement / medication units
    if _DATA_UNIT_RE.search(title):
        return False

    return True


def detect_sections(text: str) -> List[Dict[str, str]]:
    """Split clinical document text into sections using validated numbered headings.

    Two-pass algorithm:
      1. Find ALL regex candidate matches (generous pattern)
      2. Validate each candidate with _is_valid_section_heading()

    Handles both formats:
      "N TITLE"  (outpatient notes)  and  "N. TITLE" (inpatient records)

    Captures preamble text (hospital header / patient summary) that appears
    before the first numbered section.

    Returns:
        List of dicts with 'title' (str) and 'body' (str) keys.
        Body includes the heading line itself for full context.
    """
    # Pass 1: regex candidates  →  Pass 2: heuristic validation
    validated = [m for m in HEADING_RE.finditer(text)
                 if _is_valid_section_heading(m.group(2).strip())]

    if not validated:
        return [{"title": "Full Document", "body": text}]

    sections: List[Dict[str, str]] = []

    # Capture preamble (hospital header, patient summary before first section)
    preamble_end = validated[0].start()
    if preamble_end > 30:
        preamble = text[:preamble_end].strip()
        if preamble:
            sections.append({"title": "Document Header", "body": preamble})

    # Build sections from validated heading matches
    for i, m in enumerate(validated):
        start = m.start()
        end = validated[i + 1].start() if i + 1 < len(validated) else len(text)
        num = m.group(1)
        title = m.group(2).strip()
        body = text[start:end].strip()
        sections.append({"title": f"{num} {title}", "body": body})

    return sections


def _split_soap_daily_notes(body: str) -> Optional[List[Dict[str, str]]]:
    """Split a SOAP daily progress notes section into individual day entries.

    Detects "Day N —" boundaries and creates one entry per hospital day.
    Returns None if fewer than 2 day markers are found (not a daily notes section).
    """
    day_matches = list(SOAP_DAY_RE.finditer(body))
    if len(day_matches) < 2:
        return None

    entries: List[Dict[str, str]] = []

    # Capture intro text before Day 1 (section heading + description)
    if day_matches[0].start() > 30:
        intro = body[: day_matches[0].start()].strip()
        if intro:
            entries.append({"sublabel": "Introduction", "content": intro})

    # Each day: from "Day N —" to the next "Day N+1 —"
    for i, m in enumerate(day_matches):
        start = m.start()
        end = day_matches[i + 1].start() if i + 1 < len(day_matches) else len(body)
        day_text = body[start:end].strip()
        day_label = m.group(1).strip()
        entries.append({"sublabel": day_label, "content": day_text})

    return entries


def extract_mrn(text: str) -> str:
    """Extract MRN (Medical Record Number) from document text."""
    pattern = r"\b(?:mrn|medical\s+record\s+number)\s*(?:is\s*|:\s*)?([A-Z0-9\-]+)"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def extract_patient_name(text: str) -> str:
    """Extract patient name from document text.

    Handles common clinical document formats:
      - "Patient Name: Name"  /  "Full Name: Name"
      - "Patient: Name"  (inpatient header)
    """
    patterns = [
        r"(?:patient\s+name|full\s+name)\s*:\s*(.+?)(?:\n|$)",
        r"(?:^|\n)\s*patient\s*:\s*([A-Za-z][A-Za-z .\-,]+?)(?:\n|$)",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            if name and len(name) > 2 and name[0].isalpha():
                return name
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Standalone Vectorizer Class
# ──────────────────────────────────────────────────────────────────────────────

class ClinicalNotesVectorizer:
    """
    Standalone, reusable vectorizer for clinical PDF documents.

    Stores vectors using Qdrant in embedded mode (in-process, on-disk)
    with NeuML/pubmedbert-base-embeddings (768-dim, runs on CPU).
    No Docker or external server needed for development.

    This class has ZERO coupling with the chatbot or agent graph.
    It can be imported and used in any Python project.
    """

    def __init__(
        self,
        embedding_model: str = "NeuML/pubmedbert-base-embeddings",
        embedding_dimensions: int = 768,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        qdrant_url: Optional[str] = None,
        qdrant_path: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        batch_size: int = 20,
    ):
        """
        Args:
            embedding_model: HuggingFace model name for embeddings.
            embedding_dimensions: Vector dimensionality (768 for PubMedBERT).
            chunk_size: Max characters per chunk.
            chunk_overlap: Character overlap between consecutive chunks.
            qdrant_url: URL of a Qdrant server (Docker/cloud). If empty, uses embedded mode.
            qdrant_path: Local disk path for embedded Qdrant storage (no server needed).
            qdrant_api_key: Optional API key for Qdrant authentication.
            batch_size: Number of documents per upload batch.
        """
        self.embedding_model_name = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.qdrant_url = qdrant_url if qdrant_url else None
        self.qdrant_path = qdrant_path
        self.qdrant_api_key = qdrant_api_key
        self.batch_size = batch_size

        # Initialize embedding model (local CPU — no data leaves the server)
        print(f"\n[Embeddings] Loading {embedding_model} (local CPU)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        )
        print(f"  Model loaded: {embedding_model} ({embedding_dimensions}-dim)")

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    # ── PDF Processing ────────────────────────────────────────────────────

    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Process a single PDF into chunked Documents with rich metadata.

        Content-aware chunking strategy (applied per section):
          1. SOAP daily progress notes → split on "Day N —" boundaries
          2. Short sections (≤ 1.5× chunk_size) → kept whole (tables intact)
          3. Long narrative sections → RecursiveCharacterTextSplitter
          4. Tiny fragments (< 100 chars) merged with neighbors
          5. Every chunk prepended with "[Patient | MRN | Section]" context

        Args:
            pdf_path: Full path to the PDF file.

        Returns:
            List of LangChain Document objects ready for indexing.
        """
        pdf_file = os.path.basename(pdf_path)
        print(f"\n[Processing] {pdf_file}")

        # Load pages and build full text
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        full_text = "\n\n".join(page.page_content for page in pages)

        # Build char-offset → page-number mapping
        page_offsets = []
        offset = 0
        for i, page in enumerate(pages):
            page_offsets.append((offset, offset + len(page.page_content), i + 1))
            offset += len(page.page_content) + 2  # +2 for "\n\n" join

        def get_page_number(char_pos: int) -> int:
            for start, end, page_num in page_offsets:
                if start <= char_pos < end:
                    return page_num
            return page_offsets[-1][2] if page_offsets else 1

        # Extract document-level metadata
        patient_mrn = extract_mrn(full_text)
        patient_name = extract_patient_name(full_text)
        print(f"  MRN     : {patient_mrn or 'not found'}")
        print(f"  Patient : {patient_name or 'not found'}")

        # Detect sections using validated heading parser
        sections = detect_sections(full_text)
        print(f"  Sections: {len(sections)}")

        # Build context prefix parts (only include non-empty fields)
        ctx_parts = []
        if patient_name:
            ctx_parts.append(f"Patient: {patient_name}")
        if patient_mrn:
            ctx_parts.append(f"MRN: {patient_mrn}")

        # Threshold: sections at or below this size are kept whole
        # (preserves tables, allergy lists, medication lists, etc.)
        whole_section_limit = int(self.chunk_size * 1.5)

        # ── Content-aware chunking per section ──
        documents = []
        chunk_idx = 0

        # Keywords that identify SOAP daily progress note sections
        _SOAP_KEYWORDS = ("DAILY PROGRESS", "SOAP NOTES", "PROGRESS NOTES")

        for section in sections:
            section_title = section["title"]
            section_body = section["body"]

            # Full context prefix for this section
            section_ctx = "[" + " | ".join(ctx_parts + [f"Section: {section_title}"]) + "]"

            # Collect (chunk_text, optional_sublabel) pairs
            chunks_with_labels: List[tuple] = []

            # ── Strategy 1: SOAP daily notes — split by Day boundaries
            is_soap = any(kw in section_title.upper() for kw in _SOAP_KEYWORDS)
            if is_soap:
                day_entries = _split_soap_daily_notes(section_body)
                if day_entries:
                    for entry in day_entries:
                        txt = entry["content"]
                        sub = entry["sublabel"]
                        if len(txt) <= self.chunk_size * 2:
                            chunks_with_labels.append((txt, sub))
                        else:
                            # Rare: a single day's note exceeds 2× chunk_size
                            parts = self.text_splitter.split_text(txt)
                            parts = self._merge_small_chunks(parts)
                            for j, p in enumerate(parts):
                                lbl = f"{sub} (part {j+1})" if len(parts) > 1 else sub
                                chunks_with_labels.append((p, lbl))

            # ── Strategy 2 (fallback): keep short sections whole
            if not chunks_with_labels:
                if len(section_body) <= whole_section_limit:
                    chunks_with_labels.append((section_body, None))
                else:
                    # ── Strategy 3: recursive text splitting for long content
                    raw_chunks = self.text_splitter.split_text(section_body)
                    raw_chunks = self._merge_small_chunks(raw_chunks)
                    for rc in raw_chunks:
                        chunks_with_labels.append((rc, None))

            # ── Create Document objects with rich metadata
            for chunk_text, sub_label in chunks_with_labels:
                # Build contextualized page_content
                if sub_label:
                    prefix = f"{section_ctx}\n[{sub_label}]"
                else:
                    prefix = section_ctx
                contextualized = f"{prefix}\n{chunk_text}"

                # Approximate page number from char offset in full_text
                body_pos = full_text.find(chunk_text[:80])
                page_num = get_page_number(body_pos) if body_pos >= 0 else 1

                doc = Document(
                    page_content=contextualized,
                    metadata={
                        "source": pdf_file,
                        "filename": pdf_file,
                        "patient_mrn": patient_mrn,
                        "patient_name": patient_name,
                        "section_title": section_title,
                        "page_number": page_num,
                        "chunk_index": chunk_idx,
                    },
                )
                documents.append(doc)
                chunk_idx += 1

        print(f"  Chunks  : {chunk_idx}")
        return documents

    # ── Small-chunk merging ───────────────────────────────────────────────

    @staticmethod
    def _merge_small_chunks(chunks: List[str], min_size: int = 100) -> List[str]:
        """Merge chunks smaller than *min_size* characters with their neighbors.

        Prevents tiny, context-free fragments from becoming standalone chunks
        (e.g., a trailing table row or a partial sentence from splitting).
        """
        if not chunks or len(chunks) <= 1:
            return chunks

        merged: List[str] = []
        buf = ""
        for chunk in chunks:
            if not buf:
                buf = chunk
            elif len(buf) < min_size:
                buf = buf + "\n\n" + chunk
            else:
                merged.append(buf)
                buf = chunk

        # Flush remaining buffer
        if buf:
            if merged and len(buf) < min_size:
                merged[-1] = merged[-1] + "\n\n" + buf
            else:
                merged.append(buf)

        return merged

    def process_folder(self, folder_path: str) -> List[Document]:
        """Process all PDFs in a folder.

        Args:
            folder_path: Path to folder containing PDF files.

        Returns:
            Combined list of Document objects from all PDFs.
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        pdf_files = sorted(
            [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
        )
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files in: {folder_path}")

        print(f"\n[Documents] Found {len(pdf_files)} PDF(s): {pdf_files}")

        all_documents = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(folder_path, pdf_file)
            docs = self.process_pdf(pdf_path)
            all_documents.extend(docs)

        print(f"\n[Summary] Total chunks to index: {len(all_documents)}")
        return all_documents

    # ── Qdrant Collection Management ──────────────────────────────────────

    def _connect(self) -> QdrantClient:
        """Create a Qdrant client connection.

        Supports two modes:
          - Embedded (default): Runs in-process, persists to disk. No Docker needed.
          - Server: Connects to a running Qdrant server via URL.
        """
        if self.qdrant_url:
            # Server mode (Docker or cloud)
            kwargs = {"url": self.qdrant_url, "timeout": 120}
            if self.qdrant_api_key:
                kwargs["api_key"] = self.qdrant_api_key
            print(f"  Mode: Server ({self.qdrant_url})")
            return QdrantClient(**kwargs)
        elif self.qdrant_path:
            # Embedded mode — runs in-process, no Docker needed
            os.makedirs(self.qdrant_path, exist_ok=True)
            print(f"  Mode: Embedded (on-disk at {self.qdrant_path})")
            return QdrantClient(path=self.qdrant_path)
        else:
            raise ValueError(
                "Either qdrant_url or qdrant_path must be provided. "
                "Set qdrant_path for embedded mode (no Docker) or "
                "qdrant_url for server mode."
            )

    def create_collection(
        self,
        client: QdrantClient,
        collection_name: str,
    ) -> None:
        """Create an optimized Qdrant collection with HNSW + INT8 quantization.

        Payload indexes are created BEFORE data insertion (Qdrant best practice)
        so HNSW builds extra edges for filtered search from the start.
        """
        # Drop old collection if exists
        if client.collection_exists(collection_name):
            print(f"  Deleting existing collection '{collection_name}'...")
            client.delete_collection(collection_name)

        print(f"  Creating collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dimensions,
                distance=Distance.COSINE,
                on_disk=False,  # keep vectors in RAM for fast search
            ),
            hnsw_config=HnswConfigDiff(
                m=16,              # edges per node — optimal for <100K vectors
                ef_construct=128,  # build-time accuracy
            ),
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    always_ram=True,  # quantized vectors always in RAM
                )
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=5000,  # build HNSW index early
            ),
        )
        print("  HNSW(m=16, ef=128) + INT8 quantization + Cosine")

        # Create payload indexes BEFORE inserting data
        # Note: LangChain's QdrantVectorStore nests metadata under "metadata."
        # so indexes must use the full nested path (e.g., "metadata.patient_mrn")
        print("  Creating payload indexes...")
        for field in ("metadata.patient_mrn", "metadata.section_title", "metadata.source"):
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        print("  Payload indexes: metadata.patient_mrn, metadata.section_title, metadata.source")

    def index_documents(
        self,
        client: QdrantClient,
        collection_name: str,
        documents: List[Document],
    ) -> QdrantVectorStore:
        """Upload documents to Qdrant in batches.

        Returns:
            QdrantVectorStore instance (can be used for verification queries).
        """
        print(f"\n[Indexing] Uploading {len(documents)} chunks...")

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=self.embeddings,
        )

        for i in range(0, len(documents), self.batch_size):
            batch = documents[i : i + self.batch_size]
            vector_store.add_documents(batch)
            progress = min(i + self.batch_size, len(documents))
            print(f"  Uploaded {progress}/{len(documents)} chunks")

        return vector_store

    # ── High-level API ────────────────────────────────────────────────────

    def vectorize_folder(
        self,
        folder_path: str,
        collection_name: str = "PatientData",
        force_recreate: bool = True,
    ) -> None:
        """End-to-end: process all PDFs in a folder and index into Qdrant.

        Args:
            folder_path: Path to folder containing PDF files.
            collection_name: Qdrant collection name (e.g. "PatientData").
            force_recreate: If True, drop and recreate the collection.
        """
        print("=" * 70)
        print("  CLINICAL NOTES VECTORIZATION — Production Pipeline")
        print("=" * 70)
        print(f"\n[Config]")
        print(f"  Embedding model : {self.embedding_model_name}")
        print(f"  Vector dims     : {self.embedding_dimensions}")
        print(f"  Chunk size      : {self.chunk_size} chars")
        print(f"  Chunk overlap   : {self.chunk_overlap} chars")
        print(f"  Collection      : {collection_name}")
        if self.qdrant_url:
            print(f"  Qdrant          : Server @ {self.qdrant_url}")
        else:
            print(f"  Qdrant          : Embedded @ {self.qdrant_path}")

        # Process PDFs
        documents = self.process_folder(folder_path)

        # Show a sample chunk for verification
        if documents:
            sample = documents[len(documents) // 2]
            print(f"\n[Sample Chunk — #{sample.metadata['chunk_index']}]")
            print(f"  Section : {sample.metadata['section_title']}")
            print(f"  Page    : {sample.metadata['page_number']}")
            print(f"  MRN     : {sample.metadata['patient_mrn']}")
            print(f"  Content : {sample.page_content[:200]}...")

        # Connect to Qdrant
        print(f"\n[Qdrant] Connecting...")
        client = self._connect()

        # Create collection
        if force_recreate:
            self.create_collection(client, collection_name)

        # Index documents
        vector_store = self.index_documents(client, collection_name, documents)

        # Verify
        self._verify(client, collection_name, vector_store)

    def vectorize_single_pdf(
        self,
        pdf_path: str,
        collection_name: str = "PatientData",
        append: bool = True,
    ) -> None:
        """Vectorize a single PDF and add to a Qdrant collection.

        Args:
            pdf_path: Full path to the PDF file.
            collection_name: Qdrant collection name.
            append: If True, add to existing collection. If False, recreate.
        """
        documents = self.process_pdf(pdf_path)
        client = self._connect()

        if not append:
            self.create_collection(client, collection_name)

        vector_store = self.index_documents(client, collection_name, documents)

        info = client.get_collection(collection_name)
        print(f"\n[Done] '{collection_name}' now has {info.points_count} vectors")

    def _verify(
        self,
        client: QdrantClient,
        collection_name: str,
        vector_store: QdrantVectorStore,
    ) -> None:
        """Verify collection and run a sanity-check query."""
        print(f"\n[Verification]")
        info = client.get_collection(collection_name)
        print(f"  Collection      : {collection_name}")
        print(f"  Vectors stored  : {info.points_count}")
        print(f"  Vector size     : {info.config.params.vectors.size}")
        print(f"  Distance metric : {info.config.params.vectors.distance}")
        print(f"  Status          : {info.status}")

        # Quick sanity search
        print(f"\n[Sanity Check] Running test query...")
        test_results = vector_store.similarity_search_with_score(
            "What medications is the patient taking?", k=3
        )
        for doc, score in test_results:
            section = doc.metadata.get("section_title", "?")
            print(f"  Score={score:.4f} | Section: {section} | {doc.page_content[:80]}...")

        print(f"\n{'=' * 70}")
        print("  DONE — Vector database built successfully!")
        print(f"{'=' * 70}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point — reads config for convenience, but the class is standalone
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """CLI entry point. Reads config from tools_config.yml for convenience.
    The ClinicalNotesVectorizer class itself has zero config dependency.
    """
    from dotenv import load_dotenv
    from pyprojroot import here

    sys.path.insert(0, str(here("src")))
    load_dotenv()

    from agent_graph.load_tools_config import LoadToolsConfig
    cfg = LoadToolsConfig()

    # Resolve docs folder path
    docs_folder = cfg.clinical_notes_rag_unstructured_docs_directory
    if not os.path.isabs(docs_folder):
        docs_folder = str(here(docs_folder))

    # Resolve qdrant storage path
    vectordb_dir = cfg.clinical_notes_rag_vectordb_directory
    if not os.path.isabs(vectordb_dir):
        vectordb_dir = str(here(vectordb_dir))

    # If qdrant_url is set, use server mode; otherwise, embedded mode with local path
    qdrant_url = cfg.clinical_notes_rag_qdrant_url or None

    vectorizer = ClinicalNotesVectorizer(
        embedding_model=cfg.clinical_notes_rag_embedding_model,
        embedding_dimensions=cfg.clinical_notes_rag_embedding_dimensions,
        chunk_size=cfg.clinical_notes_rag_chunk_size,
        chunk_overlap=cfg.clinical_notes_rag_chunk_overlap,
        qdrant_url=qdrant_url,
        qdrant_path=vectordb_dir,
        qdrant_api_key=cfg.clinical_notes_rag_qdrant_api_key or None,
    )

    vectorizer.vectorize_folder(
        folder_path=docs_folder,
        collection_name=cfg.clinical_notes_rag_collection_name,
    )

    # ── Optionally pre-compute summaries so they're cached at runtime ────
    if "--precompute-summaries" in sys.argv:
        print("\n" + "=" * 60)
        print("PRE-COMPUTING PATIENT SUMMARIES")
        print("=" * 60)
        try:
            from agent_graph.tool_clinical_notes_rag import (
                get_rag_tool_instance, summarize_patient_record
            )
            rag = get_rag_tool_instance()
            patients_info = rag.list_patients()
            print(patients_info)
            # Parse MRNs from the listing
            import re
            mrns = re.findall(r"MRN:\s*(\S+)", patients_info)
            for mrn in mrns:
                print(f"\nGenerating summary for {mrn} ...")
                result = summarize_patient_record.invoke(mrn)
                print(f"  Done ({len(result):,} chars)")
            print("\nAll summaries pre-computed and cached.")
        except Exception as e:
            print(f"Warning: Summary pre-computation failed: {e}")
            print("Summaries will be generated on first request instead.")


if __name__ == "__main__":
    main()
