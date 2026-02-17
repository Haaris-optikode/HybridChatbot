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

  2. CHUNKING: Section-aware recursive splitting (1000 chars, 200 overlap)
     - Clinical notes are heavily structured (Demographics, Meds, Labs, SOAP…)
     - Detects numbered section headers and chunks WITHIN sections only
     - 1000 chars ≈ 250 tokens — holds a full clinical concept intact
     - Each chunk is prepended with "[Section: <title>]" for contextual grounding

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
# Text utilities (pure functions, no external dependencies)
# ──────────────────────────────────────────────────────────────────────────────

# Matches numbered headings: "1 PATIENT DEMOGRAPHICS", "6.1.2 Subjective", etc.
SECTION_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$", re.MULTILINE)


def detect_sections(text: str) -> List[Dict[str, str]]:
    """Split document text into sections based on numbered headings.

    Returns:
        List of dicts with 'title' and 'body' keys.
    """
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        return [{"title": "Full Document", "body": text}]

    sections = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = f"{m.group(1)} {m.group(2).strip()}"
        body = text[start:end].strip()
        sections.append({"title": title, "body": body})
    return sections


def extract_mrn(text: str) -> str:
    """Extract MRN (Medical Record Number) from document text."""
    pattern = r'\b(?:mrn|medical\s+record\s+number)\s*(?:is\s*|:\s*)?([A-Z0-9\-]+)'
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def extract_patient_name(text: str) -> str:
    """Extract patient name from document text."""
    pattern = r'(?:patient\s+name|name)\s*:\s*(.+?)(?:\n|$)'
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


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

        # Detect sections
        sections = detect_sections(full_text)
        print(f"  Sections: {len(sections)}")

        # Chunk each section independently — never split across sections
        documents = []
        chunk_idx = 0
        for section in sections:
            title = section["title"]
            body = section["body"]
            chunks = self.text_splitter.split_text(body)

            for chunk_text in chunks:
                # Prepend section context so the embedding captures section meaning
                contextualized = f"[Section: {title}]\n{chunk_text}"

                # Approximate page number from char offset
                body_pos = full_text.find(chunk_text[:80])
                page_num = get_page_number(body_pos) if body_pos >= 0 else 1

                doc = Document(
                    page_content=contextualized,
                    metadata={
                        "source": pdf_file,
                        "filename": pdf_file,
                        "patient_mrn": patient_mrn,
                        "patient_name": patient_name,
                        "section_title": title,
                        "page_number": page_num,
                        "chunk_index": chunk_idx,
                    },
                )
                documents.append(doc)
                chunk_idx += 1

        print(f"  Chunks  : {chunk_idx}")
        return documents

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
        print("  Creating payload indexes...")
        for field in ("patient_mrn", "section_title", "source"):
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        print("  Payload indexes: patient_mrn, section_title, source")

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
