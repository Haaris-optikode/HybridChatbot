#!/usr/bin/env python3
"""
prepare_vector_db.py — Standalone, production-grade document vectorization.

This module is fully decoupled from the chatbot/agent graph and can be reused
independently for any PDF-to-vector pipeline.

Architecture decisions:
  1. EMBEDDING: NeuML/pubmedbert-base-embeddings (768-dim, local CPU, HIPAA-safe)
     - Fine-tuned from Microsoft PubMedBERT on PubMed title-abstract pairs
     - Outperforms all general-purpose models on medical text (Pearson 95.62)
     - Runs on CPU (~50 sent/s on i5), no GPU required
     - Data never leaves your server — full HIPAA compliance

  2. CHUNKING: Adaptive element-aware chunking (via document_processing module)
     - PDF → Structured Markdown (pymupdf4llm: preserves headers, tables, lists)
     - Markdown header splitting for format-agnostic section detection
     - Content-zone strategies:
         • Tables → kept whole (split by rows if very large, header row preserved)
         • SOAP daily notes → split on Day/date boundaries (one chunk per entry)
         • Short sections (≤ 1.5× chunk_size) → kept whole
         • Long narrative → RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
     - Tiny fragments (< 150 chars) are merged with neighbors
     - Each chunk prepended with "[Patient: X | MRN: Y | Section: Z | DocType: W]"

  3. INDEXING: Qdrant embedded mode (in-process, on-disk, no Docker needed)
     - Cosine distance for normalized sentence-transformer embeddings
     - HNSW m=16, ef_construct=128 (optimal for <100K vectors)
     - Payload indexes created BEFORE insert (builds extra HNSW edges)
     - INT8 scalar quantization with always_ram=True for fast retrieval
     - Can also connect to a Qdrant server via URL if configured

  4. METADATA: Rich payload for filtered search
     - patient_mrn, patient_name, section_title, page_number, source,
       document_id, document_type
     - Enables Qdrant payload-filtered search (by MRN, document, etc.)

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
import sys
import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PayloadSchemaType,
    ScalarQuantizationConfig, ScalarType, ScalarQuantization,
    OptimizersConfigDiff, HnswConfigDiff,
)

logger = logging.getLogger(__name__)


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
        logger.info(f"\n[Embeddings] Loading {embedding_model} (local CPU)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        )
        logger.info(f"  Model loaded: {embedding_model} ({embedding_dimensions}-dim)")

    # ── PDF Processing ────────────────────────────────────────────────────

    def process_pdf(self, pdf_path: str, document_id: Optional[str] = None) -> List[Document]:
        """Process a single PDF into chunked Documents with rich metadata.

        Uses the adaptive document processing pipeline:
          1. PDF → structured markdown (pymupdf4llm: preserves tables, headers)
          2. Multi-pattern metadata extraction (MRN, name, doc type)
          3. Element-aware chunking (tables whole, SOAP per-day, narrative split)
          4. Context prefixing and metadata attachment

        Args:
            pdf_path: Full path to the PDF file.
            document_id: Optional unique document ID. Auto-generated if None.

        Returns:
            List of LangChain Document objects ready for indexing.
        """
        from document_processing import process_document

        return process_document(
            pdf_path=pdf_path,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            document_id=document_id,
        )

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

        logger.info(f"\n[Documents] Found {len(pdf_files)} PDF(s): {pdf_files}")

        all_documents = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(folder_path, pdf_file)
            docs = self.process_pdf(pdf_path)
            all_documents.extend(docs)

        logger.info(f"\n[Summary] Total chunks to index: {len(all_documents)}")
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
            logger.info(f"  Mode: Server ({self.qdrant_url})")
            return QdrantClient(**kwargs)
        elif self.qdrant_path:
            # Embedded mode — runs in-process, no Docker needed
            os.makedirs(self.qdrant_path, exist_ok=True)
            logger.info(f"  Mode: Embedded (on-disk at {self.qdrant_path})")
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
            logger.info(f"  Deleting existing collection '{collection_name}'...")
            client.delete_collection(collection_name)

        logger.info(f"  Creating collection '{collection_name}'...")
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
        logger.info("  HNSW(m=16, ef=128) + INT8 quantization + Cosine")

        # Create payload indexes BEFORE inserting data
        # Note: LangChain's QdrantVectorStore nests metadata under "metadata."
        # so indexes must use the full nested path (e.g., "metadata.patient_mrn")
        logger.info("  Creating payload indexes...")
        for field in ("metadata.patient_mrn", "metadata.section_title",
                      "metadata.source", "metadata.document_id",
                      "metadata.document_type"):
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        logger.info("  Payload indexes: patient_mrn, section_title, source, document_id, document_type")

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
        logger.info(f"\n[Indexing] Uploading {len(documents)} chunks...")

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=self.embeddings,
        )

        for i in range(0, len(documents), self.batch_size):
            batch = documents[i : i + self.batch_size]
            vector_store.add_documents(batch)
            progress = min(i + self.batch_size, len(documents))
            logger.info(f"  Uploaded {progress}/{len(documents)} chunks")

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
        logger.info("=" * 70)
        logger.info("  CLINICAL NOTES VECTORIZATION — Production Pipeline")
        logger.info("=" * 70)
        logger.info(f"\n[Config]")
        logger.info(f"  Embedding model : {self.embedding_model_name}")
        logger.info(f"  Vector dims     : {self.embedding_dimensions}")
        logger.info(f"  Chunk size      : {self.chunk_size} chars")
        logger.info(f"  Chunk overlap   : {self.chunk_overlap} chars")
        logger.info(f"  Collection      : {collection_name}")
        if self.qdrant_url:
            logger.info(f"  Qdrant          : Server @ {self.qdrant_url}")
        else:
            logger.info(f"  Qdrant          : Embedded @ {self.qdrant_path}")

        # Process PDFs
        documents = self.process_folder(folder_path)

        # Show a sample chunk for verification
        if documents:
            sample = documents[len(documents) // 2]
            logger.info(f"\n[Sample Chunk — #{sample.metadata['chunk_index']}]")
            logger.info(f"  Section : {sample.metadata['section_title']}")
            logger.info(f"  Page    : {sample.metadata['page_number']}")
            logger.info(f"  MRN     : {sample.metadata['patient_mrn']}")
            logger.info(f"  Content : {sample.page_content[:200]}...")

        # Connect to Qdrant
        logger.info(f"\n[Qdrant] Connecting...")
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
        logger.info(f"\n[Done] '{collection_name}' now has {info.points_count} vectors")

    def _verify(
        self,
        client: QdrantClient,
        collection_name: str,
        vector_store: QdrantVectorStore,
    ) -> None:
        """Verify collection and run a sanity-check query."""
        logger.info(f"\n[Verification]")
        info = client.get_collection(collection_name)
        logger.info(f"  Collection      : {collection_name}")
        logger.info(f"  Vectors stored  : {info.points_count}")
        logger.info(f"  Vector size     : {info.config.params.vectors.size}")
        logger.info(f"  Distance metric : {info.config.params.vectors.distance}")
        logger.info(f"  Status          : {info.status}")

        # Quick sanity search
        logger.info(f"\n[Sanity Check] Running test query...")
        test_results = vector_store.similarity_search_with_score(
            "What medications is the patient taking?", k=3
        )
        for doc, score in test_results:
            section = doc.metadata.get("section_title", "?")
            logger.info(f"  Score={score:.4f} | Section: {section} | {doc.page_content[:80]}...")

        logger.info(f"\n{'=' * 70}")
        logger.info("  DONE — Vector database built successfully!")
        logger.info(f"{'=' * 70}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point — reads config for convenience, but the class is standalone
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """CLI entry point. Reads config from tools_config.yml for convenience.
    The ClinicalNotesVectorizer class itself has zero config dependency.
    """
    from dotenv import load_dotenv
    from pyprojroot import here

    # Configure logging for CLI usage (human-readable, not JSON)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

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
        logger.info("\n" + "=" * 60)
        logger.info("PRE-COMPUTING PATIENT SUMMARIES")
        logger.info("=" * 60)
        try:
            from agent_graph.tool_clinical_notes_rag import (
                get_rag_tool_instance, summarize_patient_record
            )
            rag = get_rag_tool_instance()
            patients_info = rag.list_patients()
            logger.info(patients_info)
            # Parse MRNs from the listing
            import re
            mrns = re.findall(r"MRN:\s*(\S+)", patients_info)
            for mrn in mrns:
                logger.info("Generating summary for %s ...", mrn)
                result = summarize_patient_record.invoke(mrn)
                logger.info("  Done (%s chars)", f"{len(result):,}")
            logger.info("All summaries pre-computed and cached.")
        except Exception as e:
            logger.warning("Summary pre-computation failed: %s", e)
            logger.info("Summaries will be generated on first request instead.")
            logger.info("Summaries will be generated on first request instead.")


if __name__ == "__main__":
    main()
