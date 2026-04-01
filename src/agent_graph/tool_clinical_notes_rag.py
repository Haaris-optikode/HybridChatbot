# tool_clinical_notes_rag.py
"""
Clinical notes retrieval & summarization tools.

  Tool 1 — lookup_clinical_notes: Two-stage semantic search (PubMedBERT + reranker)
  Tool 2 — summarize_patient_record: Full-document clinical summary (all chunks → GPT-4.1-mini)
  Tool 3 — list_available_patients: Discover patients stored in the vector DB

All data stays local — embeddings run on CPU, Qdrant runs embedded on disk.
"""
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from agent_graph.load_tools_config import (
    LoadToolsConfig,
    build_model_chain,
    get_google_api_key,
    get_active_llm_provider,
    swap_google_api_key,
)
from qdrant_client import QdrantClient
from pyprojroot import here
import logging
import os
import json
import hashlib
import time as _time
import re as _re
import concurrent.futures as _futures
from pathlib import Path as _Path
from datetime import datetime as _dt
from dotenv import load_dotenv
import threading
from collections import OrderedDict
from contextvars import ContextVar

logger = logging.getLogger(__name__)

load_dotenv()
TOOLS_CFG = LoadToolsConfig()

# ── LRU in-memory summary cache (cross-session, process-lifetime) ────────────
# OrderedDict keeps insertion order; we move-to-end on access to track LRU.
# When the cache exceeds _CACHE_MAX_SIZE, the oldest entry is evicted.
_memory_cache: OrderedDict[str, dict] = OrderedDict()
_memory_cache_lock = threading.Lock()
_CACHE_TTL_HOURS = 24
_CACHE_MAX_SIZE = int(os.environ.get("MEDGRAPH_CACHE_MAX_SIZE", "50"))  # max patient summaries in memory
_SUMMARY_REQUEST_MODE: ContextVar[str] = ContextVar("summary_request_mode", default="normal")


def set_summary_request_mode(mode: str):
    """Set request-scoped summary mode ('normal' or 'thinking')."""
    return _SUMMARY_REQUEST_MODE.set(mode if mode in ("normal", "thinking") else "normal")


def reset_summary_request_mode(token) -> None:
    """Reset request-scoped summary mode token."""
    try:
        _SUMMARY_REQUEST_MODE.reset(token)
    except Exception:
        pass


def _current_summary_mode() -> str:
    return _SUMMARY_REQUEST_MODE.get()


def _record_cache_event(event: str, layer: str = ""):
    """Record cache hit/miss to Prometheus (best-effort, non-critical)."""
    try:
        from prometheus_client import Counter
        # Use get_or_create pattern — counters are singletons by name
        if event == "hit":
            Counter("medgraph_cache_hits_total", "Summary cache hits", ["layer"]).labels(layer=layer).inc()
        elif event == "miss":
            Counter("medgraph_cache_misses_total", "Summary cache misses").inc()
    except Exception:
        pass  # metrics are best-effort


def _make_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """Create local HuggingFace embeddings (CPU, HIPAA-safe)."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


def _load_reranker(model_name: str):
    """Lazily load cross-encoder reranker model."""
    try:
        from sentence_transformers import CrossEncoder
        logger.info("Loading reranker: %s", model_name)
        reranker = CrossEncoder(model_name, max_length=512)
        logger.info("Reranker loaded: %s", model_name)
        return reranker
    except ImportError:
        logger.warning("sentence-transformers not installed. Reranking disabled.")
        return None
    except Exception as e:
        logger.warning("Could not load reranker '%s': %s", model_name, e)
        return None


class ClinicalNotesRAGTool:
    """
    Two-stage retrieval tool for clinical notes:
      1. Dense vector search (PubMedBERT via local Qdrant)
      2. Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)

    Connects to self-hosted Qdrant on localhost.
    """
    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, embedding_model: str, vectordb_dir: str, k: int,
                collection_name: str, qdrant_url: str = None,
                qdrant_api_key: str = None, embedding_dimensions: int = None,
                fetch_k: int = 20, reranker_model: str = None,
                reranker_top_k: int = 6):
        instance_key = (embedding_model, collection_name, qdrant_url)
        if instance_key not in cls._instances:
            with cls._lock:
                if instance_key not in cls._instances:
                    instance = super().__new__(cls)
                    cls._instances[instance_key] = instance
        return cls._instances[instance_key]

    def __init__(self, embedding_model: str, vectordb_dir: str, k: int,
                 collection_name: str, qdrant_url: str = None,
                 qdrant_api_key: str = None, embedding_dimensions: int = None,
                 fetch_k: int = 20, reranker_model: str = None,
                 reranker_top_k: int = 6) -> None:
        if hasattr(self, "_initialized"):
            self.k = k
            self.fetch_k = fetch_k
            self.reranker_top_k = reranker_top_k
            return

        self._initialized = True
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.vectordb_dir = vectordb_dir
        self.k = k
        self.fetch_k = fetch_k
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.collection_name = collection_name
        self.reranker_model_name = reranker_model
        self.reranker_top_k = reranker_top_k

        # Connect to Qdrant
        client_kwargs = {}
        if qdrant_url:
            # Server mode (Docker / cloud)
            client_kwargs["url"] = qdrant_url
            if qdrant_api_key:
                client_kwargs["api_key"] = qdrant_api_key
            logger.info("Connecting to Qdrant server: %s", qdrant_url)
        else:
            # Embedded mode — runs in-process, no Docker needed
            local_path = str(here(vectordb_dir)) if vectordb_dir else None
            if not local_path:
                raise ValueError(
                    "No Qdrant URL or vectordb path configured. "
                    "Set qdrant_url for server mode or vectordb in tools_config.yml for embedded mode."
                )
            os.makedirs(local_path, exist_ok=True)
            client_kwargs["path"] = local_path
            logger.info("Using embedded Qdrant: %s", local_path)

        self.client = QdrantClient(**client_kwargs)

        # Create embeddings (local PubMedBERT)
        self.vectordb = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=_make_embeddings(self.embedding_model),
        )

        # Load cross-encoder reranker
        self.reranker = None
        if reranker_model:
            self.reranker = _load_reranker(reranker_model)

        # Build BM25 index for hybrid search (P2.24)
        self._bm25_docs = []   # parallel list of LangChain Documents
        self._bm25_index = None
        self._build_bm25_index()

        try:
            info = self.client.get_collection(collection_name)
            logger.info("Connected to '%s': %d vectors | Status: %s", collection_name, info.points_count, info.status)
        except Exception as e:
            logger.warning("Could not get collection info: %s", e)

    def _build_bm25_index(self):
        """Build a BM25 index over all documents in the collection for sparse retrieval."""
        try:
            from rank_bm25 import BM25Okapi
            from langchain_core.documents import Document

            results = self.client.scroll(
                collection_name=self.collection_name,
                limit=5000,
                with_payload=True,
                with_vectors=False,
            )
            points = results[0]
            if not points:
                logger.warning("BM25: No documents found — sparse retrieval disabled")
                return

            corpus = []
            docs = []
            for p in points:
                content = p.payload.get("page_content", "")
                meta = p.payload.get("metadata", {})
                if content.strip():
                    corpus.append(content.lower().split())
                    docs.append(Document(page_content=content, metadata=meta))

            self._bm25_index = BM25Okapi(corpus)
            self._bm25_docs = docs
            logger.info("BM25 index built: %d documents", len(docs))
        except ImportError:
            logger.warning("rank_bm25 not installed — BM25 hybrid search disabled")
        except Exception as e:
            logger.warning("BM25 index build failed: %s", e)

    def search(self, query: str, document_id: str = None) -> str:
        """
        Hybrid retrieval: dense MMR + BM25 sparse + RRF fusion + cross-encoder reranking.
        Optimized for latency: reduced candidate pool, efficient reranking.

        Args:
            query: The search query.
            document_id: Optional. If provided, restricts search to chunks from this document only.
        """
        try:
            t0 = _time.perf_counter()

            # Build optional Qdrant filter for document scoping
            _qdrant_filter = None
            if document_id:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                _qdrant_filter = Filter(
                    must=[FieldCondition(
                        key="metadata.document_id",
                        match=MatchValue(value=document_id),
                    )]
                )

            # Stage 1: Dense MMR search (original query — no augmentation)
            search_kwargs = {
                "k": self.fetch_k,
                "fetch_k": self.fetch_k * 2,
                "lambda_mult": 0.5,
            }
            if _qdrant_filter:
                search_kwargs["filter"] = _qdrant_filter
            dense_docs = self.vectordb.max_marginal_relevance_search(
                query=query,
                **search_kwargs,
            )
            t_dense = _time.perf_counter() - t0

            # Stage 2: BM25 sparse search (keyword-level)
            bm25_docs = []
            if self._bm25_index and self._bm25_docs:
                try:
                    import numpy as np
                    tokenized_query = query.lower().split()
                    scores = self._bm25_index.get_scores(tokenized_query)
                    top_indices = np.argsort(scores)[::-1][:self.fetch_k]
                    bm25_docs = [self._bm25_docs[i] for i in top_indices if scores[i] > 0]
                    # Filter by document_id if scoped
                    if document_id:
                        bm25_docs = [d for d in bm25_docs
                                     if d.metadata.get("document_id") == document_id]
                except Exception as e:
                    logger.warning("BM25 search failed, using dense only: %s", e)
            t_bm25 = _time.perf_counter() - t0 - t_dense

            # Stage 3: Reciprocal Rank Fusion (RRF) to merge rankings
            if bm25_docs:
                fused = self._reciprocal_rank_fusion(dense_docs, bm25_docs, k=60)
            else:
                fused = dense_docs

            if not fused:
                return f"No relevant documents found for: {query}"

            # Stage 4: Cross-encoder reranking with relevance filtering
            # NotebookLM-inspired: reject irrelevant chunks BEFORE they reach the LLM
            # to prevent hallucination from noisy context.
            _MIN_RERANKER_SCORE = 0.05  # cross-encoder threshold
            rerank_pool = fused[:self.fetch_k]  # cap reranker input
            if self.reranker and len(rerank_pool) > 1:
                pairs = [[query, doc.page_content] for doc in rerank_pool]
                scores = self.reranker.predict(pairs)
                scored_docs = sorted(
                    zip(rerank_pool, scores), key=lambda x: x[1], reverse=True
                )
                # Filter out chunks below relevance threshold
                scored_docs = [(doc, s) for doc, s in scored_docs if s >= _MIN_RERANKER_SCORE]
                if not scored_docs:
                    return (f"No sufficiently relevant clinical data found for: {query}. "
                            f"Try rephrasing with specific clinical terms "
                            f"(e.g., medication names, lab tests, section names, dates).")
                docs = [doc for doc, _ in scored_docs[:self.reranker_top_k]]
            else:
                docs = rerank_pool[:self.k]

            # Stage 5: Adjacent-chunk expansion
            # SOAP notes and long sections often split across chunks.
            # For each retrieved doc, also pull its chunk_index ± 1 neighbor
            # from the same source file so the LLM gets complete context
            # (e.g., full Day 7 SOAP with Assessment & Plan).
            docs = self._expand_with_neighbors(docs)

            t_total = _time.perf_counter() - t0
            logger.info("Search: %.2fs (dense=%.2fs, bm25=%.2fs, rerank+fuse+expand=%.2fs) → %d docs",
                        t_total, t_dense, t_bm25, t_total - t_dense - t_bm25, len(docs))

            # Render with source citations for LLM grounding (NotebookLM-style)
            # Each chunk is tagged with section, page, and MRN so the LLM
            # can cite exact sources — preventing hallucination.
            parts = []
            for i, d in enumerate(docs, 1):
                sec = d.metadata.get("section_title", "Section")
                page = d.metadata.get("page_number", "?")
                mrn = d.metadata.get("patient_mrn", "")
                parts.append(f"[Source {i} | Section: {sec} | Page {page} | MRN: {mrn}]\n{d.page_content}")
            return "\n\n---\n\n".join(parts)
        except Exception as e:
            return f"Error during search: {e}"

    @staticmethod
    def _reciprocal_rank_fusion(
        *rank_lists, k: int = 60
    ) -> list:
        """Merge multiple ranked document lists using RRF.
        Score = sum over lists of 1 / (k + rank).
        Documents are deduplicated by page_content hash."""
        scores: dict[str, float] = {}
        doc_map: dict[str, object] = {}

        for rank_list in rank_lists:
            for rank, doc in enumerate(rank_list):
                key = hashlib.md5(doc.page_content.encode()).hexdigest()
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
                doc_map[key] = doc

        sorted_keys = sorted(scores, key=scores.get, reverse=True)
        return [doc_map[k] for k in sorted_keys]

    def _expand_with_neighbors(self, docs: list) -> list:
        """Expand each retrieved document with its ±1 chunk-index neighbors.

        Clinical SOAP notes and long sections often split across consecutive
        chunks. A Day-specific query may retrieve the chunk that starts the
        day entry but miss the chunk that contains the Assessment & Plan.

        This method looks up neighbors from the BM25 docs cache (already in
        memory), merges overlapping text between consecutive chunks, and
        deduplicates by chunk_index to avoid sending duplicates to the LLM.

        Returns a new list of Documents, ordered by (filename, chunk_index).
        """
        from langchain_core.documents import Document

        if not self._bm25_docs:
            return docs

        # Build a lookup: (filename, chunk_index) → Document
        idx_lookup: dict[tuple[str, int], object] = {}
        for d in self._bm25_docs:
            key = (d.metadata.get("filename", ""), d.metadata.get("chunk_index", -1))
            idx_lookup[key] = d

        # Collect original + neighbor chunk keys (deduplicated, preserving order)
        seen_keys: set[tuple[str, int]] = set()
        expanded_keys: list[tuple[str, int]] = []

        for d in docs:
            fname = d.metadata.get("filename", "")
            cidx = d.metadata.get("chunk_index", -1)
            if cidx < 0:
                # No chunk_index metadata — keep original doc as-is
                key = (fname, cidx)
                if key not in seen_keys:
                    seen_keys.add(key)
                    expanded_keys.append(key)
                    idx_lookup[key] = d
                continue

            # Add prev, current, next chunk (if they exist in the corpus)
            for offset in (-1, 0, 1):
                nkey = (fname, cidx + offset)
                if nkey not in seen_keys and nkey in idx_lookup:
                    seen_keys.add(nkey)
                    expanded_keys.append(nkey)

        # Sort by (filename, chunk_index) for coherent reading order
        expanded_keys.sort(key=lambda k: (k[0], k[1]))

        # Build expanded doc list, merging overlap between consecutive chunks
        expanded: list[object] = []
        prev_content: str | None = None
        prev_key: tuple[str, int] | None = None

        for key in expanded_keys:
            d = idx_lookup[key]
            content = d.page_content
            fname, cidx = key

            # If consecutive chunks from same file, strip overlap
            if (prev_key and prev_key[0] == fname
                    and prev_key[1] == cidx - 1 and prev_content):
                overlap = self._find_overlap(prev_content, content)
                if overlap > 0:
                    content = content[overlap:]

            if content.strip():
                expanded.append(Document(
                    page_content=content,
                    metadata=d.metadata,
                ))

            prev_content = d.page_content  # original for overlap detection
            prev_key = key

        added = len(expanded) - len(docs)
        if added > 0:
            logger.info("Neighbor expansion: %d → %d chunks (+%d neighbors)",
                        len(docs), len(expanded), added)

        return expanded

    def get_all_chunks_for_patient(self, patient_mrn: str) -> str:
        """Retrieve ALL chunks for a specific patient by MRN, merge overlapping
        text to reduce token count by ~20%.

        Uses metadata filtering to fetch every chunk belonging to a patient,
        ordered by chunk_index, then strips the overlap region between
        consecutive chunks that share the same source file.
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            t0 = _time.perf_counter()

            # Payload is nested: {page_content: ..., metadata: {patient_mrn: ...}}
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.patient_mrn",
                            match=MatchValue(value=patient_mrn),
                        )
                    ]
                ),
                limit=500,
                with_payload=True,
                with_vectors=False,
            )

            points = results[0]
            if not points:
                return f"No records found for patient MRN: {patient_mrn}"

            # Sort by chunk_index to reconstruct document order
            points.sort(
                key=lambda p: p.payload.get("metadata", {}).get("chunk_index", 0)
            )

            # ── Merge overlapping text ────────────────────────────────
            # Adjacent chunks share `chunk_overlap` chars.  Instead of
            # including every chunk verbatim (20 % duplication), we strip
            # the leading overlap from each chunk after the first one in
            # the same source file.
            merged_parts: list[str] = []
            prev_content: str | None = None
            prev_source: str | None = None

            for p in points:
                meta = p.payload.get("metadata", {})
                content = p.payload.get("page_content", "")
                source = meta.get("filename", "")
                sec = meta.get("section_title", "Section")

                if prev_content and source == prev_source:
                    # Try to detect the overlap region
                    overlap = self._find_overlap(prev_content, content)
                    if overlap > 0:
                        content = content[overlap:]

                if content.strip():
                    merged_parts.append(f"[{sec}]\n{content}")

                prev_content = p.payload.get("page_content", "")  # original content for next overlap check
                prev_source = source

            elapsed = _time.perf_counter() - t0
            logger.info("Retrieved %d chunks for %s, merged to %d parts in %.2fs",
                        len(points), patient_mrn, len(merged_parts), elapsed)

            return "\n\n".join(merged_parts)
        except Exception as e:
            return f"Error retrieving patient data: {e}"

    def get_all_chunks_for_patient_structured(self, patient_mrn: str) -> list[tuple[str, dict]]:
        """Retrieve all patient chunks with metadata, ordered and overlap-trimmed.

        This is used by the structured summarization pipeline that requires
        stable source IDs and per-chunk citations.
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            mrn = (patient_mrn or "").strip()
            if not mrn:
                return []

            all_points = []
            next_offset = None
            while True:
                results = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="metadata.patient_mrn",
                                match=MatchValue(value=mrn),
                            )
                        ]
                    ),
                    limit=512,
                    with_payload=True,
                    with_vectors=False,
                    offset=next_offset,
                )
                points, next_offset = results[0], results[1]
                if points:
                    all_points.extend(points)
                if not next_offset:
                    break

            if not all_points:
                return []

            all_points.sort(
                key=lambda p: (
                    p.payload.get("metadata", {}).get("filename", ""),
                    p.payload.get("metadata", {}).get("page_number", 0),
                    p.payload.get("metadata", {}).get("chunk_index", 0),
                )
            )

            out: list[tuple[str, dict]] = []
            prev_content: str | None = None
            prev_source: str | None = None
            for p in all_points:
                meta = p.payload.get("metadata", {}) or {}
                content = p.payload.get("page_content", "") or ""
                source = meta.get("filename", "")
                if prev_content and source and source == prev_source:
                    overlap = self._find_overlap(prev_content, content)
                    if overlap > 0:
                        content = content[overlap:]
                if content.strip():
                    out.append((content, meta))
                prev_content = p.payload.get("page_content", "") or ""
                prev_source = source
            return out
        except Exception as e:
            logger.warning("Error retrieving patient chunks for %s: %s", patient_mrn, str(e)[:200])
            return []

    def get_all_chunks_for_document(self, document_id: str) -> list[tuple[str, dict]]:
        """Retrieve ALL chunks for a specific uploaded document_id.

        Returns:
            List of (page_content, metadata) tuples ordered by (page_number, chunk_index).
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            doc_id = (document_id or "").strip()
            if not doc_id:
                return []

            all_points = []
            next_offset = None
            # Use paginated scroll to avoid missing >500 chunks documents.
            while True:
                results = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="metadata.document_id",
                                match=MatchValue(value=doc_id),
                            )
                        ]
                    ),
                    limit=512,
                    with_payload=True,
                    with_vectors=False,
                    offset=next_offset,
                )
                points, next_offset = results[0], results[1]
                if points:
                    all_points.extend(points)
                if not next_offset:
                    break

            if not all_points:
                return []

            # Order by page then chunk_index for coherent reading order.
            all_points.sort(
                key=lambda p: (
                    p.payload.get("metadata", {}).get("page_number", 0),
                    p.payload.get("metadata", {}).get("chunk_index", 0),
                )
            )

            out: list[tuple[str, dict]] = []
            prev_content: str | None = None
            prev_source: str | None = None
            for p in all_points:
                meta = p.payload.get("metadata", {}) or {}
                content = p.payload.get("page_content", "") or ""
                source = meta.get("filename", "")

                # Strip overlap when consecutive chunks from same source file
                if prev_content and source and source == prev_source:
                    overlap = self._find_overlap(prev_content, content)
                    if overlap > 0:
                        content = content[overlap:]

                if content.strip():
                    out.append((content, meta))
                prev_content = p.payload.get("page_content", "") or ""
                prev_source = source

            return out
        except Exception as e:
            logger.warning("Error retrieving document chunks for %s: %s", document_id, str(e)[:200])
            return []

    # ── Section-name patterns for "all orders" retrieval ──────────
    ORDER_SECTION_PATTERNS: list[str] = [
        "18a",                    # Laboratory Orders
        "18b",                    # Medication Orders (New/Changed)
        "18c",                    # Procedure Orders
        "18d",                    # Referrals / Consultations Ordered
        "19 DISCHARGE",           # Discharge Prescriptions (37 chunks)
        "14 RECENT ORDERS",       # Recent outpatient orders
        "22 DISCHARGE SUMMARY",   # Discharge disposition / dates
        "1 PATIENT DEMOGRAPHICS", # Advance directive, code status
    ]

    def get_order_chunks_for_patient(self, patient_mrn: str) -> str:
        """Retrieve ALL order-related chunks for a patient: labs, medications,
        procedures, referrals, discharge prescriptions, care directives.

        Uses metadata filtering on section_title to pull every chunk from
        order-related sections, returning them organised by section.
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            import re as _re

            t0 = _time.perf_counter()

            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.patient_mrn",
                            match=MatchValue(value=patient_mrn),
                        )
                    ]
                ),
                limit=500,
                with_payload=True,
                with_vectors=False,
            )

            points = results[0]
            if not points:
                return f"No records found for patient MRN: {patient_mrn}"

            # Filter to order-related sections (case-insensitive prefix match)
            order_points = []
            for p in points:
                sec = p.payload.get("metadata", {}).get("section_title", "")
                if any(sec.upper().startswith(pat.upper()) for pat in self.ORDER_SECTION_PATTERNS):
                    order_points.append(p)

            if not order_points:
                return f"No order-related data found for patient MRN: {patient_mrn}"

            # Sort by section_title then chunk_index for logical ordering
            order_points.sort(
                key=lambda p: (
                    p.payload.get("metadata", {}).get("section_title", ""),
                    p.payload.get("metadata", {}).get("chunk_index", 0),
                )
            )

            # Group by section and merge overlapping text
            current_section = None
            parts: list[str] = []
            prev_content: str | None = None
            prev_source: str | None = None

            for p in order_points:
                meta = p.payload.get("metadata", {})
                content = p.payload.get("page_content", "")
                source = meta.get("filename", "")
                sec = meta.get("section_title", "Section")

                # Section separator for readability
                if sec != current_section:
                    if current_section is not None:
                        parts.append("")  # blank line between sections
                    current_section = sec
                    prev_content = None  # reset overlap tracking at section boundary
                    prev_source = None

                # Strip overlap with previous chunk in same source
                if prev_content and source == prev_source:
                    overlap = self._find_overlap(prev_content, content)
                    if overlap > 0:
                        content = content[overlap:]

                if content.strip():
                    parts.append(f"[{sec}]\n{content}")

                prev_content = p.payload.get("page_content", "")
                prev_source = source

            elapsed = _time.perf_counter() - t0
            logger.info(
                "Order retrieval: %d/%d chunks matched order sections for %s in %.2fs",
                len(order_points), len(points), patient_mrn, elapsed,
            )

            return "\n\n".join(parts)

        except Exception as e:
            return f"Error retrieving order data: {e}"

    @staticmethod
    def _find_overlap(prev: str, curr: str, max_check: int = 250) -> int:
        """Find the length of overlapping text between end of `prev` and
        start of `curr`.  Returns 0 if no overlap detected."""
        check_len = min(max_check, len(prev), len(curr))
        # Start from the longest possible overlap and shrink
        for size in range(check_len, 30, -1):
            if prev.endswith(curr[:size]):
                return size
        return 0

    def list_patients(self) -> str:
        """List all unique patients in the vector database with their MRN, name, and source file."""
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=True,
                with_vectors=False,
            )
            points = results[0]
            patients = {}
            for p in points:
                meta = p.payload.get("metadata", {})
                mrn = meta.get("patient_mrn", "")
                if mrn and mrn not in patients:
                    patients[mrn] = {
                        "mrn": mrn,
                        "name": meta.get("patient_name", "Unknown"),
                        "source": meta.get("source", "Unknown"),
                    }
            if not patients:
                return "No patients found in the database."

            lines = ["Available patients in the database:"]
            for info in patients.values():
                lines.append(f"  - MRN: {info['mrn']} | Name: {info['name']} | Source: {info['source']}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error listing patients: {e}"

    def resolve_to_mrn(self, identifier: str) -> str | None:
        """Resolve a patient name or MRN string to the canonical MRN.

        Accepts: exact MRN, partial MRN, full name, partial name (case-insensitive).
        Returns the MRN string, or None if no match found.
        Uses scored matching to pick the best candidate, not first-found.
        """
        results = self.client.scroll(
            collection_name=self.collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        patients: dict[str, str] = {}  # mrn -> name
        for p in results[0]:
            meta = p.payload.get("metadata", {})
            mrn = meta.get("patient_mrn", "")
            name = meta.get("patient_name", "")
            if mrn and mrn not in patients:
                patients[mrn] = name

        query = identifier.strip().lower()
        best_mrn: str | None = None
        best_score = 0

        for mrn, name in patients.items():
            mrn_lower = mrn.lower()
            name_lower = name.lower()
            score = 0

            # Exact MRN match
            if mrn_lower == query:
                return mrn  # perfect match, return immediately

            # Partial MRN match
            if query in mrn_lower:
                score = max(score, 90)

            # Exact name match
            if name_lower == query:
                score = max(score, 100)

            # Name contains query as substring
            if query in name_lower:
                score = max(score, 80)

            # Query contains name as substring
            if name_lower and name_lower in query:
                score = max(score, 70)

            # Token-level overlap (scored by fraction of query tokens matched)
            query_tokens = set(query.split())
            name_tokens = set(name_lower.split())
            if query_tokens and name_tokens:
                overlap = query_tokens & name_tokens
                if overlap:
                    frac = len(overlap) / max(len(query_tokens), len(name_tokens))
                    token_score = int(50 * frac)
                    score = max(score, token_score)

            if score > best_score:
                best_score = score
                best_mrn = mrn

        if best_mrn:
            logger.info("Name resolution: '%s' → %s (name='%s', score=%d)",
                        identifier, best_mrn, patients[best_mrn], best_score)
        return best_mrn


# ── Global singleton ─────────────────────────────────────────────────────────
_rag_tool_instance = None
_rag_tool_lock = threading.Lock()


def get_rag_tool_instance():
    global _rag_tool_instance
    if _rag_tool_instance is None:
        with _rag_tool_lock:
            if _rag_tool_instance is None:
                _rag_tool_instance = ClinicalNotesRAGTool(
                    embedding_model=TOOLS_CFG.clinical_notes_rag_embedding_model,
                    vectordb_dir=getattr(TOOLS_CFG, "clinical_notes_rag_vectordb_directory", ""),
                    k=TOOLS_CFG.clinical_notes_rag_k,
                    collection_name=TOOLS_CFG.clinical_notes_rag_collection_name,
                    qdrant_url=TOOLS_CFG.clinical_notes_rag_qdrant_url,
                    qdrant_api_key=TOOLS_CFG.clinical_notes_rag_qdrant_api_key or None,
                    embedding_dimensions=TOOLS_CFG.clinical_notes_rag_embedding_dimensions,
                    fetch_k=TOOLS_CFG.clinical_notes_rag_fetch_k,
                    reranker_model=TOOLS_CFG.clinical_notes_rag_reranker_model,
                    reranker_top_k=TOOLS_CFG.clinical_notes_rag_reranker_top_k,
                )
    return _rag_tool_instance


# ── Summarization helpers ────────────────────────────────────────────────────

def _cleanup_expired_cache():
    """Delete disk cache files older than TTL. Runs once per startup."""
    cache_dir = _Path(here("data")) / "summary_cache"
    if not cache_dir.exists():
        return
    now = _dt.now()
    for f in cache_dir.glob("*.json"):
        try:
            cached = json.loads(f.read_text(encoding="utf-8"))
            generated_at = cached.get("generated_at", "")
            cache_time = _dt.strptime(generated_at, "%Y-%m-%d %H:%M:%S")
            age_hours = (now - cache_time).total_seconds() / 3600
            if age_hours > _CACHE_TTL_HOURS:
                f.unlink()
                logger.info("[Cache Cleanup] Deleted expired: %s (%.1fh old)", f.name, age_hours)
        except Exception:
            pass  # skip corrupt files

# Run cleanup once on module load
try:
    _cleanup_expired_cache()
except Exception:
    pass


def _extract_text_content(content) -> str:
    """Normalize LLM .content to a plain string.
    Gemini may return a list of part dicts (including 'thinking' parts)
    instead of a plain string. This handles all forms."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "thinking":
                    continue
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content) if content else ""


def _is_quota_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(k in s for k in ["rate limit", "429", "resource_exhausted", "quota"])


def _is_server_overload(exc: Exception) -> bool:
    """Detect Gemini 504 DEADLINE_EXCEEDED and similar server-side overloads."""
    s = str(exc).lower()
    return any(k in s for k in ["504", "deadline_exceeded", "deadline expired", "503", "overloaded", "502"])


def _llm_invoke_with_fallback(llm_factory, prompt, max_retries: int = 2, model_chain=None):
    """Invoke LLM with retry for quota exhaustion AND server-side 504/deadline errors.
    
    Retry strategy:
    - Quota errors: swap API key, retry once
    - 504/DEADLINE_EXCEEDED: exponential backoff (2s, 4s), retry up to max_retries
    """
    chain = model_chain or [None]
    last_exc = None
    model_attempt = 0
    for model_override in chain:
        model_attempt += 1
        model_label = model_override or "configured-default"
        for attempt in range(max_retries + 1):
            model_name, llm = llm_factory(model_override=model_override)
            try:
                return model_name, _extract_text_content(llm.invoke(prompt).content)
            except Exception as e:
                last_exc = e
                if _is_quota_error(e) and attempt == 0 and swap_google_api_key():
                    logger.warning("Quota exhausted — switched to backup API key, retrying")
                    continue
                if _is_server_overload(e) and attempt < max_retries:
                    wait = (attempt + 1) * 2  # 2s, 4s
                    logger.warning("Server overload (model=%s, attempt %d/%d): %s — retrying in %ds",
                                   model_label, attempt + 1, max_retries + 1, str(e)[:150], wait)
                    _time.sleep(wait)
                    continue
                break
        if model_attempt < len(chain):
            logger.warning("Model fallback: switching from '%s' to next candidate after error: %s",
                           model_label, str(last_exc)[:200])
    raise last_exc  # should not reach here


def _is_reasoning_model(model_name: str) -> bool:
    """Check if a model is a reasoning model (different API params)."""
    reasoning_prefixes = ("o1", "o3", "o4", "gpt-5")
    return any(model_name.lower().startswith(p) for p in reasoning_prefixes)


def _get_summarization_llm(model_override: str = None):
    """Create the summarization LLM — supports both Google Gemini and OpenAI."""
    base_model = getattr(TOOLS_CFG, "clinical_notes_rag_summarization_llm",
                         TOOLS_CFG.clinical_notes_rag_llm)
    model = model_override or base_model
    provider = get_active_llm_provider(getattr(TOOLS_CFG, "llm_provider", "openai"))

    if provider == "google" or model.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=get_google_api_key(),
            temperature=TOOLS_CFG.clinical_notes_rag_llm_temperature,
            max_output_tokens=8192,
            timeout=120,
            max_retries=3,
            thinking_budget=0,
        )
    elif _is_reasoning_model(model):
        llm = ChatOpenAI(
            model=model,
            max_completion_tokens=16384,
            api_key=os.getenv("OPENAI_API_KEY"),
            request_timeout=120,
            max_retries=3,
        )
    else:
        llm = ChatOpenAI(
            model=model,
            temperature=TOOLS_CFG.clinical_notes_rag_llm_temperature,
            max_tokens=4096,
            api_key=os.getenv("OPENAI_API_KEY"),
            request_timeout=120,
            max_retries=3,
        )
    logger.info("Summarization LLM: %s (provider=%s, timeout=120s, max_retries=3)", model, provider)
    return model, llm


_SUMMARY_PROMPT = """You are a clinical documentation specialist. Produce a comprehensive, exhaustive structured summary of this patient record (MRN: {mrn}).

Sections (extract ALL data present in the record for each section — search the ENTIRE record thoroughly):
1. Demographics (name, DOB, age, sex, gender, race/ethnicity, marital status, blood type, BMI, advance directives)
2. Chief Complaint & Admission Reason
3. History of Present Illness
4. Past Medical History (list ALL diagnoses)
5. Surgical History (list ALL procedures — look for tables labeled surgical history, past surgical history, prior procedures)
6. Family History (list ALL entries — look for tables with family member, condition, age columns)
7. Social History (living situation, marital status, transportation, smoking/alcohol/drug use with pack-years, occupation, functional status ECOG/Karnofsky, mental health screenings PHQ-9/GAD-7, financial situation, support system)
8. Allergies (list EVERY allergy with reaction type and severity — look for allergy tables/registries. This is PATIENT SAFETY CRITICAL — never skip)
9. Medications (TWO separate complete lists: a) ALL home/admission medications with dose/route/frequency b) ALL discharge medications with dose/route/frequency. Include PRN medications, supplements, devices like CPAP)
10. Review of Systems
11. Physical Exam (admission — all systems documented)
12. Vital Signs (table format — include ALL days documented, not just admission/discharge. Look for daily vital sign tables with BP, HR, Temp, RR, SpO2, weight, glucose)
13. Laboratory Results (list ALL labs with values, units, reference ranges, and flags. Include ALL timepoints documented — admission, serial, discharge. Look for lab tables with columns. Flag abnormals with [ABNORMAL], criticals with [CRITICAL])
14. Imaging & Diagnostics (with dates and key findings)
15. Hospital Course (day-by-day key events, interventions, transitions)
16. Procedures (with dates and findings)
17. Consultations
18. Discharge Summary & Disposition (include risk scores like LACE, readmission probability)
19. Patient Education
20. Outstanding Issues & Follow-up

STRICT GROUNDING RULES:
- Use ONLY information explicitly present in the record below. Never infer, estimate, or assume.
- SCAN THE ENTIRE RECORD for each section — data may appear in unexpected locations (e.g., allergies in nursing notes, surgical history in a procedures table, family history in an oncology section).
- Use exact values: lab numbers with units, medication dosages with route/frequency, exact dates.
- For every key finding, note the source section in parentheses (e.g., "(from Vital Signs, Day 3)").
- Flag critical/abnormal findings with [CRITICAL] or [ABNORMAL] markers.
- Include ALL entries in tables — do not truncate or summarize tables. If there are 12 surgical procedures, list all 12. If there are 8 allergies, list all 8. If there are 38 lab values, list all 38.
- When values changed over time, show the full trajectory with dates (not just admission → discharge).
- Never round, paraphrase loosely, or omit documented details.
- If a section truly has no data anywhere in the record, write "No data documented in record."

RECORD:

{record}"""

# ── Uploaded-document summarization (document_id-scoped) ─────────────────────
#
# Goal: reliably summarize ANY uploaded document regardless of length, with
# clinician-safe completeness for "can't miss" domains (allergies, meds,
# critical labs, key imaging impressions, procedures, disposition/follow-up).
#
# Approach:
# - Retrieve ALL chunks for the uploaded document_id (not MRN-scoped).
# - MAP: per-group structured extraction → strict JSON with citations to Source IDs
# - MERGE: deterministic merge + dedupe across groups
# - REDUCE: generate clinician-friendly summary from merged JSON, with citations
# - VERIFY: rule-based + LLM check for missing can't-miss categories; retry targeted extraction once
#

_DOC_SUMMARY_CACHE: OrderedDict[str, dict] = OrderedDict()  # document_id -> {summary, cached_at}
_DOC_SUMMARY_CACHE_LOCK = threading.Lock()
_DOC_SUMMARY_CACHE_MAX_SIZE = int(os.environ.get("MEDGRAPH_DOC_CACHE_MAX_SIZE", "100"))
_SUMMARY_PIPELINE_VERSION = "v3_fast_adaptive"


def _get_doc_reduce_llm(model_override: str = None):
    """Higher-accuracy model for document reduce/verification."""
    provider = get_active_llm_provider(getattr(TOOLS_CFG, "llm_provider", "openai"))
    default_model = "gpt-4.1" if provider == "openai" else "gemini-2.5-pro"
    preferred = os.environ.get("MEDGRAPH_DOCUMENT_SUMMARY_REDUCE_LLM", "").strip() or default_model
    model = model_override or preferred
    mode = _current_summary_mode()
    max_out = 4096 if mode == "thinking" else 2400
    timeout_s = 120 if mode == "thinking" else 90
    thinking_budget = 1024 if ("2.5-pro" in model) else 0
    if provider == "google" or model.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=get_google_api_key(),
            temperature=0.0,
            max_output_tokens=max_out,
            timeout=timeout_s,
            max_retries=2,
            thinking_budget=thinking_budget,
        )
        return model, llm
    return _get_summarization_llm(model_override=model)


def _get_doc_map_llm(model_override: str = None):
    """Fast extractor for map phase."""
    provider = get_active_llm_provider(getattr(TOOLS_CFG, "llm_provider", "openai"))
    default_model = "gpt-4.1-mini" if provider == "openai" else "gemini-2.5-flash"
    preferred = os.environ.get("MEDGRAPH_DOCUMENT_SUMMARY_MAP_LLM", "").strip() or default_model
    model = model_override or preferred
    mode = _current_summary_mode()
    max_out = 3072 if mode == "thinking" else 2200
    timeout_s = 120 if mode == "thinking" else 90
    thinking_budget = 1024 if ("2.5-pro" in model) else 0

    if provider == "google" or model.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=get_google_api_key(),
            temperature=0.0,
            max_output_tokens=max_out,
            timeout=timeout_s,
            max_retries=2,
            thinking_budget=thinking_budget,
        )
        return model, llm

    # Fallback to OpenAI if configured (keeps repo portable)
    if _is_reasoning_model(model):
        return model, ChatOpenAI(
            model=model,
            max_completion_tokens=8192,
            api_key=os.getenv("OPENAI_API_KEY"),
            request_timeout=120,
            max_retries=2,
        )
    return model, ChatOpenAI(
        model=model,
        temperature=0.0,
        max_tokens=4096,
        api_key=os.getenv("OPENAI_API_KEY"),
        request_timeout=120,
        max_retries=2,
    )


def _safe_json_loads(text: str) -> dict:
    """Best-effort JSON parse: trims fences, extracts first {...} block."""
    if not text:
        return {}
    t = text.strip()
    # Strip markdown fences if present
    t = _re.sub(r"^```(?:json)?\s*", "", t, flags=_re.IGNORECASE).strip()
    t = _re.sub(r"\s*```$", "", t).strip()
    # If there's leading commentary, extract first JSON object
    if not t.startswith("{"):
        m = _re.search(r"\{[\s\S]*\}", t)
        if m:
            t = m.group(0)
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _norm_key(s: str) -> str:
    s = (s or "").strip().lower()
    s = _re.sub(r"\s+", " ", s)
    return s


def _merge_lists_unique(items: list, key_fields: list[str]) -> list:
    """Merge a list of dict items using a composite normalized key."""
    out = []
    seen = set()
    for it in items or []:
        if not isinstance(it, dict):
            continue
        key_parts = []
        for f in key_fields:
            key_parts.append(_norm_key(str(it.get(f, ""))))
        key = "||".join(key_parts)
        if not key.strip("|"):
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def _merge_doc_extractions(extractions: list[dict]) -> dict:
    """Deterministically merge per-group extraction JSONs into one."""
    merged: dict = {
        "document": {},
        "patient": {},
        "allergies": [],
        "problems": [],
        "medications": [],
        "labs": [],
        "vitals": [],
        "imaging": [],
        "procedures": [],
        "hospital_course": [],
        "disposition_followup": [],
        "other_key_findings": [],
        "missing_or_unclear": [],
    }

    for ex in extractions or []:
        if not isinstance(ex, dict):
            continue
        merged["document"] = {**merged.get("document", {}), **(ex.get("document") or {})}
        merged["patient"] = {**merged.get("patient", {}), **(ex.get("patient") or {})}

        merged["allergies"].extend(ex.get("allergies") or [])
        merged["problems"].extend(ex.get("problems") or [])
        merged["medications"].extend(ex.get("medications") or [])
        merged["labs"].extend(ex.get("labs") or [])
        merged["vitals"].extend(ex.get("vitals") or [])
        merged["imaging"].extend(ex.get("imaging") or [])
        merged["procedures"].extend(ex.get("procedures") or [])
        merged["hospital_course"].extend(ex.get("hospital_course") or [])
        merged["disposition_followup"].extend(ex.get("disposition_followup") or [])
        merged["other_key_findings"].extend(ex.get("other_key_findings") or [])
        merged["missing_or_unclear"].extend(ex.get("missing_or_unclear") or [])

    # Deduplicate with conservative keys (avoid accidental drops)
    merged["allergies"] = _merge_lists_unique(merged["allergies"], ["substance", "reaction", "severity"])
    merged["problems"] = _merge_lists_unique(merged["problems"], ["problem", "status"])
    merged["medications"] = _merge_lists_unique(merged["medications"], ["name", "dose", "route", "frequency", "context"])
    merged["labs"] = _merge_lists_unique(merged["labs"], ["test", "value", "unit", "date_time"])
    merged["vitals"] = _merge_lists_unique(merged["vitals"], ["type", "value", "unit", "date_time"])
    merged["imaging"] = _merge_lists_unique(merged["imaging"], ["study", "date", "impression"])
    merged["procedures"] = _merge_lists_unique(merged["procedures"], ["procedure", "date", "result"])
    merged["hospital_course"] = _merge_lists_unique(merged["hospital_course"], ["date", "event"])
    merged["disposition_followup"] = _merge_lists_unique(merged["disposition_followup"], ["item", "date", "details"])
    merged["other_key_findings"] = _merge_lists_unique(merged["other_key_findings"], ["finding"])
    merged["missing_or_unclear"] = _merge_lists_unique(merged["missing_or_unclear"], ["topic", "note"])

    return merged


def _is_sparse_merged_payload(merged: dict) -> bool:
    """Detect pathological under-extraction."""
    if not isinstance(merged, dict):
        return True
    signal_counts = 0
    for key in ("problems", "allergies", "medications", "labs", "imaging", "procedures", "hospital_course"):
        signal_counts += len(merged.get(key) or [])
    patient = merged.get("patient") or {}
    has_patient_anchor = bool((patient.get("name") or "").strip() or (patient.get("mrn") or "").strip())
    return signal_counts < 3 and not has_patient_anchor


_DOC_MAP_JSON_PROMPT = """You are a clinical data extraction engine.

Extract clinically relevant facts from the provided SOURCES (an excerpt of an uploaded medical document) into STRICT JSON.

CRITICAL RULES:
- Output MUST be valid JSON (no markdown, no commentary, no trailing commas).
- Only extract facts explicitly present in the sources. If uncertain, put it in missing_or_unclear.
- Be exhaustive for patient-safety domains: allergies, medications, anticoagulants/insulin/opioids, critical labs, imaging impressions, procedures, and discharge/follow-up.
- Explicitly capture high-risk context often missed in physician summaries: prior CAD/MI/PCI/CABG history, OSA/CKD/COPD/diabetes comorbid burden, ECG findings, echocardiogram metrics, catheterization hemodynamics, recurrence events, treatment response, and discharge-limiting complications.
- Preserve exact numbers, units, dates, and medication dose/route/frequency.
- Every item MUST include a `citations` array listing the Source IDs that support it (e.g., ["S3","S4"]). If unsure, omit the item.

JSON SCHEMA (top-level keys required; use [] when none found):
{{
  "document": {{"document_id": "{document_id}", "doc_type": "", "encounter_date": ""}},
  "patient": {{"name": "", "mrn": "", "dob": "", "sex": ""}},
  "allergies": [{{"substance": "", "reaction": "", "severity": "", "citations": ["S1"]}}],
  "problems": [{{"problem": "", "status": "", "onset_or_date": "", "citations": ["S1"]}}],
  "medications": [{{"name": "", "dose": "", "route": "", "frequency": "", "context": "home|inpatient|discharge|unknown", "start_stop_change": "", "citations": ["S2"]}}],
  "labs": [{{"test": "", "value": "", "unit": "", "ref_range": "", "flag": "ABNORMAL|CRITICAL|", "date_time": "", "citations": ["S5"]}}],
  "vitals": [{{"type": "BP|HR|Temp|RR|SpO2|Weight|Other", "value": "", "unit": "", "date_time": "", "citations": ["S6"]}}],
  "imaging": [{{"study": "", "date": "", "key_findings": "", "impression": "", "citations": ["S7"]}}],
  "procedures": [{{"procedure": "", "date": "", "result": "", "complications": "", "citations": ["S8"]}}],
  "hospital_course": [{{"date": "", "event": "", "citations": ["S9"]}}],
  "disposition_followup": [{{"item": "", "date": "", "details": "", "citations": ["S10"]}}],
  "other_key_findings": [{{"finding": "", "citations": ["S11"]}}],
  "missing_or_unclear": [{{"topic": "", "note": "", "citations": ["S12"]}}]
}}

SOURCES:
{sources}
"""


_DOC_REDUCE_PROMPT = """You are MedGraph AI generating a clinician-facing summary of an uploaded medical document.

Use ONLY the extracted JSON facts below. Do not invent. If a key domain is missing, explicitly say "Not documented in provided document."

OUTPUT REQUIREMENTS:
- Produce a concise but comprehensive clinical summary suitable for an EHR sidebar.
- Choose section headers ADAPTIVELY based on what is present in the document; do not force empty sections.
- Never output repetitive "Not documented" lines for many categories. Mention missing data only when safety-critical (e.g., allergies unknown).
- Prefer compact bullets and short paragraphs; prioritize clinically decisive facts and management impact.
- Include clinically decisive detail when present: EF trajectory, ECG findings, cath/hemodynamic numbers, significant procedures (e.g., thoracentesis), rhythm events/treatments, renal complications that changed discharge plan, final discharge labs/vitals, and scheduled follow-up services.
- Cite sources at the end of claims/sections as [S#] (e.g., [S3][S7]).

EXTRACTED_JSON:
{json_blob}
"""

_DOC_REDUCE_PROMPT_THINKING_APPEND = """

THINKING MODE REQUIREMENTS (higher depth):
- Build a decision-grade physician summary with explicit chronology (admission -> major events -> discharge).
- Include nuanced causal links when documented (e.g., recurrence trigger, treatment response, dose changes due to complications).
- Explicitly include complete allergy specificity and major background context impacting management.
- Include final discharge objective status (key vitals/labs/functional disposition) when present.
- Keep concise but do not compress away clinically actionable details.
"""

_DOC_AUGMENT_JSON_PROMPT = """You are a senior clinical abstraction QA pass.

You are given:
1) EXISTING_EXTRACTED_JSON produced from the full document.
2) SOURCES likely to contain nuanced/high-impact details.

Task:
- Return STRICT JSON in the same schema, containing ONLY additional facts that are missing or more precise than EXISTING_EXTRACTED_JSON.
- Focus on physician-critical details often omitted: full allergy specificity, major comorbid/cardiac history, ECG/echo/cath numbers, procedure outputs, complications and their management, discharge-limiting events, and final discharge status/follow-up.
- Every item must include citations.
- Do not repeat existing facts unless adding precision (e.g., exact value/date/treatment).

JSON schema: same as prior extraction schema.

EXISTING_EXTRACTED_JSON:
{existing_json}

SOURCES:
{sources}
"""


def _doc_needs_keyword_retry(raw_text: str, merged: dict) -> list[str]:
    """Heuristic: detect missing can't-miss domains when source contains signals."""
    raw = (raw_text or "").lower()
    missing = []
    if ("allerg" in raw or "nka" in raw or "no known allergy" in raw) and not (merged.get("allergies") or []):
        missing.append("allergies")
    if ("medication" in raw or "rx" in raw or "sig" in raw or "discharge medications" in raw) and not (merged.get("medications") or []):
        missing.append("medications")
    if ("lab" in raw or "cbc" in raw or "bmp" in raw or "cmp" in raw) and not (merged.get("labs") or []):
        missing.append("labs")
    if ("impression" in raw or "ct" in raw or "mri" in raw or "x-ray" in raw or "ultrasound" in raw) and not (merged.get("imaging") or []):
        missing.append("imaging")
    if ("ecg" in raw or "ekg" in raw or "echo" in raw or "ejection fraction" in raw or "rvsp" in raw) and not (merged.get("imaging") or []):
        missing.append("imaging")
    if ("procedure" in raw or "operative" in raw or "surgery" in raw) and not (merged.get("procedures") or []):
        missing.append("procedures")
    if ("catheter" in raw or "thoracentesis" in raw or "pcwp" in raw or "cardiac index" in raw or "pvr" in raw) and not (merged.get("procedures") or []):
        missing.append("procedures")
    if ("discharge" in raw or "follow-up" in raw or "appointment" in raw) and not (merged.get("disposition_followup") or []):
        missing.append("disposition_followup")
    return missing


def _build_sources_for_groups(chunks: list[tuple[int, str, dict]], max_group_chars: int = 22_000) -> list[str]:
    """Create grouped source strings with globally stable Source IDs.

    Args:
        chunks: list of (global_source_index, text, metadata) tuples.
                global_source_index must be stable across the whole document (1..N).
        max_group_chars: approximate per-group context budget.
    """
    groups: list[list[tuple[int, str, dict]]] = []
    cur: list[tuple[int, str, dict]] = []
    cur_chars = 0
    for global_idx, txt, meta in chunks:
        t = txt or ""
        if cur and cur_chars + len(t) > max_group_chars:
            groups.append(cur)
            cur = []
            cur_chars = 0
        cur.append((int(global_idx), t, meta))
        cur_chars += len(t)
    if cur:
        groups.append(cur)

    rendered = []
    for g in groups:
        parts = []
        for global_idx, txt, meta in g:
            sid = f"S{global_idx}"
            sec = (meta or {}).get("section_title", "Section")
            page = (meta or {}).get("page_number", "?")
            parts.append(f"[{sid} | Section: {sec} | Page {page}]\n{txt}")
        rendered.append("\n\n---\n\n".join(parts))
    return rendered


def _run_structured_summary_pipeline(entity_id: str, indexed_chunks: list[tuple[int, str, dict]]) -> tuple[str, str]:
    """Unified structured summary pipeline used by both patient and document summaries.

    Returns:
        (summary_text, model_name)
    """
    if not indexed_chunks:
        return "No indexed chunks available for summarization.", "none"

    mode = _current_summary_mode()
    provider = get_active_llm_provider(getattr(TOOLS_CFG, "llm_provider", "openai"))
    default_map = "gpt-4.1-mini" if provider == "openai" else "gemini-2.5-flash"
    default_reduce = "gpt-4.1-mini" if provider == "openai" else "gemini-2.5-flash"
    thinking_model = "gpt-4.1" if provider == "openai" else "gemini-2.5-pro"
    map_primary = thinking_model if mode == "thinking" else os.environ.get("MEDGRAPH_DOCUMENT_SUMMARY_MAP_LLM", default_map)
    reduce_primary = thinking_model if mode == "thinking" else default_reduce
    is_fast_mode = mode != "thinking"
    max_group_chars = 32000 if is_fast_mode else 22000
    sources_groups = _build_sources_for_groups(indexed_chunks, max_group_chars=max_group_chars)

    if is_fast_mode:
        # Fast path: single-pass adaptive synthesis directly from sources.
        # This avoids fragile, token-heavy strict JSON extraction in normal mode.
        selected_groups = sources_groups[:2]
        if len(sources_groups) > 2:
            high_signal = []
            for global_idx, txt, meta in indexed_chunks:
                low = (txt or "").lower()
                if any(k in low for k in [
                    "allerg", "medication", "discharge", "follow-up", "echo", "ecg", "ekg",
                    "catheter", "thoracentesis", "bnp", "creatinine", "proteinuria", "afib", "rvr"
                ]):
                    high_signal.append((global_idx, txt, meta))
            if high_signal:
                selected_groups.extend(_build_sources_for_groups(high_signal[:30], max_group_chars=18000)[:1])
        fast_sources = "\n\n====\n\n".join(selected_groups)
        fast_prompt = f"""You are MedGraph AI.

Create a fast, clinically useful summary for EHR use from SOURCES only.

Rules:
- Use adaptive headings only for data that exists; do not include empty sections.
- Prioritize patient-safety and decision-making details (diagnoses, allergies, meds, critical labs/imaging/procedures, major complications, discharge status/follow-up).
- Keep concise (about 250-500 words unless clearly needed).
- Use Source IDs for support (e.g., [S12], [S4][S9]).
- If allergies are not present in sources, say "Allergies: not found in provided excerpts."
- Do not invent facts.

SOURCES:
{fast_sources}
"""
        reduce_chain = build_model_chain(
            reduce_primary,
            ["gemini-2.5-flash-lite"],
        )
        model_name, summary = _llm_invoke_with_fallback(
            _get_doc_reduce_llm,
            fast_prompt,
            max_retries=2,
            model_chain=reduce_chain,
        )
        return summary, model_name

    def _extract_group(sources_text: str) -> dict:
        base_prompt = _DOC_MAP_JSON_PROMPT.format(document_id=entity_id, sources=sources_text)
        model_chain = build_model_chain(
            map_primary,
            ["gemini-2.5-flash-lite"],
        )
        _, text = _llm_invoke_with_fallback(
            _get_doc_map_llm,
            base_prompt,
            max_retries=2,
            model_chain=model_chain,
        )
        obj = _safe_json_loads(text)
        if obj:
            return obj
        strict_prompt = (
            base_prompt
            + "\n\nIMPORTANT: Your previous output was not valid JSON. "
              "Return ONLY valid JSON matching the schema."
        )
        _, text2 = _llm_invoke_with_fallback(
            _get_doc_map_llm,
            strict_prompt,
            max_retries=1,
            model_chain=model_chain,
        )
        return _safe_json_loads(text2)

    with _futures.ThreadPoolExecutor(max_workers=min(len(sources_groups), 3 if is_fast_mode else 4)) as pool:
        extractions = list(pool.map(_extract_group, sources_groups))
    merged = _merge_doc_extractions(extractions)
    if _is_sparse_merged_payload(merged):
        # Emergency fallback: force extraction on condensed high-signal chunks.
        focus_chunks = []
        for gi, txt, meta in indexed_chunks:
            low = (txt or "").lower()
            if any(k in low for k in [
                "allerg", "medication", "discharge", "history", "assessment", "plan",
                "echo", "ecg", "ekg", "procedure", "catheter", "lab", "vital",
            ]):
                focus_chunks.append((gi, txt, meta))
        if not focus_chunks:
            focus_chunks = indexed_chunks[: min(18, len(indexed_chunks))]
        focus_groups = _build_sources_for_groups(focus_chunks, max_group_chars=18000)
        with _futures.ThreadPoolExecutor(max_workers=min(len(focus_groups), 3)) as pool:
            focus_extractions = list(pool.map(_extract_group, focus_groups))
        merged = _merge_doc_extractions([merged, _merge_doc_extractions(focus_extractions)])

    # Build domain-signal text from all chunks (truncated per chunk for memory safety)
    signal_text = "\n".join((txt[:600] if txt else "") for _, txt, _ in indexed_chunks)
    missing_domains = _doc_needs_keyword_retry(signal_text, merged)

    if missing_domains:
        keyword_map = {
            "allergies": ["allerg", "nka", "reaction", "anaphyl", "urticaria", "angioedema", "contrast"],
            "medications": ["med", "medication", "rx", "sig", "dose", "tablet", "capsule", "discharge rx"],
            "labs": ["lab", "cbc", "bmp", "cmp", "mg/dl", "mmol", "wbc", "hgb", "plt", "bnp", "creatinine"],
            "imaging": ["impression", "ct", "mri", "x-ray", "ultrasound", "echo", "ecg", "ekg", "rvsp", "ef"],
            "procedures": ["procedure", "operative", "surgery", "biopsy", "endoscopy", "thoracentesis", "catheter", "pcwp", "pvr", "cardiac index"],
            "disposition_followup": ["discharge", "follow-up", "appointment", "return to", "home health", "pt", "ot", "icd"],
        }
        want = set(missing_domains)
        filtered = []
        for global_idx, txt, meta in indexed_chunks:
            low = (txt or "").lower()
            if any(any(k in low for k in keyword_map.get(dom, [])) for dom in want):
                filtered.append((global_idx, txt, meta))
        if filtered:
            retry_groups = _build_sources_for_groups(filtered, max_group_chars=22_000)
            with _futures.ThreadPoolExecutor(max_workers=min(len(retry_groups), 3)) as pool:
                retry_extractions = list(pool.map(_extract_group, retry_groups))
            merged = _merge_doc_extractions([merged, _merge_doc_extractions(retry_extractions)])

    # Augmentation pass: capture nuanced high-impact details even if domain isn't empty
    # Skip in fast mode for latency.
    if not is_fast_mode:
        augment_keywords = [
            "allerg", "anaphyl", "angioedema", "contrast", "premedication",
            "nstemi", "cad", "pci", "stent", "lad", "mi",
            "ecg", "ekg", "echo", "ejection fraction", "rvsp", "q wave", "st depression", "qtc",
            "thoracentesis", "catheter", "right-heart", "pcwp", "pvr", "cardiac index", "ra ", "pa ",
            "afib", "atrial fibrillation", "rvr", "magnesium", "diltiazem",
            "proteinuria", "nephrology", "creatinine", "discharge delayed", "arni",
            "home health", "follow-up", "repeat echo", "icd", "final discharge", "discharge vitals", "discharge labs",
        ]
        augment_chunks = []
        for global_idx, txt, meta in indexed_chunks:
            low = (txt or "").lower()
            if any(k in low for k in augment_keywords):
                augment_chunks.append((global_idx, txt, meta))
        if augment_chunks:
            augment_groups = _build_sources_for_groups(augment_chunks, max_group_chars=20_000)
            base_existing = json.dumps(merged, ensure_ascii=False)

            def _augment_group(sources_text: str) -> dict:
                prompt = _DOC_AUGMENT_JSON_PROMPT.format(existing_json=base_existing, sources=sources_text)
                model_chain = build_model_chain(
                    map_primary,
                    ["gemini-2.5-flash-lite"],
                )
                _, text = _llm_invoke_with_fallback(
                    _get_doc_map_llm,
                    prompt,
                    max_retries=2,
                    model_chain=model_chain,
                )
                return _safe_json_loads(text)

            with _futures.ThreadPoolExecutor(max_workers=min(len(augment_groups), 3)) as pool:
                augment_extractions = list(pool.map(_augment_group, augment_groups))
            merged = _merge_doc_extractions([merged, _merge_doc_extractions(augment_extractions)])

    json_blob = json.dumps(merged, ensure_ascii=False)
    reduce_prompt = _DOC_REDUCE_PROMPT.format(json_blob=json_blob)
    if is_fast_mode:
        reduce_prompt += (
            "\n\nFAST MODE REQUIREMENTS:\n"
            "- Target 250-500 words total unless the document is highly complex.\n"
            "- Keep only high-yield sections present in data.\n"
            "- Prioritize key diagnoses, allergies, major medications, critical labs/imaging, procedures, and follow-up.\n"
            "- Avoid exhaustive enumeration of every minor value.\n"
        )
    else:
        reduce_prompt += _DOC_REDUCE_PROMPT_THINKING_APPEND
    reduce_chain = build_model_chain(
        reduce_primary,
        ["gemini-2.5-flash"],
    )
    model_name, summary = _llm_invoke_with_fallback(
        _get_doc_reduce_llm,
        reduce_prompt,
        max_retries=2,
        model_chain=reduce_chain,
    )
    return summary, model_name


def _supplement_from_targeted_searches(rag: "ClinicalNotesRAGTool", patient_mrn: str) -> list[tuple[str, dict]]:
    """Get supplemental chunks from targeted semantic search for high-risk domains.

    This guards against occasional metadata extraction gaps where a critical
    section (e.g., allergy table) may not be included in strict MRN-filtered
    chunk retrieval.
    """
    queries = [
        f"{patient_mrn} allergies allergy reaction anaphylaxis contrast premedication",
        f"{patient_mrn} echocardiogram ECG EKG right heart catheterization thoracentesis hemodynamics PCWP PVR cardiac index",
        f"{patient_mrn} discharge labs final vitals follow-up home health nephrology proteinuria AFib recurrence magnesium diltiazem",
    ]
    out: list[tuple[str, dict]] = []
    seen = set()
    for q in queries:
        try:
            text = rag.search(q)
        except Exception:
            continue
        if not text or text.startswith("No relevant"):
            continue
        parts = text.split("\n\n---\n\n")
        for p in parts:
            # Strip leading source metadata header line if present.
            body = _re.sub(r"^\[Source[^\]]+\]\n?", "", p.strip())
            key = hashlib.md5(body.encode("utf-8")).hexdigest()
            if not body or key in seen:
                continue
            seen.add(key)
            out.append((body, {"section_title": "Supplemental Targeted Retrieval", "page_number": "?"}))
    return out


# ── Map-reduce parallel summarization ────────────────────────────────────────
# For large records (>30K chars), split → parallel extract → merge.
# This cuts 60s+ single-pass summarization to ~15-20s via concurrent API calls.
_MAP_REDUCE_THRESHOLD = 30_000  # chars — records above this use map-reduce

_MAP_EXTRACT_PROMPT = """You are a clinical data extractor. Condense this section of patient record (MRN: {mrn}) into concise bullet points.

RULES:
- One bullet per fact. Combine related details on a single line.
- Keep exact numbers, dates, units, dosages — but drop filler words and narrative prose.
- Use abbreviations: pt, dx, hx, rx, prn, q4h, PO, IV, etc.
- Skip section headers, formatting, and repetitive text.
- NEVER drop table data. If this section contains tables (allergies, surgical history, family history, vitals, labs), extract EVERY row — do not summarize or skip any entries.
- PATIENT SAFETY: Allergy lists, medication lists, and surgical history must be extracted COMPLETELY — every single entry.
- Target ≤50% of original length (but preserve all discrete data points).

EXAMPLE OUTPUT:
• DOB 1958-11-03, Male, MRN-2026-004782
• Admitted 2026-01-15 for acute decompensated HF (NYHA III→IV)
• ALLERGIES: Penicillin→Anaphylaxis, Sulfa→Rash, Latex→Contact dermatitis
• BNP 1842 pg/mL [ABNORMAL], Cr 1.8 mg/dL, K 3.2 mEq/L [LOW]
• Furosemide 40mg IV q12h → transitioned PO 80mg daily day 3
• Echo 2026-01-16: EF 25%, severe MR, dilated LV

RECORD SECTION:

{section}"""


def _get_map_llm(model_override: str = None):
    """Lighter LLM for the map phase — lower max_tokens to enforce brevity.
    Timeout set to 90s to give Gemini extra server-side processing time
    and avoid 504 DEADLINE_EXCEEDED on large chunks."""
    base_model = getattr(TOOLS_CFG, "clinical_notes_rag_summarization_llm",
                         TOOLS_CFG.clinical_notes_rag_llm)
    model = model_override or base_model
    provider = get_active_llm_provider(getattr(TOOLS_CFG, "llm_provider", "openai"))

    if provider == "google" or model.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=get_google_api_key(),
            temperature=0.0,
            max_output_tokens=2048,     # reduced from 4000 to enforce conciseness
            timeout=90,                 # increased from 60 to avoid Gemini 504s
            max_retries=2,
            thinking_budget=0,
        )
        return model, llm
    if _is_reasoning_model(model):
        return model, ChatOpenAI(
            model=model,
            max_completion_tokens=4096,
            api_key=os.getenv("OPENAI_API_KEY"),
            request_timeout=90,
            max_retries=2,
        )
    return model, ChatOpenAI(
        model=model,
        temperature=0.0,
        max_tokens=2048,
        api_key=os.getenv("OPENAI_API_KEY"),
        request_timeout=90,
        max_retries=2,
    )


def _map_reduce_summarize(patient_mrn: str, full_record: str) -> tuple:
    """Map-reduce parallel summarization.

    Strategy:
      1. Split record into 6-8 groups at paragraph boundaries (small enough
         to avoid Gemini 504 DEADLINE_EXCEEDED on any single call)
      2. MAP: Condense each group to concise bullet points (parallel, capped tokens)
      3. REDUCE: Synthesize bullets into final structured summary

    Target: map produces ~30-40% of original volume so reduce is fast.

    Returns:
        (summary_text, model_name)
    """
    primary_model = getattr(TOOLS_CFG, "clinical_notes_rag_summarization_llm",
                            TOOLS_CFG.clinical_notes_rag_llm)
    model_chain = build_model_chain(
        primary_model,
        getattr(TOOLS_CFG, "clinical_notes_rag_summarization_fallback_llms", []),
    )
    model_name = primary_model

    # Split record into sections at paragraph boundaries
    raw_sections = full_record.split("\n\n")
    sections = [s.strip() for s in raw_sections if s.strip()]

    # Use 6-8 groups (more groups = smaller per-call token count = avoids Gemini 504s)
    # Previous: 2-4 groups caused ~7-10K chars/group which overwhelmed Gemini's server
    NUM_GROUPS = min(8, max(3, len(sections) // 4))
    total_chars = sum(len(s) for s in sections)
    target_group_size = total_chars // NUM_GROUPS

    groups = []
    current_group = []
    current_size = 0
    for sec in sections:
        current_group.append(sec)
        current_size += len(sec)
        if current_size >= target_group_size and len(groups) < NUM_GROUPS - 1:
            groups.append("\n\n".join(current_group))
            current_group = []
            current_size = 0
    if current_group:
        groups.append("\n\n".join(current_group))

    logger.info("Map-reduce: %d chars → %d groups (target %d chars/group)",
                total_chars, len(groups), target_group_size)

    # ── MAP PHASE: Condense each group to bullet points in parallel ──
    def extract_facts(group_text: str) -> str:
        try:
            prompt = _MAP_EXTRACT_PROMPT.format(mrn=patient_mrn, section=group_text)
            _, text = _llm_invoke_with_fallback(
                _get_map_llm,
                prompt,
                max_retries=2,
                model_chain=model_chain,
            )
            return text
        except Exception as e:
            logger.warning("Map phase failed for a group: %s", str(e)[:200])
            # Fallback: return raw text (heavily truncated) so reduce still has something
            return group_text[:2000]

    t_map_start = _time.perf_counter()
    with _futures.ThreadPoolExecutor(max_workers=min(len(groups), 4)) as pool:
        partial_summaries = list(pool.map(extract_facts, groups))
    t_map = _time.perf_counter() - t_map_start

    # ── REDUCE PHASE: Merge bullets into structured summary ──
    merged_facts = "\n\n".join(partial_summaries)
    merged_chars = len(merged_facts)
    compression = (1 - merged_chars / total_chars) * 100
    logger.info("Map phase: %.1fs (%d groups), merged facts: %s chars (%.0f%% compression)",
                t_map, len(groups), f"{merged_chars:,}", compression)

    t_reduce_start = _time.perf_counter()
    reduce_prompt = _SUMMARY_PROMPT.format(mrn=patient_mrn, record=merged_facts)
    model_name, final_summary = _llm_invoke_with_fallback(
        _get_summarization_llm,
        reduce_prompt,
        model_chain=model_chain,
    )
    t_reduce = _time.perf_counter() - t_reduce_start

    logger.info("Reduce phase: %.1fs, output: %s chars", t_reduce, f"{len(final_summary):,}")
    logger.info("Map-reduce total: %.1fs (map=%.1fs + reduce=%.1fs)",
                t_map + t_reduce, t_map, t_reduce)

    return final_summary, model_name


# ── Tool definitions ─────────────────────────────────────────────────────────

# Query rewriting removed — saves ~2s per request.
# The hybrid BM25+dense+RRF search already handles synonyms and abbreviations well.
# Evaluation showed 92.5% accuracy without query rewriting being a bottleneck.


@tool
def lookup_clinical_notes(query: str, document_id: str = "") -> str:
    """Search among clinical notes to find specific information. Use this for ANY targeted question about patient data including: date of birth, demographics, allergies, medications, labs, vitals, procedures, diagnoses, imaging, or any other specific clinical detail. Input should be the clinical question. Optionally pass document_id to restrict search to a specific uploaded document (get IDs from list_uploaded_documents)."""
    try:
        import signal
        import threading
        
        rag = get_rag_tool_instance()
        result = [None]
        exception = [None]
        
        # Use document_id filter if provided (strip whitespace, treat empty as None)
        doc_filter = document_id.strip() if document_id else None
        
        def _search_with_timeout():
            try:
                result[0] = rag.search(query, document_id=doc_filter)
            except Exception as e:
                exception[0] = e
        
        # Execute search with 15 second timeout
        thread = threading.Thread(target=_search_with_timeout, daemon=False)
        thread.start()
        thread.join(timeout=15)
        
        if thread.is_alive():
            return f"Clinical notes search exceeded 15 second timeout. Please refine your query to be more specific."
        
        if exception[0]:
            return f"Error searching clinical notes: {exception[0]}. Ensure the vector database is initialized."
        
        return result[0] or "No results found."
    except Exception as e:
        return f"Error searching clinical notes: {e}. Ensure the vector database is initialized."


@tool
def lookup_patient_orders(patient_identifier: str) -> str:
    """Retrieve ALL orders for a patient — laboratory orders (with CPT/LOINC codes), medication orders (with RxNorm codes), procedure orders (with CPT/SNOMED codes), referrals/consultations, discharge prescriptions, and care directives. Input is the patient name (e.g. 'Robert Whitfield') or MRN (e.g. 'MRN-2026-004782'). Use this when asked for 'all orders', 'complete order list', 'what was ordered', or any comprehensive orders query."""
    try:
        import threading
        
        rag = get_rag_tool_instance()
        t_start = _time.perf_counter()
        
        result = [None]
        exception = [None]
        
        def _fetch_orders():
            try:
                # Resolve identifier to MRN
                patient_mrn = rag.resolve_to_mrn(patient_identifier)
                if not patient_mrn:
                    exception[0] = f"Could not find a patient matching '{patient_identifier}'. Use list_available_patients to see who is in the database."
                    return
                logger.info("[Orders] Resolved '%s' → %s", patient_identifier, patient_mrn)
                
                # Fetch all order-related chunks (section-filtered)
                result[0] = rag.get_order_chunks_for_patient(patient_mrn)
            except Exception as e:
                exception[0] = str(e)
        
        # Execute with 20 second timeout
        thread = threading.Thread(target=_fetch_orders, daemon=False)
        thread.start()
        thread.join(timeout=20)
        
        if thread.is_alive():
            return "Order retrieval exceeded 20 second timeout. Try querying a single order type instead."
        
        elapsed = _time.perf_counter() - t_start
        logger.info("[Orders] Total retrieval: %.2fs", elapsed)
        
        if exception[0]:
            return f"Error retrieving orders: {exception[0]}"
        
        return result[0] or "No orders found."
    except Exception as e:
        return f"Error retrieving orders: {e}"


@tool
def summarize_patient_record(patient_identifier: str) -> str:
    """Generate a comprehensive clinical summary for a patient. Input can be a patient name (e.g. 'Robert Whitfield') or MRN (e.g. 'MRN-2026-004782'). This retrieves the ENTIRE patient record and produces a thorough summary covering all clinical domains. Use this when asked to summarize, review, or give an overview of a patient's record."""
    try:
        rag = get_rag_tool_instance()
        t_start = _time.perf_counter()

        # ── Resolve identifier to MRN ─────────────────────────────
        patient_mrn = rag.resolve_to_mrn(patient_identifier)
        if not patient_mrn:
            return (f"Could not find a patient matching '{patient_identifier}'. "
                    f"Use list_available_patients to see who is in the database.")
        logger.info("Resolved '%s' → %s", patient_identifier, patient_mrn)
        req_mode = _current_summary_mode()
        bypass_cache = req_mode == "thinking"

        # ── Layer 1: In-memory LRU cache (instant, cross-session) ─
        with _memory_cache_lock:
            if not bypass_cache and patient_mrn in _memory_cache:
                entry = _memory_cache[patient_mrn]
                cache_version = entry.get("pipeline_version", "legacy")
                age_hours = (_dt.now() - entry["cached_at"]).total_seconds() / 3600
                if age_hours <= _CACHE_TTL_HOURS and cache_version == _SUMMARY_PIPELINE_VERSION:
                    _memory_cache.move_to_end(patient_mrn)  # mark as recently used
                    elapsed = _time.perf_counter() - t_start
                    logger.info("MEMORY CACHE HIT for %s (%.3fs, %.1fh old)", patient_mrn, elapsed, age_hours)
                    _record_cache_event("hit", "memory")
                    return entry["summary"]
                else:
                    logger.info("Memory cache EXPIRED/STALE for %s (%.1fh old, v=%s)", patient_mrn, age_hours, cache_version)
                    del _memory_cache[patient_mrn]

        # ── Layer 2: Disk cache (survives restarts, cross-session) ─
        cache_dir = _Path(here("data")) / "summary_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = hashlib.sha256(patient_mrn.encode()).hexdigest()[:16]
        cache_file = cache_dir / f"{cache_key}.json"

        if not bypass_cache and cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                generated_at = cached.get("generated_at", "")
                cache_time = _dt.strptime(generated_at, "%Y-%m-%d %H:%M:%S")
                age_hours = (_dt.now() - cache_time).total_seconds() / 3600
                cache_version = cached.get("pipeline_version", "legacy")

                if age_hours <= _CACHE_TTL_HOURS and cache_version == _SUMMARY_PIPELINE_VERSION:
                    # Valid cache — promote to memory cache (with LRU eviction) and return
                    summary = cached["summary"]
                    with _memory_cache_lock:
                        _memory_cache[patient_mrn] = {
                            "summary": summary,
                            "cached_at": cache_time,
                            "pipeline_version": _SUMMARY_PIPELINE_VERSION,
                        }
                        _memory_cache.move_to_end(patient_mrn)
                        while len(_memory_cache) > _CACHE_MAX_SIZE:
                            evicted_key, _ = _memory_cache.popitem(last=False)
                            logger.info("LRU eviction: removed %s from memory cache (size=%d)", evicted_key, len(_memory_cache))
                    elapsed = _time.perf_counter() - t_start
                    logger.info("DISK CACHE HIT for %s (%.3fs, %.1fh old)", patient_mrn, elapsed, age_hours)
                    _record_cache_event("hit", "disk")
                    return summary
                else:
                    logger.info("Disk cache EXPIRED for %s (%.1fh old), deleting", patient_mrn, age_hours)
                    cache_file.unlink(missing_ok=True)
            except (ValueError, TypeError, json.JSONDecodeError) as e:
                logger.warning("Disk cache unreadable for %s: %s, regenerating", patient_mrn, e)
                cache_file.unlink(missing_ok=True)

        # ── Cache MISS — generate fresh summary ───────────────────
        logger.info("Cache MISS for %s — generating summary... (mode=%s)", patient_mrn, req_mode)
        _record_cache_event("miss")

        # Retrieve chunks with metadata and run unified structured pipeline
        patient_chunks = rag.get_all_chunks_for_patient_structured(patient_mrn)
        if not patient_chunks:
            return f"No records found for patient MRN: {patient_mrn}"
        # Add targeted-retrieval supplements for high-risk domains that can be
        # underrepresented if metadata extraction misses a section.
        patient_chunks.extend(_supplement_from_targeted_searches(rag, patient_mrn))
        indexed_chunks: list[tuple[int, str, dict]] = [(i, txt, meta) for i, (txt, meta) in enumerate(patient_chunks, 1)]

        t_llm_start = _time.perf_counter()
        summary, model_name = _run_structured_summary_pipeline(patient_mrn, indexed_chunks)
        t_llm_end = _time.perf_counter()
        logger.info("Structured pipeline LLM: %.1fs, output=%s chars (model=%s)",
                    t_llm_end - t_llm_start, f"{len(summary):,}", model_name)

        t_total = _time.perf_counter() - t_start
        logger.info("Total summarization: %.1fs", t_total)

        # ── Write to both cache layers ────────────────────────────
        now = _dt.now()

        if not bypass_cache:
            # Memory cache (LRU eviction when full)
            with _memory_cache_lock:
                _memory_cache[patient_mrn] = {
                    "summary": summary,
                    "cached_at": now,
                    "pipeline_version": _SUMMARY_PIPELINE_VERSION,
                }
                _memory_cache.move_to_end(patient_mrn)
                # Evict oldest entries if over max size
                while len(_memory_cache) > _CACHE_MAX_SIZE:
                    evicted_key, _ = _memory_cache.popitem(last=False)
                    logger.info("LRU eviction: removed %s from memory cache (size=%d)", evicted_key, len(_memory_cache))

            # Disk cache (TTL-only, no fragile chunk_count validation)
            try:
                cache_file.write_text(json.dumps({
                    "mrn": patient_mrn,
                    "summary": summary,
                    "model": model_name,
                    "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "pipeline_version": _SUMMARY_PIPELINE_VERSION,
                }, ensure_ascii=False), encoding="utf-8")
                logger.info("Cached → %s (memory + disk)", cache_file.name)
            except Exception as ce:
                logger.warning("Disk cache write failed (non-critical): %s", ce)

        return summary

    except Exception as e:
        return f"Error generating summary: {e}"


@tool
def list_available_patients() -> str:
    """List all patients available in the clinical notes database. Returns their MRN (Medical Record Number), name, and source document. Use this to discover which patients are in the system before searching or summarizing."""
    try:
        rag = get_rag_tool_instance()
        return rag.list_patients()
    except Exception as e:
        return f"Error listing patients: {e}"


@tool
def list_uploaded_documents() -> str:
    """List all documents that have been uploaded to the system. Returns document IDs, filenames, processing status, and upload timestamps. Use this to see which documents are available for querying."""
    try:
        from pyprojroot import here
        meta_path = str(here("data/uploads/documents.json"))
        if not os.path.exists(meta_path):
            return "No documents have been uploaded yet."
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if not meta:
            return "No documents have been uploaded yet."
        lines = ["Uploaded Documents:\n"]
        for doc_id, info in meta.items():
            status = info.get("status", "unknown")
            name = info.get("original_filename", "unknown")
            uploaded = info.get("uploaded_at", "unknown")
            chunks = info.get("chunks", "N/A")
            lines.append(f"  • {name} (ID: {doc_id})")
            lines.append(f"    Status: {status} | Chunks: {chunks} | Uploaded: {uploaded}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing documents: {e}"


@tool
def summarize_uploaded_document(document_id: str) -> str:
    """Generate a clinically grounded summary of a specific uploaded document.

    Input: document_id (UUID) from list_uploaded_documents().
    Output: clinician-facing summary with citations like [S12].
    """
    try:
        doc_id = (document_id or "").strip()
        if not doc_id:
            return "Missing document_id. Use list_uploaded_documents() to get an ID."

        # ── Cache (document-scoped) ───────────────────────────────
        with _DOC_SUMMARY_CACHE_LOCK:
            if doc_id in _DOC_SUMMARY_CACHE:
                entry = _DOC_SUMMARY_CACHE[doc_id]
                age_hours = (_dt.now() - entry["cached_at"]).total_seconds() / 3600
                if age_hours <= _CACHE_TTL_HOURS:
                    _DOC_SUMMARY_CACHE.move_to_end(doc_id)
                    return entry["summary"]
                del _DOC_SUMMARY_CACHE[doc_id]

        rag = get_rag_tool_instance()
        chunks = rag.get_all_chunks_for_document(doc_id)
        if not chunks:
            return f"No indexed chunks found for document_id={doc_id}. If you just uploaded it, it may still be processing."

        # Assign global stable source IDs across the full document order
        indexed_chunks: list[tuple[int, str, dict]] = [(i, txt, meta) for i, (txt, meta) in enumerate(chunks, 1)]

        t0 = _time.perf_counter()
        summary, model_name = _run_structured_summary_pipeline(doc_id, indexed_chunks)

        elapsed = _time.perf_counter() - t0
        logger.info("[DocSummary] document_id=%s chunks=%d elapsed=%.1fs model=%s",
                    doc_id, len(chunks), elapsed, model_name)

        # ── Cache result ─────────────────────────────────────────
        with _DOC_SUMMARY_CACHE_LOCK:
            _DOC_SUMMARY_CACHE[doc_id] = {"summary": summary, "cached_at": _dt.now()}
            _DOC_SUMMARY_CACHE.move_to_end(doc_id)
            while len(_DOC_SUMMARY_CACHE) > _DOC_SUMMARY_CACHE_MAX_SIZE:
                _DOC_SUMMARY_CACHE.popitem(last=False)

        return summary

    except Exception as e:
        return f"Error generating document summary: {str(e)[:300]}"
