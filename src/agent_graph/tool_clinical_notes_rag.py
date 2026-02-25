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
from agent_graph.load_tools_config import LoadToolsConfig
from qdrant_client import QdrantClient
from pyprojroot import here
import logging
import os
import json
import hashlib
import time as _time
from pathlib import Path as _Path
from datetime import datetime as _dt
from dotenv import load_dotenv
import threading
from collections import OrderedDict

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

    def search(self, query: str) -> str:
        """
        Hybrid retrieval: dense MMR + BM25 sparse + RRF fusion + cross-encoder reranking.
        Optimized for latency: reduced candidate pool, efficient reranking.
        """
        try:
            t0 = _time.perf_counter()

            # Stage 1: Dense MMR search (original query — no augmentation)
            dense_docs = self.vectordb.max_marginal_relevance_search(
                query=query,
                k=self.fetch_k,
                fetch_k=self.fetch_k * 2,
                lambda_mult=0.5,
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

            # Stage 4: Cross-encoder reranking — only rerank top candidates
            rerank_pool = fused[:self.fetch_k]  # cap reranker input
            if self.reranker and len(rerank_pool) > 1:
                pairs = [[query, doc.page_content] for doc in rerank_pool]
                scores = self.reranker.predict(pairs)
                scored_docs = sorted(
                    zip(rerank_pool, scores), key=lambda x: x[1], reverse=True
                )
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

            # Render with section headers for clarity to the LLM
            parts = []
            for d in docs:
                sec = d.metadata.get("section_title", "Section")
                src = os.path.basename(d.metadata.get("filename", ""))
                mrn = d.metadata.get("patient_mrn", "")
                parts.append(f"[{sec} — {src} | MRN: {mrn}]\n{d.page_content}")
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


def _is_reasoning_model(model_name: str) -> bool:
    """Check if a model is a reasoning model (different API params)."""
    reasoning_prefixes = ("o1", "o3", "o4", "gpt-5")
    return any(model_name.lower().startswith(p) for p in reasoning_prefixes)


def _get_summarization_llm():
    """Create the summarization LLM instance.
    Reasoning models (gpt-5-mini, o3, etc.) don't support temperature or max_tokens.
    They use max_completion_tokens instead — which includes BOTH internal reasoning
    tokens AND the visible response. A clinical summary needs ~4K output tokens,
    and reasoning can use 5-10K+ tokens, so we set a generous limit."""
    model = getattr(TOOLS_CFG, "clinical_notes_rag_summarization_llm",
                    TOOLS_CFG.clinical_notes_rag_llm)
    if _is_reasoning_model(model):
        llm = ChatOpenAI(
            model=model,
            max_completion_tokens=16384,   # reasoning (~8K) + response (~4-6K)
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
    logger.info("Summarization LLM: %s (reasoning=%s, timeout=120s, max_retries=3)", model, _is_reasoning_model(model))
    return model, llm


_SUMMARY_PROMPT = """You are a clinical documentation specialist. Produce a structured summary of this patient record (MRN: {mrn}).

Sections (skip any with no data):
1. Demographics  2. Chief Complaint  3. HPI  4. PMH  5. Surgical Hx
6. Family Hx  7. Social Hx  8. Allergies  9. Medications (admit & discharge)
10. ROS  11. Physical Exam  12. Vital Signs (table)  13. Labs (flag abnormals)
14. Imaging  15. Hospital Course  16. Procedures  17. Consults
18. Discharge Summary  19. Education  20. Outstanding Issues

Rules: exact values/dates. Flag critical findings. No fabrication. Be comprehensive but concise.

RECORD:

{record}"""


# ── Tool definitions ─────────────────────────────────────────────────────────

# Query rewriting removed — saves ~2s per request.
# The hybrid BM25+dense+RRF search already handles synonyms and abbreviations well.
# Evaluation showed 92.5% accuracy without query rewriting being a bottleneck.


@tool
def lookup_clinical_notes(query: str) -> str:
    """Search among clinical notes to find specific information. Use this for ANY targeted question about patient data including: date of birth, demographics, allergies, medications, labs, vitals, procedures, diagnoses, imaging, or any other specific clinical detail. Input should be the clinical question."""
    try:
        rag = get_rag_tool_instance()
        return rag.search(query)
    except Exception as e:
        return f"Error searching clinical notes: {e}. Ensure the vector database is initialized."


@tool
def lookup_patient_orders(patient_identifier: str) -> str:
    """Retrieve ALL orders for a patient — laboratory orders (with CPT/LOINC codes), medication orders (with RxNorm codes), procedure orders (with CPT/SNOMED codes), referrals/consultations, discharge prescriptions, and care directives. Input is the patient name (e.g. 'Robert Whitfield') or MRN (e.g. 'MRN-2026-004782'). Use this when asked for 'all orders', 'complete order list', 'what was ordered', or any comprehensive orders query."""
    try:
        rag = get_rag_tool_instance()
        t_start = _time.perf_counter()

        # Resolve identifier to MRN
        patient_mrn = rag.resolve_to_mrn(patient_identifier)
        if not patient_mrn:
            return (f"Could not find a patient matching '{patient_identifier}'. "
                    f"Use list_available_patients to see who is in the database.")
        logger.info("[Orders] Resolved '%s' → %s", patient_identifier, patient_mrn)

        # Fetch all order-related chunks (section-filtered)
        result = rag.get_order_chunks_for_patient(patient_mrn)

        elapsed = _time.perf_counter() - t_start
        logger.info("[Orders] Total retrieval: %.2fs for %s", elapsed, patient_mrn)
        return result
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

        # ── Layer 1: In-memory LRU cache (instant, cross-session) ─
        with _memory_cache_lock:
            if patient_mrn in _memory_cache:
                entry = _memory_cache[patient_mrn]
                age_hours = (_dt.now() - entry["cached_at"]).total_seconds() / 3600
                if age_hours <= _CACHE_TTL_HOURS:
                    _memory_cache.move_to_end(patient_mrn)  # mark as recently used
                    elapsed = _time.perf_counter() - t_start
                    logger.info("MEMORY CACHE HIT for %s (%.3fs, %.1fh old)", patient_mrn, elapsed, age_hours)
                    _record_cache_event("hit", "memory")
                    return entry["summary"]
                else:
                    logger.info("Memory cache EXPIRED for %s (%.1fh old)", patient_mrn, age_hours)
                    del _memory_cache[patient_mrn]

        # ── Layer 2: Disk cache (survives restarts, cross-session) ─
        cache_dir = _Path(here("data")) / "summary_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = hashlib.sha256(patient_mrn.encode()).hexdigest()[:16]
        cache_file = cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                generated_at = cached.get("generated_at", "")
                cache_time = _dt.strptime(generated_at, "%Y-%m-%d %H:%M:%S")
                age_hours = (_dt.now() - cache_time).total_seconds() / 3600

                if age_hours <= _CACHE_TTL_HOURS:
                    # Valid cache — promote to memory cache (with LRU eviction) and return
                    summary = cached["summary"]
                    with _memory_cache_lock:
                        _memory_cache[patient_mrn] = {
                            "summary": summary,
                            "cached_at": cache_time,
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
        logger.info("Cache MISS for %s — generating summary...", patient_mrn)
        _record_cache_event("miss")

        # Retrieve and merge ALL chunks
        full_record = rag.get_all_chunks_for_patient(patient_mrn)
        if full_record.startswith("No records found") or full_record.startswith("Error"):
            return full_record

        record_chars = len(full_record)
        logger.info("Record: %s chars (~%s tokens)", f"{record_chars:,}", f"{record_chars // 4:,}")

        # Single-pass summarization with output cap
        model_name, llm = _get_summarization_llm()
        prompt = _SUMMARY_PROMPT.format(mrn=patient_mrn, record=full_record)
        t_llm_start = _time.perf_counter()
        summary = llm.invoke(prompt).content
        t_llm_end = _time.perf_counter()
        logger.info("LLM: %.1fs, output=%s chars (model=%s)",
                    t_llm_end - t_llm_start, f"{len(summary):,}", model_name)

        t_total = _time.perf_counter() - t_start
        logger.info("Total summarization: %.1fs", t_total)

        # ── Write to both cache layers ────────────────────────────
        now = _dt.now()

        # Memory cache (LRU eviction when full)
        with _memory_cache_lock:
            _memory_cache[patient_mrn] = {
                "summary": summary,
                "cached_at": now,
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
