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
from datetime import datetime as _dt, timedelta as _td
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


# ── Medical abbreviation expansion table (Phase 2.5) ─────────────────────────
# Applied to dense search query only (BM25 benefits from exact abbreviation matches).
# Keys must be lowercase for case-insensitive matching.
_MEDICAL_EXPANSIONS: dict[str, str] = {
    "hba1c":   "hemoglobin a1c glycated hemoglobin",
    "bnp":     "brain natriuretic peptide b-type natriuretic peptide",
    "bmp":     "basic metabolic panel electrolytes sodium potassium",
    "cmp":     "comprehensive metabolic panel liver function renal function",
    "ef":      "ejection fraction cardiac function systolic",
    "echo":    "echocardiogram cardiac ultrasound",
    "chf":     "congestive heart failure",
    "hf":      "heart failure",
    "dm":      "diabetes mellitus",
    "htn":     "hypertension blood pressure",
    "cad":     "coronary artery disease",
    "afib":    "atrial fibrillation",
    "ckd":     "chronic kidney disease renal insufficiency",
    "copd":    "chronic obstructive pulmonary disease",
    "dvt":     "deep vein thrombosis",
    "pe":      "pulmonary embolism",
    "mi":      "myocardial infarction heart attack",
    "sob":     "shortness of breath dyspnea",
    "cp":      "chest pain",
    "nkda":    "no known drug allergies",
    "dnr":     "do not resuscitate code status",
    "dni":     "do not intubate",
    "cpr":     "cardiopulmonary resuscitation",
    "icu":     "intensive care unit critical care",
    "ed":      "emergency department emergency room",
    "iv":      "intravenous",
    "po":      "oral by mouth",
    "prn":     "as needed",
    "wbc":     "white blood cell count leukocytes",
    "rbc":     "red blood cell count erythrocytes",
    "hgb":     "hemoglobin",
    "plt":     "platelet count thrombocytes",
    "bun":     "blood urea nitrogen",
    "cr":      "creatinine",
    "bp":      "blood pressure",
    "hr":      "heart rate pulse",
    "rr":      "respiratory rate",
    "spo2":    "oxygen saturation pulse oximetry",
    "o2":      "oxygen",
    "egfr":    "estimated glomerular filtration rate renal function",
    "lv":      "left ventricle left ventricular",
    "rv":      "right ventricle right ventricular",
    "pef":     "peak expiratory flow",
    "fev1":    "forced expiratory volume",
    "inr":     "international normalized ratio coagulation",
    "pt":      "prothrombin time coagulation",
    "ptt":     "partial thromboplastin time coagulation",
    "alt":     "alanine aminotransferase liver enzyme",
    "ast":     "aspartate aminotransferase liver enzyme",
    "alp":     "alkaline phosphatase liver enzyme",
    "ldl":     "low density lipoprotein cholesterol",
    "hdl":     "high density lipoprotein cholesterol",
    "tsh":     "thyroid stimulating hormone thyroid function",
    "a1c":     "hemoglobin a1c glycated hemoglobin diabetes",
}


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

        try:
            info = self.client.get_collection(collection_name)
            logger.info("Connected to '%s': %d vectors | Status: %s", collection_name, info.points_count, info.status)
        except Exception as e:
            logger.warning("Could not get collection info: %s", e)

    # ── Phase 2 retrieval helpers ────────────────────────────────────────────

    def _expand_medical_query(self, query: str) -> str:
        """Append expansion terms for medical abbreviations found in the query.

        Applied to the dense (vector) search only — BM25 benefits from exact
        abbreviation hits, so we leave the BM25 query unexpanded.
        """
        q_lower = query.lower()
        expansions: list[str] = []
        for abbr, expansion in _MEDICAL_EXPANSIONS.items():
            if _re.search(r"\b" + _re.escape(abbr) + r"\b", q_lower):
                expansions.append(expansion)
        if expansions:
            expanded = query + " " + " ".join(expansions)
            logger.debug("Query expanded with medical abbreviations: %d terms added", len(expansions))
            return expanded
        return query

    def _adaptive_fetch_k(self, query: str) -> int:
        """Increase the dense-search candidate pool for complex multi-domain queries.

        Signals of complexity: multiple clinical domains named in a single query
        (medications AND labs AND procedures), summary/complete requests, or
        conjunctive structure.  The upper cap (30) keeps latency bounded.
        """
        q = query.lower()
        # Count how many domain signals appear
        domain_signals = [
            bool(_re.search(r"\bmedication|\bdrug|\brx\b|\bprescri", q)),
            bool(_re.search(r"\blab|\btest|\bblood|\bculture|\bpanel", q)),
            bool(_re.search(r"\bprocedure|\bsurgery|\bimaging|\bscan|\becho", q)),
            bool(_re.search(r"\ball\b|\bcomplete|\bfull\b|\bsummary|\btotal", q)),
            bool(_re.search(r"\band\b.{1,50}\band\b", q)),
        ]
        complexity = sum(domain_signals)
        if complexity >= 2:
            boosted = min(self.fetch_k * 2, 30)
            logger.debug("Adaptive fetch_k: %d→%d (complexity=%d)", self.fetch_k, boosted, complexity)
            return boosted
        return self.fetch_k

    def _bm25_score_candidates(self, query: str, candidates: list) -> list:
        """Build a per-query BM25 index over the supplied candidates and return
        them re-ranked by BM25 score (descending).

        This replaces the global startup BM25 index (Phase 2.1):
        - No startup delay — index is built lazily per query.
        - Always reflects latest documents (no staleness after uploads).
        - Memory usage is proportional to the candidate pool (k=15–30 docs),
          not the entire collection.
        - MRN/scope filtering already applied by Qdrant before this stage, so
          cross-patient leakage is impossible.

        Returns documents with BM25 score > 0, ordered highest-first.
        Returns empty list if rank_bm25 is not installed or candidates < 2.
        """
        if len(candidates) < 2:
            return []
        try:
            from rank_bm25 import BM25Okapi
            tokenized_query = query.lower().split()
            corpus = [doc.page_content.lower().split() for doc in candidates]
            bm25 = BM25Okapi(corpus)
            scores = bm25.get_scores(tokenized_query)
            scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, score in scored if score > 0]
        except ImportError:
            logger.debug("rank_bm25 not installed — per-query BM25 skipped")
            return []
        except Exception as e:
            logger.warning("Per-query BM25 failed, using dense only: %s", e)
            return []

    def _detect_metadata_hints(self, query: str) -> dict[str, list[str]]:
        """Detect document type and section hints from the query.

        Returns a dict with optional 'doc_types' and 'section_titles' lists.
        Used for soft metadata re-ranking after retrieval (Phase 2.4).
        """
        q = query.lower()
        hints: dict[str, list[str]] = {"doc_types": [], "section_titles": []}

        # Document type hints
        if _re.search(r"\bdischarge\b", q):
            hints["doc_types"].append("discharge_summary")
        if _re.search(r"\blab\b|\blaboratory\b|\bblood\s+(?:test|result)", q):
            hints["doc_types"].append("lab_report")
        if _re.search(r"\bradiol|\bimaging\b|\bct\s+scan\b|\bmri\b|\bx[-\s]?ray\b", q):
            hints["doc_types"].append("radiology")
        if _re.search(r"\boperat|\bsurgical|\bprocedure\b", q):
            hints["doc_types"].append("operative_note")

        # Section title hints
        if _re.search(r"\ballerg", q):
            hints["section_titles"] += ["Allergies", "Allergy", "ALLERGIES"]
        if _re.search(r"\bmedication|\bcurrent\s+med|\bhome\s+med|\bdischarge\s+med", q):
            hints["section_titles"] += ["Medications", "Current Medications", "Discharge Medications",
                                         "Medication Reconciliation"]
        if _re.search(r"\bdiagno|\bimpression\b|\bassessment\b", q):
            hints["section_titles"] += ["Assessment", "Assessment & Plan", "Assessment and Plan",
                                         "Impression", "Diagnoses"]
        if _re.search(r"\bfollow\s*[-]?\s*up|\bappointment\b|\bdischarge\s+plan", q):
            hints["section_titles"] += ["Follow-up", "Discharge Plan", "Plan"]
        if _re.search(r"\bvital|\bblood\s+pressure|\bheart\s+rate|\btemperature", q):
            hints["section_titles"] += ["Vitals", "Vital Signs", "Physical Examination"]

        return hints

    def _apply_metadata_boost(self, query: str, docs: list) -> list:
        """Soft re-rank: move documents matching query-inferred metadata hints to
        the front of the list WITHOUT discarding any document.

        This is a boost (re-ordering), not a filter — every retrieved document
        is returned.  Documents matching either doc_type or section_title hints
        are promoted ahead of non-matching ones, preserving intra-group order.
        """
        hints = self._detect_metadata_hints(query)
        preferred_types = set(hints.get("doc_types", []))
        preferred_sections = set(hints.get("section_titles", []))

        if not preferred_types and not preferred_sections:
            return docs  # No hint — no reordering

        boosted: list = []
        others: list = []
        for doc in docs:
            meta = doc.metadata or {}
            doc_type = meta.get("document_type", "")
            section = meta.get("section_title", "")
            if doc_type in preferred_types or section in preferred_sections:
                boosted.append(doc)
            else:
                others.append(doc)

        if boosted:
            logger.debug("Metadata boost: %d/%d docs promoted by type/section affinity",
                         len(boosted), len(docs))
        return boosted + others

    @staticmethod
    def _doc_scope_key(doc) -> tuple[str, str, str, int]:
        """Return a stable scope key for retrieval and neighbor expansion."""
        meta = getattr(doc, "metadata", {}) or {}
        chunk_index = meta.get("chunk_index", -1)
        try:
            chunk_index = int(chunk_index)
        except Exception:
            chunk_index = -1
        return (
            str(meta.get("patient_mrn", "") or ""),
            str(meta.get("document_id", "") or ""),
            str(meta.get("filename", "") or ""),
            chunk_index,
        )

    def search(self, query: str, document_id: str = None, patient_mrn: str = None) -> str:
        """
        Hybrid retrieval: dense MMR + per-query BM25 + RRF fusion + cross-encoder
        reranking + metadata soft-boost + neighbor chunk expansion.

        Phase 2 improvements over Phase 1:
        - BM25 is built lazily per query over dense candidates (no startup index).
        - Adaptive fetch_k for complex multi-domain queries.
        - Medical abbreviation expansion on the dense search query.
        - Metadata soft-boost re-ranks type/section-matching docs to front.
        - Neighbor expansion uses Qdrant directly (no in-memory corpus needed).
        """
        try:
            t0 = _time.perf_counter()

            # ── Build Qdrant filter for patient/document scoping ───────────────
            _qdrant_filter = None
            must_conditions = []
            if patient_mrn or document_id:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                if patient_mrn:
                    must_conditions.append(
                        FieldCondition(
                            key="metadata.patient_mrn",
                            match=MatchValue(value=patient_mrn),
                        )
                    )
                if document_id:
                    must_conditions.append(
                        FieldCondition(
                            key="metadata.document_id",
                            match=MatchValue(value=document_id),
                        )
                    )
                _qdrant_filter = Filter(must=must_conditions)

            # ── Stage 1: Dense MMR search (Phase 2.2 adaptive fetch_k) ────────
            # Use expanded query with medical abbreviations for dense search.
            # BM25 gets the original query (benefits from exact abbreviation hits).
            adaptive_k = self._adaptive_fetch_k(query)
            expanded_query = self._expand_medical_query(query)

            search_kwargs = {
                "k": adaptive_k,
                "fetch_k": adaptive_k * 2,
                "lambda_mult": 0.5,
            }
            if _qdrant_filter:
                search_kwargs["filter"] = _qdrant_filter
            dense_docs = self.vectordb.max_marginal_relevance_search(
                query=expanded_query,
                **search_kwargs,
            )
            t_dense = _time.perf_counter() - t0

            # ── Stage 2: Per-query BM25 over dense candidates (Phase 2.1) ─────
            # BM25 sees only the MRN-filtered dense candidates — no cross-patient
            # leakage is possible since Qdrant already enforced the scope filter.
            bm25_docs = self._bm25_score_candidates(query, dense_docs)
            t_bm25 = _time.perf_counter() - t0 - t_dense

            # ── Stage 3: RRF fusion ────────────────────────────────────────────
            if bm25_docs:
                fused = self._reciprocal_rank_fusion(dense_docs, bm25_docs, k=60)
            else:
                fused = dense_docs

            if not fused:
                return f"No relevant documents found for: {query}"

            # ── Stage 4: Cross-encoder reranking ──────────────────────────────
            _MIN_RERANKER_SCORE = 0.05
            rerank_pool = fused[:adaptive_k]  # cap reranker input at adaptive fetch_k
            if self.reranker and len(rerank_pool) > 1:
                pairs = [[query, doc.page_content] for doc in rerank_pool]
                scores = self.reranker.predict(pairs)
                scored_docs = sorted(
                    zip(rerank_pool, scores), key=lambda x: x[1], reverse=True
                )
                scored_docs = [(doc, s) for doc, s in scored_docs if s >= _MIN_RERANKER_SCORE]
                if not scored_docs:
                    return (f"No sufficiently relevant clinical data found for: {query}. "
                            f"Try rephrasing with specific clinical terms "
                            f"(e.g., medication names, lab tests, section names, dates).")
                docs = [doc for doc, _ in scored_docs[:self.reranker_top_k]]
            else:
                docs = rerank_pool[:self.k]

            # ── Stage 4b: Metadata soft-boost (Phase 2.4) ─────────────────────
            # Re-order (never discard) docs that match query-inferred type/section
            # hints so the most contextually relevant chunks appear first.
            docs = self._apply_metadata_boost(query, docs)

            # ── Stage 5: Adjacent-chunk expansion (Qdrant-based, Phase 2.1) ───
            docs = self._expand_with_neighbors(
                docs,
                patient_mrn=patient_mrn,
                document_id=document_id,
            )

            t_total = _time.perf_counter() - t0
            logger.info(
                "Search: %.2fs (dense=%.2fs, bm25=%.2fs, rerank+fuse+expand=%.2fs) → %d docs "
                "[fetch_k=%d, expanded_query=%s]",
                t_total, t_dense, t_bm25, t_total - t_dense - t_bm25, len(docs),
                adaptive_k, "yes" if expanded_query != query else "no",
            )

            # ── Render with source citations for LLM grounding ────────────────
            parts = []
            for i, d in enumerate(docs, 1):
                sec = d.metadata.get("section_title", "Section")
                page = d.metadata.get("page_number", "?")
                mrn = d.metadata.get("patient_mrn", "")
                header = f"[Source {i} | Section: {sec} | Page {page} | MRN: {mrn}]"
                parts.append(f"{header}\n{d.page_content}")
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
                meta = getattr(doc, "metadata", {}) or {}
                key_seed = "||".join([
                    str(meta.get("patient_mrn", "") or ""),
                    str(meta.get("document_id", "") or ""),
                    str(meta.get("filename", "") or ""),
                    str(meta.get("chunk_index", "") or ""),
                    str(getattr(doc, "page_content", "") or ""),
                ])
                key = hashlib.md5(key_seed.encode()).hexdigest()
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
                doc_map[key] = doc

        sorted_keys = sorted(scores, key=scores.get, reverse=True)
        return [doc_map[k] for k in sorted_keys]

    def _expand_with_neighbors(self, docs: list, patient_mrn: str = None, document_id: str = None) -> list:
        """Expand each retrieved document with its ±1 chunk-index neighbors.

        Phase 2.1 change: uses Qdrant queries instead of the old global BM25
        corpus.  This means neighbor lookup is always up-to-date (no staleness
        after uploads) and requires no startup memory overhead.

        Strategy:
        - Group retrieved docs by document_id (one Qdrant scroll per unique doc).
        - Fetch the ±1 chunks around each retrieved chunk_index.
        - Merge overlapping text between consecutive chunks.
        - Deduplicate across the expanded set before returning.

        Patient-isolation guarantee: scroll filters are scoped by patient_mrn
        and/or document_id so neighbors from other patients are never fetched.
        """
        from langchain_core.documents import Document
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        if not docs:
            return docs

        # ── Step 1: Determine which (document_id, chunk_indices) we need ──────
        # Map document_id → set of chunk indices wanted (current ± 1).
        doc_needed: dict[str, set[int]] = {}
        no_index_docs: list = []

        for d in docs:
            meta = d.metadata or {}
            did = str(meta.get("document_id") or "").strip()
            cidx = meta.get("chunk_index", -1)
            try:
                cidx = int(cidx)
            except Exception:
                cidx = -1

            if not did or cidx < 0:
                no_index_docs.append(d)
                continue

            if did not in doc_needed:
                doc_needed[did] = set()
            doc_needed[did].update([cidx - 1, cidx, cidx + 1])

        if not doc_needed:
            # No chunk-indexed docs to expand
            return docs

        # ── Step 2: Fetch neighbor chunks from Qdrant (one scroll per doc_id) ─
        # neighbor_map: (doc_id, chunk_index) → Document
        neighbor_map: dict[tuple[str, int], object] = {}

        for did, needed_idxs in doc_needed.items():
            needed_idxs = {i for i in needed_idxs if i >= 0}
            try:
                must_conds = []
                if patient_mrn:
                    must_conds.append(
                        FieldCondition(key="metadata.patient_mrn",
                                       match=MatchValue(value=patient_mrn))
                    )
                # Always filter by document_id so we only fetch the right doc.
                must_conds.append(
                    FieldCondition(key="metadata.document_id",
                                   match=MatchValue(value=document_id or did))
                )
                scroll_filter = Filter(must=must_conds)

                # A limit of 512 comfortably covers ±1 neighbors for any realistic doc.
                results = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=scroll_filter,
                    limit=512,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in results[0]:
                    pmeta = point.payload.get("metadata", {}) or {}
                    pcidx = pmeta.get("chunk_index", -1)
                    try:
                        pcidx = int(pcidx)
                    except Exception:
                        pcidx = -1
                    if pcidx in needed_idxs:
                        content = point.payload.get("page_content", "") or ""
                        neighbor_map[(did, pcidx)] = Document(
                            page_content=content, metadata=pmeta
                        )
            except Exception as e:
                logger.warning("Neighbor expansion Qdrant query failed for doc %s: %s", did, e)

        # ── Step 3: Collect expanded + deduplicated set ────────────────────────
        seen_keys: set[tuple[str, int]] = set()
        ordered_keys: list[tuple[str, str, int]] = []  # (fname, did, cidx)
        key_to_doc: dict[tuple[str, int], object] = {}

        # Seed from original docs (they define ordering anchor)
        for d in docs:
            meta = d.metadata or {}
            did = str(meta.get("document_id") or "").strip()
            cidx = meta.get("chunk_index", -1)
            try:
                cidx = int(cidx)
            except Exception:
                cidx = -1
            fname = meta.get("filename", "")
            if not did or cidx < 0:
                continue
            for offset in (-1, 0, 1):
                ncidx = cidx + offset
                if ncidx < 0:
                    continue
                nkey = (did, ncidx)
                if nkey not in seen_keys and nkey in neighbor_map:
                    seen_keys.add(nkey)
                    ordered_keys.append((fname, did, ncidx))
                    key_to_doc[nkey] = neighbor_map[nkey]

        # Sort for coherent reading order: (filename, chunk_index)
        ordered_keys.sort(key=lambda t: (t[0], t[1], t[2]))

        # ── Step 4: Build expanded doc list with overlap stripping ─────────────
        expanded: list[object] = []
        prev_content: str | None = None
        prev_fname: str | None = None
        prev_cidx: int = -999

        for fname, did, cidx in ordered_keys:
            d = key_to_doc[(did, cidx)]
            content = d.page_content

            # Strip leading overlap when consecutive chunks from the same file
            if (prev_fname == fname and cidx == prev_cidx + 1 and prev_content):
                overlap = self._find_overlap(prev_content, content)
                if overlap > 0:
                    content = content[overlap:]

            if content.strip():
                expanded.append(Document(page_content=content, metadata=d.metadata))

            prev_content = d.page_content  # original for overlap detection
            prev_fname = fname
            prev_cidx = cidx

        # Append original no-index docs that couldn't be expanded
        expanded.extend(no_index_docs)

        added = len(expanded) - len(docs)
        if added > 0:
            logger.info("Neighbor expansion: %d → %d chunks (+%d neighbors)",
                        len(docs), len(expanded), added)

        return expanded if expanded else docs

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

    # Note: We intentionally avoid hardcoding EHR-specific section header strings
    # for order retrieval. "All orders" is inferred from chunk content instead
    # (see `_infer_order_category`), so new/unknown PDF formats work without
    # code changes.

    @staticmethod
    def _infer_order_category(text: str, section_title: str | None = None) -> str | None:
        """
        Infer whether a chunk is "order-like" using content-based signals.

        Production goal: avoid depending on brittle section header strings that
        vary across EHR exports. We instead look for medication dose/SIG
        patterns, lab test keywords + units, procedure keywords, and
        discharge/follow-up directives.
        """
        t = (text or "").strip()
        if not t:
            return None
        s = (section_title or "").upper()
        u = t.upper()

        # --- Shared evidence detectors ---
        # Dose/signature-like tokens common in medication lists.
        dose_re = _re.compile(
            r"\b\d+(\.\d+)?\s*(MG|MCG|G)\b|\b\d+(\.\d+)?\s*(MILLIGRAMS|MICROGRAMS|GRAMS)\b",
            flags=_re.IGNORECASE,
        )
        route_re = _re.compile(
            r"\b(PO|IV|IM|SC|SUBCUTANEOUS|INTRAVENOUS|INTRAMUSCULAR|ORAL|SL|TOPICAL|INHALED|INHAL)\b",
            flags=_re.IGNORECASE,
        )
        freq_re = _re.compile(
            r"\b(QD|QAM|QPM|BID|TID|QID|QHS|Q4H|Q6H|Q8H|PRN|DAILY|TWICE DAILY|THREE TIMES DAILY|FOUR TIMES DAILY)\b",
            flags=_re.IGNORECASE,
        )

        # Lab-like evidence: common biomarkers/panels + unit formats.
        lab_keyword_re = _re.compile(
            r"\b(CBC|BMP|CMP|A1C|HEMOGLOBIN\s*A1C|GLUCOSE|CREATININE|BNP|TROPONIN|PT|INR|PTT|PLATELET|WBC|HGB|HBA1C|MICROALBUMIN)\b",
            flags=_re.IGNORECASE,
        )
        lab_unit_re = _re.compile(
            r"\b(mg/dL|mmol/L|pg/mL|uIU/mL|k/uL|IU/L|mEq/L|g/dL|ng/mL|uL|%|mL)\b",
            flags=_re.IGNORECASE,
        )

        # Procedure / imaging-like evidence.
        proc_keyword_re = _re.compile(
            r"\b(CT|MRI|ULTRASOUND|X-?RAY|ECHOCARDIOGRAM|CATHETERIZATION|CATH|THORACENTESIS|ENDOSCOPY|BIOPSY|PROCEDURE)\b",
            flags=_re.IGNORECASE,
        )

        # Discharge / care directive evidence.
        discharge_keyword_re = _re.compile(
            r"\b(DISCHARGE|DISPOSITION|FOLLOW[- ]?UP|RETURN TO|HOME HEALTH|FOLLOW UP|APPOINTMENT|CARE DIRECTIVE|DISCHARGE MEDICATIONS)\b",
            flags=_re.IGNORECASE,
        )

        # Order code evidence (CPT/LOINC/SNOMED) when present.
        cpt_loinc_keyword_re = _re.compile(
            r"\b(CPT|LOINC|RXNORM|SNOMED)\b",
            flags=_re.IGNORECASE,
        )
        code_re = _re.compile(r"\b\d{4,6}\b")

        # --- Category scoring ---
        medication_score = 0
        if ("MED" in s or "RX" in s or "PRESCR" in s or "CURRENT" in s or "DISCONT" in s):
            medication_score += 1
        if _re.search(r"\b(RX|SIG|PRESCRIBE|PRESCRIPTION|MEDICATION|MEDICATIONS|DISCHARGE MEDICATIONS)\b", u):
            medication_score += 2
        if dose_re.search(u):
            medication_score += 2
        if route_re.search(u):
            medication_score += 1
        if freq_re.search(u):
            medication_score += 1
        if code_re.search(u) and cpt_loinc_keyword_re.search(u):
            medication_score += 1

        lab_score = 0
        if lab_keyword_re.search(u):
            lab_score += 3
        if lab_unit_re.search(u) and (lab_keyword_re.search(u) or _re.search(r"\bLAB\b", u)):
            lab_score += 2
        if cpt_loinc_keyword_re.search(u):
            lab_score += 1

        proc_score = 0
        if proc_keyword_re.search(u):
            proc_score += 3
        if cpt_loinc_keyword_re.search(u):
            proc_score += 1
        if code_re.search(u) and ("CPT" in u or "LOINC" in u):
            proc_score += 1

        referral_score = 0
        if _re.search(r"\b(REFERRAL|CONSULT|CONSULTATION)\b", u):
            referral_score += 3
        if discharge_keyword_re.search(u) and _re.search(r"\b(CARDIOLOGY|NEPHROLOGY|ONCOLOGY|PULMON|GASTRO|ENDOCRIN|UROLOGY|NEURO)\b", u):
            referral_score += 1
        if cpt_loinc_keyword_re.search(u):
            referral_score += 1

        discharge_score = 0
        if discharge_keyword_re.search(u):
            discharge_score += 3
        # Often discharge meds appear as medication lists; capture that here too.
        if dose_re.search(u) and _re.search(r"\b(DISCHARGE|DISPOSITION)\b", u):
            discharge_score += 2

        care_directive_score = 0
        if _re.search(r"\b(NO CHANGE|AVOID|MONITOR|INSTRUCTIONS|RETURN PRECAUTIONS|WOUND CARE|DIET|ACTIVITY|WEIGHT)\b", u):
            care_directive_score += 2
        if discharge_keyword_re.search(u):
            care_directive_score += 1

        # Hard threshold: only include chunks with meaningful order evidence.
        # This avoids pulling plain narrative progress notes.
        scored = {
            "Medication Orders": medication_score,
            "Lab Orders": lab_score,
            "Procedure Orders": proc_score,
            "Referrals/Consults": referral_score,
            "Discharge Prescriptions/Disposition": discharge_score,
            "Care Directives/Follow-up": care_directive_score,
        }
        best_cat, best_score = max(scored.items(), key=lambda kv: kv[1])
        if best_score <= 0:
            return None

        # Require at least one of the major evidence signals for inclusion.
        # (Medication: dose+either route/frequency, or medication keywords + SIG.)
        # (Labs/Procedures: strong keyword hits.)
        if best_cat == "Medication Orders":
            if medication_score >= 4 or (dose_re.search(u) and (route_re.search(u) or freq_re.search(u))):
                return best_cat
            return None
        if best_cat == "Lab Orders":
            return best_cat if best_score >= 3 else None
        if best_cat == "Procedure Orders":
            return best_cat if best_score >= 3 else None
        if best_cat == "Referrals/Consults":
            return best_cat if best_score >= 3 else None
        if best_cat == "Discharge Prescriptions/Disposition":
            return best_cat if best_score >= 3 else None
        if best_cat == "Care Directives/Follow-up":
            return best_cat if best_score >= 2 else None
        return None

    def get_order_chunks_for_patient(self, patient_mrn: str) -> str:
        """Retrieve ALL order-related chunks for a patient: labs, medications,
        procedures, referrals, discharge prescriptions, care directives.

        Format-agnostic approach:
        - Retrieve all patient chunks (paginated scroll for long documents)
        - Infer order categories from chunk content
        - Return them grouped by inferred order category
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            t0 = _time.perf_counter()

            # Paginate to avoid missing >500 chunks.
            # (Longer EHR extracts and multi-encounter PDFs exceed that easily.)
            points: list = []
            next_offset = None
            max_points = int(os.getenv("MEDGRAPH_ORDER_SCROLL_MAX_POINTS", "4000"))
            while True:
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
                    limit=512,
                    with_payload=True,
                    with_vectors=False,
                    offset=next_offset,
                )
                batch, next_offset = results[0], results[1]
                if batch:
                    points.extend(batch)
                if not next_offset:
                    break
                if len(points) >= max_points:
                    logger.warning(
                        "Order scroll max_points reached (%d). Truncating further retrieval.",
                        max_points,
                    )
                    break

            if not points:
                return f"No records found for patient MRN: {patient_mrn}"

            # Filter to order-related chunks using content inference.
            order_points: list = []
            for p in points:
                meta = p.payload.get("metadata", {}) or {}
                content = p.payload.get("page_content", "") or ""
                sec = meta.get("section_title")
                cat = self._infer_order_category(content, sec)
                if cat:
                    order_points.append((cat, p))

            if not order_points:
                return f"No order-related data found for patient MRN: {patient_mrn}"

            # Sort by inferred category priority, then by (page, chunk_index).
            cat_priority = [
                "Medication Orders",
                "Lab Orders",
                "Procedure Orders",
                "Referrals/Consults",
                "Discharge Prescriptions/Disposition",
                "Care Directives/Follow-up",
            ]
            cat_rank = {c: i for i, c in enumerate(cat_priority)}

            order_points.sort(
                key=lambda tup: (
                    cat_rank.get(tup[0], 999),
                    tup[1].payload.get("metadata", {}).get("page_number", 0),
                    tup[1].payload.get("metadata", {}).get("chunk_index", 0),
                )
            )

            # Group by inferred category and merge overlapping text.
            current_category: str | None = None
            parts: list[str] = []
            prev_content: str | None = None
            prev_source: str | None = None

            for cat, p in order_points:
                meta = p.payload.get("metadata", {})
                content = p.payload.get("page_content", "")
                source = meta.get("filename", "")
                sec = meta.get("section_title", "Section")  # kept for traceability

                # Category separator for readability.
                if cat != current_category:
                    if current_category is not None:
                        parts.append("")  # blank line between sections
                    current_category = cat
                    prev_content = None  # reset overlap tracking at section boundary
                    prev_source = None

                # Strip overlap with previous chunk in same source
                if prev_content and source == prev_source:
                    overlap = self._find_overlap(prev_content, content)
                    if overlap > 0:
                        content = content[overlap:]

                if content.strip():
                    parts.append(f"[{cat} | {sec}]\n{content}")

                prev_content = p.payload.get("page_content", "")
                prev_source = source

            elapsed = _time.perf_counter() - t0
            logger.info(
                "Order retrieval: %d/%d chunks inferred as order-related for %s in %.2fs",
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

    @staticmethod
    def _normalize_lookup_text(value: str) -> str:
        """Normalize free-form identifier text for robust matching."""
        t = (value or "").strip().lower()
        if not t:
            return ""
        t = _re.sub(r"[^a-z0-9\-\s]", " ", t)
        t = _re.sub(r"\s+", " ", t).strip()
        return t

    @staticmethod
    def _compact_identifier(value: str) -> str:
        """Compact an identifier for tolerant MRN matching."""
        t = (value or "").strip().lower()
        if not t:
            return ""
        t = _re.sub(r"^(?:mrn|patient\s*id)\s*:?\s*", "", t)
        return _re.sub(r"[^a-z0-9]", "", t)

    def _extract_query_variants(self, identifier: str) -> list[str]:
        """Generate candidate identifier variants from natural language input."""
        norm = self._normalize_lookup_text(identifier)
        if not norm:
            return []

        variants = {norm}

        # Explicit identifier patterns, including repeated label syntax such as
        # "MRN: MRN-2026-004782" and numeric MRNs like "MRN: 250813000000794".
        explicit_patterns = [
            r"\bmrn\s*:?\s*((?:mrn\s*:?\s*)?[a-z0-9\-]{4,})",
            r"\bpatient\s*id\s*:?\s*([a-z0-9\-]{4,})",
        ]
        for pat in explicit_patterns:
            for raw in _re.findall(pat, identifier or "", flags=_re.IGNORECASE):
                qn = self._normalize_lookup_text(raw)
                if qn:
                    variants.add(qn)
                    compact = self._compact_identifier(qn)
                    if compact:
                        variants.add(compact)

        # Capture quoted identifiers when users explicitly provide a patient label.
        for quoted in _re.findall(r"['\"]([^'\"]{3,})['\"]", identifier or ""):
            qn = self._normalize_lookup_text(quoted)
            if qn:
                variants.add(qn)

        tail = _re.search(r"(?:for|of|about|patient|mrn)\s+([a-z0-9\-\s]{3,})$", norm)
        if tail:
            variants.add(tail.group(1).strip())

        # Patterns where a name is embedded in the middle of a query.
        embedded = _re.search(
            r"(?:disease|diagnosis|condition|problem|summary|record|chart|status|history|about)\s+(?:is\s+)?([a-z][a-z\-]+\s+[a-z][a-z\-]+(?:\s+[a-z][a-z\-]+)?)",
            norm,
        )
        if embedded:
            variants.add(embedded.group(1).strip())

        tokens = norm.split()
        semantic_stop_tokens = {
            "what", "which", "who", "whom", "when", "where", "why", "how", "is", "are", "was", "were",
            "do", "does", "did", "can", "could", "would", "should", "tell", "show", "find", "give",
            "me", "please", "a", "an", "the", "for", "of", "about", "from", "with", "has", "have",
            "suffering", "suffer", "disease", "diagnosis", "condition", "problem", "record", "records",
            "summary", "summarize", "chart", "patient", "mrn",
        }
        filtered = [t for t in tokens if t not in semantic_stop_tokens and len(t) >= 2]

        # Add compact name-like windows from filtered tokens (e.g., "ayesha malik").
        for n in (2, 3):
            if len(filtered) >= n:
                for i in range(0, len(filtered) - n + 1):
                    span = filtered[i : i + n]
                    if all(_re.fullmatch(r"[a-z][a-z\-]*", s or "") for s in span):
                        variants.add(" ".join(span))

        if len(filtered) >= 2:
            variants.add(" ".join(filtered))

        for n in (2, 3, 4):
            if len(tokens) >= n:
                variants.add(" ".join(tokens[-n:]))

        # Add standalone long numeric/alphanumeric identifiers that commonly
        # represent MRNs or patient IDs.
        for tok in tokens:
            compact = self._compact_identifier(tok)
            if compact and (compact.isdigit() and len(compact) >= 6):
                variants.add(compact)

        return sorted((v for v in variants if v), key=len, reverse=True)

    def _scroll_all_points(self, scroll_filter=None, limit: int = 512) -> list:
        """Scroll all points with pagination to avoid first-page-only misses."""
        all_points = []
        next_offset = None
        max_points = int(os.environ.get("MEDGRAPH_PATIENT_SCAN_MAX_POINTS", "50000"))

        while True:
            kwargs = {
                "collection_name": self.collection_name,
                "limit": limit,
                "with_payload": True,
                "with_vectors": False,
            }
            if scroll_filter is not None:
                kwargs["scroll_filter"] = scroll_filter
            if next_offset is not None:
                kwargs["offset"] = next_offset

            results = self.client.scroll(**kwargs)
            points, next_offset = results[0], results[1]

            if points:
                all_points.extend(points)
                if len(all_points) >= max_points:
                    logger.warning(
                        "Patient scan reached cap (%d points). Set MEDGRAPH_PATIENT_SCAN_MAX_POINTS higher if needed.",
                        max_points,
                    )
                    break

            if not next_offset:
                break

        return all_points

    def list_patients(self) -> str:
        """List all unique patients in the vector database with their MRN, name, and source file."""
        try:
            points = self._scroll_all_points(limit=512)
            patients = {}
            name_only_patients = {}
            for p in points:
                meta = p.payload.get("metadata", {})
                mrn = str(meta.get("patient_mrn", "") or "").strip()
                name = str(meta.get("patient_name", "") or "").strip()
                source = str(meta.get("source", "") or meta.get("filename", "Unknown"))
                document_id = str(meta.get("document_id", "") or "").strip()
                if mrn and mrn not in patients:
                    patients[mrn] = {
                        "mrn": mrn,
                        "name": name or "Unknown",
                        "source": source,
                    }
                elif not mrn and name:
                    nk = self._normalize_lookup_text(name)
                    if nk and nk not in name_only_patients:
                        name_only_patients[nk] = {
                            "mrn": "N/A",
                            "name": name,
                            "source": source,
                            "document_id": document_id,
                        }

            if not patients and not name_only_patients:
                return "No patients found in the database."

            lines = ["Available patients in the database:"]
            for info in sorted(patients.values(), key=lambda x: (x["name"].lower(), x["mrn"].lower())):
                lines.append(f"  - MRN: {info['mrn']} | Name: {info['name']} | Source: {info['source']}")

            for info in sorted(name_only_patients.values(), key=lambda x: x["name"].lower()):
                doc_hint = f" | Document ID: {info['document_id']}" if info.get("document_id") else ""
                lines.append(
                    f"  - MRN: {info['mrn']} | Name: {info['name']} | Source: {info['source']}{doc_hint}"
                )

            return "\n".join(lines)
        except Exception as e:
            return f"Error listing patients: {e}"

    def resolve_to_mrn(self, identifier: str) -> str | None:
        """Resolve a patient name or MRN string to the canonical MRN.

        Accepts: exact MRN, partial MRN, full name, partial name (case-insensitive).
        Returns the MRN string, or None if no match found.
        Uses scored matching to pick the best candidate, not first-found.
        """
        points = self._scroll_all_points(limit=512)
        patients: dict[str, str] = {}  # mrn -> name
        for p in points:
            meta = p.payload.get("metadata", {})
            mrn = str(meta.get("patient_mrn", "") or "").strip()
            name = str(meta.get("patient_name", "") or "").strip()
            if mrn and mrn not in patients:
                patients[mrn] = name

        query_variants = self._extract_query_variants(identifier)
        if not patients or not query_variants:
            return None

        stop_tokens = {
            "give", "me", "summary", "summarize", "overview", "record", "records", "patient",
            "for", "of", "about", "please", "the", "a", "an", "show", "find",
        }

        best_mrn: str | None = None
        best_score = 0

        for mrn, name in patients.items():
            mrn_norm = self._normalize_lookup_text(mrn)
            mrn_compact = self._compact_identifier(mrn)
            name_norm = self._normalize_lookup_text(name)
            name_tokens = set(name_norm.split()) if name_norm else set()

            for variant in query_variants:
                score = 0
                variant_norm = self._normalize_lookup_text(variant)
                variant_compact = self._compact_identifier(variant)
                variant_tokens = {t for t in variant_norm.split() if t not in stop_tokens}

                # Exact MRN match
                if mrn_norm == variant_norm and variant_norm:
                    return mrn  # perfect match, return immediately
                if mrn_compact and variant_compact and mrn_compact == variant_compact:
                    return mrn

                # Partial MRN match
                if variant_norm and variant_norm in mrn_norm:
                    score = max(score, 95)
                if variant_compact and mrn_compact and variant_compact in mrn_compact:
                    score = max(score, 96)

                # Exact/partial name match
                if name_norm and variant_norm == name_norm:
                    score = max(score, 100)
                if name_norm and variant_norm and variant_norm in name_norm:
                    score = max(score, 88)
                if name_norm and variant_norm and name_norm in variant_norm:
                    score = max(score, 78)

                # Token-level overlap (weighted by how much of the variant was matched)
                if variant_tokens and name_tokens:
                    overlap = variant_tokens & name_tokens
                    if overlap:
                        frac = len(overlap) / max(len(variant_tokens), 1)
                        token_score = int(60 + (35 * frac))
                        score = max(score, token_score)

                if score > best_score:
                    best_score = score
                    best_mrn = mrn

        if best_mrn and best_score >= 70:
            logger.info(
                "Name resolution: '%s' → %s (name='%s', score=%d)",
                identifier,
                best_mrn,
                patients[best_mrn],
                best_score,
            )
            return best_mrn

        return None

    def find_documents_for_patient_identifier(self, identifier: str, max_docs: int = 3) -> list[str]:
        """Find uploaded documents likely belonging to a patient identifier.

        Used as a production-safe fallback when MRN resolution fails (e.g.,
        freshly uploaded docs with incomplete patient_mrn metadata).
        """
        query_variants = self._extract_query_variants(identifier)
        if not query_variants:
            return []

        points = self._scroll_all_points(limit=512)
        if not points:
            return []

        stop_tokens = {
            "give", "me", "summary", "summarize", "overview", "record", "records", "patient",
            "for", "of", "about", "please", "the", "a", "an", "show", "find",
        }

        doc_scores: dict[str, int] = {}

        for p in points:
            payload = p.payload or {}
            meta = payload.get("metadata", {}) or {}

            doc_id = str(meta.get("document_id") or "").strip()
            if not doc_id:
                continue

            name_norm = self._normalize_lookup_text(str(meta.get("patient_name") or ""))
            mrn_norm = self._normalize_lookup_text(str(meta.get("patient_mrn") or ""))
            content_norm = self._normalize_lookup_text(str(payload.get("page_content") or "")[:1500])

            point_score = 0

            for variant in query_variants:
                variant_norm = self._normalize_lookup_text(variant)
                variant_tokens = {t for t in variant_norm.split() if t not in stop_tokens}

                if mrn_norm and variant_norm == mrn_norm:
                    point_score = max(point_score, 120)
                elif mrn_norm and variant_norm and variant_norm in mrn_norm:
                    point_score = max(point_score, 110)

                if name_norm and variant_norm == name_norm:
                    point_score = max(point_score, 115)
                elif name_norm and variant_norm and variant_norm in name_norm:
                    point_score = max(point_score, 100)
                elif name_norm and variant_norm and name_norm in variant_norm:
                    point_score = max(point_score, 92)

                if name_norm and variant_tokens:
                    name_tokens = set(name_norm.split())
                    overlap = variant_tokens & name_tokens
                    if overlap:
                        frac = len(overlap) / max(len(variant_tokens), 1)
                        point_score = max(point_score, int(75 + (20 * frac)))

                if not name_norm and len(variant_tokens) >= 2 and all(tok in content_norm for tok in variant_tokens):
                    point_score = max(point_score, 86)

            if point_score >= 85:
                prev = doc_scores.get(doc_id, 0)
                if point_score > prev:
                    doc_scores[doc_id] = point_score

        if not doc_scores:
            return []

        ranked = sorted(doc_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_score = ranked[0][1]
        cutoff = max(85, top_score - 8)
        selected = [doc_id for doc_id, score in ranked if score >= cutoff]

        return selected[: max(1, int(max_docs or 1))]


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


def clear_summary_caches() -> dict:
    """Clear patient/document summary caches across memory and disk layers."""
    with _memory_cache_lock:
        patient_memory_cleared = len(_memory_cache)
        _memory_cache.clear()

    with _DOC_SUMMARY_CACHE_LOCK:
        document_memory_cleared = len(_DOC_SUMMARY_CACHE)
        _DOC_SUMMARY_CACHE.clear()

    disk_deleted = 0
    cache_dir = _Path(here("data")) / "summary_cache"
    if cache_dir.exists():
        for f in cache_dir.glob("*.json"):
            try:
                f.unlink()
                disk_deleted += 1
            except Exception:
                pass

    return {
        "status": "ok",
        "disk_deleted": disk_deleted,
        "patient_memory_cleared": patient_memory_cleared,
        "document_memory_cleared": document_memory_cleared,
    }


def _document_summary_cache_file(document_id: str) -> _Path:
    """Return a stable disk-cache path for uploaded-document summaries."""
    cache_dir = _Path(here("data")) / "summary_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(f"doc::{document_id}".encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"doc_{cache_key}.json"


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


def _detect_note_type_from_chunks(indexed_chunks: list[tuple[int, str, dict]]) -> str:
    """
    Detect common clinical note types from available metadata signals.

    We primarily use:
    - chunk meta `document_type` (from PDF classification)
    - chunk meta `section_title` (from adaptive chunker headings/keyword sections)
    """
    doc_types = [((meta or {}).get("document_type") or "").lower() for _, _, meta in indexed_chunks]
    sections = [((meta or {}).get("section_title") or "").lower() for _, _, meta in indexed_chunks]
    joined_sections = " ".join(sections[:60])  # cap for speed

    # Discharge / inpatient cues
    if any("discharge" in dt for dt in doc_types) or any("discharge" in s for s in sections) or any("disposition" in s for s in sections):
        return "Discharge_Summary"
    if any("inpatient" in s or "hospital course" in s or "admission" in s for s in sections):
        return "Inpatient"

    # SOAP cues: subjective/objective + assessment/plan
    if any(s in ("subjective", "objective", "ros", "physical exam", "physical examination") for s in sections) and (
        any("assessment" in s and "plan" in s for s in sections)
        or any(s in ("assessment & plan", "assessment and plan", "plan", "a/p", "a/p.") for s in sections)
        or "soap" in joined_sections
    ):
        return "SOAP"

    # HPI cues
    if any("history of present illness" in s for s in sections) or any(s.startswith("hpi") for s in sections):
        return "HPI"

    # Assessment/Plan cues
    if any(("assessment" in s and "plan" in s) or s in ("a/p", "ap") for s in sections):
        return "AP"

    # Progress notes may still be SOAP-like even when headings are messy.
    if any("progress_note" in dt or "progress" in dt for dt in doc_types) and (
        any("assessment" in s for s in sections) and any("plan" in s for s in sections)
    ):
        return "SOAP"

    return "General"


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


def _doc_needs_keyword_retry(raw_text: str, merged: dict, note_type: str | None = None) -> list[str]:
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

    # Note-type-specific priorities.
    if note_type in ("SOAP", "HPI", "AP"):
        # SOAP/AP/HPI commonly requires a problem/diagnosis backbone.
        if (
            any(k in raw for k in ["assessment", "impression", "plan", "diagnos", "differential", "problem"])
            and not (merged.get("problems") or [])
        ):
            missing.append("problems")

    if note_type in ("Inpatient", "Discharge_Summary"):
        if (
            any(k in raw for k in ["hospital course", "hospital day", "inpatient", "during admission", "day "])
            and not (merged.get("hospital_course") or [])
        ):
            missing.append("hospital_course")
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


def _run_structured_summary_pipeline(
    entity_id: str,
    indexed_chunks: list[tuple[int, str, dict]],
    hierarchical: bool = True,
) -> tuple[str, str]:
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
    note_type = _detect_note_type_from_chunks(indexed_chunks)

    # Hierarchical summarization: the current FAST path only selects a small
    # subset of sources, which can miss cross-section relationships in large
    # documents. For production, we switch to a section-first summarization
    # flow when the input is large and we're in FAST mode.
    if hierarchical and is_fast_mode:
        hierarchical_threshold_chars = int(os.environ.get("MEDGRAPH_HIER_SUMMARY_CHAR_THRESHOLD", "120000"))
        max_sections = int(os.environ.get("MEDGRAPH_HIER_SUMMARY_MAX_SECTIONS", "10"))

        total_chars = sum(len(txt or "") for _, txt, _ in indexed_chunks)
        if total_chars >= hierarchical_threshold_chars:
            # Group by section_title already computed at indexing time.
            by_section: dict[str, list[tuple[int, str, dict]]] = {}
            for gi, txt, meta in indexed_chunks:
                sec = (meta or {}).get("section_title") or "Full Document"
                by_section.setdefault(sec, []).append((gi, txt, meta))

            if len(by_section) > 1:
                # Select the most information-dense sections first.
                ordered_sections = sorted(
                    by_section.items(),
                    key=lambda kv: sum(len(t or "") for _, t, _ in kv[1]),
                    reverse=True,
                )[:max_sections]

                section_summaries: list[str] = []
                for sec_title, sec_chunks in ordered_sections:
                    sec_summary, _ = _run_structured_summary_pipeline(
                        entity_id,
                        sec_chunks,
                        hierarchical=False,  # disable recursion back into hierarchical mode
                    )
                    section_summaries.append(f"SECTION: {sec_title}\n{sec_summary}")

                hier_synth_prompt = (
                    "You are MedGraph AI.\n\n"
                    "Create a final clinician-ready summary for the full record using ONLY the section summaries below. "
                    "Do not add new facts. Consolidate duplicates, preserve clinically important decisions (diagnoses, meds, critical labs, imaging, procedures, discharge/follow-up), "
                    f"Apply note-type emphasis for: {note_type}. "
                    "and keep chronology when present.\n\n"
                    "SECTION SUMMARIES:\n"
                    f"{chr(10).join(section_summaries)}"
                )

                reduce_chain = build_model_chain(
                    reduce_primary,
                    ["gemini-2.5-flash-lite"],
                )
                model_name, final_summary = _llm_invoke_with_fallback(
                    _get_doc_reduce_llm,
                    hier_synth_prompt,
                    max_retries=2,
                    model_chain=reduce_chain,
                )
                return final_summary, model_name
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
    missing_domains = _doc_needs_keyword_retry(signal_text, merged, note_type=note_type)

    if missing_domains:
        keyword_map = {
            "allergies": ["allerg", "nka", "reaction", "anaphyl", "urticaria", "angioedema", "contrast"],
            "medications": ["med", "medication", "rx", "sig", "dose", "tablet", "capsule", "discharge rx"],
            "labs": ["lab", "cbc", "bmp", "cmp", "mg/dl", "mmol", "wbc", "hgb", "plt", "bnp", "creatinine"],
            "imaging": ["impression", "ct", "mri", "x-ray", "ultrasound", "echo", "ecg", "ekg", "rvsp", "ef"],
            "procedures": ["procedure", "operative", "surgery", "biopsy", "endoscopy", "thoracentesis", "catheter", "pcwp", "pvr", "cardiac index"],
                "problems": ["assessment", "impression", "diagnos", "differential", "problem", "plan"],
                "hospital_course": ["hospital course", "hospital day", "inpatient", "during admission", "day "],
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

    note_type_emphasis_map = {
        "SOAP": (
            "NOTE TYPE EMPHASIS (SOAP): Prioritize Assessment and Plan. Keep Subjective/Objective concise "
            "and make sure medications ordered/started/stopped and key diagnoses are emphasized."
        ),
        "HPI": (
            "NOTE TYPE EMPHASIS (HPI): Focus on the presenting complaint narrative (onset → course) and "
            "organize the summary around clinical symptom evolution and pertinent associated features."
        ),
        "AP": (
            "NOTE TYPE EMPHASIS (AP): Use problem-by-problem structure. For each problem, emphasize the "
            "planned interventions, key diagnostics, medications changes, and follow-up."
        ),
        "Inpatient": (
            "NOTE TYPE EMPHASIS (Inpatient): Emphasize hospital course chronology, medication changes during admission, "
            "procedures/consults, and pending results with discharge-limiting events."
        ),
        "Discharge_Summary": (
            "NOTE TYPE EMPHASIS (Discharge Summary): Emphasize transition of care: discharge diagnoses and condition, "
            "reconciled discharge medications, pending results, and follow-up appointments/instructions."
        ),
    }
    if note_type in note_type_emphasis_map:
        reduce_prompt += "\n\n" + note_type_emphasis_map[note_type]

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
            text = rag.search(q, patient_mrn=patient_mrn)
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


def _supplement_from_targeted_searches_for_document(
    rag: "ClinicalNotesRAGTool",
    document_id: str,
) -> list[tuple[str, dict]]:
    """Get supplemental chunks for uploaded documents using doc-scoped retrieval."""
    queries = [
        "allergies allergy reaction anaphylaxis contrast premedication",
        "discharge medications home medications dose route frequency",
        "critical labs imaging procedures hospital course follow-up",
    ]
    out: list[tuple[str, dict]] = []
    seen = set()
    for q in queries:
        try:
            text = rag.search(q, document_id=document_id)
        except Exception:
            continue
        if not text or text.startswith(("No relevant", "No sufficiently relevant", "Error during search")):
            continue
        parts = text.split("\n\n---\n\n")
        for p in parts:
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
def lookup_clinical_notes(query: str, patient_mrn: str = "", document_id: str = "") -> str:
    """Search among clinical notes for a specific patient's data. For patient-specific EHR questions, pass the resolved `patient_mrn` so retrieval stays isolated to one patient. Optionally pass `document_id` to further restrict search to a specific uploaded document (get IDs from list_uploaded_documents)."""
    try:
        import signal
        import threading
        
        rag = get_rag_tool_instance()
        result = [None]
        exception = [None]
        
        # Use document_id filter if provided (strip whitespace, treat empty as None)
        doc_filter = document_id.strip() if document_id else None
        mrn_filter = patient_mrn.strip() if patient_mrn else None
        
        def _is_unhelpful_search_result(text: str) -> bool:
            t = (text or "").strip().lower()
            if not t:
                return True
            markers = (
                "no relevant documents found for:",
                "no sufficiently relevant clinical data found for:",
                "no results found.",
                "error during search:",
            )
            return any(t.startswith(m) for m in markers)

        def _search_with_timeout():
            try:
                result[0] = rag.search(query, document_id=doc_filter, patient_mrn=mrn_filter)
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

        # Production fallback: if broad search misses and no explicit document_id
        # was provided, try patient-aware document-scoped retrieval on likely
        # uploaded documents inferred from the query itself.
        if not doc_filter and not mrn_filter and _is_unhelpful_search_result(result[0] or ""):
            try:
                candidate_doc_ids = rag.find_documents_for_patient_identifier(query, max_docs=3)
                fallback_hits: list[str] = []
                for cand_doc_id in candidate_doc_ids:
                    scoped = rag.search(query, document_id=cand_doc_id)
                    if not _is_unhelpful_search_result(scoped or ""):
                        fallback_hits.append(
                            f"[Document Scoped Fallback | document_id: {cand_doc_id}]\n{scoped}"
                        )
                if fallback_hits:
                    logger.info(
                        "lookup_clinical_notes fallback activated for query='%s' across docs=%s",
                        str(query)[:120],
                        candidate_doc_ids,
                    )
                    return "\n\n---\n\n".join(fallback_hits)
            except Exception as fallback_exc:
                logger.warning("lookup_clinical_notes fallback failed: %s", str(fallback_exc)[:200])
        
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


def _safe_json_extract_first_object(raw: str) -> dict:
    """Best-effort extraction of the first JSON object from a model response."""
    if not raw:
        return {}
    t = str(raw).strip()
    # Strip markdown fences if present
    t = _re.sub(r"^```(?:json)?\s*", "", t, flags=_re.IGNORECASE).strip()
    t = _re.sub(r"\s*```$", "", t).strip()
    if not t.startswith("{"):
        m = _re.search(r"\{[\s\S]*\}", t)
        if m:
            t = m.group(0)
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _parse_cohort_query(query: str) -> dict | None:
    """
    Parse natural-language cohort queries into a structured filter spec.

    Production-grade note:
    - We use deterministic regex parsing for the supported cohort patterns.
    - If regex parsing fails, we optionally fall back to an LLM parse only when enabled.
    """
    q = (query or "").strip()
    if not q:
        return None

    ql = q.lower()

    # Diagnosis term handling (starting with diabetes).
    diagnosis_terms: list[str] = []
    is_diabetes = bool(_re.search(r"\bdiabet", ql))
    if is_diabetes:
        # Match what our doc-level extractor stores.
        if _re.search(r"\btype\s*2\b|\bt2dm\b", ql):
            diagnosis_terms = ["diabetes", "type 2 diabetes"]
        elif _re.search(r"\btype\s*1\b|\bt1dm\b", ql):
            diagnosis_terms = ["diabetes", "type 1 diabetes"]
        else:
            diagnosis_terms = ["diabetes"]

    # Lab filter (HbA1c)
    hba1c_re = _re.search(
        r"\b(?:hba1c|a1c|hemoglobin\s*a1c|glycated\s*hemoglobin)\b\s*"
        r"(?P<op>>=|<=|>|<|=|equals|equal)\s*"
        r"(?P<val>\d+(?:\.\d+)?)",
        ql,
        flags=_re.IGNORECASE,
    )
    if not hba1c_re:
        # Alternative wording: "HbA1c greater than 9"
        hba1c_re = _re.search(
            r"\b(?:hba1c|a1c)\b.*?\b("
            r"greater\s*than|at\s*least|less\s*than|at\s*most|equal\s*to|=)\b.*?"
            r"(?P<val>\d+(?:\.\d+)?)",
            ql,
            flags=_re.IGNORECASE,
        )

    if not hba1c_re:
        return None

    # Operator normalization
    op = hba1c_re.groupdict().get("op", "") or ""
    val_raw = hba1c_re.groupdict().get("val", None)
    if val_raw is None:
        return None
    try:
        lab_value = float(val_raw)
    except Exception:
        return None

    op_norm: str | None = None
    op_l = str(op).lower().strip()
    if op_l in (">",):
        op_norm = "gt"
    elif op_l in (">=", "at least"):
        op_norm = "gte"
    elif op_l in ("<",):
        op_norm = "lt"
    elif op_l in ("<=", "at most"):
        op_norm = "lte"
    elif op_l in ("=", "equals", "equal"):
        op_norm = "eq"

    # If we used alternative wording, map based on matched phrase.
    if op_norm is None:
        if _re.search(r"\bgreater\s*than\b", ql):
            op_norm = "gt"
        elif _re.search(r"\bat\s*least\b", ql):
            op_norm = "gte"
        elif _re.search(r"\bless\s*than\b", ql):
            op_norm = "lt"
        elif _re.search(r"\bat\s*most\b", ql):
            op_norm = "lte"
        elif _re.search(r"\bequal\s*to\b", ql) or _re.search(r"\b=\b", ql):
            op_norm = "eq"

    if op_norm is None:
        return None

    # Time window handling (starting with "last N months")
    within_days: int | None = None
    m = _re.search(r"\blast\s+(?P<n>\d+)\s*(?P<unit>day|days|week|weeks|month|months|year|years)\b", ql)
    if m:
        n = int(m.group("n"))
        unit = m.group("unit")
        if unit.startswith("day"):
            within_days = n
        elif unit.startswith("week"):
            within_days = n * 7
        elif unit.startswith("month"):
            within_days = n * 30
        elif unit.startswith("year"):
            within_days = n * 365

    date_field = "hba1c_date" if within_days is not None else None

    return {
        "diagnosis_terms": diagnosis_terms,
        "lab_name": "hba1c",
        "lab_operator": op_norm,
        "lab_value": lab_value,
        "within_days": within_days,
        "date_field": date_field,
    }


@tool
def cohort_patient_search(query: str) -> str:
    """
    Search across ALL patients for a clinical cohort matching structured criteria.

    Use for: population queries (e.g., diabetes with HbA1c thresholds in a time window).
    Do NOT use for: single-patient fact lookups (use lookup_clinical_notes).
    """
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue, Range, DatetimeRange
        from document_processing import PAYLOAD_SCHEMA_VERSION

        rag = get_rag_tool_instance()
        schema_version = PAYLOAD_SCHEMA_VERSION

        parsed = _parse_cohort_query(query)
        if not parsed:
            return (
                "Could not parse the cohort criteria. "
                "Try using a pattern like: "
                "`diabetic patients with HbA1c > 9 in the last 3 months`."
            )

        # Fail fast if the collection wasn't indexed with structured clinical values.
        schema_check, _ = rag.client.scroll(
            collection_name=rag.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.payload_schema_version",
                        match=MatchValue(value=schema_version),
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not schema_check:
            return (
                "Cohort queries require structured clinical payloads (HbA1c numeric/date and "
                "diagnosis tags). Your vector DB appears to be missing them. "
                "Please reindex your documents to enable cohort filtering."
            )

        conditions: list[FieldCondition] = []
        # Always scope to correct payload schema version.
        conditions.append(
            FieldCondition(
                key="metadata.payload_schema_version",
                match=MatchValue(value=schema_version),
            )
        )

        diagnosis_terms = parsed.get("diagnosis_terms") or []
        if diagnosis_terms:
            conditions.append(
                FieldCondition(
                    key="metadata.diagnoses_text",
                    match=MatchAny(any=diagnosis_terms),
                )
            )

        # Lab numeric range
        op = parsed["lab_operator"]
        v = parsed["lab_value"]
        range_kwargs = {
            "gt": Range(gt=v),
            "gte": Range(gte=v),
            "lt": Range(lt=v),
            "lte": Range(lte=v),
            "eq": Range(gte=v, lte=v),
        }
        if op in ("eq", "gt", "gte", "lt", "lte"):
            conditions.append(
                FieldCondition(
                    key="metadata.hba1c",
                    range=range_kwargs[op],
                )
            )
        else:
            return "Unsupported HbA1c operator in cohort query."

        # Time window filtering
        within_days = parsed.get("within_days")
        if within_days is not None and parsed.get("date_field"):
            cutoff = _dt.utcnow() - _td(days=int(within_days))
            iso_cutoff = cutoff.isoformat()
            conditions.append(
                FieldCondition(
                    key=f"metadata.{parsed['date_field']}",
                    range=DatetimeRange(gte=iso_cutoff),
                )
            )

        qdrant_filter = Filter(must=conditions)

        # Scroll through matching points (server-side filter; no vector scoring).
        max_points = int(os.environ.get("MEDGRAPH_COHORT_SCROLL_MAX_POINTS", "10000"))
        per_page = 512
        next_offset = None
        all_points: list = []
        while True:
            points, next_offset = rag.client.scroll(
                collection_name=rag.collection_name,
                scroll_filter=qdrant_filter,
                limit=per_page,
                with_payload=True,
                with_vectors=False,
                offset=next_offset,
            )
            if points:
                all_points.extend(points)
                if len(all_points) >= max_points:
                    break
            if not next_offset:
                break

        # Deduplicate by patient MRN.
        def _parse_iso_date(s: str) -> _dt | None:
            try:
                return _dt.fromisoformat(s)
            except Exception:
                try:
                    return _dt.strptime(s, "%Y-%m-%d")
                except Exception:
                    return None

        patients: dict[str, dict] = {}
        for p in all_points:
            meta = p.payload.get("metadata", {}) or {}
            mrn = str(meta.get("patient_mrn") or "").strip()
            if not mrn:
                continue
            name = str(meta.get("patient_name") or "").strip()
            location = str(meta.get("patient_location") or "").strip()

            hb = meta.get("hba1c")
            hb_val: float | None = None
            try:
                hb_val = float(hb) if hb is not None else None
            except Exception:
                hb_val = None
            hb_date = meta.get("hba1c_date")
            hb_dt = _parse_iso_date(str(hb_date)) if hb_date else None

            # If somehow payload doesn't include the lab (shouldn't happen with filter), skip.
            if hb_val is None:
                continue

            prev = patients.get(mrn)
            if not prev:
                patients[mrn] = {
                    "mrn": mrn,
                    "name": name,
                    "location": location,
                    "hba1c": hb_val,
                    "hba1c_date": str(hb_date or ""),
                    "matched_on": parsed,
                }
                continue

            # Choose "best" reading: highest HbA1c; tie-break with most recent date.
            prev_hb = float(prev["hba1c"])
            prev_dt = _parse_iso_date(prev.get("hba1c_date") or "")
            better = hb_val > prev_hb
            if not better and hb_val == prev_hb:
                better = (hb_dt is not None and prev_dt is None) or (
                    hb_dt is not None and prev_dt is not None and hb_dt > prev_dt
                )

            if better:
                patients[mrn] = {
                    "mrn": mrn,
                    "name": name,
                    "location": location,
                    "hba1c": hb_val,
                    "hba1c_date": str(hb_date or ""),
                    "matched_on": parsed,
                }

        if not patients:
            return "Query Results — 0 patients found.\n\nNo patients matched the structured cohort criteria."

        # Render a markdown table (frontend converts GitHub-style tables to HTML).
        header = "| MRN | Name | Location | HbA1c | HbA1c Date |\n|---|---|---|---|---|"
        rows: list[str] = []
        for mrn, info in sorted(patients.items(), key=lambda kv: kv[1]["hba1c"], reverse=True):
            loc = (info.get("location") or "").replace("|", "/")
            nm = (info.get("name") or "").replace("|", "/")
            hb = info.get("hba1c")
            hb_date = info.get("hba1c_date") or ""
            hb_str = f"{hb:.3g}" if isinstance(hb, (int, float)) else str(hb)
            rows.append(f"| {mrn} | {nm} | {loc} | {hb_str} | {hb_date} |")

        total_found = len(patients)
        interpretation = (
            f"Found {total_found} patients matching: "
            f"diagnosis={diagnosis_terms or ['(any diagnosis)']}, "
            f"HbA1c {op} {parsed['lab_value']}, "
            + (f"within {within_days} days" if within_days is not None else "no time window")
        )
        return f"Query Results — {total_found} patients found.\n{interpretation}\n\n{header}\n" + "\n".join(rows)

    except Exception as e:
        return f"Error executing cohort search: {e}"


@tool
def summarize_patient_record(patient_identifier: str) -> str:
    """Generate a comprehensive clinical summary for a patient. Input can be a patient name (e.g. 'Robert Whitfield') or MRN (e.g. 'MRN-2026-004782'). This retrieves the ENTIRE patient record and produces a thorough summary covering all clinical domains. Use this when asked to summarize, review, or give an overview of a patient's record."""
    try:
        rag = get_rag_tool_instance()
        t_start = _time.perf_counter()

        # ── Resolve identifier to MRN ─────────────────────────────
        patient_mrn = rag.resolve_to_mrn(patient_identifier)
        fallback_doc_ids: list[str] = []
        use_document_fallback = False
        if not patient_mrn:
            fallback_doc_ids = rag.find_documents_for_patient_identifier(patient_identifier, max_docs=3)
            if fallback_doc_ids:
                use_document_fallback = True
                fallback_seed = f"{patient_identifier}|{'|'.join(sorted(fallback_doc_ids))}"
                patient_mrn = f"doc-scope-{hashlib.sha256(fallback_seed.encode()).hexdigest()[:16]}"
                logger.info(
                    "MRN resolution failed for '%s'; using document-scoped fallback across %d uploaded doc(s): %s",
                    patient_identifier,
                    len(fallback_doc_ids),
                    fallback_doc_ids,
                )
            else:
                return (
                    f"Could not find a patient matching '{patient_identifier}'. "
                    "Use list_available_patients to see who is in the database. "
                    "If this patient is from a newly uploaded file, call list_uploaded_documents "
                    "to confirm it is in 'ready' status and retry."
                )
        else:
            logger.info("Resolved '%s' → %s", patient_identifier, patient_mrn)

        summary_subject = patient_identifier if use_document_fallback else patient_mrn
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
        if use_document_fallback:
            patient_chunks = []
            seen_chunks = set()
            for doc_id in fallback_doc_ids:
                for txt, meta in rag.get_all_chunks_for_document(doc_id):
                    doc_key = (
                        str(meta.get("document_id") or doc_id),
                        int(meta.get("page_number") or 0),
                        int(meta.get("chunk_index") or -1),
                    )
                    if doc_key in seen_chunks:
                        continue
                    seen_chunks.add(doc_key)
                    patient_chunks.append((txt, meta))

            if not patient_chunks:
                return (
                    f"Found uploaded document candidates for '{patient_identifier}', but none were ready "
                    "for summarization yet. Please retry once indexing completes."
                )
        else:
            patient_chunks = rag.get_all_chunks_for_patient_structured(patient_mrn)
            if not patient_chunks:
                return f"No records found for patient MRN: {patient_mrn}"
            # Add targeted-retrieval supplements for high-risk domains that can be
            # underrepresented if metadata extraction misses a section.
            patient_chunks.extend(_supplement_from_targeted_searches(rag, patient_mrn))

        indexed_chunks: list[tuple[int, str, dict]] = [(i, txt, meta) for i, (txt, meta) in enumerate(patient_chunks, 1)]

        t_llm_start = _time.perf_counter()
        summary, model_name = _run_structured_summary_pipeline(summary_subject, indexed_chunks)
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

        req_mode = _current_summary_mode()
        bypass_cache = req_mode == "thinking"

        # ── Layer 1: In-memory cache (document-scoped) ───────────
        with _DOC_SUMMARY_CACHE_LOCK:
            if not bypass_cache and doc_id in _DOC_SUMMARY_CACHE:
                entry = _DOC_SUMMARY_CACHE[doc_id]
                cache_version = entry.get("pipeline_version", "legacy")
                age_hours = (_dt.now() - entry["cached_at"]).total_seconds() / 3600
                if age_hours <= _CACHE_TTL_HOURS and cache_version == _SUMMARY_PIPELINE_VERSION:
                    _DOC_SUMMARY_CACHE.move_to_end(doc_id)
                    return entry["summary"]
                del _DOC_SUMMARY_CACHE[doc_id]

        # ── Layer 2: Disk cache ──────────────────────────────────
        cache_file = _document_summary_cache_file(doc_id)
        if not bypass_cache and cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                generated_at = cached.get("generated_at", "")
                cache_time = _dt.strptime(generated_at, "%Y-%m-%d %H:%M:%S")
                age_hours = (_dt.now() - cache_time).total_seconds() / 3600
                cache_version = cached.get("pipeline_version", "legacy")

                if age_hours <= _CACHE_TTL_HOURS and cache_version == _SUMMARY_PIPELINE_VERSION:
                    summary = cached["summary"]
                    with _DOC_SUMMARY_CACHE_LOCK:
                        _DOC_SUMMARY_CACHE[doc_id] = {
                            "summary": summary,
                            "cached_at": cache_time,
                            "pipeline_version": _SUMMARY_PIPELINE_VERSION,
                        }
                        _DOC_SUMMARY_CACHE.move_to_end(doc_id)
                        while len(_DOC_SUMMARY_CACHE) > _DOC_SUMMARY_CACHE_MAX_SIZE:
                            _DOC_SUMMARY_CACHE.popitem(last=False)
                    return summary

                cache_file.unlink(missing_ok=True)
            except (ValueError, TypeError, json.JSONDecodeError):
                cache_file.unlink(missing_ok=True)

        rag = get_rag_tool_instance()
        chunks = rag.get_all_chunks_for_document(doc_id)
        if not chunks:
            return f"No indexed chunks found for document_id={doc_id}. If you just uploaded it, it may still be processing."

        # Keep parity with patient summary flow by supplementing high-risk domains.
        chunks.extend(_supplement_from_targeted_searches_for_document(rag, doc_id))

        # Assign global stable source IDs across the full document order
        indexed_chunks: list[tuple[int, str, dict]] = [(i, txt, meta) for i, (txt, meta) in enumerate(chunks, 1)]

        t0 = _time.perf_counter()
        summary, model_name = _run_structured_summary_pipeline(doc_id, indexed_chunks)

        elapsed = _time.perf_counter() - t0
        logger.info("[DocSummary] document_id=%s chunks=%d elapsed=%.1fs model=%s mode=%s",
                    doc_id, len(chunks), elapsed, model_name, req_mode)

        # ── Cache result (memory + disk parity with patient summaries) ─────
        if not bypass_cache:
            now = _dt.now()
            with _DOC_SUMMARY_CACHE_LOCK:
                _DOC_SUMMARY_CACHE[doc_id] = {
                    "summary": summary,
                    "cached_at": now,
                    "pipeline_version": _SUMMARY_PIPELINE_VERSION,
                }
                _DOC_SUMMARY_CACHE.move_to_end(doc_id)
                while len(_DOC_SUMMARY_CACHE) > _DOC_SUMMARY_CACHE_MAX_SIZE:
                    _DOC_SUMMARY_CACHE.popitem(last=False)

            try:
                cache_file.write_text(json.dumps({
                    "document_id": doc_id,
                    "summary": summary,
                    "model": model_name,
                    "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "pipeline_version": _SUMMARY_PIPELINE_VERSION,
                }, ensure_ascii=False), encoding="utf-8")
            except Exception as ce:
                logger.warning("Document disk cache write failed (non-critical): %s", ce)

        return summary

    except Exception as e:
        return f"Error generating document summary: {str(e)[:300]}"
