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
import os
import json
import hashlib
import time as _time
from pathlib import Path as _Path
from datetime import datetime as _dt
from dotenv import load_dotenv
import threading

load_dotenv()
TOOLS_CFG = LoadToolsConfig()


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
        print(f"Loading reranker: {model_name}")
        reranker = CrossEncoder(model_name, max_length=512)
        print(f"  Reranker loaded: {model_name}")
        return reranker
    except ImportError:
        print("Warning: sentence-transformers not installed. Reranking disabled.")
        return None
    except Exception as e:
        print(f"Warning: Could not load reranker '{model_name}': {e}")
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
            print(f"Connecting to Qdrant server: {qdrant_url}")
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
            print(f"Using embedded Qdrant: {local_path}")

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
            print(f"Connected to '{collection_name}': {info.points_count} vectors | Status: {info.status}")
        except Exception as e:
            print(f"Warning: Could not get collection info: {e}")

    def search(self, query: str) -> str:
        """
        Two-stage retrieval:
          1. Dense MMR search to get fetch_k candidates (diverse + relevant)
          2. Cross-encoder reranking to select top reranker_top_k results
        """
        try:
            # Stage 1: Dense MMR search for candidates
            docs = self.vectordb.max_marginal_relevance_search(
                query=query,
                k=self.fetch_k,         # get more candidates for reranking
                fetch_k=self.fetch_k * 2,
                lambda_mult=0.5,        # balance relevance vs diversity
            )
            if not docs:
                return f"No relevant documents found for: {query}"

            # Stage 2: Cross-encoder reranking (if available)
            if self.reranker and len(docs) > 1:
                pairs = [[query, doc.page_content] for doc in docs]
                scores = self.reranker.predict(pairs)

                # Sort by reranker score (descending) and take top-k
                scored_docs = sorted(
                    zip(docs, scores), key=lambda x: x[1], reverse=True
                )
                docs = [doc for doc, _ in scored_docs[:self.reranker_top_k]]
            else:
                # Fallback: just take top k from MMR results
                docs = docs[:self.k]

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
            print(f"  Retrieved {len(points)} chunks for {patient_mrn}, "
                  f"merged to {len(merged_parts)} parts in {elapsed:.2f}s")

            return "\n\n".join(merged_parts)
        except Exception as e:
            return f"Error retrieving patient data: {e}"

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
            print(f"    Name resolution: '{identifier}' → {best_mrn} "
                  f"(name='{patients[best_mrn]}', score={best_score})")
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

def _get_summarization_llm():
    """Create the fast summarization LLM instance with output cap."""
    model = getattr(TOOLS_CFG, "clinical_notes_rag_summarization_llm",
                    TOOLS_CFG.clinical_notes_rag_llm)
    return model, ChatOpenAI(
        model=model,
        temperature=0.1,
        max_tokens=4096,          # cap output to ~3K words → saves ~30 s
        api_key=os.getenv("OPENAI_API_KEY"),
    )


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

@tool
def lookup_clinical_notes(query: str) -> str:
    """Search among clinical notes to find specific information. Use this for targeted questions about patient data, medications, labs, vitals, procedures, or any specific clinical detail. Input should be the clinical question."""
    try:
        rag = get_rag_tool_instance()
        return rag.search(query)
    except Exception as e:
        return f"Error searching clinical notes: {e}. Ensure the vector database is initialized."


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
        print(f"  Resolved '{patient_identifier}' → {patient_mrn}")

        # ── Check disk cache ──────────────────────────────────────
        cache_dir = _Path(here("data")) / "summary_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = hashlib.sha256(patient_mrn.encode()).hexdigest()[:16]
        cache_file = cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            # ── 24-hour TTL check ─────────────────────────────────
            generated_at = cached.get("generated_at", "")
            cache_expired = False
            try:
                cache_time = _dt.strptime(generated_at, "%Y-%m-%d %H:%M:%S")
                age_hours = (_dt.now() - cache_time).total_seconds() / 3600
                if age_hours > 24:
                    print(f"  Summary cache EXPIRED ({age_hours:.1f}h old)")
                    cache_expired = True
            except (ValueError, TypeError):
                print(f"  Summary cache: invalid timestamp, treating as expired")
                cache_expired = True

            if not cache_expired:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                current_count = rag.client.count(
                    collection_name=rag.collection_name,
                    count_filter=Filter(must=[
                        FieldCondition(key="metadata.patient_mrn",
                                       match=MatchValue(value=patient_mrn))
                    ]),
                ).count
                if cached.get("chunk_count") == current_count:
                    elapsed = _time.perf_counter() - t_start
                    print(f"  Summary cache HIT for {patient_mrn} ({elapsed:.2f}s)")
                    return cached["summary"]
                print(f"  Summary cache STALE ({cached.get('chunk_count')} → {current_count})")

        # ── Retrieve and merge ALL chunks ─────────────────────────
        full_record = rag.get_all_chunks_for_patient(patient_mrn)
        if full_record.startswith("No records found") or full_record.startswith("Error"):
            return full_record

        record_chars = len(full_record)
        print(f"  Record: {record_chars:,} chars (~{record_chars // 4:,} tokens)")

        # ── Single-pass summarization with output cap ─────────────
        model_name, llm = _get_summarization_llm()
        prompt = _SUMMARY_PROMPT.format(mrn=patient_mrn, record=full_record)
        t_llm_start = _time.perf_counter()
        summary = llm.invoke(prompt).content
        t_llm_end = _time.perf_counter()
        print(f"  LLM: {t_llm_end - t_llm_start:.1f}s, output={len(summary):,} chars "
              f"(model={model_name})")

        t_total = _time.perf_counter() - t_start
        print(f"  Total summarization: {t_total:.1f}s")

        # ── Cache the result ──────────────────────────────────────
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            chunk_count = rag.client.count(
                collection_name=rag.collection_name,
                count_filter=Filter(must=[
                    FieldCondition(key="metadata.patient_mrn",
                                   match=MatchValue(value=patient_mrn))
                ]),
            ).count
            cache_file.write_text(json.dumps({
                "mrn": patient_mrn,
                "chunk_count": chunk_count,
                "summary": summary,
                "model": model_name,
                "generated_at": _time.strftime("%Y-%m-%d %H:%M:%S"),
            }, ensure_ascii=False), encoding="utf-8")
            print(f"  Cached → {cache_file.name}")
        except Exception as ce:
            print(f"  Cache write failed (non-critical): {ce}")

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
