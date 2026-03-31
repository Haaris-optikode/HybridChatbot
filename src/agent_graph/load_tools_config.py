import os
import yaml
from dotenv import load_dotenv
from pyprojroot import here
from typing import List

load_dotenv()

# ── LangSmith tracing (config-driven) ────────────────────────────────
# Reads the langsmith.tracing flag from tools_config.yml.
# Enable with "true" once langsmith RunTree bug is fixed, or leave "false".
# The RunTree workaround in api.py provides graceful degradation either way.
def _configure_langsmith():
    """Set LangSmith env vars from tools_config.yml before any LangChain import."""
    import yaml
    with open(here("configs/tools_config.yml")) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    ls_cfg = cfg.get("langsmith", {})
    tracing = str(ls_cfg.get("tracing", "false")).lower() == "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if tracing else "false"
    if tracing:
        os.environ["LANGCHAIN_PROJECT"] = ls_cfg.get("project_name", "rag_sqlagent_project")
        # Ensure LANGCHAIN_API_KEY is set (from .env or environment)
        if not os.environ.get("LANGCHAIN_API_KEY"):
            os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")

_configure_langsmith()


# ── Google API key manager (automatic fallback) ─────────────────────
import logging as _logging
_key_logger = _logging.getLogger(__name__)

"""
Production keying rule:
- Use ONLY the single Gemini API key for paid production stability.
- Ignore any secondary/backup key env vars (e.g. `GOOGLE_AI_API_KEY2`).
"""
def _as_list(value) -> List[str]:
    """Normalize config values that may be either a scalar or list."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    s = str(value).strip()
    return [s] if s else []


def _build_google_api_key_pool() -> List[str]:
    """Build ordered Google key pool from environment variables."""
    candidates = [
        os.getenv("GOOGLE_AI_API_KEY", "").strip(),
        os.getenv("GOOGLE_API_KEY", "").strip(),
        os.getenv("GOOGLE_API_KEY_BACKUP", "").strip(),
        os.getenv("GOOGLE_API_KEY_BACKUP_2", "").strip(),
    ]

    pool = []
    seen = set()
    for key in candidates:
        if not key:
            continue
        # Skip obvious non-Google keys (e.g., OpenAI sk-...)
        if key.startswith("sk-"):
            continue
        if key in seen:
            continue
        seen.add(key)
        pool.append(key)
    return pool


_GOOGLE_API_KEYS = _build_google_api_key_pool()


def build_model_chain(primary_model: str, configured_fallbacks: List[str] | None = None) -> List[str]:
    """Build deterministic model failover chain (primary first, then fallbacks)."""
    configured_fallbacks = configured_fallbacks or []
    chain = []

    def _add(model_name: str):
        m = (model_name or "").strip()
        if m and m not in chain:
            chain.append(m)

    _add(primary_model)
    for m in configured_fallbacks:
        _add(m)

    # Default Gemini production-stable fallbacks for resilience.
    if str(primary_model).startswith("gemini"):
        for m in ["gemini-2.5-flash", "gemini-2.5-flash-lite"]:
            _add(m)
    return chain


def get_google_api_key() -> str:
    """Return the currently active Google API key."""
    if not _GOOGLE_API_KEYS:
        return ""
    return _GOOGLE_API_KEYS[0]


def swap_google_api_key() -> bool:
    """
    Enable backup API key swapping for resilience.
    Only swaps to valid Google API keys (not OpenAI keys).
    """
    if len(_GOOGLE_API_KEYS) <= 1:
        _key_logger.warning("No valid backup Google API key available")
        return False

    current = _GOOGLE_API_KEYS.pop(0)
    _GOOGLE_API_KEYS.append(current)
    next_key = _GOOGLE_API_KEYS[0]
    os.environ["GOOGLE_AI_API_KEY"] = next_key
    os.environ["GOOGLE_API_KEY"] = next_key
    _key_logger.info("Swapped to backup Google API key for resilience")
    return True


def set_active_google_api_key(explicit_key: str) -> bool:
    """Set an explicit key as active if present in the key pool."""
    key = (explicit_key or "").strip()
    if not key:
        return False
    if key not in _GOOGLE_API_KEYS:
        return False
    while _GOOGLE_API_KEYS and _GOOGLE_API_KEYS[0] != key:
        _GOOGLE_API_KEYS.append(_GOOGLE_API_KEYS.pop(0))
    os.environ["GOOGLE_AI_API_KEY"] = _GOOGLE_API_KEYS[0]
    os.environ["GOOGLE_API_KEY"] = _GOOGLE_API_KEYS[0]
    return True


def get_google_api_key_pool() -> List[str]:
    """Expose the configured Google key pool for diagnostics."""
    return list(_GOOGLE_API_KEYS)


class LoadToolsConfig:
    def __init__(self) -> None:
        with open(here("configs/tools_config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        # Set environment variables
        os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY", "")
        # Use first configured Google key; supports key rotation/fallback.
        active_google_key = get_google_api_key()
        os.environ['GOOGLE_AI_API_KEY'] = active_google_key
        os.environ['GOOGLE_API_KEY'] = active_google_key
        os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY", "")

        # Primary agent
        self.primary_agent_llm = app_config["primary_agent"]["llm"]
        self.primary_agent_router_llm = app_config["primary_agent"].get(
            "router_llm", app_config["primary_agent"]["llm"])  # fast model for tool routing
        self.primary_agent_thinking_llm = app_config["primary_agent"].get(
            "thinking_llm", "gemini-2.5-pro")  # deeper reasoning model
        self.primary_agent_fallback_llms = _as_list(app_config["primary_agent"].get("fallback_llms"))
        self.primary_agent_router_fallback_llms = _as_list(app_config["primary_agent"].get("router_fallback_llms"))
        self.primary_agent_thinking_fallback_llms = _as_list(app_config["primary_agent"].get("thinking_fallback_llms"))
        self.primary_agent_llm_temperature = app_config["primary_agent"]["llm_temperature"]
        self.llm_provider = app_config["primary_agent"].get("llm_provider", "openai")  # "google" or "openai"

        # Internet Search config
        self.tavily_search_max_results = int(
            app_config["tavily_search_api"]["tavily_search_max_results"])

        # Clinical Notes RAG configs
        self.clinical_notes_rag_llm = app_config["clinical_notes_rag"]["llm"]
        self.clinical_notes_rag_summarization_llm = app_config["clinical_notes_rag"].get(
            "summarization_llm", app_config["clinical_notes_rag"]["llm"])
        self.clinical_notes_rag_fallback_llms = _as_list(app_config["clinical_notes_rag"].get("fallback_llms"))
        self.clinical_notes_rag_summarization_fallback_llms = _as_list(
            app_config["clinical_notes_rag"].get("summarization_fallback_llms")
        )
        self.clinical_notes_rag_llm_temperature = float(
            app_config["clinical_notes_rag"]["llm_temperature"])
        self.clinical_notes_rag_embedding_model = app_config["clinical_notes_rag"]["embedding_model"]
        self.clinical_notes_rag_embedding_dimensions = int(
            app_config["clinical_notes_rag"].get("embedding_dimensions", 768))
        self.clinical_notes_rag_vectordb_directory = str(here(
            app_config["clinical_notes_rag"]["vectordb"]))
        self.clinical_notes_rag_unstructured_docs_directory = str(here(
            app_config["clinical_notes_rag"]["unstructured_docs"]))
        self.clinical_notes_rag_k = app_config["clinical_notes_rag"]["k"]
        self.clinical_notes_rag_fetch_k = app_config["clinical_notes_rag"].get("fetch_k", 20)
        self.clinical_notes_rag_chunk_size = app_config["clinical_notes_rag"]["chunk_size"]
        self.clinical_notes_rag_chunk_overlap = app_config["clinical_notes_rag"]["chunk_overlap"]
        self.clinical_notes_rag_collection_name = app_config["clinical_notes_rag"]["collection_name"]
        self.clinical_notes_rag_reranker_model = app_config["clinical_notes_rag"].get(
            "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.clinical_notes_rag_reranker_top_k = int(
            app_config["clinical_notes_rag"].get("reranker_top_k", 6))
        self.clinical_notes_rag_qdrant_url = app_config["clinical_notes_rag"].get("qdrant_url", "http://localhost:6333")
        self.clinical_notes_rag_qdrant_api_key = app_config["clinical_notes_rag"].get("qdrant_api_key") or os.getenv("QDRANT_API_KEY", "")
        self.clinical_notes_rag_uploads_directory = str(here(
            app_config["clinical_notes_rag"].get("uploads_directory", "data/uploads")))
        self.clinical_notes_rag_max_upload_size_mb = int(
            app_config["clinical_notes_rag"].get("max_upload_size_mb", 50))
        self.clinical_notes_rag_supported_formats = app_config["clinical_notes_rag"].get(
            "supported_formats", [".pdf"])

        # Graph configs
        self.thread_id = str(
            app_config["graph_configs"]["thread_id"])