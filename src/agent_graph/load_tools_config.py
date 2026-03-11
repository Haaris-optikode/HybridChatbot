import os
import yaml
from dotenv import load_dotenv
from pyprojroot import here

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

_GOOGLE_API_KEYS = [
    os.getenv("GOOGLE_AI_API_KEY", ""),
    os.getenv("GOOGLE_AI_API_KEY2", ""),
]
_GOOGLE_API_KEYS = [k for k in _GOOGLE_API_KEYS if k]  # drop empty
_active_key_index = 0


def get_google_api_key() -> str:
    """Return the currently active Google API key."""
    if not _GOOGLE_API_KEYS:
        return ""
    return _GOOGLE_API_KEYS[_active_key_index]


def swap_google_api_key() -> bool:
    """Switch to the backup Google API key. Returns True if swap succeeded."""
    global _active_key_index
    next_idx = _active_key_index + 1
    if next_idx < len(_GOOGLE_API_KEYS) and _GOOGLE_API_KEYS[next_idx]:
        _active_key_index = next_idx
        os.environ["GOOGLE_API_KEY"] = _GOOGLE_API_KEYS[next_idx]
        _key_logger.warning("Switched to backup Google API key (index %d)", next_idx)
        return True
    _key_logger.warning("No more backup Google API keys available")
    return False


class LoadToolsConfig:
    def __init__(self) -> None:
        with open(here("configs/tools_config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        # Set environment variables
        os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY", "")
        os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_AI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
        os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY", "")

        # Primary agent
        self.primary_agent_llm = app_config["primary_agent"]["llm"]
        self.primary_agent_router_llm = app_config["primary_agent"].get(
            "router_llm", app_config["primary_agent"]["llm"])  # fast model for tool routing
        self.primary_agent_thinking_llm = app_config["primary_agent"].get(
            "thinking_llm", "gemini-2.5-pro")  # deeper reasoning model
        self.primary_agent_llm_temperature = app_config["primary_agent"]["llm_temperature"]
        self.llm_provider = app_config["primary_agent"].get("llm_provider", "openai")  # "google" or "openai"

        # Internet Search config
        self.tavily_search_max_results = int(
            app_config["tavily_search_api"]["tavily_search_max_results"])

        # Clinical Notes RAG configs
        self.clinical_notes_rag_llm = app_config["clinical_notes_rag"]["llm"]
        self.clinical_notes_rag_summarization_llm = app_config["clinical_notes_rag"].get(
            "summarization_llm", app_config["clinical_notes_rag"]["llm"])
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