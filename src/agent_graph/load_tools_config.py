import os
import yaml
from dotenv import load_dotenv
from pyprojroot import here

load_dotenv()

# Disable LangSmith tracing to prevent RunTree/Pydantic errors
# (langsmith 0.7.3 has a bug where RunTree references undefined 'Path')
os.environ["LANGCHAIN_TRACING_V2"] = "false"


class LoadToolsConfig:
    def __init__(self) -> None:
        with open(here("configs/tools_config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        # Set environment variables — use official OpenAI key
        os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY", "")
        os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY", "")

        # Primary agent
        self.primary_agent_llm = app_config["primary_agent"]["llm"]
        self.primary_agent_llm_temperature = app_config["primary_agent"]["llm_temperature"]

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

        # Hospital SQL agent configs
        self.hospital_sqlagent_llm = app_config["hospital_sqlagent_configs"]["llm"]
        self.hospital_sqlagent_llm_temperature = float(
            app_config["hospital_sqlagent_configs"]["llm_temperature"])

        # Graph configs
        self.thread_id = str(
            app_config["graph_configs"]["thread_id"])