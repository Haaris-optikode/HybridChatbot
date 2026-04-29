"""
FastAPI backend for the Clinical Intelligence Chatbot.
Replaces the Gradio UI with a REST API that serves a custom HTML/CSS/JS frontend.

P0 Production Hardening:
  - JWT bearer-token authentication on all /api/* endpoints
  - HIPAA-compliant structured audit logging
  - CORS locked to explicit allowed origins
  - Request rate limiting (slowapi)
  - LLM request timeouts + retry (configured in graph/tools)
"""
# ── Fix langsmith RunTree bug (0.7.3 + pydantic 2.9) ────────────────
# Must run BEFORE any langchain import touches the tracer.
import pathlib
try:
    from langsmith.run_trees import RunTree
    if not RunTree.__pydantic_complete__:
        RunTree.__pydantic_parent_namespace__ = RunTree.__pydantic_parent_namespace__ or {}
        RunTree.__pydantic_parent_namespace__['Path'] = pathlib.Path
        RunTree.model_rebuild()
except Exception:
    pass  # Non-critical — tracing will degrade gracefully
# ─────────────────────────────────────────────────────────────────────
import os
import sys
import time
import json
import uuid
import csv
import hashlib
import logging
import asyncio
import re
import random
import queue as _queue_mod
import threading as _threading
import uvicorn
import pandas as pd
from pathlib import Path
from datetime import datetime, date
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Tuple, Literal, Iterator
from dotenv import load_dotenv
from pyprojroot import here
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

# ── Structured logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format=json.dumps({
        "time": "%(asctime)s",
        "level": "%(levelname)s",
        "logger": "%(name)s",
        "message": "%(message)s"
    }),
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("medgraph.api")
audit_logger = logging.getLogger("medgraph.audit")
audit_logger.setLevel(logging.INFO)

# ── HIPAA Audit Log File Handler (rotating, 50 MB × 20 files ≈ 1 GB history) ──
_AUDIT_LOG_DIR = str(here("logs"))
os.makedirs(_AUDIT_LOG_DIR, exist_ok=True)
from logging.handlers import RotatingFileHandler as _RotatingFileHandler
_AUDIT_LOG_PATH = os.path.join(_AUDIT_LOG_DIR, "hipaa_audit.log")
_audit_file_handler = _RotatingFileHandler(
    _AUDIT_LOG_PATH,
    maxBytes=50 * 1024 * 1024,   # 50 MB per file
    backupCount=20,               # keep 20 files → up to 1 GB audit history
    encoding="utf-8",
)
_audit_file_handler.setFormatter(logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "event": %(message)s}',
    datefmt="%Y-%m-%dT%H:%M:%S",
))
audit_logger.addHandler(_audit_file_handler)

# Phase 11 — Harden audit log file permissions at startup (owner read/write only)
# On POSIX systems this restricts access to the process owner (0o600).
# On Windows, use icacls as a post-deployment ops step:
#   icacls logs\hipaa_audit.log /inheritance:r /grant:r "%USERNAME%:(R,W)"
try:
    import stat as _stat
    if os.path.exists(_AUDIT_LOG_PATH):
        os.chmod(_AUDIT_LOG_PATH, _stat.S_IRUSR | _stat.S_IWUSR)  # 0o600
except (OSError, NotImplementedError):
    pass  # Windows: chmod is a no-op for NTFS ACLs; ops team applies icacls

# ── LangGraph agent imports ──────────────────────────────────────────
from chatbot.load_config import LoadProjectConfig
from agent_graph.load_tools_config import LoadToolsConfig, build_model_chain, swap_google_api_key
from agent_graph.build_full_graph import build_graph, _make_llm
from utils.app_utils import create_directory
from chatbot.memory import Memory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from observability.token_cost import (
    RequestUsageCallback,
    summarize_request,
    LEDGER,
    PRICING_TABLE,
    _compute_cost_usd,
)
from agent_graph.tool_clinical_notes_rag import (
    lookup_clinical_notes,
    lookup_patient_orders,
    summarize_uploaded_document,
    summarize_uploaded_documents,
    set_summary_request_mode,
    reset_summary_request_mode,
    clear_summary_caches,
    invalidate_document_summary_cache,
    invalidate_patient_summary_cache,
    set_active_document_scope,
    reset_active_document_scope,
)

PROJECT_CFG = LoadProjectConfig()
TOOLS_CFG = LoadToolsConfig()

_DEFAULT_PROVIDER = (getattr(TOOLS_CFG, "llm_provider", "openai") or "openai").strip().lower()
if _DEFAULT_PROVIDER not in ("openai", "google"):
    _DEFAULT_PROVIDER = "openai"

_PROVIDER_MODEL_CHAINS = {
    "openai": {
        "router": build_model_chain(
            os.getenv("MEDGRAPH_OPENAI_ROUTER_MODEL", "gpt-4.1-mini"),
            [os.getenv("MEDGRAPH_OPENAI_ROUTER_FALLBACK", "gpt-4.1-nano")],
        ),
        "synth": build_model_chain(
            os.getenv("MEDGRAPH_OPENAI_SYNTH_MODEL", "gpt-4.1-mini"),
            [os.getenv("MEDGRAPH_OPENAI_SYNTH_FALLBACK", "gpt-4.1-nano")],
        ),
        "thinking": build_model_chain(
            os.getenv("MEDGRAPH_OPENAI_THINKING_MODEL", "gpt-5.4-mini"),
            [os.getenv("MEDGRAPH_OPENAI_THINKING_FALLBACK", "gpt-4.1")],
        ),
    },
    "google": {
        "router": build_model_chain(
            TOOLS_CFG.primary_agent_router_llm,
            getattr(TOOLS_CFG, "primary_agent_router_fallback_llms", []),
        ),
        "synth": build_model_chain(
            TOOLS_CFG.primary_agent_llm,
            getattr(TOOLS_CFG, "primary_agent_fallback_llms", []),
        ),
        "thinking": build_model_chain(
            TOOLS_CFG.primary_agent_thinking_llm,
            getattr(TOOLS_CFG, "primary_agent_thinking_fallback_llms", []),
        ),
    },
}
_MODEL_FALLBACK_STATE = {
    "openai": {"router_idx": 0, "synth_idx": 0, "thinking_idx": 0},
    "google": {"router_idx": 0, "synth_idx": 0, "thinking_idx": 0},
}


def _apply_graph_model_overrides(provider: str) -> None:
    """Apply current fallback-model indexes to graph build environment vars."""
    p = provider if provider in ("openai", "google") else _DEFAULT_PROVIDER
    chains = _PROVIDER_MODEL_CHAINS[p]
    state = _MODEL_FALLBACK_STATE[p]
    os.environ["MEDGRAPH_LLM_PROVIDER_OVERRIDE"] = p
    os.environ["MEDGRAPH_ROUTER_MODEL_OVERRIDE"] = chains["router"][state["router_idx"]]
    os.environ["MEDGRAPH_SYNTH_MODEL_OVERRIDE"] = chains["synth"][state["synth_idx"]]
    os.environ["MEDGRAPH_THINKING_MODEL_OVERRIDE"] = chains["thinking"][state["thinking_idx"]]


def _advance_graph_model_fallback(thinking_mode: bool, provider: str) -> bool:
    """Advance to next configured model candidate. Returns True if advanced."""
    p = provider if provider in ("openai", "google") else _DEFAULT_PROVIDER
    chains = _PROVIDER_MODEL_CHAINS[p]
    state = _MODEL_FALLBACK_STATE[p]
    if thinking_mode and state["thinking_idx"] < len(chains["thinking"]) - 1:
        state["thinking_idx"] += 1
        _apply_graph_model_overrides(p)
        logger.warning("Model fallback advanced (thinking): router=%s synth=%s thinking=%s",
                       os.environ.get("MEDGRAPH_ROUTER_MODEL_OVERRIDE"),
                       os.environ.get("MEDGRAPH_SYNTH_MODEL_OVERRIDE"),
                       os.environ.get("MEDGRAPH_THINKING_MODEL_OVERRIDE"))
        return True
    if not thinking_mode and state["synth_idx"] < len(chains["synth"]) - 1:
        state["synth_idx"] += 1
        _apply_graph_model_overrides(p)
        logger.warning("Model fallback advanced (normal): router=%s synth=%s",
                       os.environ.get("MEDGRAPH_ROUTER_MODEL_OVERRIDE"),
                       os.environ.get("MEDGRAPH_SYNTH_MODEL_OVERRIDE"))
        return True
    if state["router_idx"] < len(chains["router"]) - 1:
        state["router_idx"] += 1
        _apply_graph_model_overrides(p)
        logger.warning("Model fallback advanced (router): router=%s",
                       os.environ.get("MEDGRAPH_ROUTER_MODEL_OVERRIDE"))
        return True
    return False


_GRAPH_CACHE: dict[str, tuple] = {}


def _normalize_provider(provider: Optional[str]) -> str:
    p = (provider or "").strip().lower()
    if p in ("openai", "google"):
        return p
    return _DEFAULT_PROVIDER


def _get_graph_pair(provider: str):
    p = _normalize_provider(provider)
    # Ensure request-scoped provider/model overrides are active even on cache hits.
    _apply_graph_model_overrides(p)
    if p in _GRAPH_CACHE:
        return _GRAPH_CACHE[p]
    g = build_graph(thinking_mode=False)
    tg = build_graph(thinking_mode=True)
    _GRAPH_CACHE[p] = (g, tg)
    return g, tg


graph, thinking_graph = _get_graph_pair(_DEFAULT_PROVIDER)
create_directory("memory")

MEMORY_DIR = str(PROJECT_CFG.memory_dir)
FEEDBACK_DIR = str(here("feedback"))
os.makedirs(FEEDBACK_DIR, exist_ok=True)

# ── Prometheus Metrics (P2.22) ───────────────────────────────────────
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST,
)

PROM_REQUEST_LATENCY = Histogram(
    "medgraph_request_duration_seconds",
    "Latency of API requests",
    ["endpoint", "method"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)
PROM_REQUEST_ERRORS = Counter(
    "medgraph_request_errors_total",
    "Total API errors",
    ["endpoint", "status_code"],
)
PROM_CACHE_HITS = Counter(
    "medgraph_cache_hits_total",
    "Summary cache hits",
    ["layer"],  # "memory" or "disk"
)
PROM_CACHE_MISSES = Counter(
    "medgraph_cache_misses_total",
    "Summary cache misses",
)
PROM_ACTIVE_SESSIONS = Gauge(
    "medgraph_active_sessions",
    "Current number of active sessions",
)
PROM_LLM_INPUT_TOKENS = Counter(
    "medgraph_llm_input_tokens_total",
    "Total LLM input tokens",
    ["endpoint"],
)
PROM_LLM_OUTPUT_TOKENS = Counter(
    "medgraph_llm_output_tokens_total",
    "Total LLM output tokens",
    ["endpoint"],
)
PROM_LLM_COST_USD = Counter(
    "medgraph_llm_cost_usd_total",
    "Total estimated LLM cost in USD",
    ["endpoint"],
)

# ── Allowed CORS Origins ─────────────────────────────────────────────
# Comma-separated list of allowed origins, or "*" for dev mode.
_cors_origins_raw = os.getenv("MEDGRAPH_CORS_ORIGINS", "http://localhost:7860,http://127.0.0.1:7860")
ALLOWED_ORIGINS = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
logger.info("CORS allowed origins: %s", ALLOWED_ORIGINS)


def _normalize_content(content) -> str:
    """Normalize LLM message content to a plain string.
    Gemini returns content as a list of parts: [{"type": "text", "text": "..."}]
    Gemini thinking models include [{"type":"thinking",...}, {"type":"text",...}]
    OpenAI returns a plain string. This function handles all forms."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                # Skip thinking/reasoning parts from Gemini thinking models
                if item.get("type") == "thinking":
                    continue
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content) if content else ""


def _sentence_candidates(text: str) -> List[str]:
    t = re.sub(r"\s+", " ", (text or "")).strip()
    if not t:
        return []
    # Prefer natural sentence boundaries; fallback to line-level chunks.
    sents = [s.strip(" -\t\n") for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    if not sents:
        sents = [s.strip(" -\t\n") for s in t.split("\n") if s.strip()]
    # De-duplicate while preserving order.
    seen = set()
    out: List[str] = []
    for s in sents:
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def _bullets(items: List[str], max_items: int = 5) -> str:
    cleaned = []
    for item in items:
        it = re.sub(r"\s+", " ", (item or "")).strip(" -\t\n")
        if it:
            cleaned.append(it)
    if not cleaned:
        return "- No key points available."
    return "\n".join(f"- {x}" for x in cleaned[:max_items])


def _simple_medical_to_plain(text: str) -> str:
    replacements = {
        r"\bHbA1c\b": "A1C blood sugar test",
        r"\bA1C\b": "A1C blood sugar test",
        r"\bhypertension\b": "high blood pressure",
        r"\bdyspnea\b": "shortness of breath",
        r"\bedema\b": "swelling",
        r"\bmyocardial infarction\b": "heart attack",
        r"\bBNP\b": "BNP heart strain test",
        r"\bcreatinine\b": "kidney function blood test",
    }
    out = text
    for pat, repl in replacements.items():
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    return out


def _extract_by_keywords(sentences: List[str], keywords: List[str], max_items: int = 3) -> List[str]:
    out: List[str] = []
    for s in sentences:
        sl = s.lower()
        if any(k in sl for k in keywords):
            out.append(s)
            if len(out) >= max_items:
                break
    return out


def _followup_checks(query: str) -> List[str]:
    q = (query or "").lower()
    checks: List[str] = []
    if any(k in q for k in ["hba1c", "a1c", "glucose", "diabet"]):
        checks.extend([
            "Confirm trend over time with exact dates (not one value).",
            "Verify current diabetes medication plan and adherence.",
            "Check kidney-function context when adjusting therapy.",
        ])
    if any(k in q for k in ["medication", "dose", "drug", "rx"]):
        checks.extend([
            "Confirm dose, route, and frequency against latest order timestamp.",
            "Check duplicate therapy and high-risk interaction flags.",
        ])
    if any(k in q for k in ["lab", "critical", "abnormal"]):
        checks.extend([
            "Compare abnormal values to reference ranges and prior baseline.",
            "Verify whether repeat testing was ordered/completed.",
        ])
    if not checks:
        checks.extend([
            "Confirm date/time recency of the cited findings.",
            "Verify missing fields before making clinical decisions.",
        ])
    return checks[:4]


def _format_highlight_mode(raw_reply: str) -> str:
    sents = _sentence_candidates(raw_reply)
    diagnoses = _extract_by_keywords(sents, ["diagnos", "impression", "condition", "problem"], max_items=4)
    meds = _extract_by_keywords(sents, ["medication", "dose", "mg", "insulin", "metformin", "rx"], max_items=4)
    labs = _extract_by_keywords(sents, ["lab", "hba1c", "a1c", "glucose", "creatinine", "bnp", "wbc", "hgb"], max_items=4)
    plan = _extract_by_keywords(sents, ["plan", "recommend", "continue", "start", "stop", "monitor", "ordered"], max_items=4)
    followup = _extract_by_keywords(sents, ["follow-up", "follow up", "return", "appointment", "recheck", "within"], max_items=3)
    safety = _extract_by_keywords(sents, ["critical", "urgent", "emergency", "warning", "red flag", "severe"], max_items=3)

    def _list_or_na(items: List[str]) -> str:
        return "; ".join(items) if items else "Not documented in retrieved response."

    return (
        f"DIAGNOSES: {_list_or_na(diagnoses)}\n"
        f"ACTIVE MEDICATIONS: {_list_or_na(meds)}\n"
        f"CRITICAL LABS: {_list_or_na(labs)}\n"
        f"PLAN ACTIONS: {_list_or_na(plan)}\n"
        f"FOLLOW-UP: {_list_or_na(followup)}\n"
        f"SAFETY ALERTS: {_list_or_na(safety)}"
    )


def _format_regenerate_mode(raw_reply: str, query: str) -> str:
    sents = _sentence_candidates(raw_reply)
    if not sents:
        return raw_reply

    # Intentionally vary structure to ensure a visible, useful alternative rendering.
    style = random.choice(["timeline", "problem", "qa", "checklist"])

    if style == "timeline":
        ordered = sents[:6]
        lines = [f"{i + 1}. {s}" for i, s in enumerate(ordered)]
        return "Alternative View (Timeline):\n" + "\n".join(lines)

    if style == "problem":
        key = sents[0]
        support = sents[1:5]
        return (
            "Alternative View (Problem-Oriented):\n"
            f"Primary Problem: {key}\n"
            "Supporting Findings:\n"
            + _bullets(support, max_items=4)
        )

    if style == "qa":
        q_clean = (query or "the requested clinical question").strip()
        answer = sents[0]
        evidence = _bullets(sents[1:5], max_items=4)
        return (
            "Alternative View (Q/A Brief):\n"
            f"Q: {q_clean}\n"
            f"A: {answer}\n"
            "Evidence:\n"
            f"{evidence}"
        )

    # checklist
    items = sents[:5]
    return (
        "Alternative View (Clinical Checklist):\n"
        + "\n".join(f"[ ] {x}" for x in items)
    )


def _apply_reply_mode_transform_sync(mode: Optional[str], raw_reply: str, query: str) -> str:
    if not mode:
        return raw_reply

    base = (raw_reply or "").strip()
    if not base:
        return raw_reply

    sents = _sentence_candidates(base)
    if not sents:
        return raw_reply

    if mode == "shorten":
        return "Short Summary:\n" + _bullets(sents, max_items=5)

    if mode == "simplify":
        plain = _simple_medical_to_plain(base)
        plain_sents = _sentence_candidates(plain)
        return "Plain-English Summary:\n" + _bullets(plain_sents, max_items=5)

    if mode == "highlight":
        return _format_highlight_mode(base)

    if mode == "refine":
        top = sents[:4]
        checks = _followup_checks(query)
        return (
            "Refined Clinical Focus:\n"
            "Key Points:\n"
            f"{_bullets(top, max_items=4)}\n"
            "Suggested Follow-Up Checks:\n"
            f"{_bullets(checks, max_items=4)}"
        )

    if mode == "enhance":
        top = sents[:5]
        checks = _followup_checks(query)
        return (
            "Enhanced Clinical Brief:\n"
            "Documented Findings:\n"
            f"{_bullets(top, max_items=5)}\n"
            "Additional Clinical Context To Verify:\n"
            f"{_bullets(checks, max_items=4)}"
        )

    if mode == "regenerate":
        return _format_regenerate_mode(base, query)

    return raw_reply


async def _apply_reply_mode_transform(mode: Optional[str], raw_reply: str, query: str) -> str:
    return await asyncio.to_thread(_apply_reply_mode_transform_sync, mode, raw_reply, query)

# ── Rate Limiting ────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── FastAPI app ──────────────────────────────────────────────────────
app = FastAPI(title="MedGraph AI", version="2.0.0")
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please slow down."},
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
    allow_credentials=True,
)

# Serve static frontend files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


AUDIT_SYSTEM_USER = {"sub": "system", "role": "system"}


def _audit_log(event: str, _compat_user=None, **extra):
    """Write a structured HIPAA audit log entry."""
    if _compat_user is None:
        audit_user = AUDIT_SYSTEM_USER
    elif isinstance(_compat_user, dict):
        audit_user = _compat_user
    else:
        audit_user = {}

    entry = {
        "event": event,
        "user_id": str(audit_user.get("sub") or audit_user.get("user_id") or "unknown"),
        "role": str(audit_user.get("role") or "unknown"),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        **extra,
    }
    audit_logger.info(json.dumps(entry))


# ── Phase 11: HIPAA audit enrichment helpers ─────────────────────────────────

# Tools that query the Qdrant vector store (patient clinical data)
_QDRANT_TOOLS: frozenset = frozenset({
    "lookup_clinical_notes",
    "lookup_patient_orders",
    "cohort_patient_search",
    "list_available_patients",
    "list_uploaded_documents",
    "summarize_patient_record",
    "summarize_uploaded_document",
    "summarize_uploaded_documents",
})

# Tools that hit the open web (Tavily)
_WEB_TOOLS: frozenset = frozenset({
    "tavily_search_results_json",
})


def _classify_data_sources(tools_invoked: list) -> str:
    """Classify which data sources were accessed based on tool names.

    Returns one of: "qdrant", "web", "qdrant+web", "tool", "none".
    """
    invoked_set = set(tools_invoked)
    has_qdrant = bool(invoked_set & _QDRANT_TOOLS)
    has_web = bool(invoked_set & _WEB_TOOLS)
    if has_qdrant and has_web:
        return "qdrant+web"
    if has_qdrant:
        return "qdrant"
    if has_web:
        return "web"
    if invoked_set:
        return "tool"
    return "none"


def _extract_audit_metadata(events: list) -> dict:
    """Extract patient MRN, tools invoked, data sources, and grounding score
    from a completed LangGraph event list (stream_mode="values").

    Returns a dict with zero or more of:
      patient_mrn   — the active patient MRN resolved by the graph
      tools_invoked — ordered, deduplicated list of tool names called
      data_sources  — classification string (qdrant / web / qdrant+web / none)
      grounding_score — citation_coverage float from grounding verification
    """
    if not events:
        return {"data_sources": "none"}

    final_state = events[-1]

    # MRN / document scope — propagated through state by every node that calls a patient tool
    patient_mrn = (final_state.get("active_patient_mrn") or "").strip() or None
    active_document_ids = final_state.get("active_document_ids") or []

    # Tools invoked — ToolMessage objects carry the tool function name
    all_messages = final_state.get("messages", [])
    tools_invoked: list = list(dict.fromkeys(  # preserve insertion order, dedup
        getattr(m, "name", "")
        for m in all_messages
        if m.__class__.__name__ == "ToolMessage" and getattr(m, "name", "")
    ))

    data_sources = _classify_data_sources(tools_invoked)

    # Grounding — synthesizer node stores result in state (Phase 11 addition)
    grounding = final_state.get("grounding_result") or {}
    grounding_score = grounding.get("citation_coverage")  # float or None

    result: dict = {"data_sources": data_sources}
    if patient_mrn:
        result["patient_mrn"] = patient_mrn
    if active_document_ids:
        result["document_ids"] = [str(d) for d in active_document_ids if str(d)]
    if tools_invoked:
        result["tools_invoked"] = tools_invoked
    if grounding_score is not None:
        result["grounding_score"] = grounding_score
    return result


def _build_scope_context_block(req: "ChatRequest") -> str:
    """Return a system-prompt addition that tells the LLM which patient/documents are pre-scoped.

    Called AFTER _validate_document_scope_or_raise so that req.patient_mrn is already
    populated (auto-derived from document metadata when not explicitly supplied).

    This block is appended to SYSTEM_PROMPT before the graph is invoked so the router
    can route directly to the correct tool with the correct arguments, rather than
    calling list_uploaded_documents() first and then asking the user to choose.

    Backend enforcement of scope constraints is done separately via
    _scope_lookup_tool_calls() in build_full_graph.py — this block is guidance only.
    """
    mrn = (getattr(req, "patient_mrn", None) or "").strip()
    doc_ids: List[str] = list(getattr(req, "document_ids", None) or [])
    if not mrn and not doc_ids:
        return ""

    lines: List[str] = [
        "\n\nACTIVE REQUEST SCOPE (pre-validated by backend, do not override or ignore):",
    ]
    if mrn:
        lines.append(f"- Patient MRN in scope: {mrn}")
        lines.append(f"  Use patient_mrn='{mrn}' for ALL lookup_clinical_notes calls.")
        lines.append(f"  When user says 'this patient', 'the patient', or asks about the patient without naming one, they mean MRN {mrn}.")
    if doc_ids:
        lines.append(f"- Document ID(s) in scope: {', '.join(doc_ids)}")
        lines.append("  For insurance, benefit, copay, deductible, coverage, or other administrative questions: use lookup_clinical_notes, not a clinical summary tool.")
        lines.append("  These documents are already selected. Do NOT call list_uploaded_documents() — the scope is resolved.")
        if len(doc_ids) == 1:
            lines.append(
                f"  For document questions (what is this about, summarize, contents): "
                f"call summarize_uploaded_document(document_id='{doc_ids[0]}') directly."
            )
            lines.append(
                f"  For specific facts from this document: "
                f"call lookup_clinical_notes(query=<user question>, document_id='{doc_ids[0]}')."
            )
            lines.append(
                f"  For complete orders/medications/labs from this document: "
                f"call lookup_patient_orders(patient_identifier='{mrn or 'selected patient'}', document_id='{doc_ids[0]}')."
            )
        else:
            lines.append(
                f"  For summaries/overviews across these selected documents: "
                f"call summarize_uploaded_documents(document_ids={doc_ids!r}, patient_mrn='{mrn}')."
            )
            lines.append(
                "  For specific facts across these selected documents: call lookup_clinical_notes with the patient_mrn above. "
                "Backend scope will restrict retrieval to the selected document IDs."
            )
            lines.append(
                f"  For complete orders/medications/labs across these selected documents: "
                f"call lookup_patient_orders(patient_identifier='{mrn or 'selected patient'}', document_ids={doc_ids!r})."
            )
    return "\n".join(lines)


# ── System prompt ────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are MedGraph AI — a clinical data retrieval assistant for EHR systems.

ROLE: Retrieve and organize documented clinical data only. You are NOT a licensed physician. Never diagnose, prescribe, or advise medication changes. Recommend consulting a provider for clinical decisions.

DOMAIN: Medical/clinical queries and brief social greetings only. For non-medical requests, reply exactly: "I can only answer medical and clinical queries."

TOOL SELECTION — strict, no exceptions:
- Specific patient fact (labs, meds, vitals, allergies, diagnoses, imaging, history, DOB) → lookup_clinical_notes(query, patient_mrn)
- "All orders" / complete order list / comprehensive meds+labs+procedures → lookup_patient_orders(patient_identifier)
- Explicit full record summary or overview only → summarize_patient_record(patient_identifier)
- Population/cohort query across multiple patients → cohort_patient_search(query)
- General medical knowledge (not patient-specific) → tavily_search_results_json(query)
- Unscoped uploaded document query (no document_ids supplied by API/request) → list_uploaded_documents() FIRST to get document_id + verify "ready", then lookup_clinical_notes(query, document_id)
- Unscoped uploaded document summary (no document_ids supplied by API/request) → list_uploaded_documents() FIRST, then summarize_uploaded_document(document_id)
- Do NOT use summarize_patient_record for specific facts. Do NOT use lookup_clinical_notes for all-orders requests.

CLINICAL REASONING (comparisons, trends, medication changes, risk, critical findings):
Call lookup_clinical_notes at minimum 3 times with different targeted queries:
1. The specific clinical domain (e.g., "cardiac events atrial fibrillation EF")
2. "hospital course interventions complications [topic]"
3. "discharge summary plan follow-up warning signs [topic]"
Then synthesize results with clinical severity prioritization:
- Lead with the clinical story — what happened and what was most dangerous. Never lead with raw lab values.
- Prioritize: (1) immediately life-threatening events, (2) major complications requiring intervention, (3) notable trends, (4) monitoring needs.
- Labs are supporting evidence, not the headline. Frame them clinically: "BNP of 1,842 pg/mL confirmed severe decompensation" not "BNP 1,842 [ABNORMAL]."
- Always include discharge warning signs and return-to-ER criteria if documented.
- Include medication changes that occurred during hospitalization — they represent the clinical team's response to what mattered most.

MULTI-HOP RULES:
- "First time" / "when started" questions: search early days first, verify before answering — never answer without confirming earliest occurrence.
- Trend questions (creatinine, BNP, weight): MUST search twice — baseline/admission AND later/discharge values.
- Temporal precision: home/admission medications ≠ discharge medications. State which list explicitly.

PATIENT CONTEXT: When user says "the patient" without a name, use the most recently discussed patient. If none discussed, ask for name or MRN. NEVER fabricate. If a name doesn't match exactly, report the mismatch explicitly.

ANSWER RULES:
- Facts only from retrieved sources. If not documented, say so. Never estimate or extrapolate.
- Cite inline: [S1], [S2] after major clinical facts. Not required after every word — use for key claims only.
- Exact values: lab with units + reference range + flag if abnormal. Medications with dose/route/frequency.
- 80–150 words for simple factual queries. 200–500 words for multi-part questions. Analytical synthesis: as thorough as needed.
- Use bullet points and bold for key values. Headers only when answer has 3+ distinct categories."""

# ── Thread-safe session store (P1.8 + P1.14) ───────────────────────
# Abstraction that makes it trivial to swap in Redis later.
# All access is lock-protected to prevent race conditions.

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[float] = None


class SessionStore:
    """Thread-safe in-memory session store with idle-session eviction.
    Drop-in replacement for a plain dict. To migrate to Redis, subclass
    and override get/set/delete/clear."""

    def __init__(self, max_idle_hours: float = 24.0):
        self._lock = _threading.Lock()
        self._data: Dict[str, List[ChatMessage]] = {}
        self._last_access: Dict[str, float] = {}  # sid → timestamp
        self._max_idle_seconds = max_idle_hours * 3600

    def get(self, sid: str) -> List[ChatMessage]:
        with self._lock:
            self._last_access[sid] = time.time()
            return self._data.get(sid, [])

    def set(self, sid: str, history: List[ChatMessage]) -> None:
        with self._lock:
            self._data[sid] = history
            self._last_access[sid] = time.time()

    def delete(self, sid: str) -> None:
        with self._lock:
            self._data.pop(sid, None)
            self._last_access.pop(sid, None)

    def contains(self, sid: str) -> bool:
        with self._lock:
            return sid in self._data

    def clear_all(self) -> int:
        with self._lock:
            n = len(self._data)
            self._data.clear()
            self._last_access.clear()
            return n

    def evict_idle(self) -> int:
        """Remove sessions that haven't been accessed within max_idle_seconds."""
        now = time.time()
        evicted = 0
        with self._lock:
            expired_sids = [
                sid for sid, ts in self._last_access.items()
                if now - ts > self._max_idle_seconds
            ]
            for sid in expired_sids:
                del self._data[sid]
                del self._last_access[sid]
                evicted += 1
        if evicted:
            logger.info("Evicted %d idle sessions", evicted)
        return evicted

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


sessions = SessionStore(max_idle_hours=24.0)

# ── Conversation history trimming (P1.9) ─────────────────────────────
# Only the last MAX_HISTORY_TURNS exchanges are sent to the LLM to prevent
# unbounded token growth and increasing latency per turn.
MAX_HISTORY_TURNS = int(os.getenv("MEDGRAPH_MAX_HISTORY_TURNS", "4"))  # pairs of user+assistant

# ── Input validation constants (P1.11) ───────────────────────────────
MAX_MESSAGE_LENGTH = int(os.getenv("MEDGRAPH_MAX_MESSAGE_LENGTH", "10000"))  # characters
_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous",
    "disregard your instructions",
    "forget your system prompt",
    "you are now",
    "act as if you",
    "new persona",
    "override your",
    "bypass your",
    "reveal your system",
    "show me your prompt",
    "print your instructions",
]


def _validate_message(message: str) -> str:
    """Validate and sanitize user input. Raises HTTPException on violation."""
    if not message or not message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if len(message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Message too long ({len(message):,} chars). Maximum is {MAX_MESSAGE_LENGTH:,}.",
        )
    msg_lower = message.lower()
    for pattern in _INJECTION_PATTERNS:
        if pattern in msg_lower:
            logger.warning("Prompt injection attempt detected: %s", pattern)
            raise HTTPException(
                status_code=400,
                detail="Message contains disallowed content.",
            )
    return message.strip()


def _trim_history(history: List[ChatMessage]) -> List[ChatMessage]:
    """Return only the last MAX_HISTORY_TURNS user+assistant pairs."""
    if MAX_HISTORY_TURNS <= 0:
        return history  # 0 = no trimming
    # Each turn = 2 messages (user + assistant)
    max_messages = MAX_HISTORY_TURNS * 2
    if len(history) <= max_messages:
        return history
    return history[-max_messages:]


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    tool_override: Optional[str] = None  # "vector_db", "web_search" or None (auto)
    thinking: Optional[bool] = False     # use deeper reasoning model when True
    llm_provider: Optional[str] = None   # "openai" | "google"
    reply_mode: Optional[Literal[
        "refine",
        "shorten",
        "simplify",
        "enhance",
        "regenerate",
        "highlight"
    ]] = None
    # Phase 3 — EHR document scope
    patient_id: Optional[str] = None        # EHR patient identifier
    patient_mrn: Optional[str] = None       # Patient MRN for scoping
    tenant_id: Optional[str] = None         # Optional tenant scope for EHR requests
    org_id: Optional[str] = None            # Optional organization scope for EHR requests
    document_ids: List[str] = Field(default_factory=list)  # Selected document IDs to restrict retrieval


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    latency_ms: int
    usage: Dict[str, object]


class FeedbackRequest(BaseModel):
    session_id: str
    message_index: int  # which bot message (0-based among bot messages)
    user_query: str
    bot_response: str
    vote: str  # "up" or "down"
    comment: Optional[str] = None


def _simple_capabilities_text() -> str:
    """Prebuilt capabilities response for low-latency conversational queries."""
    return (
        "I can help you with four main things:\n"
        "1. Search specific patient details from clinical records (allergies, meds, labs, vitals, diagnoses, procedures).\n"
        "2. Generate full patient summaries from complete records.\n"
        "3. Retrieve patient order data (labs, meds, procedures, referrals, discharge prescriptions).\n"
        "4. Do web-backed medical information lookups when needed.\n\n"
        "Share a patient name or MRN and what you need, and I will fetch it."
    )


def _fast_path_reply(message: str) -> Optional[str]:
    """Return an instant canned response for simple conversational requests.

    This avoids unnecessary LLM/tool latency for greetings and capability prompts.
    """
    m = (message or "").strip().lower()
    if not m:
        return None

    is_greeting = bool(re.search(r"\b(hi|hello|hey|good morning|good afternoon|good evening|how are you)\b", m))
    is_social_ack = bool(re.fullmatch(
        r"\s*(great|awesome|nice|cool|thanks|thank you|thx|ok|okay|got it|understood|sounds good|perfect|sure)\s*[!.?]*\s*",
        m,
    ))
    asks_capabilities = (
        "what can you do" in m
        or "your capabilities" in m
        or "help me" in m
        or "who are you" in m
        or "about yourself" in m
    )

    if asks_capabilities:
        return _simple_capabilities_text()
    if is_greeting:
        return (
            "Hello, I am MedGraph AI. I can search patient records, summarize full charts, "
            "retrieve order history, and answer medical info questions with web search when needed."
        )
    if is_social_ack:
        return "Happy to help. Share any medical question when you are ready."
    return None


def _predefined_identity_reply(message: str) -> Optional[str]:
    """Return fixed branding/identity answer for builder/owner-style queries."""
    m = (message or "").strip().lower()
    if not m:
        return None

    builder_patterns = [
        r"\bwho\s+built\s+you\b",
        r"\bwho\s+made\s+you\b",
        r"\bwho\s+created\s+you\b",
        r"\bwho\s+developed\s+you\b",
        r"\bwho\s+owns\s+you\b",
        r"\bwho\s+is\s+your\s+owner\b",
        r"\bwho\s+is\s+behind\s+you\b",
    ]
    if any(re.search(p, m) for p in builder_patterns):
        return "Optikode"
    return None


def _auto_tool_override(message: str) -> Optional[str]:
    m = (message or "").strip().lower()
    if not m:
        return None

    # Full record summary
    if (("summar" in m or "overview" in m or "complete record" in m or "full record" in m)
            and ("patient" in m or "mrn" in m or "record" in m)):
        return "summarize"

    # All-orders (deterministic — never let model decide this)
    _ORDERS_TRIGGERS = [
        "all orders", "complete order", "order list", "what was ordered",
        "all medications", "all labs", "all procedures", "all referrals",
        "list all meds", "list all drugs", "full med list", "complete medication",
        "discharge prescriptions", "all prescriptions",
    ]
    if any(trigger in m for trigger in _ORDERS_TRIGGERS):
        return "orders"

    # Cohort / population queries
    if re.search(
        r"\b(all|show|find|list|which).{0,30}(patients?|diabetic|hba1c|a1c)"
        r".{0,40}(with|having|where|above|below|greater|less)\b",
        m,
    ):
        return "cohort"

    # Patient-specific factual queries (existing logic, unchanged)
    patient_fact_terms = (
        "disease", "diagnosis", "diagnosed", "condition", "suffering", "symptom", "allergy",
        "medication", "medicine", "lab", "vital", "history", "record", "chart",
    )
    has_patient_fact_intent = any(t in m for t in patient_fact_terms)
    name_like_phrases = re.findall(r"\b([a-z]{2,}(?:\s+[a-z]{2,}){1,2})\b", m)
    name_like_stop = {
        "what", "which", "who", "when", "where", "why", "how", "the", "a", "an", "is", "are",
        "do", "does", "did", "can", "could", "would", "should", "for", "of", "about", "from",
        "with", "give", "tell", "show", "find", "check", "patient", "record", "summary",
        "uploaded", "documents", "there", "someone", "anybody", "person", "people",
        "suffering", "disease", "diagnosis", "condition",
    }
    medical_term_tokens = {
        "disease", "diagnosis", "diagnosed", "condition", "suffering", "symptom", "allergy",
        "medication", "medications", "medicine", "medicines", "lab", "labs", "vital", "vitals",
        "history", "record", "records", "chart", "orders", "treatment", "therapy",
        "anxiety", "depression", "diabetes", "hypertension", "asthma", "cancer",
    }
    has_name_like_phrase = any(
        len([t for t in phrase.split()
             if t not in name_like_stop and t not in medical_term_tokens and len(t) >= 3]) >= 2
        for phrase in name_like_phrases
    )
    if has_patient_fact_intent and (has_name_like_phrase or "patient" in m or "mrn" in m):
        return "vector_db"

    return None


_ANALYTICAL_PATTERNS = [
    re.compile(r"\b(compar|differ|chang|increas|decreas|titrat|uptitrat)\b", re.I),
    re.compile(r"\b(trend|over time|throughout|during admission|progression|trajectory)\b", re.I),
    re.compile(r"\b(why|what caused|reason for|explain|how did)\b", re.I),
    re.compile(r"\b(prioriti[sz](e|ed|ing)?|priority|trigger(ed|s|ing)?|rationale|driver(s)?)\b", re.I),
    re.compile(r"\b(critical|most important|key finding|major concern|significant|urgent)\b", re.I),
    re.compile(r"\b(which (med|drug|treatment|intervention))\b", re.I),
    re.compile(r"\bmedication.{0,30}(start|stop|chang|increas|decreas|added|removed|new)\b", re.I),
    re.compile(r"\b(risk|readmission|prognosis|deteriorat|worsening)\b", re.I),
    re.compile(r"\b(when (was|did|were).{0,20}(first|start|initiat|begin))\b", re.I),
    re.compile(r"\b(what happened|hospital course|what changed|clinical events)\b", re.I),
]


def _is_analytical_query(message: str) -> bool:
    """Auto-upgrade to GPT-5.4-mini for multi-step clinical reasoning queries."""
    return any(p.search(message) for p in _ANALYTICAL_PATTERNS)


# ── Tool override prompts ────────────────────────────────────────────
TOOL_PROMPTS = {
    "vector_db": (
        "The user has explicitly selected the Clinical Notes (Vector DB) tool. "
        "You MUST call the `lookup_clinical_notes` tool with the user's query. "
        "Do NOT choose a different tool."
    ),
    "web_search": (
        "The user has explicitly selected the Web Search tool. "
        "You MUST call the `tavily_search_results_json` tool with the user's query. "
        "Do NOT choose a different tool."
    ),
    "summarize": (
        "The user has explicitly selected the Summarize Patient tool. "
        "Call `summarize_patient_record` directly with the patient name or MRN from the user's query. "
        "Do NOT call list_available_patients first — the tool resolves names automatically. "
        "Do NOT choose a different tool."
    ),
    "orders": (
        "The user is requesting order data. "
        "Call `lookup_patient_orders` with the patient name or MRN from the query. "
        "Do NOT use lookup_clinical_notes or any other tool."
    ),
    "cohort": (
        "The user is asking about a patient population or cohort. "
        "Call `cohort_patient_search` with the full query text exactly as written. "
        "Do NOT use lookup_clinical_notes or any other tool."
    ),
}


_REPLY_MODE_INSTRUCTIONS = {
    "refine": (
        "REPLY MODE: REFINE. Provide more depth on the most clinically relevant aspect of the user's request. "
        "Keep the same grounding/citation standards; do not fabricate. If key details are missing, state they "
        "are not documented in the retrieved context."
    ),
    "shorten": (
        "REPLY MODE: SHORTEN. Condense your previous/overall answer to maximum 5 bullet points. "
        "Prioritize diagnosis > plan > medications > follow-up. Drop all narrative prose."
    ),
    "simplify": (
        "REPLY MODE: SIMPLIFY. Rewrite the answer in plain English for a patient with no medical training. "
        "Maintain clinical accuracy (do not omit safety-critical information); remove medical jargon."
    ),
    "enhance": (
        "REPLY MODE: ENHANCE. Expand with clinically useful context (e.g., typical workup, key guideline framing, "
        "and red-flag symptoms). Clearly distinguish added context from source-documented facts. "
        "Do not invent any new source-specific patient facts."
    ),
    "regenerate": (
        "REPLY MODE: REGENERATE. Produce a fresh answer (do not reuse prior phrasing). "
        "Prefer a different structure while preserving all facts that are present in retrieved sources. "
        "If this request uses summary tools, server-side summary caches are cleared before generation."
    ),
    "highlight": (
        "REPLY MODE: HIGHLIGHT KEY INFO. Output the most critical clinical facts in exactly this format:\n"
        "DIAGNOSES: [list]\n"
        "ACTIVE MEDICATIONS: [list with doses when present]\n"
        "CRITICAL LABS: [values + dates when present]\n"
        "PLAN ACTIONS: [ordered list]\n"
        "FOLLOW-UP: [timeline]\n"
        "SAFETY ALERTS: [anything requiring urgent attention]\n"
    ),
}


# ── Routes ───────────────────────────────────────────────────────────
_SCOPED_SUMMARY_RE = re.compile(
    r"\b(summar(?:y|ize|ise)|overview|review|what\s+is\s+this|what\s+are\s+these|contents?)\b",
    re.I,
)
_SCOPED_ORDERS_RE = re.compile(
    r"\b(all\s+orders?|orders?|complete\s+(?:med|medication|lab|order)|medications?\s+and\s+labs?)\b",
    re.I,
)


def _scoped_document_fast_path_kind(message: str, tool_override: Optional[str]) -> str:
    """Return summary/orders/lookup for pre-selected document fast path."""
    if tool_override in ("cohort", "web_search"):
        return ""
    if _is_admin_policy_query_or_evidence(message or ""):
        return "lookup"
    if tool_override == "summarize" or _SCOPED_SUMMARY_RE.search(message or ""):
        return "summary"
    if tool_override == "orders" or _SCOPED_ORDERS_RE.search(message or ""):
        return "orders"
    return "lookup"


def _persist_chat_turn(sid: str, history: List[ChatMessage], user_text: str, assistant_text: str) -> None:
    """Persist one user/assistant turn to session memory and CSV."""
    now = time.time()
    history.append(ChatMessage(role="user", content=user_text, timestamp=now))
    history.append(ChatMessage(role="assistant", content=assistant_text, timestamp=time.time()))
    sessions.set(sid, history)

    full_history = sessions.get(sid)
    gradio_format = [
        (h.content, full_history[i + 1].content)
        for i, h in enumerate(full_history)
        if h.role == "user" and i + 1 < len(full_history)
    ]
    with _get_session_write_lock(sid):
        Memory.write_chat_history_to_file(
            gradio_chatbot=gradio_format,
            folder_path=PROJECT_CFG.memory_dir,
            thread_id=sid,
        )


def _is_admin_policy_query_or_evidence(query: str, evidence: str = "") -> bool:
    text = f"{query or ''}\n{evidence or ''}".lower()
    return bool(re.search(
        r"\b(benefit|coverage|copay|co[- ]?insurance|deductible|policy|eligibility|"
        r"allowance|covered service|claim|premium|out[- ]of[- ]pocket|administrative_benefit)\b",
        text,
    ))


def _direct_synthesize_scoped_evidence(
    query: str,
    evidence: str,
    provider: str,
    thinking: bool,
    usage_cb: RequestUsageCallback,
) -> str:
    """Synthesize a selected-document answer without binding the full toolset."""
    _apply_graph_model_overrides(provider)
    p = _normalize_provider(provider)
    chain_key = "thinking" if thinking else "synth"
    state_key = f"{chain_key}_idx"
    model_name = _PROVIDER_MODEL_CHAINS[p][chain_key][_MODEL_FALLBACK_STATE[p][state_key]]
    llm = _make_llm(
        model_name,
        timeout=120 if thinking else 60,
        max_tokens=4096 if thinking else 1600,
        thinking_budget=8192 if thinking else 0,
    )
    if _is_admin_policy_query_or_evidence(query, evidence):
        system_text = (
            "You are MedGraph AI answering from pre-selected administrative/EHR document evidence only.\n"
            "Use only the evidence below. Map [Source 1] to citation [S1], [Source 2] to [S2], etc.\n"
            "Answer the user's exact administrative or benefits question; do not use a clinical summary template.\n"
            "If the user asks for a list, table, or short answer, preserve that requested format.\n"
            "Do not introduce unrelated clinical statements. If a requested field is missing, say it was not found in the selected documents.\n"
            "Do not call tools."
        )
    else:
        system_text = (
            "You are MedGraph AI answering from pre-selected EHR document evidence only.\n"
            "Use source metadata dates as authoritative for encounter/timeline attribution when present.\n"
            "Use only the evidence below. Map [Source 1] to citation [S1], [Source 2] to [S2], etc.\n"
            "When table evidence includes a row-provided Document class or Interpretation, use that exact row value; do not recalculate a different class from reference ranges.\n"
            "Cite every medication dose, lab value, vital sign, diagnosis, procedure, allergy, and date-supported claim.\n"
            "If the evidence does not contain the answer, state that it was not found in the selected documents.\n"
            "Do not call tools. Keep the answer concise unless the question asks for a complete list."
        )
    system = SystemMessage(content=system_text)
    human = HumanMessage(content=f"Question:\n{query}\n\nSelected-document evidence:\n{evidence}")
    response = llm.invoke([system, human], config={"callbacks": [usage_cb]})
    return _normalize_content(getattr(response, "content", "") or "")


def _run_scoped_document_fast_path_sync(
    req: ChatRequest,
    query: str,
    provider: str,
    thinking: bool,
) -> tuple[str, dict, list[str]]:
    """Run deterministic selected-document path before LangGraph routing."""
    kind = _scoped_document_fast_path_kind(query, req.tool_override)
    if not kind or not req.document_ids:
        return "", {}, []

    usage_cb = RequestUsageCallback(PRICING_TABLE)
    scope_tokens = set_active_document_scope(
        document_ids=req.document_ids,
        patient_mrn=req.patient_mrn or "",
        patient_id=req.patient_id or "",
        tenant_id=req.tenant_id or "",
        org_id=req.org_id or "",
    )
    summary_mode_token = set_summary_request_mode("thinking" if thinking else "normal")
    tools_invoked: list[str] = []
    try:
        if kind == "summary":
            if len(req.document_ids) == 1:
                tools_invoked.append("summarize_uploaded_document")
                reply = summarize_uploaded_document.invoke({"document_id": req.document_ids[0]})
            else:
                tools_invoked.append("summarize_uploaded_documents")
                reply = summarize_uploaded_documents.invoke({
                    "document_ids": list(req.document_ids),
                    "patient_mrn": req.patient_mrn or "",
                })
        elif kind == "orders":
            tools_invoked.append("lookup_patient_orders")
            evidence = lookup_patient_orders.invoke({
                "patient_identifier": req.patient_mrn or req.patient_id or query,
                "document_ids": list(req.document_ids),
            })
            reply = _direct_synthesize_scoped_evidence(query, evidence, provider, thinking, usage_cb)
        else:
            tools_invoked.append("lookup_clinical_notes")
            evidence = lookup_clinical_notes.invoke({
                "query": query,
                "patient_mrn": req.patient_mrn or "",
                "document_id": "",
            })
            reply = _direct_synthesize_scoped_evidence(query, evidence, provider, thinking, usage_cb)

        usage = summarize_request(usage_cb.records)
        return reply or "", usage, tools_invoked
    finally:
        reset_summary_request_mode(summary_mode_token)
        reset_active_document_scope(scope_tokens)


async def _run_scoped_document_fast_path(
    req: ChatRequest,
    query: str,
    provider: str,
    thinking: bool,
) -> tuple[str, dict, list[str]]:
    return await asyncio.to_thread(
        _run_scoped_document_fast_path_sync,
        req,
        query,
        provider,
        thinking,
    )


@app.get("/")
async def serve_frontend():
    return FileResponse(
        str(STATIC_DIR / "index.html"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.post("/api/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat(req: ChatRequest, request: Request):
    global graph, thinking_graph
    t0 = time.perf_counter()
    validated_msg = _validate_message(req.message)
    req_provider = _normalize_provider(req.llm_provider)
    effective_tool_override = req.tool_override or _auto_tool_override(validated_msg)
    effective_reply_mode = req.reply_mode
    effective_thinking = bool(req.thinking)
    if not effective_thinking and _is_analytical_query(validated_msg):
        effective_thinking = True
        logger.info("Auto-upgraded to gpt-5.4-mini for analytical query: %s", validated_msg[:80])
    _audit_log("chat_request", AUDIT_SYSTEM_USER, tool_override=effective_tool_override or "auto", llm_provider=req_provider)

    # Session management
    sid = req.session_id or str(uuid.uuid4())
    history: List[ChatMessage] = sessions.get(sid)
    _validate_document_scope_or_raise(req)

    predefined_reply = _predefined_identity_reply(validated_msg)
    if predefined_reply:
        now = time.time()
        history.append(ChatMessage(role="user", content=validated_msg, timestamp=now))
        history.append(ChatMessage(role="assistant", content=predefined_reply, timestamp=time.time()))
        sessions.set(sid, history)

        full_history = sessions.get(sid)
        gradio_format = [(h.content, full_history[i + 1].content)
                         for i, h in enumerate(full_history)
                         if h.role == "user" and i + 1 < len(full_history)]
        with _get_session_write_lock(sid):
            Memory.write_chat_history_to_file(
                gradio_chatbot=gradio_format,
                folder_path=PROJECT_CFG.memory_dir,
                thread_id=sid
            )

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        request_usage = {
            "provider": "fast-path",
            "model": "rule-based",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "records": [],
        }
        LEDGER.record_request(sid, request_usage)
        PROM_REQUEST_LATENCY.labels(endpoint="/api/chat", method="POST").observe(elapsed_ms / 1000)
        _audit_log(
            "chat_response",
            AUDIT_SYSTEM_USER,
            session_id=sid,
            latency_ms=elapsed_ms,
            tokens_total=0,
            cost_usd=0.0,
            fast_path=True,
        )
        return ChatResponse(reply=predefined_reply, session_id=sid, latency_ms=elapsed_ms, usage=request_usage)

    # Instant fast-path for simple conversational/capabilities requests.
    fast_reply = _fast_path_reply(validated_msg)
    if fast_reply and not effective_tool_override:
        now = time.time()
        history.append(ChatMessage(role="user", content=validated_msg, timestamp=now))
        history.append(ChatMessage(role="assistant", content=fast_reply, timestamp=time.time()))
        sessions.set(sid, history)

        full_history = sessions.get(sid)
        gradio_format = [(h.content, full_history[i + 1].content)
                         for i, h in enumerate(full_history)
                         if h.role == "user" and i + 1 < len(full_history)]
        with _get_session_write_lock(sid):
            Memory.write_chat_history_to_file(
                gradio_chatbot=gradio_format,
                folder_path=PROJECT_CFG.memory_dir,
                thread_id=sid
            )

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        request_usage = {
            "provider": "fast-path",
            "model": "rule-based",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "records": [],
        }
        LEDGER.record_request(sid, request_usage)
        PROM_REQUEST_LATENCY.labels(endpoint="/api/chat", method="POST").observe(elapsed_ms / 1000)
        _audit_log(
            "chat_response",
            AUDIT_SYSTEM_USER,
            session_id=sid,
            latency_ms=elapsed_ms,
            tokens_total=0,
            cost_usd=0.0,
            fast_path=True,
        )
        return ChatResponse(reply=fast_reply, session_id=sid, latency_ms=elapsed_ms, usage=request_usage)

    # Deterministic selected-document path. When the EHR/API already supplied
    # document_ids, do not spend a router call rediscovering documents or risk
    # the model broadening scope to the whole patient corpus.
    if req.document_ids and _scoped_document_fast_path_kind(validated_msg, effective_tool_override):
        try:
            if effective_reply_mode == "regenerate":
                cache_result = clear_summary_caches()
                _audit_log(
                    "regenerate_cache_cleared",
                    AUDIT_SYSTEM_USER,
                    session_id=sid,
                    patient_memory_cleared=cache_result.get("patient_memory_cleared", 0),
                    document_memory_cleared=cache_result.get("document_memory_cleared", 0),
                    disk_deleted=cache_result.get("disk_deleted", 0),
                )
            reply, request_usage, tools_invoked = await _run_scoped_document_fast_path(
                req,
                validated_msg,
                req_provider,
                effective_thinking,
            )
            if reply:
                if effective_reply_mode:
                    reply = await _apply_reply_mode_transform(effective_reply_mode, reply, validated_msg)
                _persist_chat_turn(sid, history, validated_msg, reply)
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                LEDGER.record_request(sid, request_usage)
                PROM_REQUEST_LATENCY.labels(endpoint="/api/chat", method="POST").observe(elapsed_ms / 1000)
                PROM_LLM_INPUT_TOKENS.labels(endpoint="/api/chat").inc(request_usage.get("input_tokens", 0))
                PROM_LLM_OUTPUT_TOKENS.labels(endpoint="/api/chat").inc(request_usage.get("output_tokens", 0))
                PROM_LLM_COST_USD.labels(endpoint="/api/chat").inc(request_usage.get("cost_usd", 0.0))
                _audit_log(
                    "chat_response",
                    AUDIT_SYSTEM_USER,
                    session_id=sid,
                    latency_ms=elapsed_ms,
                    tokens_total=request_usage.get("total_tokens", 0),
                    cost_usd=request_usage.get("cost_usd", 0.0),
                    fast_path=True,
                    patient_mrn=req.patient_mrn,
                    document_ids=list(req.document_ids or []),
                    tools_invoked=tools_invoked,
                    data_sources="qdrant",
                )
                return ChatResponse(reply=reply, session_id=sid, latency_ms=elapsed_ms, usage=request_usage)
        except Exception as scoped_exc:
            logger.warning("Scoped document fast path failed; falling back to LangGraph: %s", str(scoped_exc)[:300])

    graph, thinking_graph = _get_graph_pair(req_provider)

    if effective_reply_mode == "regenerate":
        cache_result = clear_summary_caches()
        _audit_log(
            "regenerate_cache_cleared",
            AUDIT_SYSTEM_USER,
            session_id=sid,
            patient_memory_cleared=cache_result.get("patient_memory_cleared", 0),
            document_memory_cleared=cache_result.get("document_memory_cleared", 0),
            disk_deleted=cache_result.get("disk_deleted", 0),
        )

    # Build LangChain message list (trimmed to last N turns)
    system_prompt = SYSTEM_PROMPT
    if effective_tool_override and effective_tool_override in TOOL_PROMPTS:
        system_prompt += " " + TOOL_PROMPTS[effective_tool_override]

    if effective_reply_mode and effective_reply_mode in _REPLY_MODE_INSTRUCTIONS:
        system_prompt += "\n\n" + _REPLY_MODE_INSTRUCTIONS[effective_reply_mode]

    # Phase 3: inject scope context so the router routes directly to the right
    # tool/doc rather than calling list_uploaded_documents() and asking the user.
    scope_block = _build_scope_context_block(req)
    if scope_block:
        system_prompt += scope_block

    trimmed = _trim_history(history)
    messages = [SystemMessage(content=system_prompt)]
    for msg in trimmed:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        else:
            messages.append(AIMessage(content=msg.content))
    messages.append(HumanMessage(content=validated_msg))

    # Run through LangGraph agent — with retry on transient LLM errors
    usage_cb = RequestUsageCallback(PRICING_TABLE)
    config = {"callbacks": [usage_cb]}
    active_graph = thinking_graph if effective_thinking else graph
    max_retries = 2

    # Phase 3: set request-scoped document/patient context vars
    _scope_tokens = None
    if req.document_ids or req.patient_mrn or req.patient_id or req.tenant_id or req.org_id:
        _scope_tokens = set_active_document_scope(
            document_ids=req.document_ids,
            patient_mrn=req.patient_mrn or "",
            patient_id=req.patient_id or "",
            tenant_id=req.tenant_id or "",
            org_id=req.org_id or "",
        )

    summary_mode_token = set_summary_request_mode("thinking" if effective_thinking else "normal")
    try:
        for attempt in range(max_retries + 1):
            try:
                initial_state = {
                    "messages": messages,
                    "active_patient_mrn": req.patient_mrn or "",
                    "active_document_ids": list(req.document_ids or []),
                }
                events = list(active_graph.stream(
                    initial_state, config, stream_mode="values"
                ))
                reply = _normalize_content(events[-1]["messages"][-1].content)
                if effective_reply_mode:
                    reply = await _apply_reply_mode_transform(effective_reply_mode, reply, validated_msg)
                break
            except Exception as exc:
                exc_str = str(exc).lower()
                is_rate_limit = any(k in exc_str for k in ["rate limit", "429", "resource_exhausted", "quota"])
                is_transient = is_rate_limit or any(k in exc_str for k in ["timeout", "timed out", "overloaded", "503"])
                if is_rate_limit:
                    swapped = swap_google_api_key()
                    if swapped:
                        _GRAPH_CACHE.pop(req_provider, None)
                        graph, thinking_graph = _get_graph_pair(req_provider)
                        active_graph = thinking_graph if effective_thinking else graph
                        logger.warning("Rebuilt graphs with backup API key after rate limit")
                    elif effective_thinking and active_graph is not graph:
                        allow_downgrade = os.environ.get("MEDGRAPH_ALLOW_THINKING_DOWNGRADE", "").strip().lower() in ("1", "true", "yes")
                        if allow_downgrade:
                            active_graph = graph
                            logger.warning("Thinking model quota exhausted, falling back to normal graph by config")
                        else:
                            logger.warning("Thinking model quota exhausted; preserving thinking mode (no silent downgrade)")
                            raise HTTPException(status_code=503, detail="Thinking model temporarily unavailable. Please retry in a few moments.")
                if is_transient and attempt < max_retries:
                    advanced_model = _advance_graph_model_fallback(effective_thinking, req_provider)
                    if advanced_model:
                        graph, thinking_graph = _get_graph_pair(req_provider)
                        active_graph = thinking_graph if effective_thinking else graph
                        logger.warning("Rebuilt graphs with fallback model after transient error")
                    wait = (attempt + 1) * 5  # 5s, 10s
                    logger.warning("Transient LLM error (attempt %d/%d), retrying in %ds: %s",
                                   attempt + 1, max_retries + 1, wait, str(exc)[:200])
                    import asyncio
                    await asyncio.sleep(wait)
                    continue
                logger.error("LLM error after %d attempts: %s", attempt + 1, str(exc)[:500])
                raise HTTPException(status_code=503, detail=f"Upstream AI service unavailable. Please retry in a moment.")
    finally:
        reset_summary_request_mode(summary_mode_token)
        # Phase 3: reset document scope
        if _scope_tokens:
            try:
                reset_active_document_scope(_scope_tokens)
            except Exception:
                pass

    # Store in session (full history, not trimmed)
    now = time.time()
    history.append(ChatMessage(role="user", content=validated_msg, timestamp=now))
    history.append(ChatMessage(role="assistant", content=reply, timestamp=time.time()))
    sessions.set(sid, history)

    # Save to memory
    full_history = sessions.get(sid)
    gradio_format = [(h.content, full_history[i + 1].content)
                     for i, h in enumerate(full_history)
                     if h.role == "user" and i + 1 < len(full_history)]
    with _get_session_write_lock(sid):
        Memory.write_chat_history_to_file(
            gradio_chatbot=gradio_format,
            folder_path=PROJECT_CFG.memory_dir,
            thread_id=sid
        )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    request_usage = summarize_request(usage_cb.records)
    LEDGER.record_request(sid, request_usage)
    PROM_LLM_INPUT_TOKENS.labels(endpoint="/api/chat").inc(request_usage["input_tokens"])
    PROM_LLM_OUTPUT_TOKENS.labels(endpoint="/api/chat").inc(request_usage["output_tokens"])
    PROM_LLM_COST_USD.labels(endpoint="/api/chat").inc(request_usage["cost_usd"])
    PROM_REQUEST_LATENCY.labels(endpoint="/api/chat", method="POST").observe(elapsed_ms / 1000)
    # Phase 11 — enrich audit with MRN, tools, data sources, and grounding score
    _audit_meta = _extract_audit_metadata(events)
    _audit_log(
        "chat_response",
        AUDIT_SYSTEM_USER,
        session_id=sid,
        latency_ms=elapsed_ms,
        tokens_total=request_usage["total_tokens"],
        cost_usd=request_usage["cost_usd"],
        **_audit_meta,
    )
    return ChatResponse(reply=reply, session_id=sid, latency_ms=elapsed_ms, usage=request_usage)


# ── Streaming SSE endpoint ───────────────────────────────────────────

_TOOL_STATUS = {
    "summarize_patient_record": "I am reading the full chart and building a clear timeline.",
    "summarize_uploaded_document": "I am reading the uploaded file and extracting key medical points.",
    "summarize_uploaded_documents": "I am reading the selected files and extracting key medical points.",
    "lookup_clinical_notes": "I am searching the patient record for the exact details you asked for.",
    "lookup_patient_orders": "I am gathering all orders (labs, meds, and procedures) for this patient.",
    "cohort_patient_search": "I am scanning records to find all patients matching your criteria.",
    "list_available_patients": "I am checking which patients are currently available in the system.",
    "list_uploaded_documents": "I am checking recently uploaded documents and their status.",
    "tavily_search_results_json": "I am checking trusted web sources for medical background information.",
}

_SUMMARY_PROGRESS_STAGES = [
    (0.0, "I am collecting the full patient timeline from the chart."),
    (2.0, "I am identifying the most important clinical events first."),
    (5.0, "I am connecting labs, treatments, and outcomes into one story."),
    (8.0, "I am drafting a clear summary in plain language."),
    (12.0, "I am double-checking that key safety details are included."),
    (16.0, "I am preparing the final clinician-ready summary."),
]

_GENERAL_PROGRESS_STAGES = [
    (0.0, "I am reading your question and planning the best way to answer it."),
    (1.0, "I am checking the most relevant medical sources now."),
    (3.0, "I am extracting the exact details needed for your question."),
    (6.0, "I am turning the findings into a clear response."),
]

_THINKING_PROGRESS_STAGES = [
    (0.0, "I am breaking this into steps so no critical detail is missed."),
    (1.5, "I am checking timeline context and important clinical changes."),
    (4.0, "I am comparing evidence across notes to avoid contradictions."),
    (8.0, "I am preparing a careful, evidence-grounded explanation."),
]


def _select_progress_stages(is_summary: bool, is_thinking: bool) -> List[Tuple[float, str]]:
    if is_summary:
        return _SUMMARY_PROGRESS_STAGES
    if is_thinking:
        return _THINKING_PROGRESS_STAGES
    return _GENERAL_PROGRESS_STAGES


def _stream_model_hint(provider: str, is_thinking: bool) -> str:
    if provider != "openai":
        return ""
    if is_thinking:
        return (
            os.environ.get("MEDGRAPH_THINKING_MODEL_OVERRIDE", "").strip()
            or getattr(TOOLS_CFG, "primary_agent_thinking_llm", "gpt-5.4-mini")
        )
    return (
        os.environ.get("MEDGRAPH_SYNTH_MODEL_OVERRIDE", "").strip()
        or getattr(TOOLS_CFG, "primary_agent_llm", "gpt-4.1-mini")
    )


def _iter_stream_tokens(text: str, model_hint: str = "") -> Iterator[str]:
    """Yield token-sized pieces for SSE emission.

    Uses tiktoken when available (OpenAI-compatible tokenization), then falls
    back to a whitespace-preserving lexical split.
    """
    if not text:
        return

    try:
        import tiktoken  # optional dependency

        encoding = None
        if model_hint:
            try:
                encoding = tiktoken.encoding_for_model(model_hint)
            except Exception:
                encoding = None
        if encoding is None:
            encoding = tiktoken.get_encoding("cl100k_base")

        for token_id in encoding.encode(text):
            piece = encoding.decode([token_id])
            if piece:
                yield piece
        return
    except Exception:
        pass

    # Fallback path keeps spacing intact for incremental rendering.
    for piece in re.findall(r"\s+|[^\s]+", text):
        yield piece


@app.post("/api/chat/stream")
@limiter.limit("30/minute")
async def chat_stream(req: ChatRequest, request: Request):
    """SSE endpoint — runs graph in a thread, relays updates via queue."""
    global graph, thinking_graph
    t0 = time.perf_counter()
    validated_msg = _validate_message(req.message)
    req_provider = _normalize_provider(req.llm_provider)
    effective_tool_override = req.tool_override or _auto_tool_override(validated_msg)
    effective_reply_mode = req.reply_mode
    effective_thinking = bool(req.thinking)
    if not effective_thinking and _is_analytical_query(validated_msg):
        effective_thinking = True
        logger.info("Auto-upgraded to gpt-5.4-mini for analytical query: %s", validated_msg[:80])
    is_full_record_summary_request = effective_tool_override == "summarize"
    _audit_log("chat_stream_request", AUDIT_SYSTEM_USER, tool_override=effective_tool_override or "auto", llm_provider=req_provider)


    sid = req.session_id or str(uuid.uuid4())
    history: List[ChatMessage] = sessions.get(sid)
    _validate_document_scope_or_raise(req)

    predefined_reply = _predefined_identity_reply(validated_msg)
    if predefined_reply:
        async def _predefined_stream():
            now = time.time()
            history.append(ChatMessage(role="user", content=validated_msg, timestamp=now))
            history.append(ChatMessage(role="assistant", content=predefined_reply, timestamp=time.time()))
            sessions.set(sid, history)

            full_history = sessions.get(sid)
            gradio_format = [(h.content, full_history[i + 1].content)
                             for i, h in enumerate(full_history)
                             if h.role == "user" and i + 1 < len(full_history)]
            with _get_session_write_lock(sid):
                Memory.write_chat_history_to_file(
                    gradio_chatbot=gradio_format,
                    folder_path=PROJECT_CFG.memory_dir,
                    thread_id=sid
                )

            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            usage = {
                "provider": "fast-path",
                "model": "rule-based",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "records": [],
            }
            LEDGER.record_request(sid, usage)
            PROM_REQUEST_LATENCY.labels(endpoint="/api/chat/stream", method="POST").observe(elapsed_ms / 1000)

            yield f"data: {json.dumps({'type': 'token', 'content': predefined_reply})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'session_id': sid, 'latency_ms': elapsed_ms, 'usage': usage})}\n\n"
            _audit_log(
                "chat_stream_response",
                AUDIT_SYSTEM_USER,
                session_id=sid,
                latency_ms=elapsed_ms,
                tokens_total=0,
                cost_usd=0.0,
                fast_path=True,
            )

        return StreamingResponse(
            _predefined_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Instant fast-path for simple conversational/capabilities requests.
    fast_reply = _fast_path_reply(validated_msg)
    if fast_reply and not effective_tool_override:
        async def _fast_stream():
            now = time.time()
            history.append(ChatMessage(role="user", content=validated_msg, timestamp=now))
            history.append(ChatMessage(role="assistant", content=fast_reply, timestamp=time.time()))
            sessions.set(sid, history)

            full_history = sessions.get(sid)
            gradio_format = [(h.content, full_history[i + 1].content)
                             for i, h in enumerate(full_history)
                             if h.role == "user" and i + 1 < len(full_history)]
            with _get_session_write_lock(sid):
                Memory.write_chat_history_to_file(
                    gradio_chatbot=gradio_format,
                    folder_path=PROJECT_CFG.memory_dir,
                    thread_id=sid
                )

            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            usage = {
                "provider": "fast-path",
                "model": "rule-based",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "records": [],
            }
            LEDGER.record_request(sid, usage)
            PROM_REQUEST_LATENCY.labels(endpoint="/api/chat/stream", method="POST").observe(elapsed_ms / 1000)

            yield f"data: {json.dumps({'type': 'token', 'content': fast_reply})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'session_id': sid, 'latency_ms': elapsed_ms, 'usage': usage})}\n\n"
            _audit_log(
                "chat_stream_response",
                AUDIT_SYSTEM_USER,
                session_id=sid,
                latency_ms=elapsed_ms,
                tokens_total=0,
                cost_usd=0.0,
                fast_path=True,
            )

        return StreamingResponse(
            _fast_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    if req.document_ids and _scoped_document_fast_path_kind(validated_msg, effective_tool_override):
        try:
            if effective_reply_mode == "regenerate":
                cache_result = clear_summary_caches()
                _audit_log(
                    "regenerate_cache_cleared",
                    AUDIT_SYSTEM_USER,
                    session_id=sid,
                    patient_memory_cleared=cache_result.get("patient_memory_cleared", 0),
                    document_memory_cleared=cache_result.get("document_memory_cleared", 0),
                    disk_deleted=cache_result.get("disk_deleted", 0),
                )
            scoped_reply, scoped_usage, scoped_tools = await _run_scoped_document_fast_path(
                req,
                validated_msg,
                req_provider,
                effective_thinking,
            )
            if scoped_reply:
                if effective_reply_mode:
                    scoped_reply = await _apply_reply_mode_transform(effective_reply_mode, scoped_reply, validated_msg)
                _persist_chat_turn(sid, history, validated_msg, scoped_reply)
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                LEDGER.record_request(sid, scoped_usage)
                PROM_REQUEST_LATENCY.labels(endpoint="/api/chat/stream", method="POST").observe(elapsed_ms / 1000)
                PROM_LLM_INPUT_TOKENS.labels(endpoint="/api/chat/stream").inc(scoped_usage.get("input_tokens", 0))
                PROM_LLM_OUTPUT_TOKENS.labels(endpoint="/api/chat/stream").inc(scoped_usage.get("output_tokens", 0))
                PROM_LLM_COST_USD.labels(endpoint="/api/chat/stream").inc(scoped_usage.get("cost_usd", 0.0))

                async def _scoped_fast_stream():
                    yield f"data: {json.dumps({'type': 'status', 'content': 'I am answering from the selected document scope.'})}\n\n"
                    yield f"data: {json.dumps({'type': 'token', 'content': scoped_reply})}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'session_id': sid, 'latency_ms': elapsed_ms, 'usage': scoped_usage})}\n\n"
                    _audit_log(
                        "chat_stream_response",
                        AUDIT_SYSTEM_USER,
                        session_id=sid,
                        latency_ms=elapsed_ms,
                        tokens_total=scoped_usage.get("total_tokens", 0),
                        cost_usd=scoped_usage.get("cost_usd", 0.0),
                        fast_path=True,
                        patient_mrn=req.patient_mrn,
                        document_ids=list(req.document_ids or []),
                        tools_invoked=scoped_tools,
                        data_sources="qdrant",
                    )

                return StreamingResponse(
                    _scoped_fast_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )
        except Exception as scoped_exc:
            logger.warning("Scoped document stream fast path failed; falling back to LangGraph: %s", str(scoped_exc)[:300])

    if effective_reply_mode == "regenerate":
        cache_result = clear_summary_caches()
        _audit_log(
            "regenerate_cache_cleared",
            AUDIT_SYSTEM_USER,
            session_id=sid,
            patient_memory_cleared=cache_result.get("patient_memory_cleared", 0),
            document_memory_cleared=cache_result.get("document_memory_cleared", 0),
            disk_deleted=cache_result.get("disk_deleted", 0),
        )

    graph, thinking_graph = _get_graph_pair(req_provider)

    system_prompt = SYSTEM_PROMPT
    if effective_tool_override and effective_tool_override in TOOL_PROMPTS:
        system_prompt += " " + TOOL_PROMPTS[effective_tool_override]

    if effective_reply_mode and effective_reply_mode in _REPLY_MODE_INSTRUCTIONS:
        system_prompt += "\n\n" + _REPLY_MODE_INSTRUCTIONS[effective_reply_mode]

    # Phase 3: inject scope context so the router routes directly to the right
    # tool/doc rather than calling list_uploaded_documents() and asking the user.
    scope_block = _build_scope_context_block(req)
    if scope_block:
        system_prompt += scope_block

    trimmed = _trim_history(history)
    messages = [SystemMessage(content=system_prompt)]
    for msg in trimmed:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        else:
            messages.append(AIMessage(content=msg.content))
    messages.append(HumanMessage(content=validated_msg))

    usage_cb = RequestUsageCallback(PRICING_TABLE)
    config = {"callbacks": [usage_cb]}
    active_graph = thinking_graph if effective_thinking else graph

    # ── Thread-safe queue bridges sync graph → async SSE generator ──
    q: _queue_mod.Queue = _queue_mod.Queue()
    _graph_holder = {"active": active_graph}

    def _run_graph():
        """Runs sync graph.stream() in background and pushes updates immediately."""
        global graph, thinking_graph
        max_retries = 2
        _ended = False  # Track whether we already sent end/error

        summary_mode_token = set_summary_request_mode("thinking" if effective_thinking else "normal")
        try:
            initial_state = {
                "messages": messages,
                "active_patient_mrn": req.patient_mrn or "",
                "active_document_ids": list(req.document_ids or []),
            }
            for attempt in range(max_retries + 1):
                try:
                    for stream_item in _graph_holder["active"].stream(
                        initial_state, config, stream_mode=["updates", "messages"]
                    ):
                        # Multi-mode stream yields (mode, payload) tuples.
                        if isinstance(stream_item, tuple) and len(stream_item) == 2:
                            mode, payload = stream_item
                            if mode == "updates":
                                q.put(("update", payload))
                            elif mode == "messages":
                                q.put(("message", payload))
                            else:
                                q.put(("update", payload))
                        else:
                            q.put(("update", stream_item))
                    q.put(("end", None))
                    _ended = True
                    return
                except TimeoutError as exc:
                    logger.error(f"Graph execution timeout on attempt {attempt + 1}: {exc}")
                    if attempt < max_retries:
                        logger.info("Retrying with backup API key after timeout...")
                        swapped = swap_google_api_key()
                        if swapped:
                            _GRAPH_CACHE.pop(req_provider, None)
                            graph, thinking_graph = _get_graph_pair(req_provider)
                            _graph_holder["active"] = thinking_graph if effective_thinking else graph
                            import time as _t
                            _t.sleep(2)
                            continue
                    q.put(("error", f"Request timeout: {str(exc)}"))
                    _ended = True
                    return
                except Exception as exc:
                    exc_str = str(exc).lower()
                    is_rate_limit = any(k in exc_str for k in ["rate limit", "429", "resource_exhausted", "quota"])
                    is_transient = any(k in exc_str for k in ["timeout", "503", "504", "deadline"])
                    
                    if (is_rate_limit or is_transient) and attempt < max_retries:
                        swapped = swap_google_api_key()
                        if swapped:
                            _GRAPH_CACHE.pop(req_provider, None)
                            graph, thinking_graph = _get_graph_pair(req_provider)
                            _graph_holder["active"] = thinking_graph if effective_thinking else graph
                            logger.warning(f"Rebuilt graphs with backup API key (SSE), retrying... [{exc_str}]")
                        elif effective_thinking and _graph_holder["active"] is not graph:
                            allow_downgrade = os.environ.get("MEDGRAPH_ALLOW_THINKING_DOWNGRADE", "").strip().lower() in ("1", "true", "yes")
                            if allow_downgrade:
                                _graph_holder["active"] = graph
                                logger.warning("Thinking model quota exhausted (SSE), falling back to normal graph by config")
                            else:
                                logger.warning("Thinking model quota exhausted (SSE); preserving thinking mode (no silent downgrade)")
                                q.put(("error", "Thinking model temporarily unavailable. Please retry in a few moments."))
                                _ended = True
                                return
                        elif is_transient and _advance_graph_model_fallback(effective_thinking, req_provider):
                            _GRAPH_CACHE.pop(req_provider, None)
                            graph, thinking_graph = _get_graph_pair(req_provider)
                            _graph_holder["active"] = thinking_graph if effective_thinking else graph
                            logger.warning("Rebuilt graphs with fallback model (SSE), retrying...")
                        import time as _t
                        _t.sleep((attempt + 1) * 2)
                        continue
                    q.put(("error", str(exc)))
                    _ended = True
                    return
        finally:
            reset_summary_request_mode(summary_mode_token)
            # SAFETY NET: guarantee the queue always gets a terminal message
            if not _ended:
                logger.error("_run_graph exited without sending end/error — sending error now")
                q.put(("error", "Internal error: graph execution failed unexpectedly"))

    async def event_generator():
        full_response = ""
        _last_ai_content = ""   # tracks latest non-empty AI text from ANY node (safety net)
        _defer_token_emission = bool(effective_reply_mode)
        _live_token_streaming = False
        _STREAM_TIMEOUT_S = 180  # max seconds — must exceed graph timeout (180s) for summarization
        _STREAM_TOKEN_DELAY_S = 0.0
        _stream_model_name = _stream_model_hint(req_provider, effective_thinking)
        _progress_stages = _select_progress_stages(is_full_record_summary_request, effective_thinking)

        import contextvars as _cv
        _ctx = _cv.copy_context()

        def _run_graph_with_ctx():
            _ctx.run(_run_graph)

        # Phase 11 — audit metadata accumulated across streaming updates
        _audit_mrn: str = ""
        _audit_document_ids: list = list(req.document_ids or [])
        _audit_tools: list = []          # ordered, deduped tool names
        _audit_tools_seen: set = set()   # fast dedup helper
        _audit_tool_content: list = []   # tool message content for source counting
        _audit_grounding: dict = {}      # grounding result from synthesizer update

        # Fallback usage aggregation for streamed runs where callback-based token
        # accounting may be unavailable (observed with some OpenAI stream paths).
        _stream_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "calls": 0,
            "models": [],
            "providers": set(),
        }
        _stream_usage_seen: set[tuple] = set()

        def _provider_for_model(model_name: str) -> str:
            m = (model_name or "").lower()
            if "gemini" in m:
                return "google"
            if "gpt" in m or "o1" in m or "o3" in m or "o4" in m:
                return "openai"
            return "unknown"

        _announced_router = False
        _announced_tools = False
        _announced_synth = False

        async def _emit_token_stream(text: str):
            """Emit response as incremental SSE token-level chunks."""
            if not text:
                return
            for piece in _iter_stream_tokens(text, model_hint=_stream_model_name):
                yield f"data: {json.dumps({'type': 'token', 'content': piece})}\n\n"
                if _STREAM_TOKEN_DELAY_S > 0:
                    await asyncio.sleep(_STREAM_TOKEN_DELAY_S)
                else:
                    await asyncio.sleep(0)

        # Start graph in background thread (sync graph.stream is proven reliable)
        thread = _threading.Thread(target=_run_graph_with_ctx, daemon=True)
        thread.start()

        stream_start = time.perf_counter()
        stage_idx = 0
        stage_start = time.perf_counter()
        while True:
            # Timeout guard: send explicit error instead of hanging indefinitely
            if time.perf_counter() - stream_start > _STREAM_TIMEOUT_S:
                yield f"data: {json.dumps({'type': 'error', 'content': 'Request timed out. Please try again.'})}\n\n"
                return

            # Poll queue without blocking the event loop
            try:
                kind, data = q.get_nowait()
            except _queue_mod.Empty:
                if stage_idx < len(_progress_stages):
                    elapsed_stage = time.perf_counter() - stage_start
                    threshold_s, stage_text = _progress_stages[stage_idx]
                    if elapsed_stage >= threshold_s:
                        yield f"data: {json.dumps({'type': 'status', 'content': stage_text})}\n\n"
                        stage_idx += 1
                await asyncio.sleep(0.05)
                continue

            if kind == "end":
                break
            if kind == "error":
                yield f"data: {json.dumps({'type': 'error', 'content': data})}\n\n"
                return
            if kind == "message":
                # True live token streaming from LangGraph message chunks.
                # Payload format: (AIMessageChunk, metadata)
                try:
                    msg_chunk, msg_meta = data
                except Exception:
                    await asyncio.sleep(0)
                    continue

                # Capture model usage from chunk metadata as a reliable fallback
                # when callback records report zero totals for streamed requests.
                try:
                    usage_meta = getattr(msg_chunk, "usage_metadata", None) or {}
                    response_meta = getattr(msg_chunk, "response_metadata", None) or {}
                    nested_usage = {}
                    if isinstance(response_meta, dict):
                        nested = response_meta.get("token_usage") or response_meta.get("usage")
                        if isinstance(nested, dict):
                            nested_usage = nested

                    usage_source = {}
                    if nested_usage:
                        usage_source.update(nested_usage)
                    if isinstance(usage_meta, dict) and usage_meta:
                        usage_source.update(usage_meta)

                    if usage_source:
                        input_tokens = int(
                            usage_source.get("input_tokens")
                            or usage_source.get("prompt_tokens")
                            or usage_source.get("prompt_token_count")
                            or 0
                        )
                        output_tokens = int(
                            usage_source.get("output_tokens")
                            or usage_source.get("completion_tokens")
                            or usage_source.get("candidates_token_count")
                            or 0
                        )
                        total_tokens = int(
                            usage_source.get("total_tokens")
                            or usage_source.get("total_token_count")
                            or (input_tokens + output_tokens)
                        )

                        cached_input_tokens = 0
                        input_details = usage_source.get("input_token_details")
                        if isinstance(input_details, dict):
                            cached_input_tokens = int(
                                input_details.get("cache_read")
                                or input_details.get("cached_tokens")
                                or 0
                            )
                        if not cached_input_tokens:
                            prompt_details = usage_source.get("prompt_tokens_details")
                            if isinstance(prompt_details, dict):
                                cached_input_tokens = int(prompt_details.get("cached_tokens") or 0)

                        model_name = ""
                        if isinstance(response_meta, dict):
                            model_name = str(response_meta.get("model_name") or response_meta.get("model") or "")
                        if not model_name and isinstance(msg_meta, dict):
                            model_name = str(msg_meta.get("ls_model_name") or "")
                        if not model_name:
                            model_name = _stream_model_name or "unknown"

                        node_name_for_sig = str(msg_meta.get("langgraph_node", "") or "") if isinstance(msg_meta, dict) else ""
                        usage_sig = (
                            model_name,
                            node_name_for_sig,
                            input_tokens,
                            output_tokens,
                            total_tokens,
                            cached_input_tokens,
                        )
                        if total_tokens > 0 and usage_sig not in _stream_usage_seen:
                            _stream_usage_seen.add(usage_sig)
                            _stream_usage["input_tokens"] += input_tokens
                            _stream_usage["output_tokens"] += output_tokens
                            _stream_usage["total_tokens"] += total_tokens
                            _stream_usage["calls"] += 1
                            _stream_usage["models"].append(model_name)
                            _stream_usage["providers"].add(_provider_for_model(model_name))
                            _stream_usage["cost_usd"] = round(
                                _stream_usage["cost_usd"]
                                + _compute_cost_usd(
                                    model_name,
                                    input_tokens,
                                    output_tokens,
                                    PRICING_TABLE,
                                    cached_input_tokens=cached_input_tokens,
                                ),
                                8,
                            )
                except Exception:
                    pass

                if _defer_token_emission:
                    await asyncio.sleep(0)
                    continue

                node_name = ""
                if isinstance(msg_meta, dict):
                    node_name = str(msg_meta.get("langgraph_node", "") or "")

                # Stream only final-answer nodes. Router direct answers still
                # work through update-based fallback for compatibility.
                if node_name not in ("synthesizer", "chatbot"):
                    await asyncio.sleep(0)
                    continue

                has_tool_call_chunks = bool(
                    getattr(msg_chunk, "tool_call_chunks", None)
                    or getattr(msg_chunk, "tool_calls", None)
                    or getattr(msg_chunk, "invalid_tool_calls", None)
                )
                piece = _normalize_content(getattr(msg_chunk, "content", ""))
                if has_tool_call_chunks or not piece:
                    await asyncio.sleep(0)
                    continue

                _live_token_streaming = True
                full_response += piece
                yield f"data: {json.dumps({'type': 'token', 'content': piece})}\n\n"
                if _STREAM_TOKEN_DELAY_S > 0:
                    await asyncio.sleep(_STREAM_TOKEN_DELAY_S)
                else:
                    await asyncio.sleep(0)
                continue

            # Process graph node update
            for node_name, node_output in data.items():
                msgs = node_output.get("messages", [])

                if node_name == "router" and not _announced_router:
                    _announced_router = True
                    yield f"data: {json.dumps({'type': 'status', 'content': 'I am choosing the best strategy for your question.'})}\n\n"
                elif node_name == "tools" and not _announced_tools:
                    _announced_tools = True
                    yield f"data: {json.dumps({'type': 'status', 'content': 'I am now checking records and evidence for you.'})}\n\n"
                elif node_name == "synthesizer" and not _announced_synth:
                    _announced_synth = True
                    yield f"data: {json.dumps({'type': 'status', 'content': 'I have enough data. I am now writing your final answer.'})}\n\n"

                # Phase 11 — capture MRN from any node that propagates it
                if not _audit_mrn:
                    _audit_mrn = (node_output.get("active_patient_mrn") or "").strip()
                if not _audit_document_ids and node_output.get("active_document_ids"):
                    _audit_document_ids = list(node_output.get("active_document_ids") or [])

                # Phase 11 — capture grounding result from synthesizer update
                if node_name == "synthesizer" and node_output.get("grounding_result"):
                    _audit_grounding = node_output["grounding_result"]

                # Phase 11 — capture tool names and content from tool node updates
                if node_name == "tools":
                    for m in msgs:
                        t_name = getattr(m, "name", "")
                        if t_name and t_name not in _audit_tools_seen:
                            _audit_tools.append(t_name)
                            _audit_tools_seen.add(t_name)
                        t_content = str(getattr(m, "content", "") or "")
                        if t_content:
                            _audit_tool_content.append(t_content)

                # Emit responses from router, synthesizer, chatbot nodes
                if node_name in ("router", "synthesizer", "chatbot"):
                    for m in msgs:
                        content = _normalize_content(getattr(m, "content", ""))
                        tool_calls = getattr(m, "tool_calls", [])

                        # Always track the latest non-empty AI content as safety net
                        if content and content.strip():
                            _last_ai_content = content

                        # Emit tool call status
                        if tool_calls:
                            for tc in tool_calls:
                                tc_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                                status_msg = _TOOL_STATUS.get(tc_name, f"Running {tc_name}...")
                                yield f"data: {json.dumps({'type': 'status', 'content': status_msg})}\n\n"

                            if node_name == "synthesizer":
                                yield f"data: {json.dumps({'type': 'status', 'content': 'I need one more quick check before finalizing your answer.'})}\n\n"

                        # Emit content from synthesizer — ONLY when it has a final text
                        # answer (no tool_calls). If tool_calls exist on synthesizer,
                        # this is an intermediate multi-hop step, not the final answer.
                        if content and node_name in ("synthesizer", "chatbot") and not tool_calls:
                            full_response = content
                            if not _defer_token_emission and not _live_token_streaming:
                                async for _chunk in _emit_token_stream(content):
                                    yield _chunk
                        elif content and node_name == "router" and not tool_calls:
                            # Router answered directly (no tool calls) — this IS the
                            # final response (e.g., greetings, simple conversation).
                            full_response = content
                            if not _defer_token_emission and not _live_token_streaming:
                                async for _chunk in _emit_token_stream(content):
                                    yield _chunk
                        elif content and tool_calls:
                            # Intermediate step (tool invocation) — don't emit as answer,
                            # but keep tracking in _last_ai_content (already done above).
                            logger.debug("[SSE] Skipping intermediate content from %s (has tool_calls): %s…",
                                         node_name, content[:60])

        # ── Safety net: if no final token was emitted, use the last captured AI text ──
        # This handles edge cases where Gemini returns content only on intermediate
        # steps (e.g., tool-calling + reasoning in the same message) or where the
        # synthesizer's final pass was missed.
        if not full_response.strip() and _last_ai_content.strip():
            logger.warning("[SSE] full_response empty after graph — using _last_ai_content fallback")
            full_response = _last_ai_content
            if not _defer_token_emission:
                async for _chunk in _emit_token_stream(full_response):
                    yield _chunk

        # ── Fallback: if no content was collected at all ──
        if not full_response.strip():
            full_response = "I was unable to generate a response. Please try rephrasing your question or try again."
            if not _defer_token_emission:
                async for _chunk in _emit_token_stream(full_response):
                    yield _chunk

        if effective_reply_mode:
            full_response = await _apply_reply_mode_transform(effective_reply_mode, full_response, validated_msg)
            async for _chunk in _emit_token_stream(full_response):
                yield _chunk

        # ── Session & memory persistence ──────────────────────────
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        now = time.time()
        history.append(ChatMessage(role="user", content=validated_msg, timestamp=now))
        history.append(ChatMessage(role="assistant", content=full_response, timestamp=time.time()))
        sessions.set(sid, history)

        full_history = sessions.get(sid)
        gradio_format = [(h.content, full_history[i + 1].content)
                         for i, h in enumerate(full_history)
                         if h.role == "user" and i + 1 < len(full_history)]
        with _get_session_write_lock(sid):
            Memory.write_chat_history_to_file(
                gradio_chatbot=gradio_format,
                folder_path=PROJECT_CFG.memory_dir,
                thread_id=sid
            )

        request_usage = summarize_request(usage_cb.records)
        if request_usage.get("total_tokens", 0) == 0 and _stream_usage["total_tokens"] > 0:
            request_usage = {
                "input_tokens": _stream_usage["input_tokens"],
                "output_tokens": _stream_usage["output_tokens"],
                "total_tokens": _stream_usage["total_tokens"],
                "cost_usd": round(float(_stream_usage["cost_usd"]), 8),
                "calls": _stream_usage["calls"],
                "models": list(_stream_usage["models"]),
                "providers": sorted(set(_stream_usage["providers"])),
            }
            logger.info(
                "Applied stream usage fallback from message metadata: total_tokens=%s calls=%s",
                request_usage["total_tokens"],
                request_usage["calls"],
            )
        LEDGER.record_request(sid, request_usage)
        PROM_LLM_INPUT_TOKENS.labels(endpoint="/api/chat/stream").inc(request_usage["input_tokens"])
        PROM_LLM_OUTPUT_TOKENS.labels(endpoint="/api/chat/stream").inc(request_usage["output_tokens"])
        PROM_LLM_COST_USD.labels(endpoint="/api/chat/stream").inc(request_usage["cost_usd"])

        # ── Phase 2.2: Grounding warning for SSE done event ───────────────────
        # Compute grounding warning before emitting done so the client can render
        # an amber warning banner alongside the response. The warning fires when:
        #   a) citation_coverage < 0.50 (less than half of sources were cited), OR
        #   b) numeric hallucinations were detected (value in response ≠ in source)
        _grounding_warning: Optional[str] = None
        try:
            _eff_grounding = _audit_grounding or {}
            _cov = _eff_grounding.get("citation_coverage")
            _halluc = _eff_grounding.get("hallucinated_numerics", [])

            # Fallback: recompute grounding when state-level result is absent
            if not _eff_grounding and _audit_tool_content and full_response:
                from agent_graph.grounding import verify_grounding as _vg_warn
                import re as _re_warn
                _combined_warn = "\n".join(_audit_tool_content)
                _src_nums_warn = {int(n) for n in _re_warn.findall(r"\[Source (\d+)", _combined_warn)}
                _src_count_warn = max(_src_nums_warn) if _src_nums_warn else 0
                _fallback_grounding = _vg_warn(
                    full_response, _src_count_warn, tool_contents=_audit_tool_content
                )
                _cov = _fallback_grounding.get("citation_coverage")
                _halluc = _fallback_grounding.get("hallucinated_numerics", [])
                _eff_grounding = _fallback_grounding

            if _cov is not None and (_cov < 0.50 or _halluc):
                _grounding_warning = (
                    "⚠️ Some clinical values in this response could not be fully verified "
                    "against source documents. Review citations carefully before clinical use."
                )
        except Exception as _gw_exc:
            logger.debug("Grounding warning computation skipped: %s", _gw_exc)

        _done_payload: dict = {
            "type": "done",
            "session_id": sid,
            "latency_ms": elapsed_ms,
            "usage": request_usage,
        }
        if _grounding_warning is not None:
            _done_payload["grounding_warning"] = _grounding_warning
        yield f"data: {json.dumps(_done_payload)}\n\n"
        PROM_REQUEST_LATENCY.labels(endpoint="/api/chat/stream", method="POST").observe(elapsed_ms / 1000)

        # Phase 11 — build streaming audit metadata from accumulated tracking vars
        # Phase 2.2 — use _eff_grounding (already computed above for warning) when available
        _stream_audit_meta: dict = {
            "data_sources": _classify_data_sources(_audit_tools),
        }
        if _audit_mrn:
            _stream_audit_meta["patient_mrn"] = _audit_mrn
        if _audit_document_ids:
            _stream_audit_meta["document_ids"] = [str(d) for d in _audit_document_ids if str(d)]
        if _audit_tools:
            _stream_audit_meta["tools_invoked"] = _audit_tools
        # _eff_grounding is the resolved grounding dict from the warning computation above;
        # fall back to _audit_grounding if the warning block was skipped (exception path).
        _resolved_grounding = locals().get("_eff_grounding") or _audit_grounding or {}
        _gs = _resolved_grounding.get("citation_coverage")
        if _gs is not None:
            _stream_audit_meta["grounding_score"] = _gs
        _halluc_count = len(_resolved_grounding.get("hallucinated_numerics", []))
        if _halluc_count:
            _stream_audit_meta["hallucinated_numeric_count"] = _halluc_count
        if not _resolved_grounding and _audit_tool_content and full_response:
            # Secondary fallback for fast-path responses (no tool calls at all)
            try:
                import re as _re
                from agent_graph.grounding import verify_grounding as _vg
                _combined = "\n".join(_audit_tool_content)
                _src_nums = {int(n) for n in _re.findall(r"\[Source (\d+)", _combined)}
                _src_count = max(_src_nums) if _src_nums else 0
                _fb = _vg(full_response, _src_count, tool_contents=_audit_tool_content)
                if _fb.get("citation_coverage") is not None:
                    _stream_audit_meta["grounding_score"] = _fb["citation_coverage"]
                if _fb.get("hallucinated_numerics"):
                    _stream_audit_meta["hallucinated_numeric_count"] = len(_fb["hallucinated_numerics"])
            except Exception:
                pass

        _audit_log(
            "chat_stream_response",
            AUDIT_SYSTEM_USER,
            session_id=sid,
            latency_ms=elapsed_ms,
            tokens_total=request_usage["total_tokens"],
            cost_usd=request_usage["cost_usd"],
            **_stream_audit_meta,
        )

    async def _scoped_event_generator():
        _scope_tokens = None
        if req.document_ids or req.patient_mrn or req.patient_id or req.tenant_id or req.org_id:
            _scope_tokens = set_active_document_scope(
                document_ids=req.document_ids,
                patient_mrn=req.patient_mrn or "",
                patient_id=req.patient_id or "",
                tenant_id=req.tenant_id or "",
                org_id=req.org_id or "",
            )
        try:
            async for chunk in event_generator():
                yield chunk
        finally:
            if _scope_tokens:
                try:
                    reset_active_document_scope(_scope_tokens)
                except Exception:
                    pass

    return StreamingResponse(
        _scoped_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/clear")
async def clear_session(req: ChatRequest):
    sid = req.session_id
    if sid:
        sessions.delete(sid)
    _audit_log("session_cleared", AUDIT_SYSTEM_USER, session_id=sid)
    return {"status": "cleared", "session_id": sid}


@app.get("/api/health")
async def health():
    """Deep health check — verifies Qdrant and OpenAI connectivity.
    No auth required so load balancers can poll."""
    checks: dict = {"status": "ok", "model": TOOLS_CFG.primary_agent_llm}

    # ── Qdrant check ─────────────────────────────────────────────
    try:
        from agent_graph.tool_clinical_notes_rag import get_rag_tool_instance
        rag = get_rag_tool_instance()
        info = rag.client.get_collection(rag.collection_name)
        checks["qdrant"] = {
            "status": "ok",
            "collection": rag.collection_name,
            "vectors": info.points_count,
        }
    except Exception as e:
        checks["qdrant"] = {"status": "error", "detail": str(e)}
        checks["status"] = "degraded"

    # ── LLM API check (provider-aware) ──────────────────────────────
    provider = getattr(TOOLS_CFG, "llm_provider", "openai")
    try:
        import httpx
        if provider == "google" or TOOLS_CFG.primary_agent_llm.startswith("gemini"):
            api_key = os.getenv("GOOGLE_API_KEY", os.getenv("GOOGLE_AI_API_KEY", ""))
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
                )
                if resp.status_code == 200:
                    checks["llm"] = {"status": "ok", "provider": "google"}
                else:
                    checks["llm"] = {"status": "error", "provider": "google", "http_status": resp.status_code}
                    checks["status"] = "degraded"
        else:
            api_key = os.getenv("OPENAI_API_KEY", "")
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                if resp.status_code == 200:
                    checks["llm"] = {"status": "ok", "provider": "openai"}
                else:
                    checks["llm"] = {"status": "error", "provider": "openai", "http_status": resp.status_code}
                    checks["status"] = "degraded"
    except Exception as e:
        checks["llm"] = {"status": "error", "detail": str(e)}
        checks["status"] = "degraded"

    return checks


@app.get("/api/usage/summary")
async def usage_summary():
    """Global usage and cost totals across all requests."""
    _audit_log("usage_summary_view", AUDIT_SYSTEM_USER)
    return {"global": LEDGER.get_global()}


@app.get("/api/usage/session/{session_id}")
async def usage_session(session_id: str):
    """Usage and cost totals for one chat session."""
    _audit_log("usage_session_view", AUDIT_SYSTEM_USER, session_id=session_id)
    return {"session_id": session_id, "usage": LEDGER.get_session(session_id)}


# ── Prometheus Metrics Endpoint (P2.22) ───────────────────────────

@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint. No auth — scrapers need direct access."""
    PROM_ACTIVE_SESSIONS.set(len(sessions))
    from starlette.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── Clear Summary Cache API ───────────────────────────────────────

@app.post("/api/cache/clear")
async def clear_summary_cache():
    """Delete all cached patient summaries (both disk and in-memory) so they regenerate fresh."""
    _audit_log("cache_cleared", AUDIT_SYSTEM_USER)
    result = clear_summary_caches()
    # Keep backward-compatible aliases for existing UI handlers.
    result["memory_cleared"] = int(result.get("patient_memory_cleared", 0)) + int(result.get("document_memory_cleared", 0))
    result["deleted"] = int(result.get("disk_deleted", 0))
    return result


# ── Delete Chat History API ───────────────────────────────────────────

class DeleteHistoryRequest(BaseModel):
    session_id: str
    date: str  # YYYY-MM-DD


@app.post("/api/history/delete")
async def delete_session(req: DeleteHistoryRequest):
    """
    Delete a specific session from a specific date's CSV file.
    Removes only the rows matching the session_id from that date.
    """
    _audit_log("history_delete", AUDIT_SYSTEM_USER, session_id=req.session_id, date=req.date)
    csv_path = os.path.join(MEMORY_DIR, f"{req.date}.csv")
    if not os.path.exists(csv_path):
        return {"status": "not_found", "message": "CSV file not found"}

    try:
        df = pd.read_csv(csv_path)
        original_len = len(df)
        df = df[df["thread_id"].astype(str) != str(req.session_id)]

        if len(df) == original_len:
            return {"status": "not_found", "message": "Session not found in file"}

        if df.empty:
            # No rows left — delete the entire file
            os.remove(csv_path)
        else:
            df.to_csv(csv_path, index=False)

        # Also clear from in-memory sessions
        sessions.delete(req.session_id)

        return {"status": "ok", "deleted_rows": original_len - len(df)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Chat History APIs ────────────────────────────────────────────────

@app.get("/api/history/sessions")
async def list_sessions():
    """
    Scan all CSV files in memory/ and return a list of unique sessions
    grouped by date, with preview of the first user query.
    """
    all_sessions = []
    if not os.path.isdir(MEMORY_DIR):
        return {"sessions": []}

    for csv_file in sorted(os.listdir(MEMORY_DIR), reverse=True):
        if not csv_file.endswith(".csv"):
            continue
        date_str = csv_file.replace(".csv", "")
        file_path = os.path.join(MEMORY_DIR, csv_file)
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                continue
            # Group by thread_id
            date_sessions = []
            for tid, group in df.groupby("thread_id", sort=False):
                first_query = str(group.iloc[0]["user_query"])[:80]
                first_time = str(group.iloc[0]["timestamp"])
                msg_count = len(group)
                date_sessions.append({
                    "session_id": str(tid),
                    "date": date_str,
                    "time": first_time,
                    "preview": first_query,
                    "message_count": msg_count,
                })
            # Sort sessions within each date: most recent time first
            date_sessions.sort(key=lambda s: s["time"], reverse=True)
            all_sessions.extend(date_sessions)
        except Exception:
            continue

    return {"sessions": all_sessions}


@app.get("/api/history/load/{session_id}")
async def load_session(session_id: str, date: Optional[str] = None):
    """
    Load all messages for a given session_id scoped to a specific date.
    If date is provided, only loads from that day's CSV file.
    Returns messages in chronological order.
    """
    all_messages = []
    if not os.path.isdir(MEMORY_DIR):
        return {"messages": [], "session_id": session_id}

    # Determine which CSV files to scan
    if date:
        csv_files = [f"{date}.csv"]
    else:
        csv_files = sorted(os.listdir(MEMORY_DIR))

    for csv_file in csv_files:
        if not csv_file.endswith(".csv"):
            continue
        file_path = os.path.join(MEMORY_DIR, csv_file)
        if not os.path.exists(file_path):
            continue
        date_str = csv_file.replace(".csv", "")
        try:
            df = pd.read_csv(file_path)
            session_df = df[df["thread_id"].astype(str) == str(session_id)]
            for _, row in session_df.iterrows():
                all_messages.append({
                    "user_query": str(row["user_query"]),
                    "response": str(row["response"]),
                    "timestamp": str(row["timestamp"]),
                    "date": date_str,
                })
        except Exception:
            continue

    # Also populate the in-memory session so further queries keep context
    if all_messages:
        restored: List[ChatMessage] = []
        for msg in all_messages:
            restored.append(ChatMessage(role="user", content=msg["user_query"], timestamp=0))
            restored.append(ChatMessage(role="assistant", content=msg["response"], timestamp=0))
        sessions.set(session_id, restored)

    return {"messages": all_messages, "session_id": session_id}


# ── Document Upload / Management ─────────────────────────────────────

UPLOADS_DIR = TOOLS_CFG.clinical_notes_rag_uploads_directory
os.makedirs(UPLOADS_DIR, exist_ok=True)
_DOCUMENTS_META_PATH = os.path.join(UPLOADS_DIR, "documents.json")
_documents_lock = _threading.Lock()
_session_write_locks: dict = {}
_session_write_locks_meta = _threading.Lock()


def _get_session_write_lock(session_id: str) -> _threading.Lock:
    """Return the per-session write lock, creating it lazily if needed."""
    with _session_write_locks_meta:
        if session_id not in _session_write_locks:
            _session_write_locks[session_id] = _threading.Lock()
        return _session_write_locks[session_id]

MAX_UPLOAD_SIZE = TOOLS_CFG.clinical_notes_rag_max_upload_size_mb * 1024 * 1024
SUPPORTED_FORMATS = set(TOOLS_CFG.clinical_notes_rag_supported_formats)


def _load_documents_meta() -> Dict[str, dict]:
    """Load the document metadata registry from disk."""
    if os.path.exists(_DOCUMENTS_META_PATH):
        with open(_DOCUMENTS_META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_documents_meta(meta: Dict[str, dict]) -> None:
    """Persist the document metadata registry to disk."""
    with open(_DOCUMENTS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


def _vectorize_in_background(document_id: str, file_path: str, collection_name: str):
    """Run vectorization in a background thread and update metadata on completion."""
    try:
        from document_processing import process_document
        from agent_graph.tool_clinical_notes_rag import get_rag_tool_instance
        from langchain_qdrant import QdrantVectorStore

        # Process document into chunks (supports PDF, DOCX, images with OCR)
        documents = process_document(
            file_path=file_path,
            chunk_size=TOOLS_CFG.clinical_notes_rag_chunk_size,
            chunk_overlap=TOOLS_CFG.clinical_notes_rag_chunk_overlap,
            document_id=document_id,
        )

        if not documents:
            with _documents_lock:
                meta = _load_documents_meta()
                if document_id in meta:
                    meta[document_id]["status"] = "error"
                    meta[document_id]["error"] = "Document failed quality validation or produced no content"
                    _save_documents_meta(meta)
            return

        # Check for duplicate content before indexing
        content_hash = documents[0].metadata.get("content_hash") if documents else None
        if content_hash:
            rag = get_rag_tool_instance()
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            try:
                existing = rag.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=Filter(must=[
                        FieldCondition(
                            key="metadata.content_hash",
                            match=MatchValue(value=content_hash),
                        )
                    ]),
                    limit=1,
                    with_payload=["metadata.document_id"],
                    with_vectors=False,
                )
                if existing[0]:
                    existing_doc_id = existing[0][0].payload.get("metadata", {}).get("document_id", "unknown")
                    with _documents_lock:
                        meta = _load_documents_meta()
                        if document_id in meta:
                            meta[document_id]["status"] = "duplicate"
                            meta[document_id]["duplicate_of"] = existing_doc_id
                            _save_documents_meta(meta)
                    logger.info(
                        "Document %s is a duplicate of %s — skipped indexing",
                        document_id, existing_doc_id,
                    )
                    return
            except Exception as e:
                logger.warning("Dedup check failed (proceeding with indexing): %s", e)

        # Reuse the RAG tool's existing Qdrant client and vector store
        # (shared Qdrant server instance — multiple workers can call this concurrently)
        rag = get_rag_tool_instance()
        vector_store = QdrantVectorStore(
            client=rag.client,
            collection_name=collection_name,
            embedding=rag.vectordb.embeddings,
        )

        batch_size = 20
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            vector_store.add_documents(batch)
            logger.info("Uploaded %d/%d chunks for %s", min(i + batch_size, len(documents)), len(documents), document_id)

        # Phase 4.3 — invalidate cached summaries for this document so the
        # next summarize_uploaded_document call generates a fresh result.
        try:
            invalidate_document_summary_cache(document_id)
            # Also invalidate the patient-level summary if we can identify the MRN.
            patient_mrn = documents[0].metadata.get("patient_mrn") if documents else None
            if patient_mrn:
                invalidate_patient_summary_cache(patient_mrn)
        except Exception as e:
            logger.warning("Cache invalidation failed (non-critical): %s", e)

        with _documents_lock:
            meta = _load_documents_meta()
            if document_id in meta:
                meta[document_id]["status"] = "ready"
                meta[document_id]["chunks"] = len(documents)
                meta[document_id]["content_hash"] = content_hash
                meta[document_id]["completed_at"] = datetime.utcnow().isoformat() + "Z"
                _save_documents_meta(meta)

        logger.info("Document vectorized successfully: %s (%d chunks)", document_id, len(documents))

    except Exception as e:
        logger.error("Vectorization failed for %s: %s", document_id, str(e))
        with _documents_lock:
            meta = _load_documents_meta()
            if document_id in meta:
                meta[document_id]["status"] = "error"
                meta[document_id]["error"] = str(e)[:500]
                _save_documents_meta(meta)


@app.post("/api/documents/upload")
@limiter.limit("10/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
):
    """Upload a document for vectorization and RAG querying.

    Supports PDF, DOCX, and image files (PNG, JPG, TIFF, BMP).
    The file is saved to disk and vectorized in the background.
    Returns a document_id that can be used to check status or delete the document.
    """
    # Validate file extension
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}",
        )

    # Read and validate file size
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(content) / 1024 / 1024:.1f} MB). Max: {TOOLS_CFG.clinical_notes_rag_max_upload_size_mb} MB.",
        )
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    # Generate document ID and save file
    document_id = str(uuid.uuid4())
    safe_filename = f"{document_id}{ext}"
    file_path = os.path.join(UPLOADS_DIR, safe_filename)

    with open(file_path, "wb") as f:
        f.write(content)

    # Register in metadata
    with _documents_lock:
        meta = _load_documents_meta()
        meta[document_id] = {
            "document_id": document_id,
            "original_filename": filename,
            "file_path": safe_filename,
            "status": "processing",
            "uploaded_at": datetime.utcnow().isoformat() + "Z",
            "uploaded_by": "system",
            "file_size_bytes": len(content),
        }
        _save_documents_meta(meta)

    _audit_log("document_upload", AUDIT_SYSTEM_USER, document_id=document_id, filename=filename)

    # Start vectorization in background
    collection_name = TOOLS_CFG.clinical_notes_rag_collection_name
    thread = _threading.Thread(
        target=_vectorize_in_background,
        args=(document_id, file_path, collection_name),
        daemon=True,
    )
    thread.start()

    return {
        "document_id": document_id,
        "filename": filename,
        "status": "processing",
        "message": "Document uploaded. Vectorization in progress.",
    }


@app.get("/api/documents")
async def list_documents():
    """List all uploaded documents with their processing status."""
    with _documents_lock:
        meta = _load_documents_meta()
    documents = sorted(meta.values(), key=lambda d: d.get("uploaded_at", ""), reverse=True)
    return {"documents": documents}


@app.get("/api/documents/{document_id}")
async def get_document_status(document_id: str):
    """Get the status of a specific uploaded document."""
    with _documents_lock:
        meta = _load_documents_meta()
    doc = meta.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete an uploaded document and its vectors from the collection."""
    with _documents_lock:
        meta = _load_documents_meta()
        doc = meta.get(document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found.")

        # Phase 11 — capture patient MRN from metadata before deletion for audit log
        _doc_patient_mrn = (doc.get("patient_mrn") or "").strip() or None

        # Remove file from disk
        file_path = os.path.join(UPLOADS_DIR, doc.get("file_path", ""))
        if os.path.exists(file_path):
            os.remove(file_path)

        # Remove from metadata
        del meta[document_id]
        _save_documents_meta(meta)

    _delete_audit: dict = {"document_id": document_id}
    if _doc_patient_mrn:
        _delete_audit["patient_mrn"] = _doc_patient_mrn
    _audit_log("document_delete", AUDIT_SYSTEM_USER, **_delete_audit)

    # Remove vectors from Qdrant via the shared Qdrant server client
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        from agent_graph.tool_clinical_notes_rag import get_rag_tool_instance

        rag = get_rag_tool_instance()
        collection_name = TOOLS_CFG.clinical_notes_rag_collection_name
        rag.client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="metadata.document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        )
        logger.info("Deleted vectors for document %s", document_id)

    except Exception as e:
        logger.error("Failed to delete vectors for %s: %s", document_id, str(e))

    return {"status": "deleted", "document_id": document_id}


# ── Phase 3: Document ownership validation ───────────────────────────

def _validate_document_ownership(
    document_ids: List[str],
    patient_mrn: Optional[str],
    patient_id: Optional[str],
    tenant_id: Optional[str] = None,
    org_id: Optional[str] = None,
) -> List[str]:
    """Validate that every requested document belongs to the stated patient.

    Returns a list of validation error strings. Empty list means all checks pass.
    Any document_id that belongs to a different patient causes a 403 error upstream.

    Rules:
    - Every document_id must exist in the metadata store.
    - Every document_id must have indexed vectors available for that ID.
    - When patient_mrn and/or patient_id are supplied, every document must match.
    - If neither is supplied, the document's own patient_mrn is used as the implied
      scope (documents are selected from the UI so they already carry their owner).
    - tenant_id/org_id are enforced when supplied.
    """
    errors: List[str] = []
    if not document_ids:
        return errors

    with _documents_lock:
        meta = _load_documents_meta()

    # When no patient identifier is supplied, try to derive it from the first
    # document's own metadata. All selected docs must belong to the same patient.
    if not (patient_mrn or patient_id):
        for doc_id in document_ids:
            doc = meta.get(doc_id)
            if doc:
                derived_mrn = (doc.get("patient_mrn") or "").strip()
                derived_pid = (doc.get("patient_id") or "").strip()
                if derived_mrn:
                    patient_mrn = derived_mrn
                    break
                if derived_pid:
                    patient_id = derived_pid
                    break
        # If still nothing found, proceed without patient filtering —
        # document-ID scope alone restricts retrieval to the selected documents.

    for doc_id in document_ids:
        doc = meta.get(doc_id)
        if not doc:
            errors.append(f"document_id '{doc_id}' not found")
            continue
        status = doc.get("status")
        duplicate_of = (doc.get("duplicate_of") or "").strip()
        if status == "duplicate" and duplicate_of and duplicate_of != doc_id:
            errors.append(
                f"document_id '{doc_id}' is a duplicate of '{duplicate_of}' and has no indexed vectors"
            )
            continue
        if status not in ("ready", "duplicate"):
            errors.append(
                f"document_id '{doc_id}' is not ready (status={doc.get('status', 'unknown')})"
            )
            continue

        # Patient ownership check
        doc_mrn = (doc.get("patient_mrn") or "").strip()
        doc_pid = (doc.get("patient_id") or "").strip()

        if patient_mrn:
            if not doc_mrn:
                errors.append(f"document_id '{doc_id}' has no patient_mrn ownership metadata")
            elif doc_mrn != patient_mrn.strip():
                errors.append(
                    f"document_id '{doc_id}' belongs to patient '{doc_mrn}', "
                    f"not '{patient_mrn}'"
                )
        if patient_id:
            if not doc_pid:
                errors.append(f"document_id '{doc_id}' has no patient_id ownership metadata")
            elif doc_pid != patient_id.strip():
                errors.append(
                    f"document_id '{doc_id}' belongs to patient_id '{doc_pid}', "
                    f"not '{patient_id}'"
                )
        doc_tenant = (doc.get("tenant_id") or "").strip()
        if tenant_id:
            if not doc_tenant:
                errors.append(f"document_id '{doc_id}' has no tenant_id ownership metadata")
            elif doc_tenant != tenant_id.strip():
                errors.append(
                    f"document_id '{doc_id}' belongs to tenant '{doc_tenant}', "
                    f"not '{tenant_id}'"
                )
        doc_org = (doc.get("org_id") or "").strip()
        if org_id:
            if not doc_org:
                errors.append(f"document_id '{doc_id}' has no org_id ownership metadata")
            elif doc_org != org_id.strip():
                errors.append(
                    f"document_id '{doc_id}' belongs to org '{doc_org}', "
                    f"not '{org_id}'"
                )

    return errors


def _validate_document_scope_or_raise(req: ChatRequest) -> None:
    """Validate selected-document scope before any graph or stream starts.

    Also auto-populates req.patient_mrn / req.patient_id when the caller did not
    supply them but document_ids were given — the MRN is derived from the document
    metadata so that downstream scope-setting (set_active_document_scope) always
    has a patient identifier to propagate to the graph.
    """
    if not req.document_ids:
        return
    # Snapshot before validation so we can detect auto-derivation.
    _pre_mrn = req.patient_mrn
    _pre_pid = req.patient_id
    ownership_errors = _validate_document_ownership(
        req.document_ids,
        req.patient_mrn,
        req.patient_id,
        tenant_id=req.tenant_id,
        org_id=req.org_id,
    )
    if ownership_errors:
        first_err = ownership_errors[0]
        status_code = 403 if "belongs to" in first_err else 400
        raise HTTPException(status_code=status_code, detail=first_err)
    # Auto-populate patient identifiers on the request object so that
    # set_active_document_scope receives the derived MRN/patient_id.
    if not _pre_mrn and not _pre_pid:
        with _documents_lock:
            meta = _load_documents_meta()
        for doc_id in req.document_ids:
            doc = meta.get(doc_id)
            if doc:
                derived_mrn = (doc.get("patient_mrn") or "").strip()
                derived_pid = (doc.get("patient_id") or "").strip()
                if derived_mrn and not req.patient_mrn:
                    req.patient_mrn = derived_mrn
                if derived_pid and not req.patient_id:
                    req.patient_id = derived_pid
                if req.patient_mrn:
                    break


def _vectorize_batch_document(
    document_id: str,
    file_path: str,
    collection_name: str,
    metadata_overrides: Optional[dict] = None,
    batch_id: Optional[str] = None,
):
    """Vectorize a single document from a batch ingest request.

    Extends _vectorize_in_background with:
    - EHR metadata override support
    - Versioned re-index: same document_id + different content → delete old + re-index
    - batch_id tagging
    - Status always ends in 'ready', 'duplicate', or 'error'
    """
    try:
        from document_processing import process_document
        from agent_graph.tool_clinical_notes_rag import get_rag_tool_instance
        from langchain_qdrant import QdrantVectorStore

        # Build merged overrides with batch_id
        effective_overrides = dict(metadata_overrides or {})
        if batch_id:
            effective_overrides["batch_id"] = batch_id

        documents = process_document(
            file_path=file_path,
            chunk_size=TOOLS_CFG.clinical_notes_rag_chunk_size,
            chunk_overlap=TOOLS_CFG.clinical_notes_rag_chunk_overlap,
            document_id=document_id,
            metadata_overrides=effective_overrides,
        )

        if not documents:
            with _documents_lock:
                meta = _load_documents_meta()
                if document_id in meta:
                    meta[document_id]["status"] = "error"
                    meta[document_id]["error_message"] = (
                        "Document failed quality validation or produced no content"
                    )
                    _save_documents_meta(meta)
            return

        content_hash = documents[0].metadata.get("content_hash") if documents else None

        rag = get_rag_tool_instance()

        # ── Idempotency / versioning check ────────────────────────────────────
        if content_hash:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            # Check: same document_id already exists?
            try:
                existing_same_id = rag.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=Filter(must=[
                        FieldCondition(
                            key="metadata.document_id",
                            match=MatchValue(value=document_id),
                        )
                    ]),
                    limit=1,
                    with_payload=["metadata.content_hash"],
                    with_vectors=False,
                )
                if existing_same_id[0]:
                    existing_hash = (
                        existing_same_id[0][0].payload
                        .get("metadata", {}).get("content_hash", "")
                    )
                    if existing_hash == content_hash:
                        # Same content — skip re-embedding
                        with _documents_lock:
                            meta = _load_documents_meta()
                            if document_id in meta:
                                meta[document_id]["status"] = "duplicate"
                                meta[document_id]["duplicate_of"] = document_id
                                meta[document_id]["content_hash"] = content_hash
                                _save_documents_meta(meta)
                        logger.info(
                            "Document %s unchanged (same content_hash) — skipped re-index",
                            document_id,
                        )
                        return
                    else:
                        # Same document_id but different content — delete old vectors
                        logger.info(
                            "Document %s content changed — deleting old vectors and re-indexing",
                            document_id,
                        )
                        rag.client.delete(
                            collection_name=collection_name,
                            points_selector=Filter(must=[
                                FieldCondition(
                                    key="metadata.document_id",
                                    match=MatchValue(value=document_id),
                                )
                            ]),
                        )
            except Exception as dedup_exc:
                logger.warning(
                    "Dedup/version check failed for %s (proceeding with indexing): %s",
                    document_id, dedup_exc,
                )

            # Check: different document_id, same content for same patient?
            try:
                doc_patient_mrn = documents[0].metadata.get("patient_mrn") or ""
                existing_hash_match = rag.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=Filter(must=[
                        FieldCondition(
                            key="metadata.content_hash",
                            match=MatchValue(value=content_hash),
                        )
                    ]),
                    limit=1,
                    with_payload=["metadata.document_id", "metadata.patient_mrn"],
                    with_vectors=False,
                )
                if existing_hash_match[0]:
                    existing_doc_id = (
                        existing_hash_match[0][0].payload
                        .get("metadata", {}).get("document_id", "")
                    )
                    existing_mrn = (
                        existing_hash_match[0][0].payload
                        .get("metadata", {}).get("patient_mrn", "")
                    )
                    if (
                        existing_doc_id != document_id
                        and existing_mrn == doc_patient_mrn
                        and doc_patient_mrn
                    ):
                        with _documents_lock:
                            meta = _load_documents_meta()
                            if document_id in meta:
                                meta[document_id]["status"] = "duplicate"
                                meta[document_id]["duplicate_of"] = existing_doc_id
                                meta[document_id]["content_hash"] = content_hash
                                _save_documents_meta(meta)
                        logger.info(
                            "Document %s is a duplicate of %s for patient %s — skipped",
                            document_id, existing_doc_id, doc_patient_mrn,
                        )
                        return
            except Exception as cross_dedup_exc:
                logger.warning(
                    "Cross-document dedup check failed for %s: %s",
                    document_id, cross_dedup_exc,
                )

        # ── Index documents ───────────────────────────────────────────────────
        vector_store = QdrantVectorStore(
            client=rag.client,
            collection_name=collection_name,
            embedding=rag.vectordb.embeddings,
        )

        batch_size = 20
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            vector_store.add_documents(batch)

        # Invalidate summary caches
        try:
            invalidate_document_summary_cache(document_id)
            doc_patient_mrn = documents[0].metadata.get("patient_mrn") if documents else None
            if doc_patient_mrn:
                invalidate_patient_summary_cache(doc_patient_mrn)
        except Exception:
            pass

        # Update metadata
        with _documents_lock:
            meta = _load_documents_meta()
            if document_id in meta:
                meta[document_id]["status"] = "ready"
                meta[document_id]["chunks"] = len(documents)
                meta[document_id]["content_hash"] = content_hash
                meta[document_id]["completed_at"] = datetime.utcnow().isoformat() + "Z"
                _save_documents_meta(meta)

        logger.info(
            "Batch document vectorized: %s (%d chunks, batch_id=%s)",
            document_id, len(documents), batch_id or "n/a",
        )

    except Exception as exc:
        logger.error(
            "Batch vectorization failed for %s: %s", document_id, exc, exc_info=True
        )
        with _documents_lock:
            meta = _load_documents_meta()
            if document_id in meta:
                meta[document_id]["status"] = "error"
                meta[document_id]["error_message"] = str(exc)[:500]
                _save_documents_meta(meta)


@app.post("/api/documents/ingest", status_code=202)
@limiter.limit("5/minute")
async def ingest_documents_batch(
    request: Request,
    files: List[UploadFile] = File(...),
    metadata_json: str = Form(...),
):
    """Batch document ingest for EHR integration.

    Accepts multiple documents with structured EHR metadata in a single request.
    Each file must have a corresponding metadata entry in metadata_json.

    Returns 202 Accepted immediately. Vectorization runs in background threads.
    Poll GET /api/documents/{document_id} to check per-document status.

    Request format:
      - files[]: one or more UploadFile (PDF, DOCX, image)
      - metadata_json: JSON array, one object per file

    Each metadata object must include:
      - filename: matches the uploaded file name (used only for matching)
      - document_id: canonical document identity (required)
      - patient_mrn or patient_id: patient ownership (at least one required)

    Optional metadata fields:
      - document_type, document_date, encounter_id, tenant_id, org_id,
        source_system, ingestion_source
    """
    # ── Parse metadata ────────────────────────────────────────────────────
    try:
        metadata_list = json.loads(metadata_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"metadata_json is not valid JSON: {exc}",
        )
    if not isinstance(metadata_list, list):
        raise HTTPException(
            status_code=400,
            detail="metadata_json must be a JSON array.",
        )

    # ── Validate: one metadata per file, no duplicates ────────────────────
    meta_by_filename: dict = {}
    seen_document_ids: set[str] = set()
    for item in metadata_list:
        if not isinstance(item, dict):
            raise HTTPException(
                status_code=400, detail="Each metadata item must be a JSON object."
            )
        fname = item.get("filename", "").strip()
        if not fname:
            raise HTTPException(
                status_code=400,
                detail="Each metadata item must include a non-empty 'filename'.",
            )
        if fname in meta_by_filename:
            raise HTTPException(
                status_code=400,
                detail=f"Duplicate filename '{fname}' in metadata_json.",
            )
        document_id = str(item.get("document_id") or "").strip()
        if not document_id:
            raise HTTPException(
                status_code=400,
                detail=f"Metadata for '{fname}' is missing required 'document_id'.",
            )
        if document_id in seen_document_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Duplicate document_id '{document_id}' in metadata_json.",
            )
        seen_document_ids.add(document_id)
        if not (item.get("patient_mrn") or item.get("patient_id")):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Metadata for '{fname}' must include 'patient_mrn' or 'patient_id'."
                ),
            )
        meta_by_filename[fname] = item

    # ── Validate: each file has metadata; each metadata has a file ────────
    file_by_name: dict = {}
    for f in files:
        fname = (f.filename or "").strip()
        if not fname:
            raise HTTPException(status_code=400, detail="All uploaded files must have a filename.")
        if fname in file_by_name:
            raise HTTPException(
                status_code=400,
                detail=f"Duplicate filename '{fname}' in uploaded files.",
            )
        file_by_name[fname] = f

    for fname in file_by_name:
        if fname not in meta_by_filename:
            raise HTTPException(
                status_code=400,
                detail=f"Uploaded file '{fname}' has no matching metadata entry.",
            )
    for fname in meta_by_filename:
        if fname not in file_by_name:
            raise HTTPException(
                status_code=400,
                detail=f"Metadata references '{fname}' but no matching file was uploaded.",
            )

    # ── Read and validate file contents ───────────────────────────────────
    batch_id = str(uuid.uuid4())
    saved_files: dict = {}  # document_id → {file_path, meta, content}

    for fname, upload_file in file_by_name.items():
        ext = os.path.splitext(fname)[1].lower()
        if ext not in SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format '{ext}' for file '{fname}'. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}",
            )
        content = await upload_file.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"File '{fname}' is empty.")
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File '{fname}' exceeds {TOOLS_CFG.clinical_notes_rag_max_upload_size_mb} MB limit.",
            )
        item_meta = meta_by_filename[fname]
        document_id = str(item_meta["document_id"]).strip()
        safe_filename = f"{uuid.uuid4()}{ext}"
        file_path = os.path.join(UPLOADS_DIR, safe_filename)
        saved_files[document_id] = {
            "file_path": file_path,
            "meta": item_meta,
            "content": content,
            "original_filename": fname,
            "ext": ext,
        }

    # ── Save files and register metadata ─────────────────────────────────
    collection_name = TOOLS_CFG.clinical_notes_rag_collection_name
    doc_results = []

    with _documents_lock:
        existing_meta = _load_documents_meta()
        for document_id, info in saved_files.items():
            # Write file to disk
            with open(info["file_path"], "wb") as fp:
                fp.write(info["content"])

            item_meta = info["meta"]
            patient_mrn = (item_meta.get("patient_mrn") or "").strip()
            patient_id = (item_meta.get("patient_id") or "").strip()

            existing_meta[document_id] = {
                "document_id": document_id,
                "original_filename": info["original_filename"],
                "file_path": os.path.basename(info["file_path"]),
                "status": "processing",
                "uploaded_at": datetime.utcnow().isoformat() + "Z",
                "uploaded_by": "ehr_batch",
                "file_size_bytes": len(info["content"]),
                "batch_id": batch_id,
                "patient_mrn": patient_mrn,
                "patient_id": patient_id,
                "tenant_id": item_meta.get("tenant_id", ""),
                "org_id": item_meta.get("org_id", ""),
                "encounter_id": item_meta.get("encounter_id", ""),
                "document_type": item_meta.get("document_type", ""),
                "document_date": item_meta.get("document_date", ""),
                "source_system": item_meta.get("source_system", "EHR"),
            }
            doc_results.append({
                "document_id": document_id,
                "filename": info["original_filename"],
                "status": "processing",
            })
        _save_documents_meta(existing_meta)

    _audit_log(
        "batch_ingest",
        AUDIT_SYSTEM_USER,
        batch_id=batch_id,
        doc_count=len(saved_files),
    )

    # ── Start background vectorization per document ───────────────────────
    for document_id, info in saved_files.items():
        item_meta = info["meta"]
        metadata_overrides = {
            k: v for k, v in item_meta.items()
            if k not in ("filename",) and v not in (None, "")
        }
        metadata_overrides["original_filename"] = info["original_filename"]
        metadata_overrides["ingestion_source"] = metadata_overrides.get(
            "ingestion_source", "ehr_batch_multipart"
        )
        metadata_overrides["batch_id"] = batch_id

        thread = _threading.Thread(
            target=_vectorize_batch_document,
            args=(document_id, info["file_path"], collection_name),
            kwargs={"metadata_overrides": metadata_overrides, "batch_id": batch_id},
            daemon=True,
        )
        thread.start()

    return JSONResponse(
        status_code=202,
        content={
            "job_id": batch_id,
            "batch_id": batch_id,
            "status": "processing",
            "documents": doc_results,
        },
    )


# ── Feedback API ─────────────────────────────────────────────────────

_feedback_lock = _threading.Lock()


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    """
    Save thumbs-up/down feedback for a bot response to a CSV file.
    """
    _audit_log("feedback", AUDIT_SYSTEM_USER, vote=req.vote, session_id=req.session_id)
    today_str = date.today().strftime('%Y-%m-%d')
    file_path = os.path.join(FEEDBACK_DIR, f"feedback_{today_str}.csv")

    with _feedback_lock:
        file_exists = os.path.exists(file_path)
        with open(file_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "session_id", "message_index",
                                 "vote", "user_query", "bot_response", "comment"])
            writer.writerow([
                datetime.now().isoformat(),
                req.session_id,
                req.message_index,
                req.vote,
                req.user_query[:500],       # truncate for safety
                req.bot_response[:500],
                req.comment or "",
            ])

    return {"status": "ok", "vote": req.vote}


# ── Entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    # Single-worker dev launch.  For multi-worker production use:
    #   python -m uvicorn src.api:app --host 0.0.0.0 --port 7860 --workers N
    # Requires Qdrant server running (start_qdrant.ps1) — no file-lock with server mode.
    import socket as _socket
    _port = int(os.environ.get("MEDGRAPH_PORT", "7860"))
    _test_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    try:
        _test_sock.bind(("0.0.0.0", _port))
        _test_sock.close()
    except OSError:
        import sys as _sys
        print(
            f"[MedGraph] ERROR: Port {_port} is already in use by another process.\n"
            f"  Stop all other 'api.py' processes before starting a new one.\n"
            f"  (Tip: use 'netstat -ano | findstr :{_port}' to find the PID)",
            file=_sys.stderr,
        )
        _sys.exit(1)
    uvicorn.run(app, host="0.0.0.0", port=_port, log_level="info")
