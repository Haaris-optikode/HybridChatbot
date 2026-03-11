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
import logging
import asyncio
import queue as _queue_mod
import threading as _threading
import uvicorn
import jwt
import pandas as pd
from pathlib import Path
from datetime import datetime, date, timedelta
from fastapi import FastAPI, Request, Depends, HTTPException, status, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional, Dict
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

# ── HIPAA Audit Log File Handler ─────────────────────────────────────
_AUDIT_LOG_DIR = str(here("logs"))
os.makedirs(_AUDIT_LOG_DIR, exist_ok=True)
_audit_file_handler = logging.FileHandler(
    os.path.join(_AUDIT_LOG_DIR, "hipaa_audit.log"),
    encoding="utf-8",
)
_audit_file_handler.setFormatter(logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "event": %(message)s}',
    datefmt="%Y-%m-%dT%H:%M:%S",
))
audit_logger.addHandler(_audit_file_handler)

# ── LangGraph agent imports ──────────────────────────────────────────
from chatbot.load_config import LoadProjectConfig
from agent_graph.load_tools_config import LoadToolsConfig, swap_google_api_key
from agent_graph.build_full_graph import build_graph
from utils.app_utils import create_directory
from chatbot.memory import Memory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

PROJECT_CFG = LoadProjectConfig()
TOOLS_CFG = LoadToolsConfig()

graph = build_graph(thinking_mode=False)
thinking_graph = build_graph(thinking_mode=True)
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

# ── JWT Configuration ────────────────────────────────────────────────
# In production, set MEDGRAPH_JWT_SECRET to a strong random value.
# Default secret is for DEVELOPMENT ONLY — never use in production.
JWT_SECRET = os.getenv("MEDGRAPH_JWT_SECRET", "dev-secret-change-me-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("MEDGRAPH_JWT_EXPIRATION_HOURS", "24"))

if JWT_SECRET == "dev-secret-change-me-in-production":
    logger.warning("Using default JWT secret — set MEDGRAPH_JWT_SECRET env var for production!")

security_scheme = HTTPBearer(auto_error=False)

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
    allow_headers=["Authorization", "Content-Type"],
    allow_credentials=True,
)

# Serve static frontend files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Authentication helpers ───────────────────────────────────────────
def create_jwt_token(user_id: str, role: str = "clinician") -> str:
    """Generate a signed JWT token for a user."""
    payload = {
        "sub": user_id,
        "role": role,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_jwt_token(token: str) -> dict:
    """Verify and decode a JWT token. Raises HTTPException on failure."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
) -> dict:
    """Dependency that extracts and validates the JWT bearer token.
    Returns the decoded payload with 'sub' (user_id) and 'role' fields.
    Authentication is ALWAYS required — no dev-mode bypass."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide a Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return verify_jwt_token(credentials.credentials)


def _audit_log(event: str, user: dict, **extra):
    """Write a structured HIPAA audit log entry."""
    entry = {
        "event": event,
        "user_id": user.get("sub", "unknown"),
        "role": user.get("role", "unknown"),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        **extra,
    }
    audit_logger.info(json.dumps(entry))

# ── System prompt ────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are MedGraph AI, a clinical intelligence assistant. Tools:

1. **lookup_clinical_notes(query)** — Search patient records for ANY specific detail: DOB, age, allergies, medications, labs, vitals, diagnoses, procedures, imaging, history.
2. **lookup_patient_orders(patient_identifier)** — Retrieve ALL orders for a patient: lab orders (CPT/LOINC), medication orders (RxNorm), procedure orders (CPT/SNOMED), referrals, discharge prescriptions, and care directives. Accepts patient NAME or MRN.
3. **summarize_patient_record(patient_identifier)** — ONLY for full clinical summaries/overviews. Accepts patient NAME or MRN.
4. **list_available_patients()** — List all patients in the database.
5. **tavily_search_results_json(query)** — Web search for general medical knowledge.
6. **list_uploaded_documents()** — List all uploaded documents with their IDs, filenames, and processing status.

TOOL SELECTION (strict):
- ANY specific question about a patient (DOB, allergies, meds, labs, vitals, diagnoses, procedures, imaging, history) → `lookup_clinical_notes`
- Queries asking for "all orders", "complete order list", "what was ordered", "list all medications/labs/procedures/referrals", or any COMPREHENSIVE orders request → `lookup_patient_orders`
- ONLY when the user explicitly asks to "summarize the record", "give me an overview", or "review the full record" with NO specific analytical question → `summarize_patient_record`
- ANALYTICAL questions that mention a patient (readmission risk, differential diagnosis, treatment effectiveness, drug interactions, risk factors, prognosis, clinical reasoning) → `lookup_clinical_notes` (use MULTIPLE targeted searches to gather all relevant data, then synthesize)
- General medical knowledge → `tavily_search_results_json`
- Do NOT use `summarize_patient_record` for specific factual lookups like DOB, allergies, or medication lists.
- Do NOT use `summarize_patient_record` when the user asks an analytical or reasoning question — even if they say "synthesize" or "analyze". Use `lookup_clinical_notes` with multiple targeted queries instead.
- Do NOT use `lookup_clinical_notes` when the user wants ALL orders — use `lookup_patient_orders` instead.

ANSWER RULES:
- ONLY state facts from retrieved sources. If data is missing, say so.
- Use exact values (lab numbers, doses, dates). Never fabricate.
- When presenting orders, include ALL available details: CPT codes, LOINC codes, RxNorm codes, SNOMED codes, dosages, frequencies, dates, and descriptions. Present them in organized sections.
- Include patient name and MRN when discussing patient data.
- Pay careful attention to TEMPORAL context: distinguish between admission/home medications vs discharge/new medications. If the question asks about admission or home meds, report what the patient was taking BEFORE hospitalization, not what was started or changed during the stay.
- When `summarize_patient_record` output is returned, use it as context to answer the user's original question. If the user asked an analytical question, DO NOT just pass through the raw summary — synthesize a targeted answer.
- When `lookup_patient_orders` output is returned, organize the response by order category (Lab Orders, Medication Orders, Procedure Orders, Referrals, Discharge Prescriptions, Care Directives) and include ALL codes and details from the data.

RESPONSE STYLE:
- Write like a clinician briefing a colleague — clear, professional, concise.
- Synthesize retrieved data into a coherent answer. Do NOT just repeat raw chunks or bullet-dump metadata.
- Include reference ranges for abnormal lab values and flag severity (e.g., "BNP 1,842 pg/mL [Critical High, ref <100]").
- Use bullet points and bold for key values. Use headers only when the answer has 3+ distinct categories.
- For medication lists, include dose, route, frequency. Add indication only when the question asks about it.
- Keep answers focused: answer what was asked, include essential clinical context, then stop. Do NOT exhaustively list every related finding unless specifically asked for a comprehensive review.
- Target 80–150 words for simple queries, 200–500 words for complex multi-part questions. For comprehensive analytical questions (readmission risk, treatment analysis), use as many words as needed to be thorough.

SOURCE GROUNDING (Critical — prevents hallucination):
- Every factual claim MUST come from the retrieved sources. Cite as "(Source: [Section], Page X)" at the end of relevant paragraphs or sections, not after every single sentence.
- If the sources do NOT contain the answer, say: "The available clinical records do not contain information about [topic]." Do NOT guess or fill gaps.
- NEVER extrapolate beyond what is explicitly stated. If a value is not documented, do not estimate it.
- When sources contain contradictory information, report both values and note the discrepancy.

SEARCH QUERY TIPS (for `lookup_clinical_notes`):
- Include specific clinical terms, day numbers, and section names in the query.
- For 'first time' or 'when was X started' questions, mention early hospital days and the specific clinical event.
- For discharge or follow-up questions, include terms like 'discharge summary', 'after-visit summary', 'follow-up appointments'.
- For medication questions, include the medication name AND the section (e.g., 'admission medications', 'discharge prescriptions').

MULTI-HOP (important):
- If after your first search the retrieved notes only mention LATER days but the question asks about the 'first' or 'earliest' occurrence, you MUST call lookup_clinical_notes AGAIN with a query that specifically targets early hospital days (e.g., 'Day 5 through Day 10 [event]').
- Similarly, if the question asks 'when was [medication] started' and the results only show later dose changes, search again for '[medication] initiated first started early days'.
- Always verify you have found the EARLIEST event before answering a 'first time' question.
- For TREND questions (e.g., 'how did creatinine change', 'weight trend', 'BNP over time'), you MUST search at least TWICE: once for admission/baseline values and once for later/discharge values. A trend answer is incomplete without both endpoints.
- When a user asks for demographics like full name + DOB + MRN, always call lookup_clinical_notes with a query like 'patient demographics date of birth age MRN' to retrieve the full details. Do NOT stop at listing a name from list_available_patients — that tool only returns names and MRNs, not clinical details.

PATIENT DISAMBIGUATION:
- When the user says "the patient" without specifying a name or MRN, do NOT ask for clarification. Instead, proceed with the query using the most recently discussed patient in the conversation history.
- If no patient has been discussed yet in the conversation, call list_available_patients to identify available patients, then proceed with the patient who has the most comprehensive record.
- If the conversation has only one patient being discussed, always assume subsequent "the patient" references mean that same patient.
- NEVER fabricate or guess patient data. If a user mentions a name that does NOT exactly match any patient in the database (e.g., "Richard" when the record says "Robert"), clearly state the mismatch: "I found Robert James Whitfield but no patient named [queried name]." Do NOT answer as if the wrong name is correct.
"""

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


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    latency_ms: int


class FeedbackRequest(BaseModel):
    session_id: str
    message_index: int  # which bot message (0-based among bot messages)
    user_query: str
    bot_response: str
    vote: str  # "up" or "down"
    comment: Optional[str] = None


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
}


# ── Routes ───────────────────────────────────────────────────────────
@app.get("/")
async def serve_frontend():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ── Token generation endpoint (dev/test convenience) ─────────────────
class TokenRequest(BaseModel):
    user_id: str
    role: str = "clinician"  # clinician, admin, nurse


@app.post("/api/auth/token")
@limiter.limit("10/minute")
async def generate_token(req: TokenRequest, request: Request):
    """Generate a JWT token. In production, replace with real identity provider."""
    token = create_jwt_token(req.user_id, req.role)
    logger.info("Token generated for user=%s role=%s", req.user_id, req.role)
    return {"token": token, "expires_in_hours": JWT_EXPIRATION_HOURS}


@app.post("/api/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat(req: ChatRequest, request: Request, user: dict = Depends(get_current_user)):
    global graph, thinking_graph
    t0 = time.perf_counter()
    validated_msg = _validate_message(req.message)
    _audit_log("chat_request", user, tool_override=req.tool_override or "auto")

    # Session management
    sid = req.session_id or str(uuid.uuid4())
    history: List[ChatMessage] = sessions.get(sid)

    # Build LangChain message list (trimmed to last N turns)
    system_prompt = SYSTEM_PROMPT
    if req.tool_override and req.tool_override in TOOL_PROMPTS:
        system_prompt += " " + TOOL_PROMPTS[req.tool_override]

    trimmed = _trim_history(history)
    messages = [SystemMessage(content=system_prompt)]
    for msg in trimmed:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        else:
            messages.append(AIMessage(content=msg.content))
    messages.append(HumanMessage(content=validated_msg))

    # Run through LangGraph agent — with retry on transient LLM errors
    config = {"configurable": {"thread_id": sid}}
    active_graph = thinking_graph if req.thinking else graph
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            events = list(active_graph.stream(
                {"messages": messages}, config, stream_mode="values"
            ))
            reply = _normalize_content(events[-1]["messages"][-1].content)
            break
        except Exception as exc:
            exc_str = str(exc).lower()
            is_rate_limit = any(k in exc_str for k in ["rate limit", "429", "resource_exhausted", "quota"])
            is_transient = is_rate_limit or any(k in exc_str for k in ["timeout", "timed out", "overloaded", "503"])
            if is_rate_limit:
                swapped = swap_google_api_key()
                if swapped:
                    graph = build_graph(thinking_mode=False)
                    thinking_graph = build_graph(thinking_mode=True)
                    active_graph = thinking_graph if req.thinking else graph
                    logger.warning("Rebuilt graphs with backup API key after rate limit")
                elif req.thinking and active_graph is not graph:
                    # Thinking model quota exhausted on all keys — fall back to normal graph
                    active_graph = graph
                    logger.warning("Thinking model quota exhausted, falling back to normal graph")
            if is_transient and attempt < max_retries:
                wait = (attempt + 1) * 5  # 5s, 10s
                logger.warning("Transient LLM error (attempt %d/%d), retrying in %ds: %s",
                               attempt + 1, max_retries + 1, wait, str(exc)[:200])
                import asyncio
                await asyncio.sleep(wait)
                continue
            logger.error("LLM error after %d attempts: %s", attempt + 1, str(exc)[:500])
            raise HTTPException(status_code=503, detail=f"Upstream AI service unavailable. Please retry in a moment.")

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
    Memory.write_chat_history_to_file(
        gradio_chatbot=gradio_format,
        folder_path=PROJECT_CFG.memory_dir,
        thread_id=sid
    )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    PROM_REQUEST_LATENCY.labels(endpoint="/api/chat", method="POST").observe(elapsed_ms / 1000)
    _audit_log("chat_response", user, session_id=sid, latency_ms=elapsed_ms)
    return ChatResponse(reply=reply, session_id=sid, latency_ms=elapsed_ms)


# ── Streaming SSE endpoint ───────────────────────────────────────────

_TOOL_STATUS = {
    "summarize_patient_record": "Generating clinical summary...",
    "lookup_clinical_notes": "Searching clinical notes...",
    "list_available_patients": "Listing patients...",
    "tavily_search_results_json": "Searching the web...",
}


@app.post("/api/chat/stream")
@limiter.limit("30/minute")
async def chat_stream(req: ChatRequest, request: Request, user: dict = Depends(get_current_user)):
    """SSE endpoint — runs graph in a thread, relays updates via queue."""
    global graph, thinking_graph
    t0 = time.perf_counter()
    _audit_log("chat_stream_request", user, tool_override=req.tool_override or "auto")

    validated_msg = _validate_message(req.message)

    sid = req.session_id or str(uuid.uuid4())
    history: List[ChatMessage] = sessions.get(sid)

    system_prompt = SYSTEM_PROMPT
    if req.tool_override and req.tool_override in TOOL_PROMPTS:
        system_prompt += " " + TOOL_PROMPTS[req.tool_override]

    trimmed = _trim_history(history)
    messages = [SystemMessage(content=system_prompt)]
    for msg in trimmed:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        else:
            messages.append(AIMessage(content=msg.content))
    messages.append(HumanMessage(content=validated_msg))

    config = {"configurable": {"thread_id": sid}}
    active_graph = thinking_graph if req.thinking else graph

    # ── Thread-safe queue bridges sync graph → async SSE generator ──
    q: _queue_mod.Queue = _queue_mod.Queue()
    _graph_holder = {"active": active_graph}

    def _run_graph():
        """Runs the sync graph.stream() in a background thread, with key-swap retry."""
        global graph, thinking_graph
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                for update in _graph_holder["active"].stream(
                    {"messages": messages}, config, stream_mode="updates"
                ):
                    q.put(("update", update))
                q.put(("end", None))
                return
            except Exception as exc:
                exc_str = str(exc).lower()
                is_rate_limit = any(k in exc_str for k in ["rate limit", "429", "resource_exhausted", "quota"])
                if is_rate_limit and attempt < max_retries:
                    swapped = swap_google_api_key()
                    if swapped:
                        graph = build_graph(thinking_mode=False)
                        thinking_graph = build_graph(thinking_mode=True)
                        _graph_holder["active"] = thinking_graph if req.thinking else graph
                        logger.warning("Rebuilt graphs with backup API key (SSE), retrying...")
                    elif req.thinking and _graph_holder["active"] is not graph:
                        # Thinking model quota exhausted — fall back to normal graph
                        _graph_holder["active"] = graph
                        logger.warning("Thinking model quota exhausted (SSE), falling back to normal graph")
                    import time as _t
                    _t.sleep(3)
                    continue
                q.put(("error", str(exc)))
                return

    async def event_generator():
        full_response = ""

        # Start graph in background thread (sync graph.stream is proven reliable)
        thread = _threading.Thread(target=_run_graph, daemon=True)
        thread.start()

        while True:
            # Poll queue without blocking the event loop
            try:
                kind, data = q.get_nowait()
            except _queue_mod.Empty:
                await asyncio.sleep(0.05)
                continue

            if kind == "end":
                break
            if kind == "error":
                yield f"data: {json.dumps({'type': 'error', 'content': data})}\n\n"
                return

            # Process graph node update
            for node_name, node_output in data.items():
                msgs = node_output.get("messages", [])

                # router / synthesizer are the LLM nodes (split architecture)
                if node_name in ("router", "synthesizer", "chatbot"):
                    for m in msgs:
                        content = _normalize_content(getattr(m, "content", ""))
                        tool_calls = getattr(m, "tool_calls", [])
                        if tool_calls:
                            for tc in tool_calls:
                                tc_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                                status = _TOOL_STATUS.get(tc_name, f"Running {tc_name}...")
                                yield f"data: {json.dumps({'type': 'status', 'content': status})}\n\n"
                        elif content and node_name != "router":
                            # Only emit content from synthesizer — router is for tool selection only
                            full_response = content
                            yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

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
        Memory.write_chat_history_to_file(
            gradio_chatbot=gradio_format,
            folder_path=PROJECT_CFG.memory_dir,
            thread_id=sid
        )

        yield f"data: {json.dumps({'type': 'done', 'session_id': sid, 'latency_ms': elapsed_ms})}\n\n"
        PROM_REQUEST_LATENCY.labels(endpoint="/api/chat/stream", method="POST").observe(elapsed_ms / 1000)
        _audit_log("chat_stream_response", user, session_id=sid, latency_ms=elapsed_ms)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/clear")
async def clear_session(req: ChatRequest, user: dict = Depends(get_current_user)):
    sid = req.session_id
    if sid:
        sessions.delete(sid)
    _audit_log("session_cleared", user, session_id=sid)
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


# ── Prometheus Metrics Endpoint (P2.22) ───────────────────────────

@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint. No auth — scrapers need direct access."""
    PROM_ACTIVE_SESSIONS.set(len(sessions))
    from starlette.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── Clear Summary Cache API ───────────────────────────────────────

@app.post("/api/cache/clear")
async def clear_summary_cache(user: dict = Depends(get_current_user)):
    """Delete all cached patient summaries (both disk and in-memory) so they regenerate fresh."""
    _audit_log("cache_cleared", user)
    # Clear in-memory cache
    from agent_graph.tool_clinical_notes_rag import _memory_cache, _memory_cache_lock
    with _memory_cache_lock:
        memory_cleared = len(_memory_cache)
        _memory_cache.clear()

    # Clear disk cache
    cache_dir = Path(here("data")) / "summary_cache"
    disk_deleted = 0
    if cache_dir.exists():
        for f in cache_dir.glob("*.json"):
            try:
                f.unlink()
                disk_deleted += 1
            except Exception:
                pass
    return {"status": "ok", "disk_deleted": disk_deleted, "memory_cleared": memory_cleared}


# ── Delete Chat History API ───────────────────────────────────────────

class DeleteHistoryRequest(BaseModel):
    session_id: str
    date: str  # YYYY-MM-DD


@app.post("/api/history/delete")
async def delete_session(req: DeleteHistoryRequest, user: dict = Depends(get_current_user)):
    """
    Delete a specific session from a specific date's CSV file.
    Removes only the rows matching the session_id from that date.
    """
    _audit_log("history_delete", user, session_id=req.session_id, date=req.date)
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
async def list_sessions(user: dict = Depends(get_current_user)):
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
async def load_session(session_id: str, date: Optional[str] = None, user: dict = Depends(get_current_user)):
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


def _vectorize_in_background(document_id: str, pdf_path: str, collection_name: str):
    """Run vectorization in a background thread and update metadata on completion."""
    try:
        from document_processing import process_document
        from agent_graph.tool_clinical_notes_rag import get_rag_tool_instance
        from langchain_qdrant import QdrantVectorStore

        # Process PDF into chunks (no Qdrant client needed for this step)
        documents = process_document(
            pdf_path=pdf_path,
            chunk_size=TOOLS_CFG.clinical_notes_rag_chunk_size,
            chunk_overlap=TOOLS_CFG.clinical_notes_rag_chunk_overlap,
            document_id=document_id,
        )

        # Reuse the RAG tool's existing Qdrant client and vector store
        # (embedded mode only allows a single client instance per storage path)
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

        # Refresh the BM25 index
        try:
            rag._build_bm25_index()
            logger.info("BM25 index refreshed after document upload: %s", document_id)
        except Exception as e:
            logger.warning("Could not refresh BM25 index: %s", e)

        with _documents_lock:
            meta = _load_documents_meta()
            if document_id in meta:
                meta[document_id]["status"] = "ready"
                meta[document_id]["chunks"] = len(documents)
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
    user: dict = Depends(get_current_user),
):
    """Upload a PDF document for vectorization and RAG querying.

    The file is saved to disk and vectorized in the background.
    Returns a document_id that can be used to check status or delete the document.
    """
    # Validate file extension
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported: {', '.join(SUPPORTED_FORMATS)}",
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
    pdf_path = os.path.join(UPLOADS_DIR, safe_filename)

    with open(pdf_path, "wb") as f:
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
            "uploaded_by": user.get("sub", "unknown"),
            "file_size_bytes": len(content),
        }
        _save_documents_meta(meta)

    _audit_log("document_upload", user, document_id=document_id, filename=filename)

    # Start vectorization in background
    collection_name = TOOLS_CFG.clinical_notes_rag_collection_name
    thread = _threading.Thread(
        target=_vectorize_in_background,
        args=(document_id, pdf_path, collection_name),
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
async def list_documents(user: dict = Depends(get_current_user)):
    """List all uploaded documents with their processing status."""
    with _documents_lock:
        meta = _load_documents_meta()
    documents = sorted(meta.values(), key=lambda d: d.get("uploaded_at", ""), reverse=True)
    return {"documents": documents}


@app.get("/api/documents/{document_id}")
async def get_document_status(document_id: str, user: dict = Depends(get_current_user)):
    """Get the status of a specific uploaded document."""
    with _documents_lock:
        meta = _load_documents_meta()
    doc = meta.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str, user: dict = Depends(get_current_user)):
    """Delete an uploaded document and its vectors from the collection."""
    _audit_log("document_delete", user, document_id=document_id)

    with _documents_lock:
        meta = _load_documents_meta()
        doc = meta.get(document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found.")

        # Remove file from disk
        file_path = os.path.join(UPLOADS_DIR, doc.get("file_path", ""))
        if os.path.exists(file_path):
            os.remove(file_path)

        # Remove from metadata
        del meta[document_id]
        _save_documents_meta(meta)

    # Remove vectors from Qdrant (reuse RAG tool's client for embedded mode)
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

        # Refresh BM25 index
        try:
            rag._build_bm25_index()
        except Exception:
            pass

    except Exception as e:
        logger.error("Failed to delete vectors for %s: %s", document_id, str(e))

    return {"status": "deleted", "document_id": document_id}


# ── Feedback API ─────────────────────────────────────────────────────

@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest, user: dict = Depends(get_current_user)):
    """
    Save thumbs-up/down feedback for a bot response to a CSV file.
    """
    _audit_log("feedback", user, vote=req.vote, session_id=req.session_id)
    today_str = date.today().strftime('%Y-%m-%d')
    file_path = os.path.join(FEEDBACK_DIR, f"feedback_{today_str}.csv")
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
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
