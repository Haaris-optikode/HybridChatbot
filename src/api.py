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
from fastapi import FastAPI, Request, Depends, HTTPException, status
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
from agent_graph.load_tools_config import LoadToolsConfig
from agent_graph.build_full_graph import build_graph
from utils.app_utils import create_directory
from chatbot.memory import Memory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

PROJECT_CFG = LoadProjectConfig()
TOOLS_CFG = LoadToolsConfig()

graph = build_graph()
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
    allow_methods=["GET", "POST"],
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
2. **summarize_patient_record(patient_identifier)** — ONLY for full clinical summaries/overviews. Accepts patient NAME or MRN.
3. **list_available_patients()** — List all patients in the database.
4. **tavily_search_results_json(query)** — Web search for general medical knowledge.

TOOL SELECTION (strict):
- ANY specific question about a patient (DOB, allergies, meds, labs, vitals, diagnoses, procedures, imaging, history) → `lookup_clinical_notes`
- ONLY when asked to "summarize", "overview", or "review the full record" → `summarize_patient_record`
- General medical knowledge → `tavily_search_results_json`
- Do NOT use `summarize_patient_record` for specific factual lookups like DOB, allergies, or medication lists.

ANSWER RULES:
- ONLY state facts from retrieved sources. If data is missing, say so.
- Use exact values (lab numbers, doses, dates). Never fabricate.
- Be concise. Answer the question directly.
- Include patient name and MRN when discussing patient data.
- Pay careful attention to TEMPORAL context: distinguish between admission/home medications vs discharge/new medications. If the question asks about admission or home meds, report what the patient was taking BEFORE hospitalization, not what was started or changed during the stay.
- When `summarize_patient_record` output is returned, pass it through verbatim.

SEARCH QUERY TIPS (for `lookup_clinical_notes`):
- Include specific clinical terms, day numbers, and section names in the query.
- For 'first time' or 'when was X started' questions, mention early hospital days and the specific clinical event.
- For discharge or follow-up questions, include terms like 'discharge summary', 'after-visit summary', 'follow-up appointments'.
- For medication questions, include the medication name AND the section (e.g., 'admission medications', 'discharge prescriptions').

MULTI-HOP (important):
- If after your first search the retrieved notes only mention LATER days but the question asks about the 'first' or 'earliest' occurrence, you MUST call lookup_clinical_notes AGAIN with a query that specifically targets early hospital days (e.g., 'Day 5 through Day 10 [event]').
- Similarly, if the question asks 'when was [medication] started' and the results only show later dose changes, search again for '[medication] initiated first started early days'.
- Always verify you have found the EARLIEST event before answering a 'first time' question.
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

    # Run through LangGraph agent
    config = {"configurable": {"thread_id": sid}}
    events = list(graph.stream(
        {"messages": messages}, config, stream_mode="values"
    ))
    reply = events[-1]["messages"][-1].content

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

    # ── Thread-safe queue bridges sync graph → async SSE generator ──
    q: _queue_mod.Queue = _queue_mod.Queue()

    def _run_graph():
        """Runs the sync graph.stream() in a background thread."""
        try:
            for update in graph.stream(
                {"messages": messages}, config, stream_mode="updates"
            ):
                q.put(("update", update))
            q.put(("end", None))
        except Exception as exc:
            q.put(("error", str(exc)))

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
                        content = getattr(m, "content", "")
                        tool_calls = getattr(m, "tool_calls", [])
                        if tool_calls:
                            for tc in tool_calls:
                                tc_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                                status = _TOOL_STATUS.get(tc_name, f"Running {tc_name}...")
                                yield f"data: {json.dumps({'type': 'status', 'content': status})}\n\n"
                        elif content:
                            full_response = content
                            yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

                elif node_name == "summary_passthrough":
                    for m in msgs:
                        content = getattr(m, "content", "")
                        if content:
                            full_response = content
                            # Stream the summary progressively in chunks (P2.20)
                            # instead of dumping 15K+ chars as a single blob
                            _CHUNK_SIZE = 120  # chars per SSE frame
                            for i in range(0, len(content), _CHUNK_SIZE):
                                chunk = content[i:i + _CHUNK_SIZE]
                                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                                await asyncio.sleep(0.01)  # ~10ms pacing for smooth render

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

    # ── OpenAI API check (lightweight models.list call) ──────────
    try:
        import httpx
        api_key = os.getenv("OPENAI_API_KEY", "")
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if resp.status_code == 200:
                checks["openai"] = {"status": "ok"}
            else:
                checks["openai"] = {"status": "error", "http_status": resp.status_code}
                checks["status"] = "degraded"
    except Exception as e:
        checks["openai"] = {"status": "error", "detail": str(e)}
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
