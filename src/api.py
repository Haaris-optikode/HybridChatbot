"""
FastAPI backend for the Clinical Intelligence Chatbot.
Replaces the Gradio UI with a REST API that serves a custom HTML/CSS/JS frontend.
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
import asyncio
import queue as _queue_mod
import threading as _threading
import uvicorn
import pandas as pd
from pathlib import Path
from datetime import datetime, date
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from dotenv import load_dotenv
from pyprojroot import here

load_dotenv()

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

# ── FastAPI app ──────────────────────────────────────────────────────
app = FastAPI(title="MedGraph AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── System prompt ────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are MedGraph AI, a clinical intelligence assistant for healthcare professionals. You have access to a vector database containing complete patient medical records.

## YOUR TOOLS:
1. **lookup_clinical_notes(query)** — Semantic search for specific clinical details (medications, labs, vitals, diagnoses, procedures, etc.). Use for targeted questions.
2. **summarize_patient_record(patient_identifier)** — Generates a comprehensive clinical summary. Accepts patient NAME or MRN directly. Use when asked to summarize, review, or give an overview.
3. **list_available_patients()** — Lists all patients in the database with their MRN and name.
4. **tavily_search_results_json(query)** — Web search for general medical knowledge, guidelines, drug information, or non-patient-specific questions.

## DECISION LOGIC:
- Patient-specific question (e.g., "What medications is the patient on?") → Use `lookup_clinical_notes`
- Summarize / overview / review request → Call `summarize_patient_record` directly with the patient name or MRN. Do NOT call list_available_patients first — the tool resolves names automatically.
- General medical question (e.g., "What is the mechanism of furosemide?") → Use `tavily_search_results_json`
- "Who is in the database?" / "What patients?" → Use `list_available_patients`

## CRITICAL — SUMMARY PASSTHROUGH RULE:
When you receive output from `summarize_patient_record`, return it EXACTLY as-is to the user. Do NOT rephrase, reorganize, add commentary, or generate additional text. The summary is already clinically formatted. Just pass it through verbatim. This saves significant processing time.

## RESPONSE RULES (SOURCE-GROUNDED — NotebookLM-style):
1. **ONLY state facts that appear in the retrieved source material.** Never fabricate, infer, or fill in details from your training data.
2. **Cite your sources inline** using the section headers from the retrieved chunks, e.g., [Section: Current Medications]. When multiple sources support a point, cite all of them.
3. **If the retrieved data does not contain the answer**, say exactly: "The available records do not contain information about [topic]." Do NOT guess.
4. **Use exact values** from the record: lab numbers, medication doses, dates, vital signs. Never round or approximate.
5. **For clinical summaries**, be comprehensive — cover every clinical domain documented. Missing a medication, allergy, or lab result in a summary is a critical failure.
6. **Structure your responses** with clear headings and bullet points for clinical content. Use Markdown formatting.
7. **Flag critical findings** — mark abnormal lab values, critical vitals, drug allergies, or safety concerns prominently.
8. **When uncertain**, make another tool call to retrieve more context rather than guessing.

## OUTPUT FORMAT:
- For search queries: Direct answer with inline citations
- For summaries: Pass through the tool output exactly as received
- Always include the patient's name and MRN when discussing patient-specific data
"""

# ── In-memory session store ─────────────────────────────────────────
sessions: dict = {}


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[float] = None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    tool_override: Optional[str] = None  # "vector_db", "web_search", "sql" or None (auto)


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


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    t0 = time.perf_counter()

    # Session management
    sid = req.session_id or str(uuid.uuid4())
    if sid not in sessions:
        sessions[sid] = []

    history: List[ChatMessage] = sessions[sid]

    # Build LangChain message list
    system_prompt = SYSTEM_PROMPT
    if req.tool_override and req.tool_override in TOOL_PROMPTS:
        system_prompt += " " + TOOL_PROMPTS[req.tool_override]

    messages = [SystemMessage(content=system_prompt)]
    for msg in history:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        else:
            messages.append(AIMessage(content=msg.content))
    messages.append(HumanMessage(content=req.message))

    # Run through LangGraph agent
    config = {"configurable": {"thread_id": sid}}
    events = list(graph.stream(
        {"messages": messages}, config, stream_mode="values"
    ))
    reply = events[-1]["messages"][-1].content

    # Store in session
    now = time.time()
    history.append(ChatMessage(role="user", content=req.message, timestamp=now))
    history.append(ChatMessage(role="assistant", content=reply, timestamp=time.time()))
    sessions[sid] = history

    # Save to memory
    gradio_format = [(h.content, sessions[sid][i + 1].content)
                     for i, h in enumerate(sessions[sid])
                     if h.role == "user" and i + 1 < len(sessions[sid])]
    Memory.write_chat_history_to_file(
        gradio_chatbot=gradio_format,
        folder_path=PROJECT_CFG.memory_dir,
        thread_id=sid
    )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return ChatResponse(reply=reply, session_id=sid, latency_ms=elapsed_ms)


# ── Streaming SSE endpoint ───────────────────────────────────────────

_TOOL_STATUS = {
    "summarize_patient_record": "Generating clinical summary...",
    "lookup_clinical_notes": "Searching clinical notes...",
    "list_available_patients": "Listing patients...",
    "tavily_search_results_json": "Searching the web...",
}


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE endpoint — runs graph in a thread, relays updates via queue."""
    t0 = time.perf_counter()

    sid = req.session_id or str(uuid.uuid4())
    if sid not in sessions:
        sessions[sid] = []

    history: List[ChatMessage] = sessions[sid]

    system_prompt = SYSTEM_PROMPT
    if req.tool_override and req.tool_override in TOOL_PROMPTS:
        system_prompt += " " + TOOL_PROMPTS[req.tool_override]

    messages = [SystemMessage(content=system_prompt)]
    for msg in history:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        else:
            messages.append(AIMessage(content=msg.content))
    messages.append(HumanMessage(content=req.message))

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

                if node_name == "chatbot":
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
                            yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

        # ── Session & memory persistence ──────────────────────────
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        now = time.time()
        history.append(ChatMessage(role="user", content=req.message, timestamp=now))
        history.append(ChatMessage(role="assistant", content=full_response, timestamp=time.time()))
        sessions[sid] = history

        gradio_format = [(h.content, sessions[sid][i + 1].content)
                         for i, h in enumerate(sessions[sid])
                         if h.role == "user" and i + 1 < len(sessions[sid])]
        Memory.write_chat_history_to_file(
            gradio_chatbot=gradio_format,
            folder_path=PROJECT_CFG.memory_dir,
            thread_id=sid
        )

        yield f"data: {json.dumps({'type': 'done', 'session_id': sid, 'latency_ms': elapsed_ms})}\n\n"

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
async def clear_session(req: ChatRequest):
    sid = req.session_id
    if sid and sid in sessions:
        del sessions[sid]
    return {"status": "cleared", "session_id": sid}


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": TOOLS_CFG.primary_agent_llm}


# ── Clear Summary Cache API ───────────────────────────────────────

@app.post("/api/cache/clear")
async def clear_summary_cache():
    """Delete all cached patient summaries so they regenerate fresh."""
    cache_dir = Path(here("data")) / "summary_cache"
    if not cache_dir.exists():
        return {"status": "ok", "deleted": 0}
    deleted = 0
    for f in cache_dir.glob("*.json"):
        try:
            f.unlink()
            deleted += 1
        except Exception:
            pass
    return {"status": "ok", "deleted": deleted}


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
        if req.session_id in sessions:
            del sessions[req.session_id]

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
        sessions[session_id] = []
        for msg in all_messages:
            sessions[session_id].append(
                ChatMessage(role="user", content=msg["user_query"], timestamp=0))
            sessions[session_id].append(
                ChatMessage(role="assistant", content=msg["response"], timestamp=0))

    return {"messages": all_messages, "session_id": session_id}


# ── Feedback API ─────────────────────────────────────────────────────

@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    """
    Save thumbs-up/down feedback for a bot response to a CSV file.
    """
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
