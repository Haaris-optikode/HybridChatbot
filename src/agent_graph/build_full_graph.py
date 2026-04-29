import logging


from langgraph.graph import StateGraph, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage as _ToolMessage, AIMessage as _AIMessage
from agent_graph.tool_clinical_notes_rag import (
    lookup_clinical_notes,
    lookup_patient_orders,
    cohort_patient_search,
    summarize_patient_record,
    summarize_uploaded_document,
    summarize_uploaded_documents,
    list_available_patients,
    list_uploaded_documents,
    get_rag_tool_instance,
)
from agent_graph.tool_tavily_search import load_tavily_search_tool
from agent_graph.load_tools_config import LoadToolsConfig, get_google_api_key, get_active_llm_provider
from agent_graph.agent_backend import State, BasicToolNode, route_tools, plot_agent_schema
import os
import json as _json
import re as _re
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

TOOLS_CFG = LoadToolsConfig()
_UNSCOPED_PATIENT_QUERY_MSG = (
    "I need the patient name or MRN before searching clinical notes. "
    "Please specify which patient you want me to review."
)


def _is_reasoning_model(model_name: str) -> bool:
    """Check if a model name refers to a reasoning model (no temperature param)."""
    _reasoning_prefixes = ("o1", "o3", "o4", "gpt-5")
    return any(model_name.lower().startswith(p) for p in _reasoning_prefixes)


def _make_llm(model_name: str, timeout: int = 60, max_tokens: int = None, thinking_budget: int = 0):
    """Create a Chat LLM instance — auto-selects Google Gemini or OpenAI based on config."""
    provider = get_active_llm_provider(getattr(TOOLS_CFG, "llm_provider", "openai"))

    if provider == "google" or model_name.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        kwargs = dict(
            model=model_name,
            google_api_key=get_google_api_key(),
            timeout=timeout,
            max_retries=2,
        )
        if thinking_budget > 0:
            # Thinking mode: temperature must be omitted or set by the API
            kwargs["thinking_budget"] = thinking_budget
        else:
            kwargs["temperature"] = TOOLS_CFG.primary_agent_llm_temperature
            kwargs["thinking_budget"] = 0
        if max_tokens:
            kwargs["max_output_tokens"] = max_tokens
        llm = ChatGoogleGenerativeAI(**kwargs)
        return llm

    # OpenAI path (fallback)
    kwargs = dict(
        model=model_name,
        api_key=os.getenv("OPENAI_API_KEY"),
        request_timeout=timeout,
        max_retries=2,
        stream_usage=True,
    )
    if _is_reasoning_model(model_name):
        kwargs["max_completion_tokens"] = max_tokens or 16384
    else:
        kwargs["temperature"] = TOOLS_CFG.primary_agent_llm_temperature
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)


def _last_user_text(state: State) -> str:
    """Return the most recent user utterance from the graph state."""
    for m in reversed(state.get("messages", [])):
        if getattr(m, "type", "") == "human" or m.__class__.__name__ == "HumanMessage":
            return getattr(m, "content", "") or ""
    return ""


def _recent_human_texts(state: State) -> list[str]:
    """Return human messages in reverse chronological order."""
    out: list[str] = []
    for m in reversed(state.get("messages", [])):
        if getattr(m, "type", "") == "human" or m.__class__.__name__ == "HumanMessage":
            content = getattr(m, "content", "") or ""
            if content:
                out.append(content)
    return out


def _query_has_explicit_patient_identifier(query: str) -> bool:
    """Return True when the user explicitly mentions a patient identifier."""
    q = (query or "").strip().lower()
    if not q:
        return False
    if any(marker in q for marker in ("mrn", "patient id", "patient name")):
        return True
    return False


def _resolve_active_patient_mrn(state: State, current_query: str = "") -> str:
    """Resolve the active patient MRN from the current query or prior chat history."""
    existing = (state.get("active_patient_mrn", "") or "").strip()

    try:
        rag = get_rag_tool_instance()
    except Exception:
        # Still check context var before giving up
        from agent_graph.tool_clinical_notes_rag import _CTX_ACTIVE_PATIENT_MRN as _ctx_mrn_var
        return existing or (_ctx_mrn_var.get() or "").strip()

    current = (current_query or "").strip()
    if current:
        try:
            current_mrn = rag.resolve_to_mrn(current)
        except Exception:
            current_mrn = None
        if current_mrn:
            return str(current_mrn).strip()
        if _query_has_explicit_patient_identifier(current):
            # User explicitly named a different patient/identifier; do not fall
            # back to stale session context if resolution failed.
            return ""

    if existing:
        return existing

    seen = {current} if current else set()
    for text in _recent_human_texts(state):
        norm = (text or "").strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        try:
            mrn = rag.resolve_to_mrn(norm)
        except Exception:
            mrn = None
        if mrn:
            return str(mrn).strip()

    # Final fallback: use request-scoped MRN from context var (set by api.py via
    # set_active_document_scope when patient_mrn is supplied in the chat request).
    from agent_graph.tool_clinical_notes_rag import _CTX_ACTIVE_PATIENT_MRN as _ctx_mrn_var
    ctx_mrn = (_ctx_mrn_var.get() or "").strip()
    if ctx_mrn:
        return ctx_mrn
    return ""


def _resolve_active_document_ids(state: State) -> list[str]:
    """Resolve selected document IDs from graph state or request context."""
    existing = list(state.get("active_document_ids", []) or [])
    if existing:
        return [str(d).strip() for d in existing if str(d).strip()]
    from agent_graph.tool_clinical_notes_rag import _CTX_ACTIVE_DOCUMENT_IDS as _ctx_doc_ids_var
    return [str(d).strip() for d in (_ctx_doc_ids_var.get() or []) if str(d).strip()]


def _is_summary_request(query: str) -> bool:
    q = (query or "").lower()
    return bool(_re.search(r"\bsummar(?:y|ize|ise)|\boverview\b|\breview\b|\bwhat\s+is\s+this\b|\bwhat\s+are\s+these\b|\bcontents?\b", q))


def _is_orders_request(query: str) -> bool:
    q = (query or "").lower()
    return bool(_re.search(r"\ball\s+orders?\b|\border(?:ed|s)?\b|\bcomplete\s+(?:med|medication|lab|order)|\bmedications?\s+and\s+labs?\b", q))


def _normalize_document_ids(value) -> list[str]:
    """Normalize tool/model document_id inputs into a de-duplicated list."""
    if not value:
        return []
    if isinstance(value, str):
        raw = [p.strip() for p in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        raw = [str(p).strip() for p in value]
    else:
        raw = [str(value).strip()]
    return list(dict.fromkeys([p for p in raw if p]))


def _copy_ai_message_with_tool_calls(message: _AIMessage, tool_calls: list[dict]) -> _AIMessage:
    """Clone an AIMessage while replacing tool calls."""
    try:
        if hasattr(message, "model_copy"):
            return message.model_copy(update={"tool_calls": tool_calls})
        return message.copy(update={"tool_calls": tool_calls})
    except Exception:
        return _AIMessage(content=str(getattr(message, "content", "") or ""), tool_calls=tool_calls)


def _scope_lookup_tool_calls(message: _AIMessage, active_mrn: str) -> _AIMessage:
    """Enforce active scope on all tool calls that access patient/document data.

    Handles tools that touch patient/document data:
    - lookup_clinical_notes: inject MRN; validate document scope; block when unscoped.
    - lookup_patient_orders: inject MRN and selected document IDs.
    - summarize_uploaded_document(s): substitute/inject scoped document_id(s).
    - list_uploaded_documents: block when document scope is already resolved (doc IDs known).

    The model may narrow scope but must never broaden it.
    """
    tool_calls = getattr(message, "tool_calls", []) or []
    if not tool_calls:
        return message

    from agent_graph.tool_clinical_notes_rag import (
        _CTX_ACTIVE_DOCUMENT_IDS as _ctx_doc_ids_var,
        _CTX_ACTIVE_PATIENT_MRN as _ctx_mrn_var,
    )
    ctx_doc_ids: list = list(_ctx_doc_ids_var.get() or [])
    ctx_has_document_scope = bool(ctx_doc_ids)
    # Use context-var MRN when the graph-state MRN is absent (already resolved by
    # _resolve_active_patient_mrn, but guard here for synthesizer multi-hop calls).
    effective_mrn = active_mrn or (_ctx_mrn_var.get() or "").strip()

    updated_calls = []
    for tool_call in tool_calls:
        tool_name = tool_call.get("name", "")
        args = dict(tool_call.get("args") or {})

        # ── lookup_clinical_notes ─────────────────────────────────────────────
        if tool_name == "lookup_clinical_notes":
            has_doc_scope = (
                bool(str(args.get("document_id", "") or "").strip())
                or ctx_has_document_scope
            )
            if effective_mrn:
                args["patient_mrn"] = effective_mrn
            elif not has_doc_scope:
                # Neither MRN nor document scope — cannot safely run retrieval.
                return _AIMessage(content=_UNSCOPED_PATIENT_QUERY_MSG)
            updated_call = dict(tool_call)
            updated_call["args"] = args
            updated_calls.append(updated_call)

        # ── lookup_patient_orders ───────────────────────────────────────────
        elif tool_name == "lookup_patient_orders":
            if effective_mrn:
                args["patient_identifier"] = effective_mrn
            elif not ctx_has_document_scope and not str(args.get("patient_identifier", "") or "").strip():
                return _AIMessage(content=_UNSCOPED_PATIENT_QUERY_MSG)

            if ctx_has_document_scope:
                requested_ids = _normalize_document_ids(args.get("document_ids")) or _normalize_document_ids(args.get("document_id"))
                if requested_ids:
                    scoped_ids = [d for d in requested_ids if d in ctx_doc_ids]
                    if not scoped_ids:
                        logger.warning(
                            "_scope_lookup_tool_calls: lookup_patient_orders requested docs %s outside active scope %s; using active scope",
                            requested_ids,
                            ctx_doc_ids,
                        )
                        scoped_ids = ctx_doc_ids
                else:
                    scoped_ids = ctx_doc_ids
                args["document_ids"] = scoped_ids
                args.pop("document_id", None)

            updated_call = dict(tool_call)
            updated_call["args"] = args
            updated_calls.append(updated_call)

        # ── summarize_uploaded_document ───────────────────────────────────────
        elif tool_name == "summarize_uploaded_document":
            if ctx_has_document_scope:
                requested = (args.get("document_id") or "").strip()
                if not requested:
                    # Model didn't supply a doc_id — inject from scope.
                    if len(ctx_doc_ids) == 1:
                        args["document_id"] = ctx_doc_ids[0]
                    else:
                        updated_call = dict(tool_call)
                        updated_call["name"] = "summarize_uploaded_documents"
                        updated_call["args"] = {"document_ids": ctx_doc_ids, "patient_mrn": effective_mrn}
                        updated_calls.append(updated_call)
                        continue
                elif requested not in ctx_doc_ids:
                    # Model requested a doc outside the active scope.
                    # Substitute with the single scoped doc, or drop to first in scope.
                    logger.warning(
                        "_scope_lookup_tool_calls: summarize_uploaded_document requested "
                        "doc '%s' outside active scope %s — substituting with scoped doc",
                        requested, ctx_doc_ids,
                    )
                    args["document_id"] = ctx_doc_ids[0]
                updated_call = dict(tool_call)
                updated_call["args"] = args
                updated_calls.append(updated_call)
            else:
                updated_calls.append(tool_call)

        # ── summarize_uploaded_documents ─────────────────────────────────────
        elif tool_name == "summarize_uploaded_documents":
            if ctx_has_document_scope:
                requested_ids = _normalize_document_ids(args.get("document_ids")) or _normalize_document_ids(args.get("document_id"))
                scoped_ids = [d for d in requested_ids if d in ctx_doc_ids] if requested_ids else list(ctx_doc_ids)
                if not scoped_ids:
                    logger.warning(
                        "_scope_lookup_tool_calls: summarize_uploaded_documents requested docs %s outside active scope %s; using active scope",
                        requested_ids,
                        ctx_doc_ids,
                    )
                    scoped_ids = list(ctx_doc_ids)
                args["document_ids"] = scoped_ids
                if effective_mrn:
                    args["patient_mrn"] = effective_mrn
            updated_call = dict(tool_call)
            updated_call["args"] = args
            updated_calls.append(updated_call)

        # ── list_uploaded_documents ───────────────────────────────────────────
        # When the client has already selected specific documents (ctx_has_document_scope),
        # calling list_uploaded_documents adds nothing useful and causes the router to
        # present all documents to the user, ignoring the selection. Block it and let
        # the synthesizer answer directly using the scoped context.
        elif tool_name == "list_uploaded_documents" and ctx_has_document_scope:
            doc_summary = ", ".join(ctx_doc_ids)
            logger.info(
                "_scope_lookup_tool_calls: suppressed list_uploaded_documents — "
                "document scope already active: %s", doc_summary,
            )
            # Replace this call with a synthetic tool result injected as an AIMessage
            # note: returning early here replaces the whole message with a direct answer
            # instruction so the synthesizer proceeds without the useless listing call.
            # We do this only when list_uploaded_documents is the ONLY tool call.
            if len(tool_calls) == 1:
                return _AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "summarize_uploaded_document" if len(ctx_doc_ids) == 1 else "summarize_uploaded_documents",
                        "args": (
                            {"document_id": ctx_doc_ids[0]}
                            if len(ctx_doc_ids) == 1
                            else {"document_ids": ctx_doc_ids, "patient_mrn": effective_mrn}
                        ),
                        "id": tool_call.get("id", "scope_reroute_0"),
                    }]
                )
            # list_uploaded_documents alongside other calls — just drop it
            # (other calls will be processed normally).
        else:
            updated_calls.append(tool_call)

    return _copy_ai_message_with_tool_calls(message, updated_calls)


def build_graph(thinking_mode: bool = False):
    """
    Builds an agent decision-making graph with a split router/synthesizer
    architecture for optimal latency:

      router  (gemini-flash) — fast tool selection (~1s)
      synthesizer (gemini-flash / gemini-2.5-pro) — answer generation

    When thinking_mode=True, the synthesizer uses a deeper reasoning model
    with extended thinking budget for complex clinical queries.

    The synthesizer always produces the final answer, even for summaries,
    so it can tailor the response to the user's actual question.
    """

    # ── Build the LLMs ────────────────────────────────────────────────
    active_provider = get_active_llm_provider(getattr(TOOLS_CFG, "llm_provider", "openai"))
    router_model = os.getenv("MEDGRAPH_ROUTER_MODEL_OVERRIDE", "").strip() or getattr(
        TOOLS_CFG, "primary_agent_router_llm", TOOLS_CFG.primary_agent_llm
    )

    # Router is always fast (same regardless of thinking mode)
    router_llm = _make_llm(router_model, timeout=20, max_tokens=300)

    if thinking_mode:
        synth_model = os.getenv("MEDGRAPH_THINKING_MODEL_OVERRIDE", "").strip() or getattr(
            TOOLS_CFG, "primary_agent_thinking_llm", "gemini-2.5-pro"
        )
        # Deep thinking mode uses a higher-capability model and prompt guidance.
        thinking_max_out = int(os.getenv("MEDGRAPH_THINKING_MAX_OUTPUT_TOKENS", "4096"))
        synth_llm = _make_llm(synth_model, timeout=180, max_tokens=thinking_max_out, thinking_budget=8192)
        logger.info("Router LLM: %s (timeout=20s)", router_model)
        logger.info("Synthesizer LLM [THINKING]: %s (timeout=180s, max_tokens=%s, thinking=8192)", synth_model, thinking_max_out)
    else:
        synth_model = os.getenv("MEDGRAPH_SYNTH_MODEL_OVERRIDE", "").strip() or TOOLS_CFG.primary_agent_llm
        synth_llm = _make_llm(synth_model, timeout=60, max_tokens=4096)
        logger.info("Router LLM: %s (timeout=20s)", router_model)
        logger.info("Synthesizer LLM: %s (timeout=60s, max_tokens=4096)", synth_model)

    # ── Build the graph ───────────────────────────────────────────────
    graph_builder = StateGraph(State)
    search_tool = load_tavily_search_tool(TOOLS_CFG.tavily_search_max_results)
    tools = [search_tool,
             lookup_clinical_notes,
             lookup_patient_orders,
             cohort_patient_search,
             summarize_patient_record,
             summarize_uploaded_document,
             summarize_uploaded_documents,
             list_available_patients,
             list_uploaded_documents,
             ]

    # Router: fast model with tools bound — decides WHICH tool to call
    router_with_tools = router_llm.bind_tools(tools)

    def router(state: State):
        """Fast tool-routing + deterministic decomposition for multi-part questions."""
        last_user_text = _last_user_text(state)
        active_mrn = _resolve_active_patient_mrn(state, current_query=last_user_text)
        active_doc_ids = _resolve_active_document_ids(state)
        q_lower = (last_user_text or "").lower()

        # Deterministic fast path for already selected documents. The API/EHR
        # supplied the hard scope, so do not spend a router call rediscovering
        # documents or risk the LLM dropping document_id.
        if active_doc_ids and last_user_text.strip():
            if _is_summary_request(last_user_text):
                if len(active_doc_ids) == 1:
                    call = {
                        "name": "summarize_uploaded_document",
                        "args": {"document_id": active_doc_ids[0]},
                        "id": "scoped_summary_0",
                    }
                else:
                    call = {
                        "name": "summarize_uploaded_documents",
                        "args": {"document_ids": active_doc_ids, "patient_mrn": active_mrn},
                        "id": "scoped_summary_multi_0",
                    }
            elif _is_orders_request(last_user_text):
                call = {
                    "name": "lookup_patient_orders",
                    "args": {
                        "patient_identifier": active_mrn or last_user_text,
                        "document_ids": active_doc_ids,
                    },
                    "id": "scoped_orders_0",
                }
            else:
                call = {
                    "name": "lookup_clinical_notes",
                    "args": {
                        "query": last_user_text,
                        "patient_mrn": active_mrn,
                        "document_id": "",
                    },
                    "id": "scoped_lookup_0",
                }
            return {
                "messages": [_AIMessage(content="", tool_calls=[call])],
                "active_patient_mrn": active_mrn,
                "active_document_ids": active_doc_ids,
            }

        # Heuristic: multi-part analytical questions often contain multiple domains.
        domains = set()
        if any(k in q_lower for k in ["medication", "medications", "meds", "drug", "rx", "prescription"]):
            domains.add("meds")
        if any(k in q_lower for k in ["lab", "labs", "hba1c", "a1c", "cbc", "bmp", "cmp", "creatinine", "glucose", "test", "ordered"]):
            domains.add("labs")
        if any(k in q_lower for k in ["diagnosis", "diagnoses", "impression", "differential"]):
            domains.add("dx")
        if any(k in q_lower for k in ["procedure", "procedures", "surgery", "imaging", "ct", "mri", "x-ray", "ultrasound", "referral", "consult"]):
            domains.add("proc")

        should_decompose = (
            (" and " in q_lower or "&" in q_lower)
            and len(domains) >= 2
            and ("patients" not in q_lower)
            and ("cohort" not in q_lower)
            and ("diabetic" not in q_lower or "hba1c" in q_lower)
        )

        if should_decompose and last_user_text.strip():
            # Ask the router LLM for sub-queries, then trigger multiple lookup_clinical_notes calls.
            try:
                from langchain_core.messages import SystemMessage as _SysMsg2, HumanMessage as _HumanMsg2

                decomp_prompt = _SysMsg2(
                    content=(
                        "You are a clinical query decomposition engine. "
                        "Decompose the user's question into focused sub-queries optimized for lookup_clinical_notes. "
                        "Each sub-query must be a standalone search for one clinical domain "
                        "(e.g., medications, labs, diagnoses, procedures) and keep the original time context "
                        "(e.g., 'last encounter', 'during hospitalization'). "
                        "Return STRICT JSON only in this format: {\"sub_queries\": [\"...\", \"...\" ]}."
                    )
                )
                decomp_user = _HumanMsg2(content=last_user_text)
                raw = router_llm.invoke([decomp_prompt, decomp_user]).content

                # Safe JSON extraction (strip fences).
                t = str(raw or "").strip()
                t = t.replace("```json", "```").replace("```", "")
                if not t.startswith("{"):
                    m = _re.search(r"\{[\s\S]*\}", t)
                    # Fallback to no match.
                try:
                    obj = _json.loads(t)
                except Exception:
                    # Last chance: extract first JSON object substring.
                    import re as _re2
                    m2 = _re2.search(r"\{[\s\S]*\}", str(raw or ""))
                    if not m2:
                        raise
                    obj = _json.loads(m2.group(0))

                sub_queries = obj.get("sub_queries") or []
                sub_queries = [str(s).strip() for s in sub_queries if str(s).strip()]
                if len(sub_queries) >= 2:
                    from agent_graph.tool_clinical_notes_rag import _CTX_ACTIVE_DOCUMENT_IDS as _ctx_doc_ids_var2
                    _ctx_has_doc_scope = bool(_ctx_doc_ids_var2.get())
                    if not active_mrn and not _ctx_has_doc_scope:
                        return {
                            "messages": [_AIMessage(content=_UNSCOPED_PATIENT_QUERY_MSG)],
                            "active_patient_mrn": "",
                            "active_document_ids": active_doc_ids,
                        }
                    tool_calls = []
                    for i, sq in enumerate(sub_queries[:4]):
                        tool_calls.append(
                            {
                                "name": "lookup_clinical_notes",
                                "args": {
                                    "query": sq,
                                    "patient_mrn": active_mrn,
                                    "document_id": "",
                                },
                                "id": f"decomp_{i}",
                            }
                        )
                    return {
                        "messages": [_AIMessage(content="", tool_calls=tool_calls)],
                        "active_patient_mrn": active_mrn,
                        "active_document_ids": active_doc_ids,
                    }
            except Exception:
                # Fall back to normal router behavior.
                pass

        # Default: rely on the router's tool selection.
        routed = router_with_tools.invoke(state["messages"])
        routed = _scope_lookup_tool_calls(routed, active_mrn)
        return {
            "messages": [routed],
            "active_patient_mrn": active_mrn,
            "active_document_ids": active_doc_ids,
        }

    # Synthesizer: generates final answer from tool output.
    # Binds tools so it CAN call more if needed (multi-hop), but we prefer
    # it to synthesize from the tool results already in the message history.
    synth_with_tools = synth_llm.bind_tools(tools)

    # Prepend a lightweight synthesis instruction to guide the synthesizer.
    from langchain_core.messages import SystemMessage as _SysMsg
    _SYNTH_HINT = _SysMsg(content=(
        "Tool results are now in the conversation above. Each source block is labelled [Source N | ...]."
        " Synthesize a clear, grounded answer to the user's original question using those results."
        "\n\nCITATION RULES (mandatory for all clinical responses):"
        "\n1. Cite sources inline using [SN] format (e.g. [S1], [S2]) immediately after each clinical fact."
        "\n2. Every medication dose, lab value, vital sign, diagnosis, and procedure MUST have a [SN] citation."
        "\n3. Multiple sources may support one fact: e.g. [S1][S3]."
        "\n4. If a fact cannot be attributed to any retrieved source, do NOT include it."
        "\n5. If the sources do not contain the answer, state: \'This information was not found in the available records.\'"
        "\n\nOnly call additional tools if the existing results are truly insufficient to answer."
        " When in doubt, answer from what you have."
    ))

    _OPENAI_THINKING_HINT = _SysMsg(content=(
        "Deep thinking mode (OpenAI GPT-5.4-mini):\n"
        "- Verify every factual claim is directly supported by the provided tool sources.\n"
        "- If a key detail is missing, explicitly state it is not documented in the retrieved context.\n"
        "- If multiple retrieved sources disagree, report the discrepancy rather than averaging.\n"
        "- Then provide the final clinician-ready answer."
    ))

    def synthesizer(state: State):
        """High-quality answer synthesis from retrieved context."""
        msgs = state["messages"]
        active_mrn = (state.get("active_patient_mrn", "") or "").strip()
        active_doc_ids = _resolve_active_document_ids(state)
        user_query = _last_user_text(state)

        # OpenAI thinking mode can hit "input limit exceeded" when tool outputs
        # are large. Truncate ToolMessage contents to keep the prompt within
        # safe bounds while preserving the most informative parts.
        if thinking_mode and active_provider == "openai":
            max_tool_chars = int(os.getenv("MEDGRAPH_OPENAI_THINKING_MAX_TOOL_CHARS", "8000"))
            max_total_tool_chars = int(os.getenv("MEDGRAPH_OPENAI_THINKING_MAX_TOTAL_TOOL_CHARS", "22000"))

            total_tool_chars = 0
            truncated_msgs = []
            for m in msgs:
                if isinstance(m, _ToolMessage):
                    content = getattr(m, "content", "") or ""
                    content_str = str(content)
                    if len(content_str) > max_tool_chars:
                        half = max_tool_chars // 2
                        content_str = content_str[:half] + "\n\n...[TRUNCATED]...\n\n" + content_str[-half:]

                    # Soft cap: if we already used the budget, replace further tool
                    # content with an indicator instead of leaving it huge.
                    if total_tool_chars + len(content_str) > max_total_tool_chars:
                        allowed = max(0, max_total_tool_chars - total_tool_chars)
                        if allowed <= 200:
                            content_str = "...[TRUNCATED - tool context omitted]..."
                        else:
                            content_str = content_str[:allowed] + "\n...[TRUNCATED]..."

                    total_tool_chars += len(content_str)
                    try:
                        truncated_msgs.append(m.copy(update={"content": content_str}))
                    except Exception:
                        # Fallback: keep message object but with no tool content
                        truncated_msgs.append(m)
                else:
                    truncated_msgs.append(m)
            msgs = truncated_msgs

        # Accuracy-first pass-through for full-summary tools:
        # if a dedicated summarizer tool already produced a comprehensive,
        # citation-grounded summary, avoid re-summarizing it (which can drop details).
        for m in reversed(msgs):
            if isinstance(m, _ToolMessage):
                tool_name = getattr(m, "name", "") or ""
                if tool_name in ("summarize_patient_record", "summarize_uploaded_document", "summarize_uploaded_documents"):
                    # Phase 2.4 — wire grounding check for summarization pass-through.
                    # The summary itself may carry [SN] citations; run a basic
                    # citation-validity check so grounding_result is never empty.
                    summary_content = str(getattr(m, "content", "") or "")
                    _passthru_grounding: dict = {}
                    if summary_content:
                        try:
                            from agent_graph.grounding import verify_grounding as _vg_passthru
                            import re as _re_passthru
                            # Derive source count from max [SN] in the summary text
                            _sn_nums = {int(n) for n in _re_passthru.findall(r"\[S(\d+)", summary_content)}
                            _passthru_src_count = max(_sn_nums) if _sn_nums else 0
                            _passthru_grounding = _vg_passthru(
                                summary_content,
                                _passthru_src_count,
                                check_uncited=False,  # Summaries use [SN], not inline citations
                            )
                        except Exception:
                            pass
                    return {
                        "messages": [_AIMessage(content=summary_content)],
                        "active_patient_mrn": active_mrn,
                        "active_document_ids": active_doc_ids,
                        "grounding_result": _passthru_grounding,
                    }
                # Stop at the first tool message seen in reverse order
                break

        # Inject synthesis hint right before synthesizer's call (idempotent for multi-hop).
        if not any(
            isinstance(m, _SysMsg) and "Synthesize a clear" in str(getattr(m, "content", ""))
            for m in msgs
        ):
            msgs = [msgs[0], _SYNTH_HINT] + list(msgs[1:])

        if thinking_mode and active_provider == "openai":
            if not any(
                isinstance(m, _SysMsg) and "Deep thinking mode" in str(getattr(m, "content", ""))
                for m in msgs
            ):
                # Insert after _SYNTH_HINT to keep the system ordering stable.
                # We expect msgs[1] to be _SYNTH_HINT due to the previous block.
                insert_at = 2 if len(msgs) > 2 else len(msgs)
                msgs = list(msgs[:insert_at]) + [_OPENAI_THINKING_HINT] + list(msgs[insert_at:])

        synth_msg = synth_with_tools.invoke(msgs)
        synth_msg = _scope_lookup_tool_calls(synth_msg, active_mrn)

        # ── Phase 3: Grounding verification (non-blocking, < 10 ms) ──────────
        response_content = getattr(synth_msg, "content", "") or ""
        has_tool_calls = bool(getattr(synth_msg, "tool_calls", None))
        _grounding_result: dict = {}
        if response_content and not has_tool_calls:
            from agent_graph.grounding import count_sources_in_tool_messages, verify_grounding as _verify_grounding
            _tool_msgs = [m for m in msgs if isinstance(m, _ToolMessage)]
            _source_count = count_sources_in_tool_messages(_tool_msgs)
            # Phase 2.1 — pass raw tool content for numeric hallucination detection
            _tool_contents = [str(getattr(m, "content", "") or "") for m in _tool_msgs if getattr(m, "content", "")]
            _grounding_result = _verify_grounding(
                response_content,
                _source_count,
                tool_contents=_tool_contents if _tool_contents else None,
            )
            if not _grounding_result["is_grounded"]:
                logger.warning(
                    "[Grounding] %d issue(s) in synthesizer response (coverage=%.0f%%, hallucinated=%d): %s",
                    len(_grounding_result["issues"]),
                    _grounding_result["citation_coverage"] * 100,
                    len(_grounding_result.get("hallucinated_numerics", [])),
                    "; ".join(_grounding_result["issues"][:3]),
                )

        return {
            "messages": [synth_msg],
            "active_patient_mrn": active_mrn,
            "active_document_ids": active_doc_ids,
            "grounding_result": _grounding_result,  # Phase 11 — surfaced to api.py
        }


    graph_builder.add_node("router", router)
    graph_builder.add_node("synthesizer", synthesizer)
    tool_node = BasicToolNode(
        tools=[
            search_tool,
            lookup_clinical_notes,
            lookup_patient_orders,
            cohort_patient_search,
            summarize_patient_record,
            summarize_uploaded_document,
            summarize_uploaded_documents,
            list_available_patients,
            list_uploaded_documents,
        ])
    graph_builder.add_node("tools", tool_node)

    # Router decides: call a tool or end directly
    # When no tools needed (greetings, simple conversation), router's response
    # IS the final answer — skip synthesizer to avoid redundant LLM call
    graph_builder.add_conditional_edges(
        "router",
        route_tools,
        {"tools": "tools", "__end__": "__end__"},
    )

    # After tools: always go to synthesizer so it can answer the user's actual question
    graph_builder.add_edge("tools", "synthesizer")

    # Synthesizer may call additional tools (multi-hop) or finish
    graph_builder.add_conditional_edges(
        "synthesizer",
        route_tools,
        {"tools": "tools", "__end__": "__end__"},
    )

    graph_builder.add_edge(START, "router")
    graph = graph_builder.compile()
    plot_agent_schema(graph)
    return graph
