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
    )
    if _is_reasoning_model(model_name):
        kwargs["max_completion_tokens"] = 16384
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
        return existing

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
    return ""


def _copy_ai_message_with_tool_calls(message: _AIMessage, tool_calls: list[dict]) -> _AIMessage:
    """Clone an AIMessage while replacing tool calls."""
    try:
        if hasattr(message, "model_copy"):
            return message.model_copy(update={"tool_calls": tool_calls})
        return message.copy(update={"tool_calls": tool_calls})
    except Exception:
        return _AIMessage(content=str(getattr(message, "content", "") or ""), tool_calls=tool_calls)


def _scope_lookup_tool_calls(message: _AIMessage, active_mrn: str) -> _AIMessage:
    """Inject MRN scoping into lookup_clinical_notes calls or block unscoped calls."""
    tool_calls = getattr(message, "tool_calls", []) or []
    if not tool_calls:
        return message

    updated_calls = []
    for tool_call in tool_calls:
        if tool_call.get("name") != "lookup_clinical_notes":
            updated_calls.append(tool_call)
            continue

        args = dict(tool_call.get("args") or {})
        has_document_scope = bool(str(args.get("document_id", "") or "").strip())
        if active_mrn:
            args["patient_mrn"] = active_mrn
        elif not has_document_scope:
            return _AIMessage(content=_UNSCOPED_PATIENT_QUERY_MSG)

        updated_call = dict(tool_call)
        updated_call["args"] = args
        updated_calls.append(updated_call)

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
        # GPT-4.1 is a non-reasoning model (no "reasoning effort" knob). In
        # practice, "deep thinking mode" here is prompt + higher accuracy
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
             list_available_patients,
             list_uploaded_documents,
             ]

    # Router: fast model with tools bound — decides WHICH tool to call
    router_with_tools = router_llm.bind_tools(tools)

    def router(state: State):
        """Fast tool-routing + deterministic decomposition for multi-part questions."""
        last_user_text = _last_user_text(state)
        active_mrn = _resolve_active_patient_mrn(state, current_query=last_user_text)
        q_lower = (last_user_text or "").lower()

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
                    if not active_mrn:
                        return {
                            "messages": [_AIMessage(content=_UNSCOPED_PATIENT_QUERY_MSG)],
                            "active_patient_mrn": "",
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
        "Deep thinking mode (OpenAI GPT-4.1):\n"
        "- Verify every factual claim is directly supported by the provided tool sources.\n"
        "- If a key detail is missing, explicitly state it is not documented in the retrieved context.\n"
        "- If multiple retrieved sources disagree, report the discrepancy rather than averaging.\n"
        "- Then provide the final clinician-ready answer."
    ))

    def synthesizer(state: State):
        """High-quality answer synthesis from retrieved context."""
        msgs = state["messages"]
        active_mrn = (state.get("active_patient_mrn", "") or "").strip()
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
                if tool_name in ("summarize_patient_record", "summarize_uploaded_document"):
                    return {
                        "messages": [_AIMessage(content=str(getattr(m, "content", "") or ""))],
                        "active_patient_mrn": active_mrn,
                        "grounding_result": {},  # pass-through: grounding N/A for dedicated summarizers
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
            _source_count = count_sources_in_tool_messages(
                [m for m in msgs if isinstance(m, _ToolMessage)]
            )
            _grounding_result = _verify_grounding(response_content, _source_count)
            if not _grounding_result["is_grounded"]:
                logger.warning(
                    "[Grounding] %d issue(s) in synthesizer response (coverage=%.0f%%): %s",
                    len(_grounding_result["issues"]),
                    _grounding_result["citation_coverage"] * 100,
                    "; ".join(_grounding_result["issues"][:3]),
                )

        return {
            "messages": [synth_msg],
            "active_patient_mrn": active_mrn,
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
