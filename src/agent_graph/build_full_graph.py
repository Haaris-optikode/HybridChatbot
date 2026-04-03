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
)
from agent_graph.tool_tavily_search import load_tavily_search_tool
from agent_graph.load_tools_config import LoadToolsConfig, get_google_api_key, get_active_llm_provider
from agent_graph.agent_backend import State, BasicToolNode, route_tools, plot_agent_schema
import os
import json as _json
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

TOOLS_CFG = LoadToolsConfig()


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
        # Detect last user query text.
        last_user_text = ""
        for m in reversed(state.get("messages", [])):
            # Avoid depending on exact message classes; content is enough.
            if getattr(m, "type", "") == "human" or m.__class__.__name__ == "HumanMessage":
                last_user_text = getattr(m, "content", "") or ""
                break
        q_lower = (last_user_text or "").lower()

        # Heuristic: multi-part analytical questions often contain multiple domains.
        domains = set()
        if any(k in q_lower for k in ["medication", "medications", "meds", "drug", "rx", "prescription"]):
            domains.add("meds")
        if any(k in q_lower for k in ["lab", "labs", "hba1c", "a1c", "cbc", "bmp", "cmp", "creatinine", "glucose", "test", "ordered"]):
            domains.add("labs")
        if any(k in q_lower for k in ["diagnosis", "diagnoses", "impression", "differential", "icd"]):
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
                    m = _json.search(r"\{[\s\S]*\}", t) if hasattr(_json, "search") else None
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
                    tool_calls = []
                    for i, sq in enumerate(sub_queries[:4]):
                        tool_calls.append(
                            {
                                "name": "lookup_clinical_notes",
                                "args": {"query": sq, "document_id": ""},
                                "id": f"decomp_{i}",
                            }
                        )
                    return {"messages": [_AIMessage(content="", tool_calls=tool_calls)]}
            except Exception:
                # Fall back to normal router behavior.
                pass

        # Default: rely on the router's tool selection.
        return {"messages": [router_with_tools.invoke(state["messages"])]}

    # Synthesizer: generates final answer from tool output.
    # Binds tools so it CAN call more if needed (multi-hop), but we prefer
    # it to synthesize from the tool results already in the message history.
    synth_with_tools = synth_llm.bind_tools(tools)

    # Prepend a lightweight synthesis instruction to guide the synthesizer.
    from langchain_core.messages import SystemMessage as _SysMsg
    _SYNTH_HINT = _SysMsg(content=(
        "Tool results are now in the conversation above. "
        "Synthesize a clear, grounded answer to the user's original question "
        "using those results. Only call additional tools if the existing results "
        "are truly insufficient to answer. When in doubt, answer from what you have."
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
                    return {"messages": [_AIMessage(content=str(getattr(m, "content", "") or ""))]}
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

        return {"messages": [synth_with_tools.invoke(msgs)]}


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