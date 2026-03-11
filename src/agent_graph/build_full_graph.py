import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage as _ToolMessage, AIMessage as _AIMessage
from agent_graph.tool_clinical_notes_rag import lookup_clinical_notes, lookup_patient_orders, summarize_patient_record, list_available_patients, list_uploaded_documents
from agent_graph.tool_tavily_search import load_tavily_search_tool
from agent_graph.load_tools_config import LoadToolsConfig, get_google_api_key
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
    provider = getattr(TOOLS_CFG, "llm_provider", "openai")

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
    router_model = getattr(TOOLS_CFG, "primary_agent_router_llm", TOOLS_CFG.primary_agent_llm)

    # Router is always fast (same regardless of thinking mode)
    router_llm = _make_llm(router_model, timeout=15, max_tokens=300)

    if thinking_mode:
        synth_model = getattr(TOOLS_CFG, "primary_agent_thinking_llm", "gemini-2.5-pro")
        synth_llm = _make_llm(synth_model, timeout=120, max_tokens=16384, thinking_budget=8192)
        logger.info("Router LLM: %s (timeout=15s)", router_model)
        logger.info("Synthesizer LLM [THINKING]: %s (timeout=120s, max_tokens=16384, thinking=8192)", synth_model)
    else:
        synth_model = TOOLS_CFG.primary_agent_llm
        synth_llm = _make_llm(synth_model, timeout=30, max_tokens=4096)
        logger.info("Router LLM: %s (timeout=15s)", router_model)
        logger.info("Synthesizer LLM: %s (timeout=30s, max_tokens=4096)", synth_model)

    # ── Build the graph ───────────────────────────────────────────────
    graph_builder = StateGraph(State)
    search_tool = load_tavily_search_tool(TOOLS_CFG.tavily_search_max_results)
    tools = [search_tool,
             lookup_clinical_notes,
             lookup_patient_orders,
             summarize_patient_record,
             list_available_patients,
             list_uploaded_documents,
             ]

    # Router: fast model with tools bound — decides WHICH tool to call
    router_with_tools = router_llm.bind_tools(tools)

    def router(state: State):
        """Fast tool-routing: picks which tool to call."""
        return {"messages": [router_with_tools.invoke(state["messages"])]}

    # Synthesizer: generates final answer from tool output
    synth_with_tools = synth_llm.bind_tools(tools)

    def synthesizer(state: State):
        """High-quality answer synthesis from retrieved context."""
        return {"messages": [synth_with_tools.invoke(state["messages"])]}

    graph_builder.add_node("router", router)
    graph_builder.add_node("synthesizer", synthesizer)
    tool_node = BasicToolNode(
        tools=[
            search_tool,
            lookup_clinical_notes,
            lookup_patient_orders,
            summarize_patient_record,
            list_available_patients,
            list_uploaded_documents,
        ])
    graph_builder.add_node("tools", tool_node)

    # Router decides: call a tool or pass to synthesizer
    # Router NEVER goes to __end__ — synthesizer always generates the final answer
    graph_builder.add_conditional_edges(
        "router",
        route_tools,
        {"tools": "tools", "__end__": "synthesizer"},
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
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    plot_agent_schema(graph)
    return graph