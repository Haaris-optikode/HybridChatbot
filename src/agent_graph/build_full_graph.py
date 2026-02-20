import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage as _ToolMessage, AIMessage as _AIMessage
from agent_graph.tool_clinical_notes_rag import lookup_clinical_notes, summarize_patient_record, list_available_patients
from agent_graph.tool_tavily_search import load_tavily_search_tool
from agent_graph.load_tools_config import LoadToolsConfig
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


def _make_llm(model_name: str, timeout: int = 60, max_tokens: int = None) -> ChatOpenAI:
    """Create a ChatOpenAI instance with appropriate params for reasoning vs non-reasoning models."""
    kwargs = dict(
        model=model_name,
        api_key=os.getenv("OPENAI_API_KEY"),
        request_timeout=timeout,
        max_retries=2,
    )
    if _is_reasoning_model(model_name):
        # Reasoning models: max_completion_tokens instead of temperature + max_tokens
        kwargs["max_completion_tokens"] = 16384
    else:
        kwargs["temperature"] = TOOLS_CFG.primary_agent_llm_temperature
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)


def build_graph():
    """
    Builds an agent decision-making graph with a split router/synthesizer
    architecture for optimal latency:

      router  (gpt-4.1-mini) — fast tool selection (~1s)
      synthesizer (gpt-5-mini) — high-quality answer generation

    Includes a summary bypass: when summarize_patient_record is the only tool
    called, the tool output is converted directly to an AIMessage (skipping
    the synthesizer round-trip entirely).
    """

    # ── Summary bypass helpers ────────────────────────────────────────
    def route_after_tools(state: State):
        """Route to summary_passthrough when summarize is the sole tool call."""
        tool_msgs = []
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, _ToolMessage):
                tool_msgs.append(msg)
            else:
                break
        if len(tool_msgs) == 1 and tool_msgs[0].name == "summarize_patient_record":
            return "summary_passthrough"
        return "synthesizer"

    def summary_passthrough(state: State):
        """Convert the summarize ToolMessage to an AIMessage — zero LLM cost."""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, _ToolMessage) and msg.name == "summarize_patient_record":
                content = msg.content
                try:
                    content = _json.loads(content)
                except (ValueError, TypeError):
                    pass
                return {"messages": [_AIMessage(content=content)]}
        return {"messages": [_AIMessage(content="Summary generation completed.")]}

    # ── Build the two LLMs ────────────────────────────────────────────
    router_model = getattr(TOOLS_CFG, "primary_agent_router_llm", TOOLS_CFG.primary_agent_llm)
    synth_model = TOOLS_CFG.primary_agent_llm

    router_llm = _make_llm(router_model, timeout=15, max_tokens=300)
    synth_llm = _make_llm(synth_model, timeout=30, max_tokens=2048)

    logger.info("Router LLM: %s (timeout=15s)", router_model)
    logger.info("Synthesizer LLM: %s (timeout=30s, max_tokens=2048)", synth_model)

    # ── Build the graph ───────────────────────────────────────────────
    graph_builder = StateGraph(State)
    search_tool = load_tavily_search_tool(TOOLS_CFG.tavily_search_max_results)
    tools = [search_tool,
             lookup_clinical_notes,
             summarize_patient_record,
             list_available_patients,
             ]

    # Router: fast model with tools bound — decides WHICH tool to call
    router_with_tools = router_llm.bind_tools(tools)

    def router(state: State):
        """Fast tool-routing: picks which tool to call (~1s with gpt-4.1-mini)."""
        return {"messages": [router_with_tools.invoke(state["messages"])]}

    # Synthesizer: reasoning model — generates final answer from tool output
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
            summarize_patient_record,
            list_available_patients,
        ])
    graph_builder.add_node("tools", tool_node)

    # Router decides: call a tool or respond directly
    graph_builder.add_conditional_edges(
        "router",
        route_tools,
        {"tools": "tools", "__end__": "__end__"},
    )

    # After tools: summary bypass skips synthesizer; everything else → synthesizer
    graph_builder.add_node("summary_passthrough", summary_passthrough)
    graph_builder.add_conditional_edges(
        "tools",
        route_after_tools,
        {"synthesizer": "synthesizer", "summary_passthrough": "summary_passthrough"},
    )

    # Synthesizer may call additional tools (multi-hop) or finish
    graph_builder.add_conditional_edges(
        "synthesizer",
        route_tools,
        {"tools": "tools", "__end__": "__end__"},
    )

    graph_builder.add_edge("summary_passthrough", "__end__")
    graph_builder.add_edge(START, "router")
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    plot_agent_schema(graph)
    return graph