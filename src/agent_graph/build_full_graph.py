from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage as _ToolMessage, AIMessage as _AIMessage
from agent_graph.tool_clinical_notes_rag import lookup_clinical_notes, summarize_patient_record, list_available_patients
from agent_graph.tool_tavily_search import load_tavily_search_tool
#from agent_graph.tool_hospital_sqlagent import query_hospital_database
from agent_graph.load_tools_config import LoadToolsConfig
from agent_graph.agent_backend import State, BasicToolNode, route_tools, plot_agent_schema
import os
import json as _json
from dotenv import load_dotenv

load_dotenv()

TOOLS_CFG = LoadToolsConfig()


def build_graph():
    """
    Builds an agent decision-making graph by combining an LLM with various tools
    and defining the flow of interactions between them.

    Includes a summary bypass: when summarize_patient_record is the only tool
    called, the tool output is converted directly to an AIMessage (skipping a
    costly second LLM round-trip that would just copy the already-formatted summary).
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
        return "chatbot"

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

    # ── Build the graph ───────────────────────────────────────────────
    primary_llm = ChatOpenAI(
        model=TOOLS_CFG.primary_agent_llm,
        temperature=TOOLS_CFG.primary_agent_llm_temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    graph_builder = StateGraph(State)
    # Load tools with their proper configs
    search_tool = load_tavily_search_tool(TOOLS_CFG.tavily_search_max_results)
    tools = [search_tool,
             lookup_clinical_notes,
             summarize_patient_record,
             list_available_patients,
             #query_hospital_database,
             ]
    # Tell the LLM which tools it can call by binding tools
    primary_llm_with_tools = primary_llm.bind_tools(tools)

    def chatbot(state: State):
        """Executes the primary language model with tools bound and returns the generated message."""
        return {"messages": [primary_llm_with_tools.invoke(state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)
    tool_node = BasicToolNode(
        tools=[
            search_tool,
            lookup_clinical_notes,
            summarize_patient_record,
            list_available_patients,
            #query_hospital_database,
        ])
    graph_builder.add_node("tools", tool_node)
    # The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "__end__" if
    # it is fine directly responding. This conditional routing defines the main agent loop.
    graph_builder.add_conditional_edges(
        "chatbot",
        route_tools,
        {"tools": "tools", "__end__": "__end__"},
    )

    # Summary bypass: summarize → summary_passthrough → __end__ (skip agent rewrite)
    # Everything else: tools → chatbot (normal agent loop)
    graph_builder.add_node("summary_passthrough", summary_passthrough)
    graph_builder.add_conditional_edges(
        "tools",
        route_after_tools,
        {"chatbot": "chatbot", "summary_passthrough": "summary_passthrough"},
    )
    graph_builder.add_edge("summary_passthrough", "__end__")
    graph_builder.add_edge(START, "chatbot")
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    plot_agent_schema(graph)
    return graph