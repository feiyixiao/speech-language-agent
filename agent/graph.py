from typing import TypedDict
from langgraph.graph import StateGraph, END
from agent.nodes import router_node, grammar_node, vocabulary_node, pronunciation_node, rag_node

class AgentState(TypedDict):
    transcript: str
    target_language: str
    task: str
    result: str
    error_history: list[str]
    audio_path: str

def route_decision(state: AgentState) -> str:
    return state["task"]

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("grammar", grammar_node)
    graph.add_node("vocabulary", vocabulary_node)
    graph.add_node("pronunciation", pronunciation_node)
    graph.add_node("knowledge_query", rag_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_decision, {
        "grammar": "grammar",
        "vocabulary": "vocabulary",
        "pronunciation": "pronunciation",
        "knowledge_query": "knowledge_query",
    })
    graph.add_edge("grammar", END)
    graph.add_edge("vocabulary", END)
    graph.add_edge("pronunciation", END)
    graph.add_edge("knowledge_query", END)

    return graph.compile()
