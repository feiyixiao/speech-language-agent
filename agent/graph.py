"""Agent graph.

    router ──practice──▶ grammar ┐
       │                 vocabulary ├─▶ END   (run in parallel)
       │                 pronunciation ┘      (only if audio)
       └──question───▶ rag ─▶ END

The prototype ran grammar -> vocabulary -> pronunciation one after another and
never used the router in the UI; now the router decides and the practice
branch fans out concurrently.
"""
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

from agent.nodes import (grammar_node, pronunciation_node, rag_node, router_node,
                         vocabulary_node)


class AgentState(TypedDict, total=False):
    transcript: str
    target_language: str
    audio_path: str
    error_history: list[str]
    intent: str
    grammar: dict
    repeated_error_types: list[str]
    vocabulary: dict
    pronunciation: dict
    answer: dict
    retrieved_sections: list[str]
    degraded: Annotated[list[str], operator.add]  # parallel nodes may all append


def route(state: AgentState) -> list[str]:
    if state["intent"] == "question":
        return ["rag"]
    branches = ["grammar", "vocabulary"]
    if state.get("audio_path"):
        branches.append("pronunciation")
    return branches


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("router", router_node)
    g.add_node("grammar", grammar_node)
    g.add_node("vocabulary", vocabulary_node)
    g.add_node("pronunciation", pronunciation_node)
    g.add_node("rag", rag_node)
    g.set_entry_point("router")
    g.add_conditional_edges("router", route, ["grammar", "vocabulary", "pronunciation", "rag"])
    for n in ("grammar", "vocabulary", "pronunciation", "rag"):
        g.add_edge(n, END)
    return g.compile()


graph = build_graph()
