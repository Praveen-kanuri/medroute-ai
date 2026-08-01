"""Phase 2B: compiles the MedRoute AI conversation graph.

    normalize_intake -> safety_gate -> [clarification | response_composition]
    clarification -> [normalize_intake (resumed) | specialty_routing]
    specialty_routing -> [provider_search | response_composition]
    provider_search -> response_composition -> text_to_speech -> END

Reuses existing services as controlled tools inside each node (see
app/graph/nodes.py) — this module only wires up the graph's shape.
"""

from functools import lru_cache

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.nodes import (
    clarification_node,
    normalize_intake_node,
    provider_search_node,
    response_composition_node,
    route_after_clarification,
    route_after_safety_gate,
    route_after_specialty_routing,
    safety_gate_node,
    specialty_routing_node,
    text_to_speech_node,
)
from app.graph.state import ConversationState

ConversationGraph = CompiledStateGraph[
    ConversationState, None, ConversationState, ConversationState
]


def build_conversation_graph() -> ConversationGraph:
    """Build and compile a fresh graph with its own InMemorySaver.

    Exposed separately from get_conversation_graph() so tests can compile
    an isolated graph with a fresh checkpointer instead of sharing the
    process-wide singleton (whose state must persist across requests).
    """
    graph: StateGraph[ConversationState] = StateGraph(ConversationState)

    graph.add_node("normalize_intake", normalize_intake_node)
    graph.add_node("safety_gate", safety_gate_node)
    graph.add_node("clarification", clarification_node)
    graph.add_node("specialty_routing", specialty_routing_node)
    graph.add_node("provider_search", provider_search_node)
    graph.add_node("response_composition", response_composition_node)
    graph.add_node("text_to_speech", text_to_speech_node)

    graph.add_edge(START, "normalize_intake")
    graph.add_edge("normalize_intake", "safety_gate")
    graph.add_conditional_edges(
        "safety_gate",
        route_after_safety_gate,
        {"clarification": "clarification", "response_composition": "response_composition"},
    )
    graph.add_conditional_edges(
        "clarification",
        route_after_clarification,
        {"normalize_intake": "normalize_intake", "specialty_routing": "specialty_routing"},
    )
    graph.add_conditional_edges(
        "specialty_routing",
        route_after_specialty_routing,
        {"provider_search": "provider_search", "response_composition": "response_composition"},
    )
    graph.add_edge("provider_search", "response_composition")
    graph.add_edge("response_composition", "text_to_speech")
    graph.add_edge("text_to_speech", END)

    return graph.compile(checkpointer=InMemorySaver())


@lru_cache
def get_conversation_graph() -> ConversationGraph:
    """Process-wide singleton compiled graph.

    Cached so its InMemorySaver checkpointer persists across separate HTTP
    requests within this process's lifetime — the demo-appropriate scope
    for Phase 2B (no cross-process or cross-restart persistence).
    """
    return build_conversation_graph()
