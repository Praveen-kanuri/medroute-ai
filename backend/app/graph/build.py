"""Phase 2B-2D/3A/3B: compiles the MedRoute AI multi-agent conversation graph.

    supervisor_router -> [conversation_agent | vision_agent | concern_relevance_agent]
    conversation_agent -> response_agent
    vision_agent -> medical_intake_agent
    concern_relevance_agent -> [conversation_agent (general chat) | medical_intake_agent]
    medical_intake_agent -> safety_gate
    safety_gate -> [clarification | response_agent]
    clarification -> [medical_intake_agent (resumed) | clinical_intake_agent]
    clinical_intake_agent -> [clinical_intake_agent (resumed, next question) |
                              clinical_red_flag_gate | specialty_routing_agent]
    clinical_red_flag_gate -> [response_agent (emergency) | clinical_intake_agent]
    specialty_routing_agent -> [provider_search_agent | response_agent]
    provider_search_agent -> response_agent -> text_to_speech -> END

supervisor_router is the graph's single routing authority (see
app/graph/nodes.py's supervisor_router_node): it inspects the current
input, its modality, pending media, and any declared emergency, then
makes one explicit structured decision recorded in state
(detected_intent/next_agent) before any other agent runs. A greeting is
routed straight to conversation_agent and structurally cannot reach
medical_intake_agent, specialty_routing_agent, or provider_search_agent
that turn.

clinical_intake_agent (Phase 3A) is a complete no-op for any concern that
matches no known complaint protocol (see
app.services.clinical_intake_service.match_protocol) -- such a turn falls
straight through to specialty_routing_agent exactly as it did before this
phase, preserving every existing (non-protocol) conversation unchanged.
For a matched protocol, it asks its questions one at a time via the same
interrupt/resume mechanism clarification_node already uses, self-looping
until its questions are answered; clinical_red_flag_gate is a dedicated,
deterministic safety step (never a model call) that always runs
immediately after the protocol's red-flag question is answered and always
before specialty/provider routing -- see app/safety/red_flag_rules.py.

concern_relevance_agent (Phase 3B, optional/opt-in via
Settings.conversation_mode="groq") sits between supervisor_router and
medical_intake_agent for the text/voice (non-media) path only -- vision_
agent's own handoff to medical_intake_agent is unaffected. It is a
complete zero-cost no-op (falls straight through to medical_intake_agent)
whenever conversation_mode stays "deterministic" (the default) or Groq
isn't configured, and it is never even consulted for a user-declared
emergency turn -- see app.graph.nodes.concern_relevance_agent_node's
docstring for the safety guarantee this relies on.

Reuses existing services as controlled tools inside each node (see
app/graph/nodes.py) — this module only wires up the graph's shape. Each
agent has a bounded responsibility and restricted tools; text_to_speech is
deliberately not treated as an agent (see nodes.py's module docstring).
"""

from functools import lru_cache

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.nodes import (
    clarification_node,
    clinical_intake_agent_node,
    clinical_red_flag_gate_node,
    concern_relevance_agent_node,
    conversation_agent_node,
    medical_intake_agent_node,
    provider_search_agent_node,
    response_agent_node,
    route_after_clarification,
    route_after_clinical_intake,
    route_after_clinical_red_flag_gate,
    route_after_concern_relevance,
    route_after_safety_gate,
    route_after_specialty_routing,
    route_after_supervisor,
    safety_gate_node,
    specialty_routing_agent_node,
    supervisor_router_node,
    text_to_speech_node,
    vision_agent_node,
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

    graph.add_node("supervisor_router", supervisor_router_node)
    graph.add_node("conversation_agent", conversation_agent_node)
    graph.add_node("vision_agent", vision_agent_node)
    graph.add_node("concern_relevance_agent", concern_relevance_agent_node)
    graph.add_node("medical_intake_agent", medical_intake_agent_node)
    graph.add_node("safety_gate", safety_gate_node)
    graph.add_node("clarification", clarification_node)
    graph.add_node("clinical_intake_agent", clinical_intake_agent_node)
    graph.add_node("clinical_red_flag_gate", clinical_red_flag_gate_node)
    graph.add_node("specialty_routing_agent", specialty_routing_agent_node)
    graph.add_node("provider_search_agent", provider_search_agent_node)
    graph.add_node("response_agent", response_agent_node)
    graph.add_node("text_to_speech", text_to_speech_node)

    graph.add_edge(START, "supervisor_router")
    graph.add_conditional_edges(
        "supervisor_router",
        route_after_supervisor,
        {
            "conversation_agent": "conversation_agent",
            "vision_agent": "vision_agent",
            "medical_intake_agent": "concern_relevance_agent",
        },
    )
    graph.add_edge("conversation_agent", "response_agent")
    graph.add_edge("vision_agent", "medical_intake_agent")
    graph.add_conditional_edges(
        "concern_relevance_agent",
        route_after_concern_relevance,
        {
            "conversation_agent": "conversation_agent",
            "medical_intake_agent": "medical_intake_agent",
        },
    )
    graph.add_edge("medical_intake_agent", "safety_gate")
    graph.add_conditional_edges(
        "safety_gate",
        route_after_safety_gate,
        {"clarification": "clarification", "response_composition": "response_agent"},
    )
    graph.add_conditional_edges(
        "clarification",
        route_after_clarification,
        {
            "normalize_intake": "medical_intake_agent",
            "specialty_routing": "clinical_intake_agent",
        },
    )
    graph.add_conditional_edges(
        "clinical_intake_agent",
        route_after_clinical_intake,
        {
            "clinical_intake_agent": "clinical_intake_agent",
            "red_flag_gate": "clinical_red_flag_gate",
            "specialty_routing": "specialty_routing_agent",
        },
    )
    graph.add_conditional_edges(
        "clinical_red_flag_gate",
        route_after_clinical_red_flag_gate,
        {
            "response_composition": "response_agent",
            "clinical_intake_agent": "clinical_intake_agent",
        },
    )
    graph.add_conditional_edges(
        "specialty_routing_agent",
        route_after_specialty_routing,
        {"provider_search": "provider_search_agent", "response_composition": "response_agent"},
    )
    graph.add_edge("provider_search_agent", "response_agent")
    graph.add_edge("response_agent", "text_to_speech")
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
