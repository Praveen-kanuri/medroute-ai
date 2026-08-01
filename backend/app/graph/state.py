"""Phase 2B: typed LangGraph conversation state for MedRoute AI's
end-to-end navigation graph (intake -> safety gate -> clarification ->
specialty routing -> provider search -> response composition -> TTS).

All fields are JSON-serializable (plain dicts/lists/strings, never raw
pydantic model instances or SQLAlchemy rows) so the state round-trips
cleanly through LangGraph's checkpointer for pause/resume across separate
HTTP requests. Nothing beyond this typed shape — no chain-of-thought, no
raw model prompt — is ever persisted or returned to a caller.
"""

from typing import Any, TypedDict


class VisionObservation(TypedDict):
    """Phase 2C placeholder: a single controlled, non-diagnostic visual
    observation (e.g. "visible rash on forearm"), never a fabricated or
    inferred medical finding. Not populated or read anywhere in Phase 2B —
    see docs/roadmap.md's Phase 2C entry: image upload -> a dedicated
    vision node -> a list of these -> the same graph from
    specialty_routing onward. Video is controlled frame sampling through
    that same vision node, not a separate pipeline."""

    label: str
    confidence: float | None


class ConversationState(TypedDict, total=False):
    """LangGraph state for one MedRoute AI conversation thread."""

    # The Phase 1C intake request, as a JSON-serializable dict
    # (MultimodalIntakeRequest.model_dump(mode="json")). Updated in place
    # only by the clarification node merging a confirmed answer — a
    # confirmed voice_input.transcript, once set, is never cleared or
    # rewritten by any node.
    intake_request: dict[str, Any]

    # Whether the caller wants a synthesized-speech response this turn.
    generate_speech: bool

    # Phase 1C intake evaluation (MultimodalIntakeResponse.model_dump()).
    intake_response: dict[str, Any] | None
    missing_fields: list[str]

    # Set only by safety_gate; never derived from transcript/text content.
    is_emergency: bool

    # Set only by clarification, to choose its own outgoing conditional
    # edge after an interrupt/resume round-trip.
    just_resumed_clarification: bool

    # Phase 1D specialty routing (SpecialtyRoutingOut-shaped dict) and
    # Phase 1B provider search (ProviderSearchResponse-shaped dict).
    routing: dict[str, Any] | None
    provider_search: dict[str, Any] | None
    media_note: str | None

    # Phase 2B: the final safe, non-diagnostic response text and its
    # optional synthesized speech (AudioOutput-shaped dict).
    response_text: str | None
    audio: dict[str, Any] | None

    # Phase 2C placeholder only — always empty in Phase 2B. See
    # VisionObservation above. Never populated with fabricated results.
    vision_observations: list[VisionObservation]
