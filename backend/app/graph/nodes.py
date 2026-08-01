"""Phase 2B: LangGraph node implementations for the MedRoute AI
conversation graph.

Every node is a thin wrapper around an already-existing, already-tested
service function — this module contains no business logic of its own,
only orchestration glue and the JSON-dict <-> pydantic-model boundary
LangGraph's checkpointer needs. Dependencies that don't belong in
checkpointed state (Settings, the DB session, and — only in tests — a
fake TTS provider) are threaded through via each node's `config`
parameter (`config["configurable"]`), never a module-level global.
"""

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import Settings
from app.graph.state import ConversationState
from app.providers.text_to_speech.base import TextToSpeechProvider
from app.schemas.multimodal_intake import IntakeStatus, MultimodalIntakeRequest
from app.services.multimodal_intake_service import evaluate_intake
from app.services.provider_search_service import (
    DEFAULT_LIMIT,
    build_provider_search_response,
    search_providers_page,
)
from app.services.response_composition_service import compose_response
from app.services.specialty_routing_service import route_to_specialty
from app.services.text_to_speech_service import build_text_to_speech_provider, synthesize_speech

MEDIA_UNAVAILABLE_NOTE = (
    "Image/video analysis is not available in this demo. Only text and voice-transcript "
    "content were used for routing."
)


def _configurable(config: RunnableConfig) -> dict[str, Any]:
    return config.get("configurable") or {}


def _merge_clarification_answer(intake_request: dict[str, Any], answer: object) -> dict[str, Any]:
    """Merge a typed clarification answer into the intake request dict.

    Only the two structured fields the clarification node ever asks about
    are considered; anything else in `answer` is ignored. Every other key
    already in intake_request — including a confirmed voice_input — is
    carried over unchanged."""
    merged = dict(intake_request)
    if isinstance(answer, dict):
        duration = answer.get("duration")
        if duration:
            merged["duration"] = duration
        main_concern = answer.get("main_concern")
        if main_concern:
            merged["main_concern"] = main_concern
    return merged


def normalize_intake_node(state: ConversationState) -> dict[str, Any]:
    """Phase 1C intake normalization, reused unchanged."""
    request = MultimodalIntakeRequest.model_validate(state["intake_request"])
    response = evaluate_intake(request)
    media_note = MEDIA_UNAVAILABLE_NOTE if response.multimodal_status.vision_received else None
    return {
        "intake_response": response.model_dump(mode="json"),
        "missing_fields": response.missing_fields,
        "media_note": media_note,
    }


def safety_gate_node(state: ConversationState) -> dict[str, Any]:
    """Emergency precedence check. Reads only the already-evaluated intake
    status (itself derived only from user-declared emergency_concern /
    emergency_signals — see evaluate_intake) — never inspects transcript
    or symptom text directly, and never infers an emergency from it."""
    intake_response = state.get("intake_response") or {}
    return {"is_emergency": intake_response.get("status") == IntakeStatus.EMERGENCY.value}


def route_after_safety_gate(state: ConversationState) -> str:
    return "response_composition" if state.get("is_emergency") else "clarification"


def clarification_node(state: ConversationState) -> dict[str, Any]:
    """Pauses the graph (via LangGraph's interrupt()) whenever a required
    field is missing, reporting the same typed missing_fields and
    human-readable clarification_questions the API layer already returns.
    Resuming with Command(resume=<ClarificationAnswer-shaped dict>) merges
    the answer into intake_request and loops back to normalize_intake so
    missing_fields is recomputed with the new information — this may
    interrupt again if something else is still missing."""
    missing = state.get("missing_fields") or []
    if not missing:
        return {"just_resumed_clarification": False}

    intake_response = state.get("intake_response") or {}
    answer = interrupt(
        {
            "missing_fields": missing,
            "clarification_questions": intake_response.get("clarification_questions", []),
        }
    )
    merged_request = _merge_clarification_answer(state["intake_request"], answer)
    return {"intake_request": merged_request, "just_resumed_clarification": True}


def route_after_clarification(state: ConversationState) -> str:
    if state.get("just_resumed_clarification"):
        return "normalize_intake"
    return "specialty_routing"


async def specialty_routing_node(
    state: ConversationState, config: RunnableConfig
) -> dict[str, Any]:
    """Phase 1D deterministic specialty routing, reused unchanged as a
    controlled tool — it only ever selects a specialty already present in
    the curated catalog."""
    settings: Settings = _configurable(config)["settings"]
    request = state["intake_request"]
    voice_input = request.get("voice_input") or {}

    result = await route_to_specialty(
        preferred_specialty=request.get("preferred_specialty"),
        symptoms=request.get("symptoms") or [],
        main_concern=request.get("main_concern"),
        voice_transcript=voice_input.get("transcript"),
        settings=settings,
    )
    return {
        "routing": {
            "specialty_slug": result.specialty_slug,
            "specialty_display_name": result.specialty_display_name,
            "method": result.method.value,
            "note": result.note,
        }
    }


def route_after_specialty_routing(state: ConversationState) -> str:
    routing = state.get("routing") or {}
    return "provider_search" if routing.get("specialty_slug") else "response_composition"


async def provider_search_node(state: ConversationState, config: RunnableConfig) -> dict[str, Any]:
    """Phase 1B deterministic provider search, reused unchanged as a
    controlled tool — no LLM call, no fabricated results."""
    session: AsyncSession = _configurable(config)["session"]
    routing = state["routing"] or {}
    specialty_slug = routing["specialty_slug"]

    page = await search_providers_page(session, specialty_slug=specialty_slug, limit=DEFAULT_LIMIT)
    response = build_provider_search_response(page, specialty_slug=specialty_slug)
    return {"provider_search": response.model_dump(mode="json")}


async def response_composition_node(
    state: ConversationState, config: RunnableConfig
) -> dict[str, Any]:
    """Concise, non-diagnostic response text for a completed turn
    (emergency, unmatched, or routed with/without providers). Never
    called for a paused clarification turn — that question is returned
    directly by the orchestration service before this node would run."""
    settings: Settings = _configurable(config)["settings"]
    text = await compose_response(
        is_emergency=state.get("is_emergency", False),
        missing_fields=state.get("missing_fields") or [],
        clarification_questions=(state.get("intake_response") or {}).get(
            "clarification_questions", []
        ),
        routing=state.get("routing"),
        provider_search=state.get("provider_search"),
        settings=settings,
    )
    return {"response_text": text}


async def text_to_speech_node(state: ConversationState, config: RunnableConfig) -> dict[str, Any]:
    """Best-effort speech synthesis over the final response text. Always
    returns a state update (never raises) — a failed or skipped synthesis
    still lets the text response reach the caller."""
    if not state.get("generate_speech"):
        return {"audio": None}

    configurable = _configurable(config)
    settings: Settings = configurable["settings"]
    provider: TextToSpeechProvider | None = configurable.get("tts_provider")
    if provider is None:
        provider = build_text_to_speech_provider(settings)

    audio = await synthesize_speech(
        text=state.get("response_text"), settings=settings, provider=provider
    )
    return {"audio": audio.model_dump(mode="json") if audio else None}
