"""Phase 2B-2D: orchestrates one conversation turn against the compiled
multi-agent LangGraph conversation graph (app/graph/build.py). This is the
only module the API route depends on for graph execution, keeping the
route a thin HTTP adapter with no orchestration logic of its own.

Intent classification (greeting vs. medical concern vs. unsupported) and
all routing decisions happen *inside* the graph (see
app/graph/nodes.py's supervisor_router_node) — this module never
classifies a turn itself; it only starts a new turn or resumes a paused
one and translates the graph's final state into the typed API response.
"""

import uuid
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import Settings
from app.graph.build import get_conversation_graph
from app.graph.state import ConversationState
from app.providers.text_to_speech.base import TextToSpeechProvider
from app.providers.vision.base import VisionProvider
from app.safety.constants import INTAKE_DISCLAIMER
from app.schemas.clinical_context import ClinicalNavigationSummary
from app.schemas.conversation import (
    AudioOutput,
    ClarificationAnswer,
    ConversationIntent,
    ConversationResponse,
)
from app.schemas.multimodal_intake import IntakeStatus, MultimodalIntakeRequest
from app.schemas.navigation import SpecialtyRoutingOut
from app.schemas.provider_search import ProviderSearchResponse
from app.schemas.vision import VisionObservation
from app.services.media_validation_service import PreparedMedia
from app.services.text_to_speech_service import build_text_to_speech_provider, synthesize_speech


class UnknownConversationThreadError(Exception):
    """Raised when resuming a thread_id that has no pending clarification
    (never started, already completed, or unknown to this process)."""


def _infer_input_modality(intake: MultimodalIntakeRequest, media: PreparedMedia | None) -> str:
    """A caller-supplied signal (never content-inferred beyond "which
    channel produced this turn") recorded in graph state for
    supervisor_router and observability — never used to classify intent
    itself (that stays purely text-based; see
    app.services.intent_classification_service)."""
    if media is not None:
        return "vision"
    if intake.voice_input is not None and intake.voice_input.transcript:
        return "voice"
    return "text"


def _final_status_and_disclaimer(
    values: dict[str, Any], detected_intent: str | None
) -> tuple[IntakeStatus, str] | None:
    """The final status/disclaimer for a completed (non-interrupted) turn.

    A Phase 3A clinical red-flag affirmation (see
    app.graph.nodes.clinical_red_flag_gate_node) sets is_emergency exactly
    like a user-declared emergency already did — both are reported as
    IntakeStatus.EMERGENCY here, regardless of what Phase 1C's own
    intake_response.status happened to record (it has no notion of a
    red-flag-derived emergency, only a user-declared one). Returns None
    when the state doesn't represent a completed turn at all (e.g. a
    greeting that somehow never reached response_agent) — callers must
    fall back to normal processing in that case."""
    detected_intent_value = detected_intent
    if detected_intent_value == ConversationIntent.GREETING.value:
        return IntakeStatus.READY_FOR_MULTIMODAL_PROCESSING, INTAKE_DISCLAIMER

    intake_response = values.get("intake_response")
    if not intake_response:
        return None
    if values.get("is_emergency"):
        return IntakeStatus.EMERGENCY, intake_response["disclaimer"]
    return IntakeStatus(intake_response["status"]), intake_response["disclaimer"]


def _clinical_navigation_from_values(values: dict[str, Any]) -> ClinicalNavigationSummary | None:
    navigation = values.get("clinical_navigation")
    return ClinicalNavigationSummary.model_validate(navigation) if navigation else None


def _response_from_completed_cache(
    values: dict[str, Any], resolved_thread_id: str
) -> ConversationResponse | None:
    """Reconstructs a ConversationResponse from an already-checkpointed,
    fully-completed turn's state — used only when a caller-supplied
    turn_fingerprint exactly matches one this thread already processed
    (see run_conversation_turn's duplicate-submission guard, backing the
    typed processed_fingerprints state field). Returns None when the
    cached state doesn't represent a completed turn (e.g. a greeting that
    somehow never reached response_agent) — callers must fall back to
    normal processing in that case; this is a pure optimization, never a
    correctness requirement."""
    detected_intent = values.get("detected_intent")
    resolved = _final_status_and_disclaimer(values, detected_intent)
    if resolved is None:
        return None
    status, disclaimer = resolved

    routing = values.get("routing")
    provider_search = values.get("provider_search")
    audio_dict = values.get("audio")
    return ConversationResponse(
        thread_id=resolved_thread_id,
        status=status,
        intent=ConversationIntent(detected_intent)
        if detected_intent
        else ConversationIntent.MEDICAL_CONCERN,
        missing_fields=[],
        clarification_questions=[],
        response_text=values.get("response_text"),
        routing=SpecialtyRoutingOut.model_validate(routing) if routing else None,
        provider_search=(
            ProviderSearchResponse.model_validate(provider_search) if provider_search else None
        ),
        media_note=values.get("media_note"),
        audio=AudioOutput.model_validate(audio_dict) if audio_dict else None,
        disclaimer=disclaimer,
        vision_observations=[
            VisionObservation.model_validate(o) for o in values.get("vision_observations") or []
        ],
        media_analysis_note=values.get("media_analysis_note"),
        clinical_navigation=_clinical_navigation_from_values(values),
    )


async def run_conversation_turn(
    *,
    thread_id: str | None,
    intake: MultimodalIntakeRequest | None,
    clarification_answer: ClarificationAnswer | None,
    generate_speech: bool,
    settings: Settings,
    session: AsyncSession,
    media: PreparedMedia | None = None,
    tts_provider_override: TextToSpeechProvider | None = None,
    vision_provider_override: VisionProvider | None = None,
    turn_fingerprint: str | None = None,
) -> ConversationResponse:
    """Run one conversation turn.

    Starts a new conversation when `intake` is supplied (a fresh
    thread_id is generated if the caller didn't provide one), or resumes
    a paused one when `clarification_answer` is supplied against an
    existing `thread_id`. `tts_provider_override`/`vision_provider_override`
    exist solely so tests can inject fake providers without ever
    configuring or calling Deepgram/Groq.

    `media` (Phase 2C) is only ever accepted alongside a new `intake` —
    never on a resume — and is passed transiently via
    config["configurable"] (never checkpointed state) so raw media bytes
    are never persisted beyond this one graph execution; only the
    resulting VisionObservation dicts a node writes to state are.

    `turn_fingerprint` (Phase 2D, optional) identifies this specific turn
    for duplicate-submission protection: if it exactly matches a
    fingerprint this thread already fully processed, the cached response
    is returned immediately — the graph is never re-invoked, so specialty
    routing, provider search, and TTS never run twice for the same turn
    (see _response_from_completed_cache and ConversationState's
    processed_fingerprints field). Omitting it (the default) simply
    disables this optimization; correctness never depends on it.
    """
    resolved_thread_id = thread_id or str(uuid.uuid4())
    turn_id = str(uuid.uuid4())
    graph = get_conversation_graph()
    config: RunnableConfig = {
        "configurable": {
            "thread_id": resolved_thread_id,
            "settings": settings,
            "session": session,
            "tts_provider": tts_provider_override,
            "media": media,
            "vision_provider": vision_provider_override,
        }
    }

    snapshot = None
    if thread_id is not None:
        snapshot = await graph.aget_state(config)
        if turn_fingerprint is not None and not snapshot.next:
            already_processed = turn_fingerprint in (
                snapshot.values.get("processed_fingerprints") or []
            )
            if already_processed:
                cached = _response_from_completed_cache(snapshot.values, resolved_thread_id)
                if cached is not None:
                    return cached

    if clarification_answer is not None:
        if snapshot is None:
            snapshot = await graph.aget_state(config)
        if not snapshot.next:
            raise UnknownConversationThreadError(
                f"No pending clarification for thread_id={resolved_thread_id!r}."
            )
        resume_value = clarification_answer.model_dump(mode="json", exclude_none=True)
        resume_update = {"processed_fingerprints": [turn_fingerprint]} if turn_fingerprint else None
        result = await graph.ainvoke(
            Command(resume=resume_value, update=resume_update), config=config
        )
    else:
        assert intake is not None
        initial_state: ConversationState = {
            "intake_request": intake.model_dump(mode="json"),
            "generate_speech": generate_speech,
            "speech_requested": generate_speech,
            "media_pending": media is not None,
            "thread_id": resolved_thread_id,
            "turn_id": turn_id,
            "input_modality": _infer_input_modality(intake, media),
            "processed_fingerprints": [turn_fingerprint] if turn_fingerprint else [],
            # Explicit reset (state-isolation hardening): a brand new,
            # non-resume turn must never inherit *any* prior turn's result
            # on the same thread_id. LangGraph's checkpointer persists state
            # across separate ainvoke calls on one thread — a field simply
            # omitted here would otherwise carry over unchanged. Most result
            # fields (response_text, audio/audio_result, intake_response,
            # missing_fields) are safe without this because the node that
            # produces them (response_agent/text_to_speech/
            # medical_intake_agent) runs unconditionally on every turn,
            # greeting included. The fields below are the ones a *greeting*
            # turn's path (supervisor_router -> conversation_agent ->
            # response_agent -> text_to_speech) never touches at all, so
            # without an explicit reset they would otherwise still be
            # readable in state from an earlier, unrelated turn on this
            # same thread_id: routing/provider_search/media_note/
            # vision_observations/media_analysis_note (set only by
            # medical_intake_agent/vision_agent/specialty_routing_agent/
            # provider_search_agent, all skipped for a greeting) and
            # is_emergency/clinical_* (set only by safety_gate/
            # clinical_red_flag_gate_node, likewise skipped).
            "routing": None,
            "provider_search": None,
            "media_note": None,
            "vision_observations": [],
            "media_analysis_note": None,
            "clinical_protocol": None,
            "clinical_context": None,
            "clinical_just_resumed": False,
            "clinical_next_step": "clinical_intake_agent",
            "clinical_red_flag_affirmed": False,
            "clinical_navigation": None,
            "is_emergency": False,
        }
        result = await graph.ainvoke(initial_state, config=config)

    interrupts = result.get("__interrupt__")
    detected_intent = result.get("detected_intent") or ConversationIntent.MEDICAL_CONCERN.value
    if interrupts:
        interrupt_value = interrupts[0].value
        clarification_questions: list[str] = interrupt_value.get("clarification_questions", [])
        questions_text = " ".join(clarification_questions)
        response_text = (
            f"I can help with that. {questions_text}"
            if questions_text
            else "Additional information is needed."
        )
        audio: AudioOutput | None = None
        if generate_speech:
            provider = tts_provider_override or build_text_to_speech_provider(settings)
            audio = await synthesize_speech(
                text=response_text, settings=settings, provider=provider
            )
        return ConversationResponse(
            thread_id=resolved_thread_id,
            status=IntakeStatus.NEEDS_CLARIFICATION,
            intent=ConversationIntent(detected_intent),
            missing_fields=interrupt_value.get("missing_fields", []),
            clarification_questions=clarification_questions,
            response_text=response_text,
            routing=None,
            provider_search=None,
            media_note=None,
            audio=audio,
            disclaimer=INTAKE_DISCLAIMER,
            vision_observations=[
                VisionObservation.model_validate(o) for o in result.get("vision_observations") or []
            ],
            media_analysis_note=result.get("media_analysis_note"),
        )

    resolved = _final_status_and_disclaimer(result, detected_intent)
    assert resolved is not None, (
        "a non-interrupted turn always has intake_response or is a greeting"
    )
    status, disclaimer = resolved

    routing = result.get("routing")
    provider_search = result.get("provider_search")
    audio_dict = result.get("audio")

    return ConversationResponse(
        thread_id=resolved_thread_id,
        status=status,
        intent=ConversationIntent(detected_intent),
        missing_fields=[],
        clarification_questions=[],
        response_text=result.get("response_text"),
        routing=SpecialtyRoutingOut.model_validate(routing) if routing else None,
        provider_search=(
            ProviderSearchResponse.model_validate(provider_search) if provider_search else None
        ),
        media_note=result.get("media_note"),
        audio=AudioOutput.model_validate(audio_dict) if audio_dict else None,
        disclaimer=disclaimer,
        vision_observations=[
            VisionObservation.model_validate(o) for o in result.get("vision_observations") or []
        ],
        media_analysis_note=result.get("media_analysis_note"),
        clinical_navigation=_clinical_navigation_from_values(result),
    )
