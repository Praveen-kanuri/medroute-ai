"""Phase 2B: typed contract for POST /api/v1/converse.

Wraps the existing Phase 1C intake contract and Phase 1D routing/provider-
search response shapes unchanged, adding only what the LangGraph
conversation adds: a thread_id to resume a paused clarification turn, a
typed clarification_answer to resume with, a short non-diagnostic
response_text, and an optional synthesized-speech AudioOutput. Never a
diagnosis, treatment, urgency score, or medical-certainty claim.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.clinical_context import ClinicalNavigationSummary
from app.schemas.multimodal_intake import Duration, IntakeStatus, MultimodalIntakeRequest
from app.schemas.navigation import SpecialtyRoutingOut
from app.schemas.provider_search import ProviderSearchResponse
from app.schemas.vision import VisionObservation

_MAX_MAIN_CONCERN_LENGTH = 300


_MAX_TURN_FINGERPRINT_LENGTH = 128
_MAX_CLARIFICATION_VOICE_TRANSCRIPT_LENGTH = 5000
_MAX_CLINICAL_ANSWER_TEXT_LENGTH = 2000


class ConversationIntent(StrEnum):
    """A lightweight, deterministic classification of what this turn's
    request represents — set once per turn by the graph's supervisor
    router (see app.graph.nodes.supervisor_router_node), never inferred by
    an LLM. Used only to decide whether medical intake validation,
    specialty routing, and provider search should run at all; it never
    overrides emergency handling (supervisor_router checks emergency
    signals before ever classifying a turn as a greeting)."""

    GREETING = "greeting"
    MEDICAL_CONCERN = "medical_concern"
    CLARIFICATION_ANSWER = "clarification_answer"
    UNSUPPORTED_OR_UNCLEAR = "unsupported_or_unclear"


class ClarificationAnswer(BaseModel):
    """A structured answer to one previously reported missing_fields entry.
    Only the fields the graph actually asked about are ever considered —
    supplying one that was not requested has no effect.

    `voice_transcript` (Phase 2E) is a caller-confirmed speech-to-text
    transcript of the user's *spoken* answer to the clarification question
    — the voice-workflow equivalent of typing into the duration/main_concern
    form fields. It is never stored as-is: the graph only ever derives
    `duration` (via the same conservative extraction a fresh voice-only
    intake already gets — see app.services.duration_extraction) or
    `main_concern` from it, and only when the caller didn't already supply
    that field directly. It never overwrites an already-confirmed concern
    from earlier in the same turn — see
    app.graph.nodes._merge_clarification_answer."""

    model_config = ConfigDict(extra="forbid")

    duration: Duration | None = None
    main_concern: str | None = Field(default=None, max_length=_MAX_MAIN_CONCERN_LENGTH)
    voice_transcript: str | None = Field(
        default=None, max_length=_MAX_CLARIFICATION_VOICE_TRANSCRIPT_LENGTH
    )
    # Phase 3A: a typed (or voice-transcribed) free-text answer to a pending
    # app.services.clinical_intake_service protocol question (e.g. "was the
    # onset sudden or gradual?") — reported back as the single fixed
    # missing_fields entry "clinical_answer", never a protocol-specific field
    # name, since only one such question is ever pending at a time. Ignored
    # unless that is in fact what is currently pending; a caller may also
    # reuse voice_transcript for a spoken answer instead (both are accepted
    # equivalently — see app.graph.nodes.clinical_intake_agent_node).
    clinical_answer_text: str | None = Field(
        default=None, max_length=_MAX_CLINICAL_ANSWER_TEXT_LENGTH
    )


class ConversationRequest(BaseModel):
    """Either a new conversation turn (`intake` populated; `thread_id`
    optional — the server generates one if omitted) or a resume turn
    (`thread_id` required, `clarification_answer` populated, `intake`
    omitted)."""

    model_config = ConfigDict(extra="forbid")

    thread_id: str | None = None
    intake: MultimodalIntakeRequest | None = None
    clarification_answer: ClarificationAnswer | None = None
    generate_speech: bool = False
    # Optional, caller-computed duplicate-submission guard (Phase 2D) — see
    # app.services.conversation_service.run_conversation_turn. Omitting it
    # simply disables the optimization; it is never required for
    # correctness, only to avoid redundant specialty routing, provider
    # search, or TTS work on an exact resubmission of the same turn.
    turn_fingerprint: str | None = Field(default=None, max_length=_MAX_TURN_FINGERPRINT_LENGTH)


class AudioOutput(BaseModel):
    """Synthesized speech for the final response_text only. Never
    persisted server-side; the caller (Streamlit) plays it directly."""

    audio_base64: str
    content_type: str
    model: str


class ConversationResponse(BaseModel):
    thread_id: str
    status: IntakeStatus
    intent: ConversationIntent
    missing_fields: list[str]
    clarification_questions: list[str]
    response_text: str | None
    routing: SpecialtyRoutingOut | None
    provider_search: ProviderSearchResponse | None
    media_note: str | None
    audio: AudioOutput | None
    disclaimer: str

    # Phase 2C: controlled, non-diagnostic visual observations from a
    # directly-uploaded image or sampled video frames (see
    # POST /api/v1/media/analyze), and a short safe status note about that
    # analysis. Both are always empty/None for a text/voice-only turn.
    vision_observations: list[VisionObservation] = Field(default_factory=list)
    media_analysis_note: str | None = None

    # Phase 3A: a bounded, non-diagnostic clinical-navigation summary (see
    # app.schemas.clinical_context.ClinicalNavigationSummary), set only once
    # a matched complaint protocol's questions are all answered and no red
    # flag was affirmed. Always None for a concern matching no protocol, a
    # turn still awaiting a protocol question, or an emergency.
    clinical_navigation: ClinicalNavigationSummary | None = None
