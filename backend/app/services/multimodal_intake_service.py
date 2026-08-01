"""Pure evaluation of a Phase 1C multimodal intake request.

No database, no network, no LLM/vision-model calls, no media fetching.
Only deterministic normalization and state evaluation over already-
validated Pydantic input, so this is fully unit-testable in isolation.
"""

import uuid

from app.safety.constants import EMERGENCY_SAFETY_MESSAGE, INTAKE_DISCLAIMER
from app.schemas.multimodal_intake import (
    Duration,
    IntakeStatus,
    MultimodalIntakeRequest,
    MultimodalIntakeResponse,
    MultimodalStatus,
    NextAction,
    NormalizedIntake,
)
from app.services.duration_extraction import extract_duration_from_transcript

_CONCERN_QUESTION = (
    "Please describe your main concern or provide a text, voice-transcript, image, or video input."
)
_DURATION_QUESTION = "How long have you been experiencing this concern?"

_MISSING_FIELD_QUESTIONS = {
    "concern": _CONCERN_QUESTION,
    "duration": _DURATION_QUESTION,
}


def _has_text(request: MultimodalIntakeRequest) -> bool:
    return bool(request.symptoms) or request.main_concern is not None


def _has_voice(request: MultimodalIntakeRequest) -> bool:
    return request.voice_input is not None and request.voice_input.transcript is not None


def _image_count(request: MultimodalIntakeRequest) -> int:
    return len(request.vision_inputs.image_urls) if request.vision_inputs else 0


def _video_count(request: MultimodalIntakeRequest) -> int:
    return len(request.vision_inputs.video_urls) if request.vision_inputs else 0


def _is_emergency(request: MultimodalIntakeRequest) -> bool:
    return request.emergency_concern or bool(request.emergency_signals)


def _effective_duration(request: MultimodalIntakeRequest) -> Duration | None:
    """The structured duration to use: the user-supplied field if present,
    otherwise a conservative deterministic extraction from a confirmed
    voice transcript (see app.services.duration_extraction). Returns None
    whenever neither is available or the transcript's phrasing is
    ambiguous — callers must then fall back to asking the user directly,
    never guess."""
    if request.duration is not None:
        return request.duration
    if _has_voice(request):
        assert request.voice_input is not None
        return extract_duration_from_transcript(request.voice_input.transcript)
    return None


def evaluate_intake(request: MultimodalIntakeRequest) -> MultimodalIntakeResponse:
    """Evaluate a validated intake request into a typed response.

    Never mutates the input. Deterministic apart from the generated
    intake_id. Never returns a diagnosis, treatment advice, urgency score,
    media interpretation, or inferred specialty.
    """
    text_received = _has_text(request)
    voice_received = _has_voice(request)
    image_count = _image_count(request)
    video_count = _video_count(request)
    vision_received = image_count > 0 or video_count > 0
    concern_present = text_received or voice_received or vision_received
    duration = _effective_duration(request)

    missing_fields: list[str] = []
    if _is_emergency(request):
        status = IntakeStatus.EMERGENCY
        next_action = NextAction(
            code="stop_and_seek_emergency_care",
            message=EMERGENCY_SAFETY_MESSAGE,
        )
        safety_message = EMERGENCY_SAFETY_MESSAGE
    else:
        if not concern_present:
            missing_fields.append("concern")
        if duration is None:
            missing_fields.append("duration")

        if missing_fields:
            status = IntakeStatus.NEEDS_CLARIFICATION
            next_action = NextAction(
                code="provide_missing_information",
                message="Additional information is needed before continuing.",
            )
        else:
            status = IntakeStatus.READY_FOR_MULTIMODAL_PROCESSING
            next_action = NextAction(
                code="continue_to_multimodal_routing",
                message="The intake is structurally ready for the next processing stage.",
            )
        safety_message = None

    clarification_questions = [_MISSING_FIELD_QUESTIONS[field] for field in missing_fields]

    normalized_intake = NormalizedIntake(
        symptoms=request.symptoms,
        main_concern=request.main_concern,
        duration=duration,
        location=request.location,
        preferred_specialty=request.preferred_specialty,
        emergency_concern=request.emergency_concern,
        emergency_signals=request.emergency_signals,
        voice_input=request.voice_input,
        vision_inputs=request.vision_inputs,
    )

    return MultimodalIntakeResponse(
        intake_id=str(uuid.uuid4()),
        status=status,
        normalized_intake=normalized_intake,
        multimodal_status=MultimodalStatus(
            text_received=text_received,
            voice_received=voice_received,
            vision_received=vision_received,
            image_count=image_count,
            video_count=video_count,
        ),
        missing_fields=missing_fields,
        clarification_questions=clarification_questions,
        next_action=next_action,
        safety_message=safety_message,
        disclaimer=INTAKE_DISCLAIMER,
    )
