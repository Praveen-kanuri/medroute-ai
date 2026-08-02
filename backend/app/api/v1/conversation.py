"""Phase 2B: LangGraph-orchestrated conversational navigation endpoint.

Stateless HTTP adapter only — all orchestration logic lives in
app/services/conversation_service.py and app/graph/. Chain-of-thought,
internal prompts, and raw graph state are never exposed; only the typed
ConversationResponse contract below is ever returned.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import Settings, get_settings
from app.db.session import get_db_session
from app.schemas.conversation import ConversationRequest, ConversationResponse
from app.services.conversation_service import (
    UnknownConversationThreadError,
    run_conversation_turn,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/converse",
    summary="LangGraph multi-agent conversational navigation turn, with optional spoken output",
    description=(
        "Runs one turn of the Phase 2D multi-agent conversation graph: "
        "supervisor_router (a structured routing decision recorded in graph state, "
        "never a Streamlit-side if/else) -> conversation_agent (greetings/small talk, "
        "short-circuited before any medical processing) or vision_agent + "
        "medical_intake_agent -> safety_gate -> clarification (pausing and returning a "
        "typed missing_fields list when information is missing) -> "
        "specialty_routing_agent -> provider_search_agent -> response_agent -> "
        "optional Deepgram text-to-speech. Pass the same thread_id with a "
        "clarification_answer to resume a paused turn — the original confirmed "
        "voice transcript and all other intake fields are preserved automatically; "
        "no re-upload or re-transcription is ever needed."
    ),
)
async def converse(
    payload: ConversationRequest,
    session: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> ConversationResponse:
    if payload.clarification_answer is not None:
        if payload.thread_id is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="thread_id is required when supplying a clarification_answer.",
            )
        if payload.intake is not None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    "Supply either intake (a new turn) or clarification_answer "
                    "(a resume), not both."
                ),
            )
    elif payload.intake is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="intake is required to start a new conversation turn.",
        )

    try:
        response = await run_conversation_turn(
            thread_id=payload.thread_id,
            intake=payload.intake,
            clarification_answer=payload.clarification_answer,
            generate_speech=payload.generate_speech,
            settings=settings,
            session=session,
            turn_fingerprint=payload.turn_fingerprint,
        )
    except UnknownConversationThreadError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Unknown or expired conversation thread_id.",
        ) from exc

    logger.info(
        "conversation_turn_evaluated thread_id=%s intent=%s status=%s routing_method=%s "
        "specialty_matched=%s provider_result_count=%d audio_generated=%s audio_model=%s",
        response.thread_id,
        response.intent.value,
        response.status.value,
        response.routing.method if response.routing else "not_attempted",
        response.routing.specialty_slug is not None if response.routing else False,
        len(response.provider_search.results) if response.provider_search else 0,
        response.audio is not None,
        response.audio.model if response.audio else "none",
    )

    return response
