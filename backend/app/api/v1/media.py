"""Phase 2C: direct image/video upload, feeding the same LangGraph
conversation graph as POST /api/v1/converse (intake normalization -> safety
gate -> clarification -> specialty routing -> provider search -> response
composition -> optional text-to-speech), with a dedicated vision_analysis
step run first.

Stateless HTTP adapter only — all upload validation lives in
app/services/media_validation_service.py and all orchestration logic lives
in app/services/conversation_service.py / app/graph/. Uploaded media is
never persisted; only safe operational metadata (counts, kind, status) is
logged, never filenames or media bytes. Vision analysis is best-effort:
if it is not configured or fails, this endpoint still completes the full
turn using any text/voice information supplied, with a safe status note —
it never returns an error for that reason alone.
"""

import json
import logging
import time

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import Settings, get_settings
from app.db.session import get_db_session
from app.schemas.conversation import ConversationResponse
from app.schemas.multimodal_intake import MultimodalIntakeRequest
from app.services.conversation_service import run_conversation_turn
from app.services.media_validation_service import (
    EmptyMediaUploadError,
    MalformedMediaError,
    MediaDimensionExceededError,
    MediaTooLargeError,
    UnsupportedMediaFormatError,
    VideoDurationExceededError,
    prepare_media_upload,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/media/analyze",
    summary="Analyze an uploaded image or video and run a full conversation turn",
    description=(
        "Accepts one directly-uploaded image (JPEG, PNG, WebP) or short video (MP4, MOV, "
        "WebM) plus the same intake fields POST /api/v1/converse accepts (as a JSON string "
        "in the intake_json form field), validates the upload by inspecting its file "
        "signature (never trusting the filename or client-declared content type), and runs "
        "it through the same LangGraph conversation graph with a dedicated vision-analysis "
        "step first. Vision analysis produces only controlled, non-diagnostic visual "
        "observations and never a diagnosis, treatment suggestion, urgency score, or "
        "emergency classification. Uploaded media is never persisted. This always starts a "
        "new conversation turn (a fresh thread_id is generated); resuming a clarification "
        "turn still uses POST /api/v1/converse."
    ),
)
async def analyze_media(
    file: UploadFile,
    intake_json: str = Form(...),
    generate_speech: bool = Form(False),
    session: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> ConversationResponse:
    started = time.monotonic()
    data = await file.read()

    try:
        prepared = prepare_media_upload(data, settings=settings)
    except EmptyMediaUploadError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    except UnsupportedMediaFormatError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    except MalformedMediaError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    except MediaDimensionExceededError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    except VideoDurationExceededError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    except MediaTooLargeError as exc:
        raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=str(exc)) from exc

    try:
        intake_payload = json.loads(intake_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="intake_json is not valid JSON.",
        ) from exc

    try:
        intake = MultimodalIntakeRequest.model_validate(intake_payload)
    except ValidationError as exc:
        # exc.errors() can embed a raw (non-JSON-serializable) exception
        # object in some entries' "ctx" — exc.json() is pydantic's own
        # guaranteed-JSON-safe serialization of the same error list.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=json.loads(exc.json())
        ) from exc

    response = await run_conversation_turn(
        thread_id=None,
        intake=intake,
        clarification_answer=None,
        generate_speech=generate_speech,
        settings=settings,
        session=session,
        media=prepared,
    )

    elapsed_ms = (time.monotonic() - started) * 1000
    logger.info(
        "media_analyzed thread_id=%s media_kind=%s frame_count=%d status=%s "
        "observation_count=%d processing_ms=%.2f",
        response.thread_id,
        prepared.kind,
        len(prepared.frames),
        response.status.value,
        len(response.vision_observations),
        elapsed_ms,
    )

    return response
