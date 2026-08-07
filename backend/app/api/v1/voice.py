"""Phase 2A/2D: voice transcription endpoint.

Stateless: audio and transcripts are never persisted, and only safe
operational metadata is logged (never filenames, audio bytes, or
transcript content). No vendor-specific code lives in this module — see
app/services/voice_transcription_service.py (Deepgram primary, Groq
fallback) and app/providers/speech_to_text/{deepgram,groq}.py.
"""

import logging
import time

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status

from app.config.settings import Settings, get_settings
from app.providers.speech_to_text.base import SpeechToTextProvider
from app.schemas.voice_intake import TranscriptionResponse
from app.services.voice_transcription_service import (
    AudioTooLargeError,
    EmptyAudioUploadError,
    TranscriptionNotConfiguredError,
    TranscriptionProviderError,
    UnsupportedAudioFormatError,
    build_speech_to_text_provider,
    transcribe_audio,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def get_speech_to_text_provider(
    settings: Settings = Depends(get_settings),
) -> SpeechToTextProvider:
    """FastAPI dependency: builds the configured primary+fallback provider
    (see build_speech_to_text_provider), or raises a clean 503 if neither
    is configured. Tests override this dependency directly to inject a
    fake provider — Deepgram/Groq are never called in tests."""
    try:
        return build_speech_to_text_provider(settings)
    except TranscriptionNotConfiguredError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Voice transcription is not configured on this server.",
        ) from exc


@router.post(
    "/voice/transcribe",
    summary="Transcribe an uploaded audio file (Deepgram Nova-3, Groq Whisper fallback)",
    description=(
        "Accepts one multipart audio upload (mp3, wav, m4a, flac, or webm) and returns "
        "a speech-to-text transcript. Deepgram Nova-3 is the primary provider; Groq "
        "Whisper is attempted once, automatically, only on an eligible transient "
        "Deepgram failure (timeout, connection error, rate limiting, or temporary "
        "unavailability) — never in parallel. This is raw transcription only — not "
        "verified medical information, and not interpreted, diagnosed, or scored for "
        "urgency. Audio and transcripts are never persisted; only safe operational "
        "metadata (counts, model, status, processing time) is logged. Remote audio "
        "URLs are not accepted — upload the file directly."
    ),
)
async def transcribe_voice(
    file: UploadFile,
    language: str | None = Form(default=None),
    settings: Settings = Depends(get_settings),
    provider: SpeechToTextProvider = Depends(get_speech_to_text_provider),
) -> TranscriptionResponse:
    started = time.monotonic()
    audio_bytes = await file.read()
    audio_byte_count = len(audio_bytes)

    try:
        response = await transcribe_audio(
            audio_bytes=audio_bytes,
            filename=file.filename,
            content_type=file.content_type,
            language=language,
            settings=settings,
            provider=provider,
        )
    except EmptyAudioUploadError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    except UnsupportedAudioFormatError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    except AudioTooLargeError as exc:
        raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=str(exc)) from exc
    except TranscriptionProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Voice transcription is temporarily unavailable.",
        ) from exc

    elapsed_ms = (time.monotonic() - started) * 1000
    logger.info(
        "voice_transcribed transcription_id=%s model=%s status=%s language=%s "
        "audio_bytes=%d processing_ms=%.2f",
        response.transcription_id,
        response.model,
        response.status.value,
        response.language,
        audio_byte_count,
        elapsed_ms,
    )

    return response
