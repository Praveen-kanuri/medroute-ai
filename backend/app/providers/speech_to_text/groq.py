"""Groq Whisper speech-to-text provider — the configured fallback for
Deepgram (see app.services.voice_transcription_service.FallbackSTTService).

All Groq SDK usage is isolated to this module. The API route
(app/api/v1/voice.py) and orchestration service
(app/services/voice_transcription_service.py) only ever depend on the
abstract SpeechToTextProvider interface.
"""

from groq import AsyncGroq

from app.providers.speech_provider_errors import classify_groq_exception
from app.providers.speech_to_text.base import (
    SpeechToTextError,
    SpeechToTextProvider,
    TranscriptionResult,
)


class GroqSTTProvider(SpeechToTextProvider):
    """Speech-to-text provider using Groq's hosted Whisper models."""

    def __init__(self, *, api_key: str, model: str, timeout_seconds: float) -> None:
        self._client = AsyncGroq(api_key=api_key, timeout=timeout_seconds)
        self._model = model

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str | None = None,
        content_type: str | None = None,
        language: str | None = None,
    ) -> TranscriptionResult:
        request_kwargs: dict[str, object] = {
            "model": self._model,
            "file": (
                filename or "audio",
                audio_bytes,
                content_type or "application/octet-stream",
            ),
        }
        if language:
            request_kwargs["language"] = language

        try:
            response = await self._client.audio.transcriptions.create(**request_kwargs)  # type: ignore[arg-type]
        except Exception as exc:
            # Never leak raw SDK/network exception details further up.
            eligible, category = classify_groq_exception(exc)
            raise SpeechToTextError(
                "Groq transcription request failed.",
                eligible_for_fallback=eligible,
                category=category,
            ) from exc

        text: str | None = getattr(response, "text", None)
        if not text or not text.strip():
            raise SpeechToTextError(
                "Groq returned an empty transcript.",
                eligible_for_fallback=True,
                category="empty_result",
            )
        return TranscriptionResult(
            text=text,
            provider="groq",
            model=self._model,
            language=language,
            duration_seconds=None,
        )
