"""Deepgram Nova speech-to-text provider — the primary STT provider (see
app.services.voice_transcription_service.FallbackSTTService).

Uses Deepgram's pre-recorded transcription API only (one uploaded audio
file in, one transcript out) — never the Flux streaming/turn-detection
API, which remains a future milestone once MedRoute adds continuous
streaming audio (see docs/roadmap.md). All Deepgram SDK usage is isolated
to this module.
"""

from deepgram import AsyncDeepgramClient
from deepgram.core.api_error import ApiError as DeepgramApiError
from deepgram.core.request_options import RequestOptions

from app.providers.speech_provider_errors import classify_deepgram_exception
from app.providers.speech_to_text.base import (
    SpeechToTextError,
    SpeechToTextProvider,
    TranscriptionResult,
)


class DeepgramSTTProvider(SpeechToTextProvider):
    """Speech-to-text provider using Deepgram's hosted Nova models."""

    def __init__(self, *, api_key: str, model: str, timeout_seconds: float) -> None:
        self._client = AsyncDeepgramClient(api_key=api_key)
        self._model = model
        self._timeout_seconds = timeout_seconds

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str | None = None,
        content_type: str | None = None,
        language: str | None = None,
    ) -> TranscriptionResult:
        try:
            response = await self._client.listen.v1.media.transcribe_file(
                request=audio_bytes,
                model=self._model,
                language=language,
                smart_format=True,
                punctuate=True,
                request_options=RequestOptions(timeout_in_seconds=int(self._timeout_seconds)),
            )
        except DeepgramApiError as exc:
            eligible, category = classify_deepgram_exception(exc)
            raise SpeechToTextError(
                "Deepgram transcription request failed.",
                eligible_for_fallback=eligible,
                category=category,
            ) from exc
        except Exception as exc:
            eligible, category = classify_deepgram_exception(exc)
            raise SpeechToTextError(
                "Deepgram transcription request failed.",
                eligible_for_fallback=eligible,
                category=category,
            ) from exc

        results = getattr(response, "results", None)
        channels = results.channels if results is not None else None
        alternatives = channels[0].alternatives if channels else None
        transcript = alternatives[0].transcript if alternatives else None
        if not transcript or not transcript.strip():
            raise SpeechToTextError(
                "Deepgram returned an empty transcript.",
                eligible_for_fallback=True,
                category="empty_result",
            )

        metadata = getattr(response, "metadata", None)
        duration = metadata.duration if metadata is not None else None
        detected_language = channels[0].detected_language if channels else None
        return TranscriptionResult(
            text=transcript,
            provider="deepgram",
            model=self._model,
            language=language or detected_language,
            duration_seconds=duration,
        )
