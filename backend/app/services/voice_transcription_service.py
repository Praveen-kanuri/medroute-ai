"""Phase 2A/2D: audio-upload validation and transcription orchestration.

Deepgram is the primary speech-to-text provider; Groq is the single
configured fallback (see FallbackSTTService below). Vendor-specific
construction lives only in build_speech_to_text_provider() and the two
provider modules themselves (app/providers/speech_to_text/deepgram.py,
.../groq.py) — never in the API route. This module never persists audio
or transcripts; callers are responsible for not logging sensitive content
either.
"""

import logging
import uuid
from pathlib import Path

from app.config.settings import Settings
from app.providers.speech_to_text.base import (
    SpeechToTextError,
    SpeechToTextProvider,
    TranscriptionResult,
)
from app.schemas.voice_intake import TranscriptionResponse, TranscriptionStatus

logger = logging.getLogger(__name__)

# A small, documented starting set appropriate for Deepgram/Groq transcription.
SUPPORTED_AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac", ".webm")


class EmptyAudioUploadError(ValueError):
    """Raised when the uploaded audio file has zero bytes."""


class AudioTooLargeError(ValueError):
    """Raised when the uploaded audio file exceeds the configured size limit."""


class UnsupportedAudioFormatError(ValueError):
    """Raised when the uploaded file's extension is not supported."""


class TranscriptionNotConfiguredError(Exception):
    """Raised when no speech-to-text provider is configured (neither
    DEEPGRAM_API_KEY nor GROQ_API_KEY)."""


class TranscriptionProviderError(Exception):
    """Raised when the configured provider (and, if eligible, its
    fallback) fails (timeout, network, API error) or returns no usable
    transcript."""


class FallbackSTTService(SpeechToTextProvider):
    """Wraps a primary and an optional fallback SpeechToTextProvider
    behind the same provider-neutral interface, so every caller (the API
    route, tests, Streamlit via the API) depends only on
    SpeechToTextProvider — never on which concrete vendor actually ran.

    Tries the primary provider first. On an eligible failure (see
    SpeechToTextError.eligible_for_fallback), makes exactly one fallback
    attempt — never in parallel, and never a second hop back to the
    primary. A non-eligible failure (a locally-detected validation
    problem that would fail identically against any provider) propagates
    immediately without ever calling the fallback."""

    def __init__(
        self,
        *,
        primary: SpeechToTextProvider | None,
        fallback: SpeechToTextProvider | None,
    ) -> None:
        if primary is None and fallback is None:
            raise ValueError("At least one of primary/fallback must be configured.")
        self._primary = primary
        self._fallback = fallback

    @property
    def primary(self) -> SpeechToTextProvider | None:
        return self._primary

    @property
    def fallback(self) -> SpeechToTextProvider | None:
        return self._fallback

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str | None = None,
        content_type: str | None = None,
        language: str | None = None,
    ) -> TranscriptionResult:
        if self._primary is None:
            assert self._fallback is not None
            return await self._fallback.transcribe(
                audio_bytes, filename=filename, content_type=content_type, language=language
            )

        try:
            return await self._primary.transcribe(
                audio_bytes, filename=filename, content_type=content_type, language=language
            )
        except SpeechToTextError as exc:
            if not exc.eligible_for_fallback or self._fallback is None:
                raise
            logger.info("stt_primary_failed_falling_back category=%s", exc.category)
            return await self._fallback.transcribe(
                audio_bytes, filename=filename, content_type=content_type, language=language
            )


def validate_audio_upload(*, filename: str | None, size: int, settings: Settings) -> None:
    """Reject empty, oversized, or unsupported uploads before calling any provider."""
    if size <= 0:
        raise EmptyAudioUploadError("Uploaded audio file is empty.")
    if size > settings.voice_max_upload_bytes:
        raise AudioTooLargeError(
            f"Uploaded audio file exceeds the {settings.voice_max_upload_bytes}-byte limit."
        )
    extension = Path(filename or "").suffix.lower()
    if extension not in SUPPORTED_AUDIO_EXTENSIONS:
        supported = ", ".join(SUPPORTED_AUDIO_EXTENSIONS)
        raise UnsupportedAudioFormatError(
            f"Unsupported audio format {extension or '(none)'!r}. Supported: {supported}."
        )


def _build_single_stt_provider(
    provider_name: str, settings: Settings
) -> SpeechToTextProvider | None:
    """Constructs one named provider, or None if it isn't configured (no
    API key) or isn't a recognized name. Vendor imports are local so
    importing this module never requires either SDK to be exercised."""
    if provider_name == "deepgram":
        if not settings.deepgram_configured:
            return None
        from app.providers.speech_to_text.deepgram import DeepgramSTTProvider

        assert settings.deepgram_api_key is not None
        return DeepgramSTTProvider(
            api_key=settings.deepgram_api_key.get_secret_value(),
            model=settings.deepgram_stt_model,
            timeout_seconds=settings.deepgram_stt_timeout_seconds,
        )
    if provider_name == "groq":
        if not settings.groq_configured:
            return None
        from app.providers.speech_to_text.groq import GroqSTTProvider

        assert settings.groq_api_key is not None
        return GroqSTTProvider(
            api_key=settings.groq_api_key.get_secret_value(),
            model=settings.groq_stt_model,
            timeout_seconds=settings.groq_stt_timeout_seconds,
        )
    return None


def build_speech_to_text_provider(settings: Settings) -> SpeechToTextProvider:
    """Construct the configured primary/fallback speech-to-text providers,
    wrapped in FallbackSTTService. Raises TranscriptionNotConfiguredError
    only when *neither* settings.stt_primary_provider nor
    settings.stt_fallback_provider resolves to a configured provider."""
    primary = _build_single_stt_provider(settings.stt_primary_provider, settings)
    fallback = _build_single_stt_provider(settings.stt_fallback_provider, settings)
    if primary is None and fallback is None:
        raise TranscriptionNotConfiguredError("No speech-to-text provider is configured.")
    return FallbackSTTService(primary=primary, fallback=fallback)


async def transcribe_audio(
    *,
    audio_bytes: bytes,
    filename: str | None,
    content_type: str | None,
    language: str | None,
    settings: Settings,
    provider: SpeechToTextProvider,
) -> TranscriptionResponse:
    """Validate the upload, call the provider (which may itself be a
    FallbackSTTService), and build the typed response.

    Never fabricates a transcript: any provider failure or empty result
    raises TranscriptionProviderError rather than returning placeholder
    text — never submitted onward as an empty conversation turn.
    """
    validate_audio_upload(filename=filename, size=len(audio_bytes), settings=settings)

    try:
        result = await provider.transcribe(
            audio_bytes, filename=filename, content_type=content_type, language=language
        )
    except TranscriptionProviderError:
        raise
    except Exception as exc:
        raise TranscriptionProviderError("Speech-to-text provider failed.") from exc

    if not result.text or not result.text.strip():
        raise TranscriptionProviderError("Provider returned an empty transcript.")

    return TranscriptionResponse(
        transcription_id=str(uuid.uuid4()),
        transcript=result.text,
        language=result.language or language,
        model=result.model,
        status=TranscriptionStatus.COMPLETED,
    )
