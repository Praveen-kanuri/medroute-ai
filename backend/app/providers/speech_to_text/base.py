from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptionResult:
    """A normalized transcription result — the same shape regardless of
    which provider produced it, so callers never need to know whether
    Deepgram or Groq actually ran."""

    text: str
    provider: str
    model: str
    language: str | None
    duration_seconds: float | None


class SpeechToTextError(Exception):
    """Raised by any SpeechToTextProvider (including FallbackSTTService —
    see app.services.voice_transcription_service) on failure.

    `eligible_for_fallback` tells the orchestrating fallback service
    whether this looks like a transient, provider-side failure worth one
    fallback attempt (timeout, connection failure, rate limiting,
    temporary unavailability, an eligible 5xx, or an authentication/
    configuration problem specific to that provider) — never a
    locally-detected/client-side problem (empty or invalid audio, an
    unsupported format) that would fail identically against any provider.
    `category` is a short, fixed, content-free label used only for
    observability logging — never a raw vendor exception message."""

    def __init__(self, message: str, *, eligible_for_fallback: bool, category: str) -> None:
        super().__init__(message)
        self.eligible_for_fallback = eligible_for_fallback
        self.category = category


class SpeechToTextProvider(ABC):
    """Abstract interface for a speech-to-text provider."""

    @abstractmethod
    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str | None = None,
        content_type: str | None = None,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio bytes into a normalized TranscriptionResult.

        Args:
            audio_bytes: Raw audio data to transcribe.
            filename: Original filename, if known. Used only to help the
                provider infer the audio format; never persisted.
            content_type: The upload's declared MIME type, if known.
            language: An optional language hint (e.g. "en"). Providers may
                use this to improve accuracy; it is not a guarantee of the
                detected language.

        Returns:
            The transcribed text plus provider/model metadata.

        Raises:
            SpeechToTextError on any network, authentication, timeout, or
            unsupported-audio failure, or an empty result — implementations
            must never fabricate a transcript.
        """
        raise NotImplementedError
