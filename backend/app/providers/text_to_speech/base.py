from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class SpeechResult:
    """A normalized speech-synthesis result — the same shape regardless of
    which provider produced it, so callers never need to know whether
    Deepgram or Groq actually ran."""

    audio_bytes: bytes
    content_type: str
    provider: str
    model: str


class TextToSpeechError(Exception):
    """Raised by any TextToSpeechProvider (including FallbackTTSService —
    see app.services.text_to_speech_service) on failure.

    `eligible_for_fallback` tells the orchestrating fallback service
    whether this looks like a transient, provider-side failure worth one
    fallback attempt (timeout, connection failure, rate limiting,
    temporary unavailability, an eligible 5xx, or an authentication/
    configuration problem specific to that provider) — never a
    locally-detected/client-side problem (unsafe or empty text, text over
    the configured length cap) that would fail identically against any
    provider. `category` is a short, fixed, content-free label used only
    for observability logging — never a raw vendor exception message."""

    def __init__(self, message: str, *, eligible_for_fallback: bool, category: str) -> None:
        super().__init__(message)
        self.eligible_for_fallback = eligible_for_fallback
        self.category = category


class TextToSpeechProvider(ABC):
    """Abstract interface for a text-to-speech provider."""

    @abstractmethod
    async def synthesize(self, text: str, *, voice: str | None = None) -> SpeechResult:
        """Synthesize speech audio from the given text.

        Args:
            text: The input text to synthesize. Callers must only ever
                pass already-validated, user-facing response text — never
                raw graph state, provider metadata, or exception details.
            voice: An optional voice/model identifier. Providers may use
                this to select a specific voice; it is not a guarantee
                every provider supports every value.

        Returns:
            The synthesized audio bytes plus provider/model metadata.

        Raises:
            TextToSpeechError on any network, authentication, timeout, or
            unsupported-text failure, or empty audio — implementations
            must never fabricate audio, and callers must always still
            return the text response regardless.
        """
        raise NotImplementedError
