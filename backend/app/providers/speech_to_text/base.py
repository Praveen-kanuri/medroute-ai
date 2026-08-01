from abc import ABC, abstractmethod


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
    ) -> str:
        """Transcribe audio bytes into text.

        Args:
            audio_bytes: Raw audio data to transcribe.
            filename: Original filename, if known. Used only to help the
                provider infer the audio format; never persisted.
            content_type: The upload's declared MIME type, if known.
            language: An optional language hint (e.g. "en"). Providers may
                use this to improve accuracy; it is not a guarantee of the
                detected language.

        Returns:
            The transcribed text.

        Raises:
            Implementations may raise provider-specific exceptions on
            network, authentication, timeout, or unsupported-audio
            failures. Callers should treat any exception as a failed
            transcription — never fabricate a transcript when this raises
            or returns empty.
        """
        raise NotImplementedError
