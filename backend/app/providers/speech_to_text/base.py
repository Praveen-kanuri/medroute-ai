from abc import ABC, abstractmethod


class SpeechToTextProvider(ABC):
    """Abstract interface for a speech-to-text provider."""

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio bytes into text.

        Args:
            audio_bytes: Raw audio data to transcribe.

        Returns:
            The transcribed text.

        Raises:
            Implementations may raise provider-specific exceptions on
            network, authentication, or unsupported-audio failures. No
            such exceptions are defined at this Phase 0 stage.
        """
        raise NotImplementedError
