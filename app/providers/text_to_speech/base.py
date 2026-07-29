from abc import ABC, abstractmethod


class TextToSpeechProvider(ABC):
    """Abstract interface for a text-to-speech provider."""

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Synthesize speech audio bytes from the given text.

        Args:
            text: The input text to synthesize.

        Returns:
            The synthesized audio, as raw bytes.

        Raises:
            Implementations may raise provider-specific exceptions on
            network, authentication, or unsupported-text failures. No
            such exceptions are defined at this Phase 0 stage.
        """
        raise NotImplementedError
