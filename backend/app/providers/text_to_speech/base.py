from abc import ABC, abstractmethod


class TextToSpeechProvider(ABC):
    """Abstract interface for a text-to-speech provider."""

    @abstractmethod
    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        """Synthesize speech audio bytes from the given text.

        Args:
            text: The input text to synthesize.
            voice: An optional voice/model identifier. Providers may use
                this to select a specific voice; it is not a guarantee
                every provider supports every value.

        Returns:
            The synthesized audio, as raw bytes.

        Raises:
            Implementations may raise provider-specific exceptions on
            network, authentication, timeout, or unsupported-text
            failures. Callers should treat any exception as a failed
            synthesis — never fabricate audio when this raises or
            returns empty, and always still return the text response.
        """
        raise NotImplementedError
