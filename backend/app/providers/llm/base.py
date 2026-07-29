from abc import ABC, abstractmethod


class TextLLMProvider(ABC):
    """Abstract interface for a text-generation LLM provider."""

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate a text completion for the given prompt.

        Args:
            prompt: The input text prompt.

        Returns:
            The generated completion text.

        Raises:
            Implementations may raise provider-specific exceptions on
            network, authentication, or rate-limit failures. No such
            exceptions are defined at this Phase 0 stage.
        """
        raise NotImplementedError
