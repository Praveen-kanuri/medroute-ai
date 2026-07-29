from app.providers.llm.base import TextLLMProvider


class FakeTextLLMProvider(TextLLMProvider):
    """Deterministic fake LLM provider for local development and tests."""

    async def generate(self, prompt: str) -> str:
        return f"[fake-llm-response] echo: {prompt}"
