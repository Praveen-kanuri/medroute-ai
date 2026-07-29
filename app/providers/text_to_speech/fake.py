from app.providers.text_to_speech.base import TextToSpeechProvider


class FakeTextToSpeechProvider(TextToSpeechProvider):
    """Deterministic fake text-to-speech provider for local development and tests."""

    async def synthesize(self, text: str) -> bytes:
        return f"[fake-audio]{text}".encode()
