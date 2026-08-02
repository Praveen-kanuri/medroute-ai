from app.providers.text_to_speech.base import SpeechResult, TextToSpeechProvider


class FakeTextToSpeechProvider(TextToSpeechProvider):
    """Deterministic fake text-to-speech provider for local development and tests."""

    async def synthesize(self, text: str, *, voice: str | None = None) -> SpeechResult:
        return SpeechResult(
            audio_bytes=f"[fake-audio]{text}".encode(),
            content_type="audio/mpeg",
            provider="fake",
            model="fake-tts-model",
        )
