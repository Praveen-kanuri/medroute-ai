from app.providers.speech_to_text.base import SpeechToTextProvider


class FakeSpeechToTextProvider(SpeechToTextProvider):
    """Deterministic fake speech-to-text provider for local development and tests."""

    async def transcribe(self, audio_bytes: bytes) -> str:
        return f"[fake-transcript] {len(audio_bytes)} bytes received"
