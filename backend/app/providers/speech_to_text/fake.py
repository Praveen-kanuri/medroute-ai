from app.providers.speech_to_text.base import SpeechToTextProvider, TranscriptionResult


class FakeSpeechToTextProvider(SpeechToTextProvider):
    """Deterministic fake speech-to-text provider for local development and tests."""

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str | None = None,
        content_type: str | None = None,
        language: str | None = None,
    ) -> TranscriptionResult:
        return TranscriptionResult(
            text=f"[fake-transcript] {len(audio_bytes)} bytes received",
            provider="fake",
            model="fake-stt-model",
            language=language,
            duration_seconds=None,
        )
