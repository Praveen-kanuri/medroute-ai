"""Phase 2D: integration-level tests for the Deepgram-primary/Groq-fallback
speech architecture — proving the pieces that consume FallbackSTTService/
FallbackTTSService directly (local upload validation, the transcription
HTTP endpoint, and run_conversation_turn's greeting/clarification/
completed paths) all behave correctly and consistently. Unit-level
fallback-eligibility behavior is covered in test_voice_transcription_service.py
and test_text_to_speech_service.py; this file focuses on the gaps between
those layers. Never calls a real Deepgram or Groq endpoint.
"""

import pytest
from fastapi.testclient import TestClient

from app.api.v1.voice import get_speech_to_text_provider
from app.config.settings import Settings
from app.main import app
from app.providers.speech_to_text.base import (
    SpeechToTextError,
    SpeechToTextProvider,
    TranscriptionResult,
)
from app.providers.text_to_speech.base import SpeechResult, TextToSpeechError, TextToSpeechProvider
from app.schemas.multimodal_intake import (
    Duration,
    DurationUnit,
    MultimodalIntakeRequest,
    VoiceInput,
)
from app.services.conversation_service import run_conversation_turn
from app.services.text_to_speech_service import FallbackTTSService
from app.services.voice_transcription_service import (
    FallbackSTTService,
    UnsupportedAudioFormatError,
    transcribe_audio,
)


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


class _FakeSTT(SpeechToTextProvider):
    def __init__(
        self,
        *,
        result: str | None = None,
        error: Exception | None = None,
        provider_name: str = "fake",
    ) -> None:
        self._result = result
        self._error = error
        self._provider_name = provider_name
        self.call_count = 0

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str | None = None,
        content_type: str | None = None,
        language: str | None = None,
    ) -> TranscriptionResult:
        self.call_count += 1
        if self._error is not None:
            raise self._error
        return TranscriptionResult(
            text=self._result or "synthetic transcript",
            provider=self._provider_name,
            model=f"{self._provider_name}-model",
            language=language,
            duration_seconds=None,
        )


class _FakeTTS(TextToSpeechProvider):
    def __init__(self, *, error: Exception | None = None, provider_name: str = "fake") -> None:
        self._error = error
        self._provider_name = provider_name
        self.call_count = 0

    async def synthesize(self, text: str, *, voice: str | None = None) -> SpeechResult:
        self.call_count += 1
        if self._error is not None:
            raise self._error
        return SpeechResult(
            audio_bytes=b"fake-audio-bytes",
            content_type="audio/mpeg",
            provider=self._provider_name,
            model=f"{self._provider_name}-model",
        )


def _eligible_stt_error() -> SpeechToTextError:
    return SpeechToTextError(
        "synthetic transient failure", eligible_for_fallback=True, category="timeout"
    )


def _eligible_tts_error() -> TextToSpeechError:
    return TextToSpeechError(
        "synthetic transient failure", eligible_for_fallback=True, category="timeout"
    )


# 1. Deepgram Nova-3 is the default STT provider.
def test_deepgram_nova_3_is_the_default_stt_configuration() -> None:
    settings = _settings()
    assert settings.stt_primary_provider == "deepgram"
    assert settings.deepgram_stt_model == "nova-3"


# 6. Deepgram Aura-2 is the default TTS provider.
def test_deepgram_aura_2_is_the_default_tts_configuration() -> None:
    settings = _settings()
    assert settings.tts_primary_provider == "deepgram"
    assert settings.deepgram_tts_model == "aura-2-thalia-en"


def test_groq_fallback_models_match_configured_defaults() -> None:
    settings = _settings()
    assert settings.stt_fallback_provider == "groq"
    assert settings.groq_stt_model == "whisper-large-v3-turbo"
    assert settings.tts_fallback_provider == "groq"
    assert settings.groq_tts_model == "canopylabs/orpheus-v1-english"


# 4. Invalid audio does not trigger fallback.
async def test_invalid_audio_never_calls_either_stt_provider() -> None:
    primary = _FakeSTT(provider_name="deepgram")
    fallback = _FakeSTT(provider_name="groq")
    service = FallbackSTTService(primary=primary, fallback=fallback)
    with pytest.raises(UnsupportedAudioFormatError):
        await transcribe_audio(
            audio_bytes=b"0123456789",
            filename="clip.ogg",
            content_type="audio/ogg",
            language=None,
            settings=_settings(),
            provider=service,
        )
    assert primary.call_count == 0
    assert fallback.call_count == 0


# 5. Both STT failures stop without submitting an empty graph turn.
def test_both_stt_failures_return_503_without_reaching_the_graph() -> None:
    primary = _FakeSTT(error=_eligible_stt_error(), provider_name="deepgram")
    fallback = _FakeSTT(error=_eligible_stt_error(), provider_name="groq")
    service = FallbackSTTService(primary=primary, fallback=fallback)
    app.dependency_overrides[get_speech_to_text_provider] = lambda: service
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/voice/transcribe",
                files={"file": ("clip.wav", b"0123456789", "audio/wav")},
            )
        assert response.status_code == 503
        # Exactly one attempt each -- both exhausted, no request ever
        # reaches /api/v1/converse from this failed endpoint call.
        assert primary.call_count == 1
        assert fallback.call_count == 1
    finally:
        app.dependency_overrides.pop(get_speech_to_text_provider, None)


# 9. Both TTS failures preserve the assistant's text.
async def test_both_tts_failures_preserve_response_text() -> None:
    primary = _FakeTTS(error=_eligible_tts_error(), provider_name="deepgram")
    fallback = _FakeTTS(error=_eligible_tts_error(), provider_name="groq")
    service = FallbackTTSService(primary=primary, fallback=fallback)
    response = await run_conversation_turn(
        thread_id=None,
        intake=MultimodalIntakeRequest(
            voice_input=VoiceInput(transcript="zzz qqq unrelated words", language="en"),
            duration=Duration(value=1, unit=DurationUnit.DAYS),
        ),
        clarification_answer=None,
        generate_speech=True,
        settings=_settings(),
        session=None,  # type: ignore[arg-type]
        tts_provider_override=service,
    )
    assert response.response_text
    assert response.audio is None
    assert primary.call_count == 1
    assert fallback.call_count == 1


# 13. Voice greeting responses use the same provider strategy (fallback).
async def test_greeting_voice_response_uses_fallback_tts_service() -> None:
    primary = _FakeTTS(error=_eligible_tts_error(), provider_name="deepgram")
    fallback = _FakeTTS(provider_name="groq")
    service = FallbackTTSService(primary=primary, fallback=fallback)
    response = await run_conversation_turn(
        thread_id=None,
        intake=MultimodalIntakeRequest(voice_input=VoiceInput(transcript="Hi", language="en")),
        clarification_answer=None,
        generate_speech=True,
        settings=_settings(),
        session=None,  # type: ignore[arg-type]
        tts_provider_override=service,
    )
    assert response.intent == "greeting"
    assert response.audio is not None
    assert response.audio.model == "groq-model"
    assert fallback.call_count == 1


# 12. Voice clarification responses use the same provider strategy (fallback).
async def test_clarification_voice_response_uses_fallback_tts_service() -> None:
    primary = _FakeTTS(error=_eligible_tts_error(), provider_name="deepgram")
    fallback = _FakeTTS(provider_name="groq")
    service = FallbackTTSService(primary=primary, fallback=fallback)
    response = await run_conversation_turn(
        thread_id=None,
        intake=MultimodalIntakeRequest(main_concern="zzz qqq unrelated words"),
        clarification_answer=None,
        generate_speech=True,
        settings=_settings(),
        session=None,  # type: ignore[arg-type]
        tts_provider_override=service,
    )
    assert response.status == "needs_clarification"
    assert response.audio is not None
    assert response.audio.model == "groq-model"
    assert fallback.call_count == 1
