"""Unit tests for the Phase 2A/2D voice transcription service layer,
including the Deepgram-primary/Groq-fallback FallbackSTTService.

These tests never construct a real DeepgramSTTProvider/GroqSTTProvider
connection and never call either vendor — build_speech_to_text_provider()
is exercised only far enough to confirm which providers it wires up
(without invoking .transcribe()); the fallback *behavior* itself is
exercised entirely against fake in-process providers.
"""

import pytest

from app.config.settings import Settings
from app.providers.speech_to_text.base import (
    SpeechToTextError,
    SpeechToTextProvider,
    TranscriptionResult,
)
from app.providers.speech_to_text.deepgram import DeepgramSTTProvider
from app.providers.speech_to_text.groq import GroqSTTProvider
from app.schemas.voice_intake import TranscriptionStatus
from app.services.voice_transcription_service import (
    AudioTooLargeError,
    EmptyAudioUploadError,
    FallbackSTTService,
    TranscriptionNotConfiguredError,
    TranscriptionProviderError,
    UnsupportedAudioFormatError,
    build_speech_to_text_provider,
    transcribe_audio,
    validate_audio_upload,
)


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


class _FakeProvider(SpeechToTextProvider):
    def __init__(
        self,
        *,
        result: str = "synthetic transcript",
        provider_name: str = "fake",
        error: Exception | None = None,
    ) -> None:
        self._result = result
        self._provider_name = provider_name
        self._error = error
        self.calls: list[dict[str, object]] = []

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str | None = None,
        content_type: str | None = None,
        language: str | None = None,
    ) -> TranscriptionResult:
        self.calls.append(
            {
                "audio_bytes": audio_bytes,
                "filename": filename,
                "content_type": content_type,
                "language": language,
            }
        )
        if self._error is not None:
            raise self._error
        return TranscriptionResult(
            text=self._result,
            provider=self._provider_name,
            model=f"{self._provider_name}-model",
            language=language,
            duration_seconds=None,
        )


def _eligible_error(category: str = "timeout") -> SpeechToTextError:
    return SpeechToTextError(
        "synthetic eligible failure", eligible_for_fallback=True, category=category
    )


def _ineligible_error(category: str = "validation") -> SpeechToTextError:
    return SpeechToTextError(
        "synthetic ineligible failure", eligible_for_fallback=False, category=category
    )


# --- validate_audio_upload -------------------------------------------------


def test_validate_audio_upload_rejects_empty_file() -> None:
    with pytest.raises(EmptyAudioUploadError):
        validate_audio_upload(filename="clip.mp3", size=0, settings=_settings())


def test_validate_audio_upload_rejects_oversized_file() -> None:
    settings = _settings(voice_max_upload_bytes=100)
    with pytest.raises(AudioTooLargeError):
        validate_audio_upload(filename="clip.mp3", size=101, settings=settings)


def test_validate_audio_upload_rejects_unsupported_extension() -> None:
    with pytest.raises(UnsupportedAudioFormatError):
        validate_audio_upload(filename="clip.ogg", size=10, settings=_settings())


def test_validate_audio_upload_rejects_missing_extension() -> None:
    with pytest.raises(UnsupportedAudioFormatError):
        validate_audio_upload(filename=None, size=10, settings=_settings())


@pytest.mark.parametrize("extension", [".mp3", ".wav", ".m4a", ".flac", ".webm"])
def test_validate_audio_upload_accepts_each_supported_extension(extension: str) -> None:
    validate_audio_upload(filename=f"clip{extension}", size=10, settings=_settings())


# --- transcribe_audio -------------------------------------------------------


async def test_transcribe_audio_success_with_fake_provider() -> None:
    provider = _FakeProvider(result="patient reports a persistent cough", provider_name="deepgram")
    response = await transcribe_audio(
        audio_bytes=b"0123456789",
        filename="clip.wav",
        content_type="audio/wav",
        language="en",
        settings=_settings(),
        provider=provider,
    )
    assert response.transcript == "patient reports a persistent cough"
    assert response.language == "en"
    assert response.status == TranscriptionStatus.COMPLETED
    assert response.model == "deepgram-model"
    assert response.transcription_id
    assert provider.calls == [
        {
            "audio_bytes": b"0123456789",
            "filename": "clip.wav",
            "content_type": "audio/wav",
            "language": "en",
        }
    ]


async def test_transcribe_audio_rejects_invalid_upload_before_calling_provider() -> None:
    provider = _FakeProvider()
    with pytest.raises(UnsupportedAudioFormatError):
        await transcribe_audio(
            audio_bytes=b"0123456789",
            filename="clip.ogg",
            content_type="audio/ogg",
            language=None,
            settings=_settings(),
            provider=provider,
        )
    assert provider.calls == []


async def test_transcribe_audio_wraps_provider_failure() -> None:
    provider = _FakeProvider(error=RuntimeError("boom"))
    with pytest.raises(TranscriptionProviderError):
        await transcribe_audio(
            audio_bytes=b"0123456789",
            filename="clip.mp3",
            content_type="audio/mpeg",
            language=None,
            settings=_settings(),
            provider=provider,
        )


@pytest.mark.parametrize("blank_result", ["", "   "])
async def test_transcribe_audio_never_fabricates_empty_transcript(blank_result: str) -> None:
    provider = _FakeProvider(result=blank_result)
    with pytest.raises(TranscriptionProviderError):
        await transcribe_audio(
            audio_bytes=b"0123456789",
            filename="clip.mp3",
            content_type="audio/mpeg",
            language=None,
            settings=_settings(),
            provider=provider,
        )


# --- build_speech_to_text_provider -----------------------------------------


def test_build_speech_to_text_provider_raises_when_neither_configured() -> None:
    with pytest.raises(TranscriptionNotConfiguredError):
        build_speech_to_text_provider(_settings(groq_api_key=None, deepgram_api_key=None))


def test_build_speech_to_text_provider_wires_deepgram_primary_and_groq_fallback() -> None:
    provider = build_speech_to_text_provider(
        _settings(
            groq_api_key="synthetic-test-key-not-real",
            deepgram_api_key="synthetic-test-key-not-real",
        )
    )
    assert isinstance(provider, FallbackSTTService)
    assert isinstance(provider.primary, DeepgramSTTProvider)
    assert isinstance(provider.fallback, GroqSTTProvider)


def test_build_speech_to_text_provider_only_groq_configured_has_no_primary() -> None:
    provider = build_speech_to_text_provider(
        _settings(groq_api_key="synthetic-test-key-not-real", deepgram_api_key=None)
    )
    assert isinstance(provider, FallbackSTTService)
    assert provider.primary is None
    assert isinstance(provider.fallback, GroqSTTProvider)


def test_build_speech_to_text_provider_only_deepgram_configured_has_no_fallback() -> None:
    provider = build_speech_to_text_provider(
        _settings(groq_api_key=None, deepgram_api_key="synthetic-test-key-not-real")
    )
    assert isinstance(provider, FallbackSTTService)
    assert isinstance(provider.primary, DeepgramSTTProvider)
    assert provider.fallback is None


# --- FallbackSTTService -------------------------------------------------------


async def test_fallback_stt_service_success_never_calls_fallback() -> None:
    primary = _FakeProvider(result="primary result", provider_name="deepgram")
    fallback = _FakeProvider(result="fallback result", provider_name="groq")
    service = FallbackSTTService(primary=primary, fallback=fallback)
    result = await service.transcribe(b"bytes")
    assert result.text == "primary result"
    assert result.provider == "deepgram"
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 0


async def test_fallback_stt_service_eligible_failure_calls_fallback_once() -> None:
    primary = _FakeProvider(error=_eligible_error())
    fallback = _FakeProvider(result="fallback result", provider_name="groq")
    service = FallbackSTTService(primary=primary, fallback=fallback)
    result = await service.transcribe(b"bytes")
    assert result.text == "fallback result"
    assert result.provider == "groq"
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1


async def test_fallback_stt_service_ineligible_failure_never_calls_fallback() -> None:
    primary = _FakeProvider(error=_ineligible_error())
    fallback = _FakeProvider(result="fallback result", provider_name="groq")
    service = FallbackSTTService(primary=primary, fallback=fallback)
    with pytest.raises(SpeechToTextError):
        await service.transcribe(b"bytes")
    assert len(fallback.calls) == 0


async def test_fallback_stt_service_never_loops_back_to_primary() -> None:
    primary = _FakeProvider(error=_eligible_error())
    fallback = _FakeProvider(error=_eligible_error())
    service = FallbackSTTService(primary=primary, fallback=fallback)
    with pytest.raises(SpeechToTextError):
        await service.transcribe(b"bytes")
    # Exactly one attempt each -- the fallback's own failure must never
    # trigger a second attempt against the primary.
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1


async def test_fallback_stt_service_no_primary_configured_uses_fallback_directly() -> None:
    fallback = _FakeProvider(result="fallback only", provider_name="groq")
    service = FallbackSTTService(primary=None, fallback=fallback)
    result = await service.transcribe(b"bytes")
    assert result.text == "fallback only"
    assert len(fallback.calls) == 1
