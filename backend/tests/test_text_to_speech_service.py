"""Unit tests for the Phase 2B/2D text-to-speech orchestration layer,
including the Deepgram-primary/Groq-fallback FallbackTTSService.

Never calls Deepgram or Groq: build_text_to_speech_provider() is only
exercised far enough to confirm which providers it wires up — without
ever invoking .synthesize(). Every synthesis test uses a fake in-process
provider.
"""

import logging
import socket

import pytest

from app.config.settings import Settings
from app.providers.text_to_speech.base import SpeechResult, TextToSpeechError, TextToSpeechProvider
from app.providers.text_to_speech.deepgram import DeepgramTTSProvider
from app.providers.text_to_speech.groq import GroqTTSProvider
from app.services.text_to_speech_service import (
    FallbackTTSService,
    build_text_to_speech_provider,
    synthesize_speech,
)

SYNTHETIC_TEXT_MARKER = "synthetic-marker-response-text-7c1e9a"
SYNTHETIC_AUDIO_MARKER = b"synthetic-marker-audio-bytes-4b2f"


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


class _FakeProvider(TextToSpeechProvider):
    def __init__(
        self,
        *,
        result: bytes | None = None,
        provider_name: str = "fake",
        error: Exception | None = None,
    ) -> None:
        self._result = result if result is not None else SYNTHETIC_AUDIO_MARKER
        self._provider_name = provider_name
        self._error = error
        self.calls: list[dict[str, object]] = []

    async def synthesize(self, text: str, *, voice: str | None = None) -> SpeechResult:
        self.calls.append({"text": text, "voice": voice})
        if self._error is not None:
            raise self._error
        return SpeechResult(
            audio_bytes=self._result,
            content_type="audio/mpeg",
            provider=self._provider_name,
            model=f"{self._provider_name}-model",
        )


def _eligible_error(category: str = "timeout") -> TextToSpeechError:
    return TextToSpeechError(
        "synthetic eligible failure", eligible_for_fallback=True, category=category
    )


def _ineligible_error(category: str = "validation") -> TextToSpeechError:
    return TextToSpeechError(
        "synthetic ineligible failure", eligible_for_fallback=False, category=category
    )


# --- build_text_to_speech_provider ------------------------------------------


def test_build_provider_returns_none_when_neither_configured() -> None:
    assert (
        build_text_to_speech_provider(_settings(deepgram_api_key=None, groq_api_key=None)) is None
    )


def test_build_provider_wires_deepgram_primary_and_groq_fallback() -> None:
    provider = build_text_to_speech_provider(
        _settings(
            deepgram_api_key="synthetic-test-key-not-real",
            groq_api_key="synthetic-test-key-not-real",
        )
    )
    assert isinstance(provider, FallbackTTSService)
    assert isinstance(provider.primary, DeepgramTTSProvider)
    assert isinstance(provider.fallback, GroqTTSProvider)


def test_build_provider_only_groq_configured_has_no_primary() -> None:
    provider = build_text_to_speech_provider(
        _settings(deepgram_api_key=None, groq_api_key="synthetic-test-key-not-real")
    )
    assert isinstance(provider, FallbackTTSService)
    assert provider.primary is None
    assert isinstance(provider.fallback, GroqTTSProvider)


# --- synthesize_speech -------------------------------------------------------


async def test_synthesize_speech_success_with_fake_provider() -> None:
    provider = _FakeProvider(result=SYNTHETIC_AUDIO_MARKER, provider_name="deepgram")
    result = await synthesize_speech(text="hello there", settings=_settings(), provider=provider)
    assert result is not None
    assert result.content_type == "audio/mpeg"
    assert result.model == "deepgram-model"
    assert provider.calls == [{"text": "hello there", "voice": None}]


async def test_synthesize_speech_returns_none_when_provider_is_none() -> None:
    result = await synthesize_speech(text="hello there", settings=_settings(), provider=None)
    assert result is None


@pytest.mark.parametrize("blank_text", [None, ""])
async def test_synthesize_speech_returns_none_for_blank_text(blank_text: str | None) -> None:
    provider = _FakeProvider()
    result = await synthesize_speech(text=blank_text, settings=_settings(), provider=provider)
    assert result is None
    assert provider.calls == []


async def test_synthesize_speech_skips_text_over_max_length() -> None:
    provider = _FakeProvider()
    settings = _settings(tts_max_text_length=10)
    result = await synthesize_speech(
        text="this text is definitely longer than ten characters",
        settings=settings,
        provider=provider,
    )
    assert result is None
    assert provider.calls == []


async def test_synthesize_speech_never_raises_on_provider_failure() -> None:
    provider = _FakeProvider(error=RuntimeError("synthetic provider failure"))
    result = await synthesize_speech(text="hello there", settings=_settings(), provider=provider)
    assert result is None


async def test_synthesize_speech_returns_none_for_empty_audio() -> None:
    provider = _FakeProvider(result=b"")
    result = await synthesize_speech(text="hello there", settings=_settings(), provider=provider)
    assert result is None


async def test_synthesize_speech_makes_no_network_calls_with_fake_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _forbidden_create_connection(*args: object, **kwargs: object) -> None:
        raise AssertionError("text-to-speech synthesis attempted an outbound network connection")

    monkeypatch.setattr(socket, "create_connection", _forbidden_create_connection)
    provider = _FakeProvider()
    result = await synthesize_speech(text="hello there", settings=_settings(), provider=provider)
    assert result is not None


async def test_synthesize_speech_logging_never_contains_text_or_audio(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _FakeProvider(error=RuntimeError(SYNTHETIC_TEXT_MARKER))
    with caplog.at_level(logging.DEBUG):
        result = await synthesize_speech(
            text=SYNTHETIC_TEXT_MARKER, settings=_settings(), provider=provider
        )
    assert result is None
    assert SYNTHETIC_TEXT_MARKER not in caplog.text


# --- FallbackTTSService -------------------------------------------------------


async def test_fallback_tts_service_success_never_calls_fallback() -> None:
    primary = _FakeProvider(provider_name="deepgram")
    fallback = _FakeProvider(provider_name="groq")
    service = FallbackTTSService(primary=primary, fallback=fallback)
    result = await service.synthesize("hello")
    assert result.provider == "deepgram"
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 0


async def test_fallback_tts_service_eligible_failure_calls_fallback_once() -> None:
    primary = _FakeProvider(error=_eligible_error())
    fallback = _FakeProvider(provider_name="groq")
    service = FallbackTTSService(primary=primary, fallback=fallback)
    result = await service.synthesize("hello")
    assert result.provider == "groq"
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1


async def test_fallback_tts_service_ineligible_failure_never_calls_fallback() -> None:
    primary = _FakeProvider(error=_ineligible_error())
    fallback = _FakeProvider(provider_name="groq")
    service = FallbackTTSService(primary=primary, fallback=fallback)
    with pytest.raises(TextToSpeechError):
        await service.synthesize("hello")
    assert len(fallback.calls) == 0


async def test_fallback_tts_service_never_loops_back_to_primary() -> None:
    primary = _FakeProvider(error=_eligible_error())
    fallback = _FakeProvider(error=_eligible_error())
    service = FallbackTTSService(primary=primary, fallback=fallback)
    with pytest.raises(TextToSpeechError):
        await service.synthesize("hello")
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1


async def test_both_tts_failures_preserved_by_synthesize_speech_as_none() -> None:
    # synthesize_speech's job is to turn even a fully-exhausted fallback
    # into a safe "no audio" result -- never raise, never block the text
    # response the caller must still show.
    primary = _FakeProvider(error=_eligible_error())
    fallback = _FakeProvider(error=_eligible_error())
    service = FallbackTTSService(primary=primary, fallback=fallback)
    result = await synthesize_speech(text="hello there", settings=_settings(), provider=service)
    assert result is None


# --- GroqTTSProvider (real class, mocked SDK client -- never a real call) ---


class _FakeBinaryResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data

    async def read(self) -> bytes:
        return self._data


def _patch_groq_speech_create(
    provider: GroqTTSProvider,
    monkeypatch: pytest.MonkeyPatch,
    *,
    audio_bytes: bytes = b"fake-wav-bytes",
) -> dict[str, object]:
    """Replaces the real (never-called-in-tests) Groq SDK request with an
    in-process fake, capturing the exact kwargs passed so tests can assert
    on them (voice, model, response_format) without any network access."""
    captured: dict[str, object] = {}

    async def fake_create(**kwargs: object) -> _FakeBinaryResponse:
        captured.update(kwargs)
        return _FakeBinaryResponse(audio_bytes)

    monkeypatch.setattr(provider._client.audio.speech, "create", fake_create)  # noqa: SLF001
    return captured


def test_default_groq_tts_voice_setting_is_hannah() -> None:
    assert _settings().groq_tts_voice == "hannah"


async def test_groq_tts_provider_uses_configured_default_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = GroqTTSProvider(
        api_key="synthetic-test-key-not-real",
        model="canopylabs/orpheus-v1-english",
        voice="hannah",
        timeout_seconds=5.0,
    )
    captured = _patch_groq_speech_create(provider, monkeypatch)
    await provider.synthesize("hello there")
    assert captured["voice"] == "hannah"


async def test_groq_tts_provider_requests_wav_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = GroqTTSProvider(
        api_key="synthetic-test-key-not-real",
        model="canopylabs/orpheus-v1-english",
        voice="hannah",
        timeout_seconds=5.0,
    )
    captured = _patch_groq_speech_create(provider, monkeypatch)
    await provider.synthesize("hello there")
    assert captured["response_format"] == "wav"


async def test_groq_tts_provider_result_reports_audio_wav_content_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = GroqTTSProvider(
        api_key="synthetic-test-key-not-real",
        model="canopylabs/orpheus-v1-english",
        voice="hannah",
        timeout_seconds=5.0,
    )
    _patch_groq_speech_create(provider, monkeypatch, audio_bytes=b"real-wav-bytes")
    result = await provider.synthesize("hello there")
    assert result.content_type == "audio/wav"
    assert result.audio_bytes == b"real-wav-bytes"
    assert result.provider == "groq"


async def test_groq_tts_provider_rejects_text_over_200_characters_without_calling_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = GroqTTSProvider(
        api_key="synthetic-test-key-not-real",
        model="canopylabs/orpheus-v1-english",
        voice="hannah",
        timeout_seconds=5.0,
    )
    call_count = {"n": 0}

    async def fake_create(**kwargs: object) -> _FakeBinaryResponse:
        call_count["n"] += 1
        return _FakeBinaryResponse(b"should never be reached")

    monkeypatch.setattr(provider._client.audio.speech, "create", fake_create)  # noqa: SLF001

    over_limit_text = "x" * 201
    with pytest.raises(TextToSpeechError) as exc_info:
        await provider.synthesize(over_limit_text)
    assert exc_info.value.eligible_for_fallback is False
    assert exc_info.value.category == "validation"
    assert call_count["n"] == 0


async def test_text_over_200_characters_skips_groq_fallback_and_preserves_text_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Deepgram (primary) fails with an eligible transient error; Groq
    # (fallback) is attempted next but the response text exceeds Groq's
    # 200-character limit -- synthesize_speech must still return None
    # (no audio) rather than truncating the text or crashing, and the
    # underlying Groq API must never actually be called.
    deepgram_like_primary = _FakeProvider(error=_eligible_error(), provider_name="deepgram")
    groq_provider = GroqTTSProvider(
        api_key="synthetic-test-key-not-real",
        model="canopylabs/orpheus-v1-english",
        voice="hannah",
        timeout_seconds=5.0,
    )
    call_count = {"n": 0}

    async def fake_create(**kwargs: object) -> _FakeBinaryResponse:
        call_count["n"] += 1
        return _FakeBinaryResponse(b"should never be reached")

    monkeypatch.setattr(groq_provider._client.audio.speech, "create", fake_create)  # noqa: SLF001

    service = FallbackTTSService(primary=deepgram_like_primary, fallback=groq_provider)
    over_limit_text = "This response is intentionally longer than two hundred characters. " * 4
    assert len(over_limit_text) > 200

    result = await synthesize_speech(
        text=over_limit_text,
        settings=_settings(tts_max_text_length=2000),
        provider=service,
    )
    assert result is None
    assert call_count["n"] == 0


# --- DeepgramTTSProvider (real class, mocked SDK client -- never a real call) ---


def _patch_deepgram_speak_generate(
    provider: DeepgramTTSProvider,
    monkeypatch: pytest.MonkeyPatch,
    *,
    audio_chunks: tuple[bytes, ...] = (b"fake-mp3-chunk-1", b"fake-mp3-chunk-2"),
) -> dict[str, object]:
    """Replaces the real (never-called-in-tests) Deepgram SDK request with
    an in-process fake, captures the exact kwargs passed, and — critically
    — is itself an async *generator* function (matching
    speak.v1.audio.generate's real AsyncIterator[bytes] return shape, not
    a coroutine) so a regression that re-adds an `await` on the call
    itself fails loudly with the same TypeError the real SDK produces,
    rather than silently passing against a too-forgiving fake."""
    captured: dict[str, object] = {}

    async def fake_generate(**kwargs: object):  # noqa: ANN202 -- async generator, not a coroutine
        captured.update(kwargs)
        for chunk in audio_chunks:
            yield chunk

    monkeypatch.setattr(provider._client.speak.v1.audio, "generate", fake_generate)  # noqa: SLF001
    return captured


async def test_deepgram_tts_provider_calls_generate_without_awaiting_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression test: speak.v1.audio.generate is an async generator
    # function, not a coroutine function -- `await`-ing the call itself
    # raises "object async_generator can't be used in 'await' expression".
    # This test's fake is itself a real async generator function (see
    # _patch_deepgram_speak_generate), so it reproduces that failure if
    # the provider ever regresses to awaiting the call.
    provider = DeepgramTTSProvider(
        api_key="synthetic-test-key-not-real", model="aura-2-thalia-en", timeout_seconds=5.0
    )
    _patch_deepgram_speak_generate(provider, monkeypatch)
    result = await provider.synthesize("hello there")
    assert result.audio_bytes == b"fake-mp3-chunk-1fake-mp3-chunk-2"
    assert result.content_type == "audio/mpeg"
    assert result.provider == "deepgram"


async def test_deepgram_tts_provider_never_sends_container_with_mp3_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression test: Deepgram's API rejects `container` when
    # `encoding="mp3"` (mp3 is its own container) with a 400
    # UNSUPPORTED_AUDIO_FORMAT error.
    provider = DeepgramTTSProvider(
        api_key="synthetic-test-key-not-real", model="aura-2-thalia-en", timeout_seconds=5.0
    )
    captured = _patch_deepgram_speak_generate(provider, monkeypatch)
    await provider.synthesize("hello there")
    assert captured["encoding"] == "mp3"
    assert "container" not in captured
