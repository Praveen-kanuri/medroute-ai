"""HTTP-level tests for POST /api/v1/voice/transcribe.

Never calls Groq: every test either overrides get_speech_to_text_provider
with a fake in-process provider, or (for the "not configured" case)
deliberately leaves it un-overridden against a Settings override with no
GROQ_API_KEY, which fails fast before any network attempt is possible.
"""

import logging
import socket

import pytest
from fastapi.testclient import TestClient

from app.api.v1.voice import get_speech_to_text_provider
from app.config.settings import Settings, get_settings
from app.main import app
from app.providers.speech_to_text.base import SpeechToTextProvider, TranscriptionResult

ENDPOINT = "/api/v1/voice/transcribe"

SYNTHETIC_FILENAME = "synthetic-marker-clip-9f21ab.wav"
SYNTHETIC_TRANSCRIPT = "synthetic-marker-transcript-3d8c7e"


class _FakeProvider(SpeechToTextProvider):
    def __init__(self, *, result: str | None = None, error: Exception | None = None):
        self._result = result if result is not None else SYNTHETIC_TRANSCRIPT
        self._error = error

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str | None = None,
        content_type: str | None = None,
        language: str | None = None,
    ) -> TranscriptionResult:
        if self._error is not None:
            raise self._error
        return TranscriptionResult(
            text=self._result,
            provider="fake",
            model="fake-stt-model",
            language=language,
            duration_seconds=None,
        )


def _override_provider(provider: SpeechToTextProvider) -> None:
    app.dependency_overrides[get_speech_to_text_provider] = lambda: provider


def _clear_overrides() -> None:
    app.dependency_overrides.pop(get_speech_to_text_provider, None)
    app.dependency_overrides.pop(get_settings, None)


def test_successful_transcription_returns_typed_response() -> None:
    _override_provider(_FakeProvider(result="patient reports intermittent chest tightness"))
    try:
        with TestClient(app) as client:
            response = client.post(
                ENDPOINT,
                files={"file": ("clip.wav", b"0123456789", "audio/wav")},
                data={"language": "en"},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["transcript"] == "patient reports intermittent chest tightness"
        assert body["language"] == "en"
        assert body["status"] == "completed"
        assert body["model"]
        assert body["transcription_id"]
    finally:
        _clear_overrides()


def test_empty_upload_returns_422() -> None:
    _override_provider(_FakeProvider())
    try:
        with TestClient(app) as client:
            response = client.post(
                ENDPOINT,
                files={"file": ("clip.wav", b"", "audio/wav")},
            )
        assert response.status_code == 422
    finally:
        _clear_overrides()


def test_unsupported_file_type_returns_422() -> None:
    _override_provider(_FakeProvider())
    try:
        with TestClient(app) as client:
            response = client.post(
                ENDPOINT,
                files={"file": ("clip.ogg", b"0123456789", "audio/ogg")},
            )
        assert response.status_code == 422
    finally:
        _clear_overrides()


def test_oversized_file_returns_413() -> None:
    _override_provider(_FakeProvider())
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, voice_max_upload_bytes=5
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                ENDPOINT,
                files={"file": ("clip.wav", b"0123456789", "audio/wav")},
            )
        assert response.status_code == 413
    finally:
        _clear_overrides()


def test_missing_provider_configuration_returns_503() -> None:
    # Deliberately do NOT override get_speech_to_text_provider: this exercises
    # the real build_speech_to_text_provider() path, which must fail fast when
    # *neither* Deepgram nor Groq is configured, rather than attempting any
    # network call.
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, groq_api_key=None, deepgram_api_key=None
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                ENDPOINT,
                files={"file": ("clip.wav", b"0123456789", "audio/wav")},
            )
        assert response.status_code == 503
    finally:
        _clear_overrides()


def test_provider_failure_returns_503() -> None:
    _override_provider(_FakeProvider(error=RuntimeError("synthetic provider failure")))
    try:
        with TestClient(app) as client:
            response = client.post(
                ENDPOINT,
                files={"file": ("clip.wav", b"0123456789", "audio/wav")},
            )
        assert response.status_code == 503
    finally:
        _clear_overrides()


def test_provider_empty_transcript_returns_503_not_fabricated() -> None:
    _override_provider(_FakeProvider(result="   "))
    try:
        with TestClient(app) as client:
            response = client.post(
                ENDPOINT,
                files={"file": ("clip.wav", b"0123456789", "audio/wav")},
            )
        assert response.status_code == 503
    finally:
        _clear_overrides()


def test_endpoint_makes_no_network_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    # socket.create_connection is the higher-level primitive HTTP clients use
    # to reach a remote host:port. Patching the raw socket.socket constructor
    # instead would also break asyncio's own internal self-pipe plumbing on
    # Windows, which is unrelated to whether *our* code made an outbound call.
    def _forbidden_create_connection(*args: object, **kwargs: object) -> None:
        raise AssertionError("voice endpoint attempted to open an outbound network connection")

    monkeypatch.setattr(socket, "create_connection", _forbidden_create_connection)
    _override_provider(_FakeProvider())
    try:
        with TestClient(app) as client:
            response = client.post(
                ENDPOINT,
                files={"file": ("clip.wav", b"0123456789", "audio/wav")},
            )
        assert response.status_code == 200
    finally:
        _clear_overrides()


def test_no_sensitive_content_in_captured_logs(caplog: pytest.LogCaptureFixture) -> None:
    _override_provider(_FakeProvider(result=SYNTHETIC_TRANSCRIPT))
    try:
        with caplog.at_level(logging.DEBUG):
            with TestClient(app) as client:
                response = client.post(
                    ENDPOINT,
                    files={"file": (SYNTHETIC_FILENAME, b"0123456789", "audio/wav")},
                )
        assert response.status_code == 200
        assert SYNTHETIC_FILENAME not in caplog.text
        assert SYNTHETIC_TRANSCRIPT not in caplog.text
        assert b"0123456789" not in caplog.text.encode()
    finally:
        _clear_overrides()


def test_no_secrets_in_response_body() -> None:
    _override_provider(_FakeProvider())
    try:
        with TestClient(app) as client:
            response = client.post(
                ENDPOINT,
                files={"file": ("clip.wav", b"0123456789", "audio/wav")},
            )
        assert "postgresql" not in response.text
        assert "groq_api_key" not in response.text.lower()
    finally:
        _clear_overrides()


def test_voice_route_registered_in_openapi() -> None:
    with TestClient(app) as client:
        openapi = client.get("/openapi.json").json()
    assert "/api/v1/voice/transcribe" in openapi["paths"]
    assert "post" in openapi["paths"]["/api/v1/voice/transcribe"]


def test_existing_navigate_endpoint_still_registered() -> None:
    with TestClient(app) as client:
        openapi = client.get("/openapi.json").json()
    assert "/api/v1/navigate" in openapi["paths"]


def test_existing_navigate_flow_unaffected_by_voice_changes() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/navigate",
            json={
                "symptoms": ["chest pain", "heart palpitations"],
                "duration": {"value": 2, "unit": "days"},
            },
        )
    assert response.status_code == 200
    assert response.json()["routing"]["specialty_slug"] == "cardiology"
