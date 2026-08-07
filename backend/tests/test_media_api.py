"""HTTP-level tests for POST /api/v1/media/analyze.

Never calls Groq or Deepgram — response_mode/routing_mode default to
deterministic and no API key is configured, so vision analysis always
takes its "not configured" safe-fallback path at this layer (populated
vision_observations, from a fake provider, are proven at the graph level
in test_conversation_graph.py — mirroring how Deepgram TTS audio is never
exercised at this HTTP layer either, only its "not configured" path).
"""

import json
import logging
import socket

import pytest
from fastapi.testclient import TestClient

from app.config.settings import Settings, get_settings
from app.main import app
from tests.media_fixtures import make_jpeg_bytes, make_mp4_bytes

ENDPOINT = "/api/v1/media/analyze"

SYNTHETIC_FILENAME = "synthetic-marker-clip-9f21ab.jpg"
UNMATCHED_INTAKE = json.dumps(
    {"symptoms": ["zzz qqq unrelated words"], "duration": {"value": 1, "unit": "days"}}
)


def _clear_overrides() -> None:
    app.dependency_overrides.pop(get_settings, None)


def test_valid_image_upload_returns_conversation_response() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            files={"file": ("clip.jpg", make_jpeg_bytes(), "image/jpeg")},
            data={"intake_json": UNMATCHED_INTAKE},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready_for_multimodal_processing"
    assert body["thread_id"]
    assert body["vision_observations"] == []
    assert body["media_analysis_note"] is not None
    assert "not available" in body["media_analysis_note"].lower()


def test_valid_video_upload_returns_conversation_response() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            files={"file": ("clip.mp4", make_mp4_bytes(), "video/mp4")},
            data={"intake_json": UNMATCHED_INTAKE},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready_for_multimodal_processing"


def test_empty_upload_returns_422() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            files={"file": ("clip.jpg", b"", "image/jpeg")},
            data={"intake_json": UNMATCHED_INTAKE},
        )
    assert response.status_code == 422


def test_unsupported_file_type_returns_422() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            files={"file": ("clip.txt", b"just plain text, not media", "text/plain")},
            data={"intake_json": UNMATCHED_INTAKE},
        )
    assert response.status_code == 422


def test_malformed_image_returns_422() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            files={"file": ("clip.jpg", b"\xff\xd8\xffnotarealjpeg", "image/jpeg")},
            data={"intake_json": UNMATCHED_INTAKE},
        )
    assert response.status_code == 422


def test_mime_and_extension_spoofing_still_validated_by_signature() -> None:
    # Claims to be a PDF via filename/content-type, but the bytes are a
    # real JPEG signature followed by garbage — still correctly rejected
    # as malformed (not silently accepted because of the spoofed labels,
    # and not silently misclassified either).
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            files={
                "file": ("clip.pdf", b"\xff\xd8\xffnotarealjpeg", "application/pdf"),
            },
            data={"intake_json": UNMATCHED_INTAKE},
        )
    assert response.status_code == 422


def test_oversized_image_returns_413() -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, image_max_upload_bytes=5
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                ENDPOINT,
                files={"file": ("clip.jpg", make_jpeg_bytes(), "image/jpeg")},
                data={"intake_json": UNMATCHED_INTAKE},
            )
        assert response.status_code == 413
    finally:
        _clear_overrides()


def test_video_duration_exceeded_returns_422() -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, video_max_duration_seconds=0.5
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                ENDPOINT,
                files={"file": ("clip.mp4", make_mp4_bytes(fps=10.0, frame_count=20), "video/mp4")},
                data={"intake_json": UNMATCHED_INTAKE},
            )
        assert response.status_code == 422
    finally:
        _clear_overrides()


def test_video_dimension_exceeded_returns_422() -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, video_max_dimension_px=16
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                ENDPOINT,
                files={
                    "file": ("clip.mp4", make_mp4_bytes(width=64, height=48), "video/mp4"),
                },
                data={"intake_json": UNMATCHED_INTAKE},
            )
        assert response.status_code == 422
    finally:
        _clear_overrides()


def test_invalid_intake_json_returns_422() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            files={"file": ("clip.jpg", make_jpeg_bytes(), "image/jpeg")},
            data={"intake_json": "{not valid json"},
        )
    assert response.status_code == 422


def test_invalid_intake_schema_returns_422() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            files={"file": ("clip.jpg", make_jpeg_bytes(), "image/jpeg")},
            data={"intake_json": json.dumps({"preferred_specialty": "Not A Valid Slug!"})},
        )
    assert response.status_code == 422


def test_emergency_precedence_unchanged_with_media_upload() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            files={"file": ("clip.jpg", make_jpeg_bytes(), "image/jpeg")},
            data={"intake_json": json.dumps({"emergency_concern": True})},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "emergency"
    assert body["routing"] is None
    assert body["provider_search"] is None
    assert "911" in body["response_text"]


def test_endpoint_makes_no_external_network_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    def _forbidden_create_connection(*args: object, **kwargs: object) -> None:
        raise AssertionError("media endpoint attempted to open an outbound network connection")

    monkeypatch.setattr(socket, "create_connection", _forbidden_create_connection)
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            files={"file": ("clip.jpg", make_jpeg_bytes(), "image/jpeg")},
            data={"intake_json": UNMATCHED_INTAKE},
        )
    assert response.status_code == 200


def test_no_sensitive_content_in_captured_logs(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        with TestClient(app) as client:
            response = client.post(
                ENDPOINT,
                files={"file": (SYNTHETIC_FILENAME, make_jpeg_bytes(), "image/jpeg")},
                data={"intake_json": UNMATCHED_INTAKE},
            )
    assert response.status_code == 200
    assert SYNTHETIC_FILENAME not in caplog.text
    assert "zzz qqq unrelated words" not in caplog.text


def test_media_route_registered_in_openapi() -> None:
    with TestClient(app) as client:
        openapi = client.get("/openapi.json").json()
    assert ENDPOINT in openapi["paths"]
    assert "post" in openapi["paths"][ENDPOINT]


def test_existing_converse_and_navigate_endpoints_still_registered() -> None:
    with TestClient(app) as client:
        openapi = client.get("/openapi.json").json()
    assert "/api/v1/converse" in openapi["paths"]
    assert "/api/v1/navigate" in openapi["paths"]
