"""HTTP-level tests for POST /api/v1/intake/validate.

These tests never call any external model, media, speech, or medical API —
the endpoint itself makes no network calls at all.
"""

import logging
import socket

import pytest
from fastapi.testclient import TestClient

from app.main import app

ENDPOINT = "/api/v1/intake/validate"

SYNTHETIC_SYMPTOM = "unique-synthetic-symptom-marker-9f3c2b"
SYNTHETIC_TRANSCRIPT = "unique-synthetic-transcript-marker-7a1e4d"
SYNTHETIC_IMAGE_URL = "https://media.example.org/intake/unique-synthetic-marker-4b8f1a.jpg"


def test_text_only_request_returns_200() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            json={
                "symptoms": ["knee pain"],
                "duration": {"value": 3, "unit": "days"},
            },
        )
    assert response.status_code == 200
    assert response.json()["status"] == "ready_for_multimodal_processing"


def test_voice_only_request_returns_200() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            json={
                "voice_input": {"transcript": "synthetic transcript text"},
                "duration": {"value": 1, "unit": "hours"},
            },
        )
    assert response.status_code == 200
    assert response.json()["multimodal_status"]["voice_received"] is True


def test_vision_only_request_returns_200() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            json={
                "vision_inputs": {"image_urls": ["https://media.example.org/intake/a.jpg"]},
                "duration": {"value": 1, "unit": "weeks"},
            },
        )
    assert response.status_code == 200
    assert response.json()["multimodal_status"]["vision_received"] is True


def test_mixed_modality_request_returns_200() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            json={
                "symptoms": ["knee pain"],
                "voice_input": {"transcript": "synthetic transcript text"},
                "vision_inputs": {"image_urls": ["https://media.example.org/intake/a.jpg"]},
                "duration": {"value": 1, "unit": "days"},
            },
        )
    assert response.status_code == 200
    body = response.json()
    status = body["multimodal_status"]
    assert (status["text_received"], status["voice_received"], status["vision_received"]) == (
        True,
        True,
        True,
    )


def test_emergency_request_returns_200_with_emergency_status() -> None:
    with TestClient(app) as client:
        response = client.post(ENDPOINT, json={"emergency_concern": True})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "emergency"
    assert "911" in body["safety_message"]


def test_needs_clarification_response() -> None:
    with TestClient(app) as client:
        response = client.post(ENDPOINT, json={})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "needs_clarification"
    assert body["missing_fields"] == ["concern", "duration"]


def test_ready_for_multimodal_processing_response() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            json={"symptoms": ["knee pain"], "duration": {"value": 3, "unit": "days"}},
        )
    assert response.status_code == 200
    assert response.json()["status"] == "ready_for_multimodal_processing"


def test_invalid_request_returns_422() -> None:
    with TestClient(app) as client:
        response = client.post(ENDPOINT, json={"duration": {"value": -1, "unit": "days"}})
    assert response.status_code == 422


def test_unknown_field_returns_422() -> None:
    with TestClient(app) as client:
        response = client.post(ENDPOINT, json={"unexpected_field": "value"})
    assert response.status_code == 422


def test_oversized_text_returns_422() -> None:
    with TestClient(app) as client:
        response = client.post(ENDPOINT, json={"main_concern": "x" * 500})
    assert response.status_code == 422


def test_invalid_url_returns_422() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            json={"vision_inputs": {"image_urls": ["http://media.example.org/a.jpg"]}},
        )
    assert response.status_code == 422


def test_existing_health_endpoint_still_works() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_existing_specialties_endpoint_route_still_registered() -> None:
    with TestClient(app) as client:
        openapi = client.get("/openapi.json").json()
    assert "/api/v1/specialties" in openapi["paths"]


def test_intake_route_registered_in_openapi() -> None:
    with TestClient(app) as client:
        openapi = client.get("/openapi.json").json()
    assert "/api/v1/intake/validate" in openapi["paths"]
    assert "post" in openapi["paths"]["/api/v1/intake/validate"]


def test_no_secrets_in_response_body() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            json={"symptoms": ["knee pain"], "duration": {"value": 1, "unit": "days"}},
        )
    assert "postgresql" not in response.text
    assert "DATABASE_URL" not in response.text
    assert "password" not in response.text.lower()


def test_endpoint_makes_no_network_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    # socket.create_connection is the higher-level primitive HTTP clients use
    # to reach a remote host:port. Patching the raw socket.socket constructor
    # instead would also break asyncio's own internal self-pipe plumbing on
    # Windows, which is unrelated to whether *our* code made an outbound call.
    def _forbidden_create_connection(*args: object, **kwargs: object) -> None:
        raise AssertionError("intake endpoint attempted to open an outbound network connection")

    monkeypatch.setattr(socket, "create_connection", _forbidden_create_connection)

    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            json={"symptoms": ["knee pain"], "duration": {"value": 1, "unit": "days"}},
        )
    assert response.status_code == 200


def test_no_synthetic_symptom_text_in_captured_logs(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        with TestClient(app) as client:
            response = client.post(
                ENDPOINT,
                json={
                    "symptoms": [SYNTHETIC_SYMPTOM],
                    "duration": {"value": 1, "unit": "days"},
                },
            )
    assert response.status_code == 200
    assert SYNTHETIC_SYMPTOM not in caplog.text


def test_no_synthetic_transcript_text_in_captured_logs(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        with TestClient(app) as client:
            response = client.post(
                ENDPOINT,
                json={
                    "voice_input": {"transcript": SYNTHETIC_TRANSCRIPT},
                    "duration": {"value": 1, "unit": "days"},
                },
            )
    assert response.status_code == 200
    assert SYNTHETIC_TRANSCRIPT not in caplog.text


def test_no_media_urls_in_captured_logs(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        with TestClient(app) as client:
            response = client.post(
                ENDPOINT,
                json={
                    "vision_inputs": {"image_urls": [SYNTHETIC_IMAGE_URL]},
                    "duration": {"value": 1, "unit": "days"},
                },
            )
    assert response.status_code == 200
    assert SYNTHETIC_IMAGE_URL not in caplog.text


def test_log_line_contains_only_safe_operational_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="app.api.v1.intake"):
        with TestClient(app) as client:
            response = client.post(
                ENDPOINT,
                json={
                    "symptoms": [SYNTHETIC_SYMPTOM],
                    "voice_input": {"transcript": SYNTHETIC_TRANSCRIPT},
                    "vision_inputs": {"image_urls": [SYNTHETIC_IMAGE_URL]},
                    "duration": {"value": 1, "unit": "days"},
                },
            )
    assert response.status_code == 200
    intake_records = [r for r in caplog.records if r.name == "app.api.v1.intake"]
    assert len(intake_records) == 1
    message = intake_records[0].getMessage()
    assert "intake_validated" in message
    assert SYNTHETIC_SYMPTOM not in message
    assert SYNTHETIC_TRANSCRIPT not in message
    assert SYNTHETIC_IMAGE_URL not in message
