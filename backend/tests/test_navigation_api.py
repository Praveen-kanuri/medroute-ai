"""HTTP-level tests for POST /api/v1/navigate.

Never calls Groq, Deepgram, OpenAI, Qdrant, or any other external service —
routing_mode defaults to "deterministic" and no API key is configured.
"""

import logging

import pytest
from fastapi.testclient import TestClient

from app.main import app

ENDPOINT = "/api/v1/navigate"

SYNTHETIC_SYMPTOM = "unique-synthetic-symptom-marker-2f6a9c"


def test_emergency_short_circuits_before_routing() -> None:
    with TestClient(app) as client:
        response = client.post(ENDPOINT, json={"emergency_concern": True})
    assert response.status_code == 200
    body = response.json()
    assert body["intake"]["status"] == "emergency"
    assert body["routing"] is None
    assert body["provider_search"] is None


def test_needs_clarification_short_circuits_before_routing() -> None:
    with TestClient(app) as client:
        response = client.post(ENDPOINT, json={})
    assert response.status_code == 200
    body = response.json()
    assert body["intake"]["status"] == "needs_clarification"
    assert body["routing"] is None
    assert body["provider_search"] is None


def test_ready_intake_triggers_keyword_routing() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            json={
                "symptoms": ["chest pain", "heart palpitations"],
                "duration": {"value": 2, "unit": "days"},
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["intake"]["status"] == "ready_for_multimodal_processing"
    assert body["routing"]["specialty_slug"] == "cardiology"
    assert body["routing"]["method"] == "keyword_match"
    assert body["provider_search"] is not None


def test_preferred_specialty_bypasses_keyword_routing() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            json={
                "main_concern": "text that would not otherwise match anything",
                "duration": {"value": 1, "unit": "days"},
                "preferred_specialty": "dermatology",
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["routing"]["specialty_slug"] == "dermatology"
    assert body["routing"]["method"] == "user_selected"


def test_invalid_preferred_specialty_yields_unmatched_routing() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            json={
                "main_concern": "something",
                "duration": {"value": 1, "unit": "days"},
                "preferred_specialty": "not-a-real-specialty",
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["routing"]["specialty_slug"] is None
    assert body["routing"]["method"] == "unmatched"
    assert body["provider_search"] is None


def test_voice_only_transcript_can_route_to_specialty() -> None:
    # Phase 2A: no symptoms/main_concern at all — only a confirmed voice
    # transcript — must still be able to drive specialty routing.
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            json={
                "duration": {"value": 2, "unit": "days"},
                "voice_input": {
                    "transcript": "chest pain and heart palpitations",
                    "language": "en",
                },
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["intake"]["status"] == "ready_for_multimodal_processing"
    assert body["routing"]["specialty_slug"] == "cardiology"
    assert body["routing"]["method"] == "keyword_match"
    assert body["provider_search"] is not None


def test_voice_transcript_with_explicit_duration_skips_clarification() -> None:
    # Phase 2A clarification-workflow fix: an explicit duration stated in
    # the confirmed transcript itself must satisfy the duration
    # requirement — no structured "duration" field, and no clarification.
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            json={
                "voice_input": {
                    "transcript": "chest pain and heart palpitations for the past three days",
                    "language": "en",
                },
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["intake"]["status"] == "ready_for_multimodal_processing"
    assert "duration" not in body["intake"]["missing_fields"]
    assert body["intake"]["normalized_intake"]["duration"] == {"value": 3, "unit": "days"}
    assert body["routing"]["specialty_slug"] == "cardiology"
    assert body["provider_search"] is not None


def test_voice_transcript_with_ambiguous_duration_still_needs_clarification() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            json={
                "voice_input": {
                    "transcript": "chest pain and heart palpitations for a while",
                    "language": "en",
                },
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["intake"]["status"] == "needs_clarification"
    assert body["intake"]["missing_fields"] == ["duration"]
    assert body["routing"] is None


def test_followup_resubmission_merges_duration_and_preserves_voice_transcript() -> None:
    # Simulates the Streamlit follow-up workflow at the HTTP level: an
    # initial submission with only a confirmed voice transcript (no
    # explicit duration in it) triggers needs_clarification; merging just
    # a "duration" answer into that same payload and resubmitting must
    # preserve the original transcript untouched and reach routing.
    transcript = "chest pain and heart palpitations"
    initial_payload = {
        "emergency_concern": False,
        "voice_input": {"transcript": transcript, "language": "en"},
    }

    with TestClient(app) as client:
        first_response = client.post(ENDPOINT, json=initial_payload)
        assert first_response.status_code == 200
        first_body = first_response.json()
        assert first_body["intake"]["status"] == "needs_clarification"
        assert first_body["intake"]["missing_fields"] == ["duration"]

        merged_payload = {**initial_payload, "duration": {"value": 3, "unit": "days"}}
        second_response = client.post(ENDPOINT, json=merged_payload)

    assert second_response.status_code == 200
    second_body = second_response.json()
    assert second_body["intake"]["status"] == "ready_for_multimodal_processing"
    assert second_body["intake"]["normalized_intake"]["voice_input"]["transcript"] == transcript
    assert second_body["routing"]["specialty_slug"] == "cardiology"
    assert second_body["provider_search"] is not None


def test_combined_text_and_voice_input_routes_correctly() -> None:
    # Both a typed symptom and a confirmed voice transcript are present —
    # routing must consider both, not just one or the other.
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            json={
                "symptoms": ["knee pain"],
                "duration": {"value": 1, "unit": "days"},
                "voice_input": {"transcript": "chest palpitations", "language": "en"},
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["routing"]["specialty_slug"] == "cardiology"
    assert body["routing"]["method"] == "keyword_match"


def test_emergency_precedence_unchanged_with_voice_transcript() -> None:
    # A declared emergency must short-circuit before routing runs, even when
    # the voice transcript itself contains text that would otherwise match a
    # specialty — the transcript is never used to detect or override the
    # emergency, and routing must not run at all.
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            json={
                "emergency_concern": True,
                "voice_input": {
                    "transcript": "chest pain and heart palpitations",
                    "language": "en",
                },
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["intake"]["status"] == "emergency"
    assert body["routing"] is None
    assert body["provider_search"] is None


def test_unmatched_symptoms_skip_provider_search() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            json={
                "main_concern": "zzz qqq unrelated words",
                "duration": {"value": 1, "unit": "days"},
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["routing"]["specialty_slug"] is None
    assert body["provider_search"] is None


def test_vision_input_produces_media_unavailable_note() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            json={
                "symptoms": ["knee pain"],
                "duration": {"value": 1, "unit": "days"},
                "vision_inputs": {"image_urls": ["https://media.example.org/intake/a.jpg"]},
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["media_note"] is not None
    assert "not available" in body["media_note"]
    # Media never blocks text-based routing.
    assert body["routing"]["specialty_slug"] == "orthopaedic-surgery"


def test_no_vision_input_has_no_media_note() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            json={"symptoms": ["chest pain"], "duration": {"value": 1, "unit": "days"}},
        )
    assert response.status_code == 200
    assert response.json()["media_note"] is None


def test_invalid_request_returns_422() -> None:
    with TestClient(app) as client:
        response = client.post(ENDPOINT, json={"duration": {"value": -1, "unit": "days"}})
    assert response.status_code == 422


def test_unknown_field_returns_422() -> None:
    with TestClient(app) as client:
        response = client.post(ENDPOINT, json={"unexpected_field": "value"})
    assert response.status_code == 422


def test_existing_intake_endpoint_still_works() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/intake/validate",
            json={"symptoms": ["knee pain"], "duration": {"value": 1, "unit": "days"}},
        )
    assert response.status_code == 200


def test_existing_specialties_endpoint_still_registered() -> None:
    with TestClient(app) as client:
        openapi = client.get("/openapi.json").json()
    assert "/api/v1/specialties" in openapi["paths"]


def test_navigate_route_registered_in_openapi() -> None:
    with TestClient(app) as client:
        openapi = client.get("/openapi.json").json()
    assert "/api/v1/navigate" in openapi["paths"]
    assert "post" in openapi["paths"]["/api/v1/navigate"]


def test_no_secrets_in_response_body() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            json={"symptoms": ["chest pain"], "duration": {"value": 1, "unit": "days"}},
        )
    assert "postgresql" not in response.text
    assert "DATABASE_URL" not in response.text
    assert "password" not in response.text.lower()
    assert "groq_api_key" not in response.text.lower()


def test_response_never_contains_diagnosis_or_urgency_fields() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            json={"symptoms": ["chest pain"], "duration": {"value": 1, "unit": "days"}},
        )
    dumped_keys = set(response.json().keys())
    forbidden = {"diagnosis", "urgency", "urgency_score", "confidence", "treatment"}
    assert dumped_keys.isdisjoint(forbidden)


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
