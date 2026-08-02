"""HTTP-level tests for POST /api/v1/converse (Phase 2B).

Never calls Groq or Deepgram — routing_mode/response_mode default to
"deterministic", and every test either supplies no generate_speech (no
provider is ever attempted) or, for the one test that explicitly checks
for no network calls, overrides get_settings to guarantee neither speech
provider is configured — this must never depend on whether the
developer's local .env file happens to have real API keys set.
"""

import logging
import socket

import pytest
from fastapi.testclient import TestClient

from app.config.settings import Settings, get_settings
from app.main import app

ENDPOINT = "/api/v1/converse"

SYNTHETIC_TRANSCRIPT_MARKER = "synthetic-marker-transcript-9d3e7c"


def test_fresh_turn_unmatched_routing_returns_response_text() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            json={
                "intake": {
                    "main_concern": "zzz qqq unrelated words",
                    "duration": {"value": 1, "unit": "days"},
                }
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready_for_multimodal_processing"
    assert body["routing"]["method"] == "unmatched"
    assert body["provider_search"] is None
    assert body["response_text"]
    assert body["audio"] is None
    assert body["thread_id"]


def test_fresh_turn_missing_duration_returns_typed_missing_fields() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT, json={"intake": {"main_concern": "zzz qqq unrelated words"}}
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "needs_clarification"
    assert body["missing_fields"] == ["duration"]
    assert body["clarification_questions"]
    assert body["routing"] is None
    assert body["thread_id"]


def test_resume_with_clarification_answer_reaches_routing() -> None:
    with TestClient(app) as client:
        first = client.post(
            ENDPOINT,
            json={
                "intake": {
                    "voice_input": {
                        "transcript": "chest pain and heart palpitations",
                        "language": "en",
                    }
                }
            },
        )
        assert first.status_code == 200
        first_body = first.json()
        assert first_body["status"] == "needs_clarification"
        thread_id = first_body["thread_id"]

        second = client.post(
            ENDPOINT,
            json={
                "thread_id": thread_id,
                "clarification_answer": {"duration": {"value": 3, "unit": "days"}},
            },
        )
    assert second.status_code == 200
    second_body = second.json()
    assert second_body["thread_id"] == thread_id
    assert second_body["status"] == "ready_for_multimodal_processing"
    assert second_body["routing"]["specialty_slug"] == "cardiology"


def test_emergency_short_circuits_before_routing() -> None:
    with TestClient(app) as client:
        response = client.post(ENDPOINT, json={"intake": {"emergency_concern": True}})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "emergency"
    assert body["routing"] is None
    assert body["provider_search"] is None
    assert "911" in body["response_text"]


def test_specialty_routing_and_provider_search_invoked() -> None:
    # Same confirmed-working NPPES fixture example used by
    # test_navigation_api.py and test_conversation_graph.py.
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            json={
                "intake": {
                    "symptoms": ["annual checkup"],
                    "location": {"city": "Springfield", "state": "CA"},
                    "duration": {"value": 1, "unit": "days"},
                }
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["routing"]["specialty_slug"] == "family-medicine"
    assert body["provider_search"] is not None
    assert len(body["provider_search"]["results"]) == 1
    assert "Family Medicine" in body["response_text"]


def test_clarification_answer_without_thread_id_returns_422() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            json={"clarification_answer": {"duration": {"value": 1, "unit": "days"}}},
        )
    assert response.status_code == 422


def test_neither_intake_nor_clarification_answer_returns_422() -> None:
    with TestClient(app) as client:
        response = client.post(ENDPOINT, json={})
    assert response.status_code == 422


def test_both_intake_and_clarification_answer_returns_422() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            json={
                "thread_id": "some-thread",
                "intake": {"main_concern": "something"},
                "clarification_answer": {"duration": {"value": 1, "unit": "days"}},
            },
        )
    assert response.status_code == 422


def test_unknown_thread_id_resume_returns_404() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            json={
                "thread_id": "does-not-exist-thread-id",
                "clarification_answer": {"duration": {"value": 1, "unit": "days"}},
            },
        )
    assert response.status_code == 404


def test_invalid_request_body_returns_422() -> None:
    with TestClient(app) as client:
        response = client.post(ENDPOINT, json={"unexpected_field": "value"})
    assert response.status_code == 422


def test_endpoint_makes_no_external_network_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force neither speech provider to be configured, regardless of what
    # the developer's local .env file happens to contain -- this test's
    # "no network calls" guarantee must never depend on ambient
    # environment state. socket.create_connection is additionally patched
    # as a second line of defense for any synchronous connection attempt
    # (it does not intercept httpx's async connections, which is exactly
    # why the settings override above is the real guarantee here).
    def _forbidden_create_connection(*args: object, **kwargs: object) -> None:
        raise AssertionError("converse endpoint attempted an outbound network connection")

    monkeypatch.setattr(socket, "create_connection", _forbidden_create_connection)
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, groq_api_key=None, deepgram_api_key=None
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                ENDPOINT,
                json={
                    "intake": {
                        "main_concern": "zzz qqq unrelated words",
                        "duration": {"value": 1, "unit": "days"},
                    },
                    "generate_speech": True,
                },
            )
        assert response.status_code == 200
        assert response.json()["audio"] is None
    finally:
        app.dependency_overrides.pop(get_settings, None)


def test_no_secrets_in_response_body() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            json={
                "intake": {
                    "main_concern": "zzz qqq unrelated words",
                    "duration": {"value": 1, "unit": "days"},
                }
            },
        )
    assert "postgresql" not in response.text
    assert "groq_api_key" not in response.text.lower()
    assert "deepgram_api_key" not in response.text.lower()


def test_no_synthetic_transcript_text_in_captured_logs(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        with TestClient(app) as client:
            response = client.post(
                ENDPOINT,
                json={
                    "intake": {
                        "voice_input": {
                            "transcript": SYNTHETIC_TRANSCRIPT_MARKER,
                            "language": "en",
                        },
                        "duration": {"value": 1, "unit": "days"},
                    }
                },
            )
    assert response.status_code == 200
    assert SYNTHETIC_TRANSCRIPT_MARKER not in caplog.text


def test_existing_endpoints_still_registered_alongside_converse() -> None:
    with TestClient(app) as client:
        openapi = client.get("/openapi.json").json()
    paths = openapi["paths"]
    assert "/api/v1/converse" in paths
    assert "post" in paths["/api/v1/converse"]
    assert "/api/v1/navigate" in paths
    assert "/api/v1/voice/transcribe" in paths
    assert "/api/v1/intake/validate" in paths
    assert "/api/v1/specialties" in paths


def test_existing_navigate_flow_unaffected() -> None:
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


# --- conversational-intent classification ------------------------------------


def test_greeting_is_classified_and_bypasses_routing() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT, json={"intake": {"voice_input": {"transcript": "Hi", "language": "en"}}}
        )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "greeting"
    assert body["status"] == "ready_for_multimodal_processing"
    assert body["missing_fields"] == []
    assert body["clarification_questions"] == []
    assert body["routing"] is None
    assert body["provider_search"] is None
    assert "How long have you been experiencing this concern?" not in (body["response_text"] or "")


def test_good_evening_greeting_gets_a_time_of_day_reply() -> None:
    with TestClient(app) as client:
        response = client.post(ENDPOINT, json={"intake": {"main_concern": "Good evening"}})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "greeting"
    assert "evening" in body["response_text"].lower()


def test_thank_you_gets_a_thanks_reply() -> None:
    with TestClient(app) as client:
        response = client.post(ENDPOINT, json={"intake": {"main_concern": "Thank you"}})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "greeting"
    assert "welcome" in body["response_text"].lower()


def test_medical_concern_still_reaches_routing_with_medical_concern_intent() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            json={
                "intake": {
                    "symptoms": ["annual checkup"],
                    "location": {"city": "Springfield", "state": "CA"},
                    "duration": {"value": 1, "unit": "days"},
                }
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "medical_concern"
    assert body["routing"]["specialty_slug"] == "family-medicine"


def test_medical_concern_response_acknowledges_the_users_concern() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT,
            json={
                "intake": {
                    "symptoms": ["annual checkup"],
                    "location": {"city": "Springfield", "state": "CA"},
                    "duration": {"value": 1, "unit": "days"},
                }
            },
        )
    body = response.json()
    assert "annual checkup" in body["response_text"]
    assert "not a diagnosis" in body["response_text"]


def test_fresh_turn_clarification_question_sounds_conversational() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT, json={"intake": {"main_concern": "zzz qqq unrelated words"}}
        )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "medical_concern"
    assert body["response_text"].startswith("I can help with that.")
    assert "How long have you been experiencing this concern?" in body["response_text"]


def test_unsupported_or_unclear_intent_when_nothing_is_provided() -> None:
    with TestClient(app) as client:
        response = client.post(ENDPOINT, json={"intake": {}})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "unsupported_or_unclear"
    assert body["status"] == "needs_clarification"


def test_clarification_resume_has_clarification_answer_intent() -> None:
    with TestClient(app) as client:
        first = client.post(
            ENDPOINT,
            json={
                "intake": {
                    "voice_input": {
                        "transcript": "chest pain and heart palpitations",
                        "language": "en",
                    }
                }
            },
        )
        thread_id = first.json()["thread_id"]

        second = client.post(
            ENDPOINT,
            json={
                "thread_id": thread_id,
                "clarification_answer": {"duration": {"value": 3, "unit": "days"}},
            },
        )
    assert second.status_code == 200
    assert second.json()["intent"] == "clarification_answer"


def test_emergency_declared_with_greeting_like_text_is_not_treated_as_a_greeting() -> None:
    with TestClient(app) as client:
        response = client.post(
            ENDPOINT, json={"intake": {"main_concern": "hi", "emergency_concern": True}}
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "emergency"
    assert body["intent"] == "medical_concern"
    assert "911" in body["response_text"]
