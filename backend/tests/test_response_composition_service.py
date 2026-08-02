"""Unit tests for the Phase 2B conversational response composition
service.

No database, no network, no real model call — the optional Groq path is
exercised only with a monkeypatched stand-in, never the real SDK/network.
"""

import pytest

from app.config.settings import Settings
from app.safety.constants import EMERGENCY_SAFETY_MESSAGE
from app.services.response_composition_service import FORBIDDEN_TERMS, compose_response


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


async def test_emergency_returns_safety_message_verbatim() -> None:
    text = await compose_response(
        is_emergency=True,
        missing_fields=[],
        clarification_questions=[],
        routing=None,
        provider_search=None,
        settings=_settings(),
    )
    assert text == EMERGENCY_SAFETY_MESSAGE


async def test_missing_fields_produces_a_question_based_response() -> None:
    text = await compose_response(
        is_emergency=False,
        missing_fields=["duration"],
        clarification_questions=["How long have you been experiencing this concern?"],
        routing=None,
        provider_search=None,
        settings=_settings(),
    )
    assert "How long have you been experiencing this concern?" in text


async def test_unmatched_routing_is_explicit_and_non_diagnostic() -> None:
    text = await compose_response(
        is_emergency=False,
        missing_fields=[],
        clarification_questions=[],
        routing={"specialty_slug": None},
        provider_search=None,
        settings=_settings(),
    )
    assert "not a diagnosis" in text
    assert "navigation" in text.lower()


@pytest.mark.parametrize(
    "greeting", ["hi", "Hello", "hey there", "How are you?", "  hello  ", "Hi!", "good morning"]
)
async def test_greeting_gets_a_friendlier_unmatched_message(greeting: str) -> None:
    text = await compose_response(
        is_emergency=False,
        missing_fields=[],
        clarification_questions=[],
        routing={"specialty_slug": None},
        provider_search=None,
        settings=_settings(),
        concern_text=greeting,
    )
    assert "try describing a symptom" in text.lower()
    assert "could not match your concern" not in text
    assert "not a diagnosis" in text


async def test_non_greeting_unmatched_text_keeps_original_message() -> None:
    # A real (if uncommon) concern that just doesn't hit any catalog
    # keyword must still get the original, more actionable message — not
    # be mistaken for a greeting.
    text = await compose_response(
        is_emergency=False,
        missing_fields=[],
        clarification_questions=[],
        routing={"specialty_slug": None},
        provider_search=None,
        settings=_settings(),
        concern_text="my elbow makes a strange clicking noise",
    )
    assert "could not match your concern" in text


async def test_omitted_concern_text_behaves_like_before() -> None:
    text = await compose_response(
        is_emergency=False,
        missing_fields=[],
        clarification_questions=[],
        routing={"specialty_slug": None},
        provider_search=None,
        settings=_settings(),
    )
    assert "could not match your concern" in text


async def test_routed_with_providers_mentions_specialty_and_count() -> None:
    text = await compose_response(
        is_emergency=False,
        missing_fields=[],
        clarification_questions=[],
        routing={"specialty_slug": "cardiology", "specialty_display_name": "Cardiology"},
        provider_search={"results": [{"npi": "1"}, {"npi": "2"}]},
        settings=_settings(),
    )
    assert "Cardiology" in text
    assert "2" in text
    assert "not a diagnosis" in text


async def test_routed_without_providers_says_so() -> None:
    text = await compose_response(
        is_emergency=False,
        missing_fields=[],
        clarification_questions=[],
        routing={"specialty_slug": "cardiology", "specialty_display_name": "Cardiology"},
        provider_search={"results": []},
        settings=_settings(),
    )
    assert "Cardiology" in text
    assert "No matching providers" in text


@pytest.mark.parametrize(
    ("is_emergency", "routing", "provider_search"),
    [
        (True, None, None),
        (False, None, None),
        (False, {"specialty_slug": None}, None),
        (
            False,
            {"specialty_slug": "cardiology", "specialty_display_name": "Cardiology"},
            {"results": [{"npi": "1"}]},
        ),
    ],
)
async def test_response_never_contains_forbidden_terms(
    is_emergency: bool, routing: dict | None, provider_search: dict | None
) -> None:
    text = await compose_response(
        is_emergency=is_emergency,
        missing_fields=[],
        clarification_questions=[],
        routing=routing,
        provider_search=provider_search,
        settings=_settings(),
    )
    # The required safe disclaimer phrase "not a diagnosis" is expected and
    # scrubbed before checking — see _validate_model_response_text's exact
    # same exception. Anything else in FORBIDDEN_TERMS must still be absent.
    scrubbed = text.lower().replace("not a diagnosis", "")
    assert not any(term in scrubbed for term in FORBIDDEN_TERMS)


async def test_media_analysis_note_included_when_missing_fields() -> None:
    text = await compose_response(
        is_emergency=False,
        missing_fields=["duration"],
        clarification_questions=["How long have you been experiencing this concern?"],
        routing=None,
        provider_search=None,
        settings=_settings(),
        media_analysis_note="Reviewed the uploaded image; see the visual observations below.",
    )
    assert "Reviewed the uploaded image" in text
    assert "How long have you been experiencing this concern?" in text


async def test_vision_observations_summary_included_in_routed_response() -> None:
    text = await compose_response(
        is_emergency=False,
        missing_fields=[],
        clarification_questions=[],
        routing={"specialty_slug": "dermatology", "specialty_display_name": "Dermatology"},
        provider_search={"results": []},
        settings=_settings(),
        media_analysis_note="Reviewed the uploaded image; see the visual observations below.",
        vision_observations=[
            {
                "visual_description": "mild redness on the forearm",
                "visible_attributes": ["redness"],
            }
        ],
    )
    assert "mild redness on the forearm" in text
    assert "Dermatology" in text
    assert "not a diagnosis" in text


async def test_no_media_params_unchanged_response() -> None:
    # Omitting media_analysis_note/vision_observations entirely must behave
    # exactly like Phase 2B (no Phase 2C regression on text/voice-only turns).
    text = await compose_response(
        is_emergency=False,
        missing_fields=[],
        clarification_questions=[],
        routing={"specialty_slug": "cardiology", "specialty_display_name": "Cardiology"},
        provider_search={"results": []},
        settings=_settings(),
    )
    assert "Cardiology" in text
    assert "Visible findings" not in text


async def test_groq_mode_without_api_key_falls_back_to_deterministic() -> None:
    text = await compose_response(
        is_emergency=False,
        missing_fields=[],
        clarification_questions=[],
        routing={"specialty_slug": "cardiology", "specialty_display_name": "Cardiology"},
        provider_search={"results": []},
        settings=_settings(response_mode="groq"),
    )
    assert "Cardiology" in text


async def test_groq_mode_falls_back_when_client_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class _RaisingGroq:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated network failure — never a real call")

    monkeypatch.setattr("groq.AsyncGroq", _RaisingGroq)

    text = await compose_response(
        is_emergency=False,
        missing_fields=[],
        clarification_questions=[],
        routing={"specialty_slug": "cardiology", "specialty_display_name": "Cardiology"},
        provider_search={"results": []},
        settings=_settings(response_mode="groq", groq_api_key="fake-test-key-not-real"),
    )
    assert "Cardiology" in text
    assert "No matching providers" in text


async def test_groq_mode_rejects_response_with_forbidden_term(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeMessage:
        content = '{"response_text": "This looks like a treatment plan for your diagnosis."}'

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeResponse:
        choices = [_FakeChoice()]

    class _FakeCompletions:
        async def create(self, *args: object, **kwargs: object) -> _FakeResponse:
            return _FakeResponse()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeAsyncGroq:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.chat = _FakeChat()

    monkeypatch.setattr("groq.AsyncGroq", _FakeAsyncGroq)

    text = await compose_response(
        is_emergency=False,
        missing_fields=[],
        clarification_questions=[],
        routing={"specialty_slug": "cardiology", "specialty_display_name": "Cardiology"},
        provider_search={"results": []},
        settings=_settings(response_mode="groq", groq_api_key="fake-test-key-not-real"),
    )
    # Rejected for containing forbidden terms -> falls back to deterministic.
    assert "Cardiology" in text
    assert "diagnosis" not in text.lower() or "not a diagnosis" in text.lower()


async def test_groq_mode_accepts_valid_rephrasing(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeMessage:
        content = '{"response_text": "Cardiology looks like a good fit. Not a diagnosis."}'

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeResponse:
        choices = [_FakeChoice()]

    class _FakeCompletions:
        async def create(self, *args: object, **kwargs: object) -> _FakeResponse:
            return _FakeResponse()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeAsyncGroq:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.chat = _FakeChat()

    monkeypatch.setattr("groq.AsyncGroq", _FakeAsyncGroq)

    text = await compose_response(
        is_emergency=False,
        missing_fields=[],
        clarification_questions=[],
        routing={"specialty_slug": "cardiology", "specialty_display_name": "Cardiology"},
        provider_search={"results": []},
        settings=_settings(response_mode="groq", groq_api_key="fake-test-key-not-real"),
    )
    assert text == "Cardiology looks like a good fit. Not a diagnosis."
