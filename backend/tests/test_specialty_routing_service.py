"""Unit tests for Phase 1D deterministic specialty routing.

No database, no network, no real model call — the optional Groq path is
exercised only with a monkeypatched stand-in, never the real SDK/network.
"""

import pytest

from app.config.settings import Settings
from app.services.specialty_routing_service import (
    RoutingMethod,
    route_to_specialty,
)


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_preferred_specialty_bypasses_keyword_matching() -> None:
    result = await route_to_specialty(
        preferred_specialty="dermatology",
        symptoms=["completely unrelated text"],
        main_concern=None,
        settings=_settings(),
    )
    assert result.specialty_slug == "dermatology"
    assert result.method == RoutingMethod.USER_SELECTED


@pytest.mark.asyncio
async def test_preferred_specialty_not_in_catalog_is_unmatched() -> None:
    result = await route_to_specialty(
        preferred_specialty="not-a-real-specialty",
        symptoms=[],
        main_concern=None,
        settings=_settings(),
    )
    assert result.specialty_slug is None
    assert result.method == RoutingMethod.UNMATCHED


@pytest.mark.asyncio
async def test_keyword_match_routes_to_cardiology() -> None:
    result = await route_to_specialty(
        preferred_specialty=None,
        symptoms=["chest pain", "heart palpitations"],
        main_concern=None,
        settings=_settings(),
    )
    assert result.specialty_slug == "cardiology"
    assert result.method == RoutingMethod.KEYWORD_MATCH


@pytest.mark.asyncio
async def test_keyword_match_routes_to_dermatology_from_main_concern() -> None:
    result = await route_to_specialty(
        preferred_specialty=None,
        symptoms=[],
        main_concern="itchy skin rash on my arm",
        settings=_settings(),
    )
    assert result.specialty_slug == "dermatology"


@pytest.mark.asyncio
async def test_keyword_match_routes_leg_swelling_to_internal_medicine() -> None:
    # Phase 3A vertical slice: the leg-swelling protocol relies on this
    # existing deterministic keyword match (no protocol-specific routing
    # code) -- see app/catalog/nucc_specialties.py's internal-medicine
    # keywords.
    result = await route_to_specialty(
        preferred_specialty=None,
        symptoms=[],
        main_concern="My left leg has been swollen for two days.",
        settings=_settings(),
    )
    assert result.specialty_slug == "internal-medicine"
    assert result.method == RoutingMethod.KEYWORD_MATCH


@pytest.mark.asyncio
async def test_no_keyword_overlap_is_unmatched() -> None:
    result = await route_to_specialty(
        preferred_specialty=None,
        symptoms=["zzz qqq unrelated words"],
        main_concern=None,
        settings=_settings(),
    )
    assert result.specialty_slug is None
    assert result.method == RoutingMethod.UNMATCHED


@pytest.mark.asyncio
async def test_empty_input_is_unmatched_without_crashing() -> None:
    result = await route_to_specialty(
        preferred_specialty=None, symptoms=[], main_concern=None, settings=_settings()
    )
    assert result.specialty_slug is None
    assert result.method == RoutingMethod.UNMATCHED


@pytest.mark.asyncio
async def test_result_never_contains_diagnosis_or_urgency_language() -> None:
    result = await route_to_specialty(
        preferred_specialty=None,
        symptoms=["chest pain"],
        main_concern=None,
        settings=_settings(),
    )
    forbidden_terms = ("diagnos", "treatment", "urgent", "urgency", "prescri")
    combined = f"{result.note}".lower()
    assert not any(term in combined for term in forbidden_terms)


@pytest.mark.asyncio
async def test_groq_mode_without_api_key_falls_back_to_deterministic() -> None:
    # groq_configured is False when no key is set, so the Groq path must
    # never even attempt a call.
    result = await route_to_specialty(
        preferred_specialty=None,
        symptoms=["chest pain"],
        main_concern=None,
        settings=_settings(routing_mode="groq"),
    )
    assert result.method == RoutingMethod.KEYWORD_MATCH


@pytest.mark.asyncio
async def test_groq_mode_falls_back_when_client_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class _RaisingGroq:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated network failure — never a real call")

    monkeypatch.setattr("groq.AsyncGroq", _RaisingGroq)

    result = await route_to_specialty(
        preferred_specialty=None,
        symptoms=["chest pain"],
        main_concern=None,
        settings=_settings(routing_mode="groq", groq_api_key="fake-test-key-not-real"),
    )
    # Falls back to deterministic keyword matching rather than raising.
    assert result.method == RoutingMethod.KEYWORD_MATCH
    assert result.specialty_slug == "cardiology"


@pytest.mark.asyncio
async def test_groq_mode_rejects_slug_outside_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeMessage:
        content = '{"specialty_slug": "not-in-the-catalog"}'

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

    result = await route_to_specialty(
        preferred_specialty=None,
        symptoms=["chest pain"],
        main_concern=None,
        settings=_settings(routing_mode="groq", groq_api_key="fake-test-key-not-real"),
    )
    # Invalid/unvalidated slug is rejected -> falls back to deterministic.
    assert result.method == RoutingMethod.KEYWORD_MATCH
    assert result.specialty_slug == "cardiology"


@pytest.mark.asyncio
async def test_voice_transcript_alone_routes_to_specialty() -> None:
    # Phase 2A: a confirmed voice transcript with no typed symptoms/main_concern
    # must still be able to drive deterministic keyword matching.
    result = await route_to_specialty(
        preferred_specialty=None,
        symptoms=[],
        main_concern=None,
        voice_transcript="chest pain and heart palpitations",
        settings=_settings(),
    )
    assert result.specialty_slug == "cardiology"
    assert result.method == RoutingMethod.KEYWORD_MATCH


@pytest.mark.asyncio
async def test_omitted_voice_transcript_behaves_like_text_only() -> None:
    # Explicitly passing voice_transcript=None must be identical to the
    # pre-Phase-2A behavior of omitting the parameter entirely.
    with_default = await route_to_specialty(
        preferred_specialty=None,
        symptoms=["chest pain"],
        main_concern=None,
        settings=_settings(),
    )
    with_explicit_none = await route_to_specialty(
        preferred_specialty=None,
        symptoms=["chest pain"],
        main_concern=None,
        voice_transcript=None,
        settings=_settings(),
    )
    assert with_default == with_explicit_none
    assert with_default.specialty_slug == "cardiology"


@pytest.mark.asyncio
async def test_combined_symptoms_and_voice_transcript_are_scored_together() -> None:
    # symptoms alone ("knee pain") would match orthopaedic-surgery (score 1 on
    # "knee"). A confirmed voice transcript contributing two cardiology
    # keywords ("chest", "palpitations") must combine with, not replace, that
    # signal and win on total keyword-overlap score (2 > 1) — proving the two
    # sources are merged into one token set rather than one silently
    # overriding the other.
    symptoms_only = await route_to_specialty(
        preferred_specialty=None,
        symptoms=["knee pain"],
        main_concern=None,
        settings=_settings(),
    )
    assert symptoms_only.specialty_slug == "orthopaedic-surgery"

    combined = await route_to_specialty(
        preferred_specialty=None,
        symptoms=["knee pain"],
        main_concern=None,
        voice_transcript="chest palpitations",
        settings=_settings(),
    )
    assert combined.specialty_slug == "cardiology"
    assert combined.method == RoutingMethod.KEYWORD_MATCH


@pytest.mark.asyncio
async def test_vision_observation_text_alone_routes_to_specialty() -> None:
    # Phase 2C: flattened, already schema-validated visual-observation text
    # (with no typed symptoms/main_concern/voice transcript at all) must
    # still be able to drive deterministic keyword matching.
    result = await route_to_specialty(
        preferred_specialty=None,
        symptoms=[],
        main_concern=None,
        vision_observation_text="visible itchy rash on the forearm",
        settings=_settings(),
    )
    assert result.specialty_slug == "dermatology"
    assert result.method == RoutingMethod.KEYWORD_MATCH


@pytest.mark.asyncio
async def test_omitted_vision_observation_text_behaves_like_before() -> None:
    with_default = await route_to_specialty(
        preferred_specialty=None,
        symptoms=["chest pain"],
        main_concern=None,
        settings=_settings(),
    )
    with_explicit_none = await route_to_specialty(
        preferred_specialty=None,
        symptoms=["chest pain"],
        main_concern=None,
        vision_observation_text=None,
        settings=_settings(),
    )
    assert with_default == with_explicit_none


@pytest.mark.asyncio
async def test_combined_voice_transcript_and_vision_observation_scored_together() -> None:
    # A voice transcript alone ("knee pain") matches orthopaedic-surgery
    # (score 1). A vision observation contributing two cardiology keywords
    # ("chest", "palpitations") must combine with, not replace, that signal
    # and win on total keyword-overlap score (2 > 1).
    voice_only = await route_to_specialty(
        preferred_specialty=None,
        symptoms=[],
        main_concern=None,
        voice_transcript="knee pain",
        settings=_settings(),
    )
    assert voice_only.specialty_slug == "orthopaedic-surgery"

    combined = await route_to_specialty(
        preferred_specialty=None,
        symptoms=[],
        main_concern=None,
        voice_transcript="knee pain",
        vision_observation_text="chest palpitations",
        settings=_settings(),
    )
    assert combined.specialty_slug == "cardiology"


@pytest.mark.asyncio
async def test_groq_mode_accepts_valid_catalog_slug(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeMessage:
        content = '{"specialty_slug": "dermatology"}'

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

    result = await route_to_specialty(
        preferred_specialty=None,
        symptoms=["chest pain"],  # would deterministically match cardiology
        main_concern=None,
        settings=_settings(routing_mode="groq", groq_api_key="fake-test-key-not-real"),
    )
    assert result.method == RoutingMethod.GROQ_STRUCTURED
    assert result.specialty_slug == "dermatology"
