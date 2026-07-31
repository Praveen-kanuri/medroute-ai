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
