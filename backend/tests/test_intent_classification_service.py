"""Unit tests for the deterministic conversational-intent classifier (see
app.services.intent_classification_service) — no LLM, no network, no
database.
"""

import pytest

from app.config.settings import Settings
from app.schemas.conversation import ConversationIntent
from app.services.intent_classification_service import (
    classify_concern_relevance,
    classify_new_turn_intent,
    compose_general_chat_reply,
    concern_text,
    greeting_reply_text,
    is_greeting_message,
)


@pytest.mark.parametrize(
    "text",
    [
        "Hi",
        "hello",
        "Hey there",
        "Good morning",
        "Good Afternoon",
        "good evening",
        "How are you?",
        "how's it going",
        "Thank you",
        "thanks!",
        "  hello  ",
        "Hi!",
    ],
)
def test_is_greeting_message_true_for_common_greetings(text: str) -> None:
    assert is_greeting_message(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "chest pain for the past two days",
        "hi, I have a headache",
        "how are you supposed to treat a sprained ankle",
        None,
        "",
        "   ",
    ],
)
def test_is_greeting_message_false_for_non_greetings(text: str | None) -> None:
    assert is_greeting_message(text) is False


def test_greeting_reply_text_for_thanks() -> None:
    text = greeting_reply_text("Thank you so much")
    assert (
        text == "You're welcome. Let me know if you need help finding the appropriate type of care."
    )


@pytest.mark.parametrize(
    ("greeting", "expected_period"),
    [("Good morning", "morning"), ("Good afternoon", "afternoon"), ("Good evening", "evening")],
)
def test_greeting_reply_text_echoes_time_of_day(greeting: str, expected_period: str) -> None:
    text = greeting_reply_text(greeting)
    assert expected_period in text.lower()
    assert "MedAI" in text


def test_greeting_reply_text_for_how_are_you() -> None:
    text = greeting_reply_text("How are you?")
    assert "help" in text.lower()


def test_greeting_reply_text_generic_default() -> None:
    text = greeting_reply_text("Hi")
    assert "MedAI" in text


def test_greeting_reply_text_never_contains_forbidden_terms() -> None:
    for greeting in ("Hi", "Good morning", "Thank you", "How are you?"):
        text = greeting_reply_text(greeting).lower()
        for forbidden in ("diagnos", "treatment", "prescri", "urgent"):
            assert forbidden not in text


def test_concern_text_flattens_main_concern_symptoms_and_transcript() -> None:
    text = concern_text(
        {
            "main_concern": "chest pain",
            "symptoms": ["shortness of breath"],
            "voice_input": {"transcript": "started yesterday"},
        }
    )
    assert text is not None
    assert "chest pain" in text
    assert "shortness of breath" in text
    assert "started yesterday" in text


def test_concern_text_none_when_nothing_supplied() -> None:
    assert concern_text({}) is None
    assert concern_text({"symptoms": [], "main_concern": None}) is None


def test_classify_new_turn_intent_greeting() -> None:
    intent = classify_new_turn_intent(concern="Hi", has_media=False, is_emergency=False)
    assert intent == ConversationIntent.GREETING


def test_classify_new_turn_intent_medical_concern() -> None:
    intent = classify_new_turn_intent(
        concern="chest pain for two days", has_media=False, is_emergency=False
    )
    assert intent == ConversationIntent.MEDICAL_CONCERN


def test_classify_new_turn_intent_unsupported_when_nothing_provided() -> None:
    intent = classify_new_turn_intent(concern=None, has_media=False, is_emergency=False)
    assert intent == ConversationIntent.UNSUPPORTED_OR_UNCLEAR


def test_classify_new_turn_intent_medical_concern_when_media_only() -> None:
    intent = classify_new_turn_intent(concern=None, has_media=True, is_emergency=False)
    assert intent == ConversationIntent.MEDICAL_CONCERN


def test_classify_new_turn_intent_emergency_overrides_greeting_looking_text() -> None:
    # Emergency handling must never be bypassed by text that happens to
    # look like small talk -- callers are expected to check emergency
    # status *before* ever short-circuiting to a greeting reply (see
    # run_conversation_turn), but the classifier itself also refuses to
    # ever call an emergency-declared turn a "greeting".
    intent = classify_new_turn_intent(concern="hi", has_media=False, is_emergency=True)
    assert intent == ConversationIntent.MEDICAL_CONCERN


# --- Phase 3B: optional Groq-backed general-conversation layer -------------
#
# No database, no network, no real model call — the optional Groq path is
# exercised only with a monkeypatched stand-in, never the real SDK/network.


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def _fake_groq(content: str) -> type:
    class _FakeMessage:
        pass

    _FakeMessage.content = content  # type: ignore[attr-defined]

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

    return _FakeAsyncGroq


async def test_classify_concern_relevance_returns_none_without_api_key() -> None:
    result = await classify_concern_relevance("I'm feeling so bored today", _settings())
    assert result is None


async def test_classify_concern_relevance_true_for_health_concern(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("groq.AsyncGroq", _fake_groq('{"is_health_concern": true}'))
    result = await classify_concern_relevance(
        "I think I have a headache", _settings(groq_api_key="fake-test-key-not-real")
    )
    assert result is True


async def test_classify_concern_relevance_false_for_general_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("groq.AsyncGroq", _fake_groq('{"is_health_concern": false}'))
    result = await classify_concern_relevance(
        "I'm feeling so bored, what should I do?",
        _settings(groq_api_key="fake-test-key-not-real"),
    )
    assert result is False


async def test_classify_concern_relevance_falls_back_to_none_when_client_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RaisingGroq:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated network failure — never a real call")

    monkeypatch.setattr("groq.AsyncGroq", _RaisingGroq)
    result = await classify_concern_relevance(
        "I'm bored", _settings(groq_api_key="fake-test-key-not-real")
    )
    assert result is None


async def test_classify_concern_relevance_falls_back_to_none_on_malformed_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("groq.AsyncGroq", _fake_groq('{"is_health_concern": "yes"}'))
    result = await classify_concern_relevance(
        "I'm bored", _settings(groq_api_key="fake-test-key-not-real")
    )
    assert result is None


async def test_compose_general_chat_reply_deterministic_default() -> None:
    text = await compose_general_chat_reply("I'm feeling so bored", _settings())
    assert "describe a symptom" in text.lower()


async def test_compose_general_chat_reply_groq_mode_without_key_falls_back() -> None:
    text = await compose_general_chat_reply(
        "I'm feeling so bored", _settings(conversation_mode="groq")
    )
    assert "describe a symptom" in text.lower()


async def test_compose_general_chat_reply_groq_mode_accepts_valid_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "groq.AsyncGroq",
        _fake_groq(
            '{"response_text": "That sounds rough! I\'m here whenever you want to describe '
            'a symptom so I can help you find care."}'
        ),
    )
    text = await compose_general_chat_reply(
        "I'm feeling so bored",
        _settings(conversation_mode="groq", groq_api_key="fake-test-key-not-real"),
    )
    assert text == (
        "That sounds rough! I'm here whenever you want to describe a symptom so I can "
        "help you find care."
    )


async def test_compose_general_chat_reply_rejects_forbidden_term(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "groq.AsyncGroq",
        _fake_groq('{"response_text": "It sounds like you might need urgent treatment."}'),
    )
    text = await compose_general_chat_reply(
        "I'm feeling so bored",
        _settings(conversation_mode="groq", groq_api_key="fake-test-key-not-real"),
    )
    # Rejected for containing forbidden terms -> falls back to deterministic.
    assert "describe a symptom" in text.lower()


async def test_compose_general_chat_reply_falls_back_when_client_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RaisingGroq:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated network failure — never a real call")

    monkeypatch.setattr("groq.AsyncGroq", _RaisingGroq)
    text = await compose_general_chat_reply(
        "I'm feeling so bored",
        _settings(conversation_mode="groq", groq_api_key="fake-test-key-not-real"),
    )
    assert "describe a symptom" in text.lower()
