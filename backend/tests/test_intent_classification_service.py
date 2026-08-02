"""Unit tests for the deterministic conversational-intent classifier (see
app.services.intent_classification_service) — no LLM, no network, no
database.
"""

import pytest

from app.schemas.conversation import ConversationIntent
from app.services.intent_classification_service import (
    classify_new_turn_intent,
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
