"""Phase 3A: tests for the generic negation-aware phrase classifier."""

from app.schemas.clinical_context import MentionStatus
from app.services.mention_classification_service import classify_mention

_CHEST_PAIN_PHRASES = ["chest pain", "chest discomfort", "chest pressure"]
_BREATHING_PHRASES = ["difficulty breathing", "trouble breathing", "shortness of breath"]


def test_classify_mention_affirms_plain_statement() -> None:
    assert classify_mention("I have chest pain", _CHEST_PAIN_PHRASES) == MentionStatus.AFFIRMED


def test_classify_mention_negates_leading_no() -> None:
    assert classify_mention("No chest pain", _CHEST_PAIN_PHRASES) == MentionStatus.NEGATED


def test_classify_mention_negates_dont_have() -> None:
    assert (
        classify_mention("I don't have trouble breathing", _BREATHING_PHRASES)
        == MentionStatus.NEGATED
    )


def test_classify_mention_unknown_on_blanket_uncertainty() -> None:
    assert classify_mention("I'm not sure", _CHEST_PAIN_PHRASES) == MentionStatus.UNKNOWN


def test_classify_mention_unknown_when_phrase_never_mentioned() -> None:
    assert classify_mention("My knee hurts a bit", _CHEST_PAIN_PHRASES) == MentionStatus.UNKNOWN


def test_classify_mention_unknown_on_empty_text() -> None:
    assert classify_mention("", _CHEST_PAIN_PHRASES) == MentionStatus.UNKNOWN
    assert classify_mention("   ", _CHEST_PAIN_PHRASES) == MentionStatus.UNKNOWN


def test_classify_mention_recognizes_denies_phrasing() -> None:
    assert (
        classify_mention("Patient denies chest pain", _CHEST_PAIN_PHRASES) == MentionStatus.NEGATED
    )


def test_classify_mention_blanket_no_reply_negates() -> None:
    assert classify_mention("no", _CHEST_PAIN_PHRASES) == MentionStatus.NEGATED
    assert classify_mention("none of those", _BREATHING_PHRASES) == MentionStatus.NEGATED


def test_classify_mention_does_not_false_positive_on_substring() -> None:
    # "known" contains "no" as a raw substring -- word-tokenization must
    # prevent that from being treated as a negation marker.
    assert (
        classify_mention("It is a known chest pain trigger", _CHEST_PAIN_PHRASES)
        == MentionStatus.AFFIRMED
    )


def test_classify_mention_affirms_one_negates_another_in_same_reply() -> None:
    text = "I have chest pain but no trouble breathing"
    assert classify_mention(text, _CHEST_PAIN_PHRASES) == MentionStatus.AFFIRMED
    assert classify_mention(text, _BREATHING_PHRASES) == MentionStatus.NEGATED
