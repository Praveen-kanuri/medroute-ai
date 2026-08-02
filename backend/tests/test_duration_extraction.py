"""Unit tests for conservative, deterministic duration extraction from a
voice transcript (Phase 2A clarification-workflow follow-up).

Pure text parsing — no database, no network, no model call.
"""

import pytest

from app.schemas.multimodal_intake import DurationUnit
from app.services.duration_extraction import extract_duration_from_transcript


def test_for_the_past_three_days_extracts_value_and_unit() -> None:
    duration = extract_duration_from_transcript(
        "I have been having this pain for the past three days."
    )
    assert duration is not None
    assert duration.value == 3
    assert duration.unit == DurationUnit.DAYS


def test_for_digit_days_extracts_value_and_unit() -> None:
    duration = extract_duration_from_transcript("I've had a cough for 3 days.")
    assert duration is not None
    assert duration.value == 3
    assert duration.unit == DurationUnit.DAYS


def test_since_yesterday_extracts_one_day() -> None:
    duration = extract_duration_from_transcript("This started since yesterday.")
    assert duration is not None
    assert duration.value == 1
    assert duration.unit == DurationUnit.DAYS


@pytest.mark.parametrize(
    ("phrase", "expected_value", "expected_unit"),
    [
        ("for 2 hours", 2, DurationUnit.HOURS),
        ("for one week", 1, DurationUnit.WEEKS),
        ("for six months", 6, DurationUnit.MONTHS),
        ("for two years", 2, DurationUnit.YEARS),
    ],
)
def test_various_explicit_expressions(
    phrase: str, expected_value: int, expected_unit: DurationUnit
) -> None:
    duration = extract_duration_from_transcript(f"I have had this {phrase}.")
    assert duration is not None
    assert duration.value == expected_value
    assert duration.unit == expected_unit


@pytest.mark.parametrize(
    "ambiguous_phrase",
    [
        "I have been having this pain for a while.",
        "This has been going on for a few days.",
        "It has been bothering me recently.",
        "I've had this for a long time.",
        "",
    ],
)
def test_ambiguous_expressions_return_none(ambiguous_phrase: str) -> None:
    assert extract_duration_from_transcript(ambiguous_phrase) is None


def test_none_transcript_returns_none() -> None:
    assert extract_duration_from_transcript(None) is None


def test_never_extracts_diagnosis_or_urgency_language() -> None:
    # Sanity check that this module has no concept of diagnosis/urgency at
    # all: an explicit-duration transcript that also contains alarming
    # words must still only ever produce a plain Duration object, nothing
    # else, regardless of the surrounding text.
    duration = extract_duration_from_transcript(
        "This is a severe emergency and I need urgent treatment, for 3 days."
    )
    assert duration is not None
    assert duration.value == 3
    assert duration.unit == DurationUnit.DAYS
    assert set(duration.model_dump().keys()) == {"value", "unit"}
