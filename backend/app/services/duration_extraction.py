"""Phase 2A follow-up: conservative duration extraction from a voice transcript.

This performs no medical reasoning whatsoever — it only recognizes a small
set of plain, explicit time-span expressions the user stated in their own
words (e.g. "for 3 days", "for the past three days", "since yesterday") and
converts them into the same structured `Duration` a form field would
produce. Anything not clearly and unambiguously one of these patterns
returns None so the caller falls back to asking the user directly — this
never guesses, and never extracts a diagnosis, treatment, urgency, or
emergency signal.
"""

import re

from app.schemas.multimodal_intake import Duration, DurationUnit

_WORD_NUMBERS: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

_UNIT_WORDS: dict[str, DurationUnit] = {
    "hour": DurationUnit.HOURS,
    "hours": DurationUnit.HOURS,
    "day": DurationUnit.DAYS,
    "days": DurationUnit.DAYS,
    "week": DurationUnit.WEEKS,
    "weeks": DurationUnit.WEEKS,
    "month": DurationUnit.MONTHS,
    "months": DurationUnit.MONTHS,
    "year": DurationUnit.YEARS,
    "years": DurationUnit.YEARS,
}

_NUMBER_ALTERNATION = r"\d+|" + "|".join(_WORD_NUMBERS)
_UNIT_ALTERNATION = "|".join(_UNIT_WORDS)

_FOR_DURATION_RE = re.compile(
    rf"\bfor\s+(?:the\s+past\s+)?({_NUMBER_ALTERNATION})\s+({_UNIT_ALTERNATION})\b",
    re.IGNORECASE,
)
_SINCE_YESTERDAY_RE = re.compile(r"\bsince\s+yesterday\b", re.IGNORECASE)


def extract_duration_from_transcript(transcript: str | None) -> Duration | None:
    """Return a Duration only when the transcript contains one clearly
    recognized, explicit expression. Returns None on anything ambiguous
    ("a while", "recently", "a few days") — callers must never guess and
    should fall back to the follow-up clarification UI instead.
    """
    if not transcript:
        return None

    if _SINCE_YESTERDAY_RE.search(transcript):
        return Duration(value=1, unit=DurationUnit.DAYS)

    match = _FOR_DURATION_RE.search(transcript)
    if match is None:
        return None

    raw_number, raw_unit = match.group(1), match.group(2)
    value = int(raw_number) if raw_number.isdigit() else _WORD_NUMBERS.get(raw_number.lower())
    if not value:
        return None

    return Duration(value=value, unit=_UNIT_WORDS[raw_unit.lower()])
