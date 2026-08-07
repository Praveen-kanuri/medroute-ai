"""Phase 2A follow-up: conservative duration extraction from a voice transcript.

This performs no medical reasoning whatsoever — it only recognizes a small
set of plain, explicit time-span expressions the user stated in their own
words (e.g. "for 3 days", "for the past three days", "since yesterday",
"it's been a couple of hours") and converts them into the same structured
`Duration` a form field would produce. Anything not clearly and
unambiguously one of these patterns returns None so the caller falls back
to asking the user directly — this never guesses, and never extracts a
diagnosis, treatment, urgency, or emergency signal.

"a couple"/"a couple of" is treated as the specific number 2 (a
conventional, unambiguous English usage) — genuinely vague quantities like
"a few days" or "a while" are deliberately NOT recognized, since "a few"
does not reliably mean any one specific number.
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

# "(?:a\s+)?couple(?:\s+of)?" matches "couple", "a couple", "couple of", and
# "a couple of" — all conventionally mean exactly 2 in this context.
_NUMBER_ALTERNATION = r"\d+|(?:a\s+)?couple(?:\s+of)?|" + "|".join(_WORD_NUMBERS)
_UNIT_ALTERNATION = "|".join(_UNIT_WORDS)

_FOR_DURATION_RE = re.compile(
    rf"\bfor\s+(?:the\s+past\s+)?({_NUMBER_ALTERNATION})\s+({_UNIT_ALTERNATION})\b",
    re.IGNORECASE,
)
# "it's been"/"it has been"/"this has been going on for" + a duration, with
# no leading "for" required — covers everyday phrasing like "it's been a
# couple of hours" or "it's been 3 days" that _FOR_DURATION_RE alone would
# miss (there is no "for" immediately before the number there).
_BEEN_DURATION_RE = re.compile(
    rf"\b(?:it'?s\s+been|it\s+has\s+been)\s+({_NUMBER_ALTERNATION})\s+({_UNIT_ALTERNATION})\b",
    re.IGNORECASE,
)
_SINCE_YESTERDAY_RE = re.compile(r"\bsince\s+yesterday\b", re.IGNORECASE)


def _parse_number(raw: str) -> int | None:
    normalized = " ".join(raw.lower().split())
    if normalized.isdigit():
        return int(normalized)
    if "couple" in normalized:
        return 2
    return _WORD_NUMBERS.get(normalized)


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

    match = _FOR_DURATION_RE.search(transcript) or _BEEN_DURATION_RE.search(transcript)
    if match is None:
        return None

    raw_number, raw_unit = match.group(1), match.group(2)
    value = _parse_number(raw_number)
    if not value:
        return None

    return Duration(value=value, unit=_UNIT_WORDS[raw_unit.lower()])
