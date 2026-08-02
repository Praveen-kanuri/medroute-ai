"""Deterministic, non-LLM conversational-intent classification for
POST /api/v1/converse.

Recognizes a small, fixed set of greetings/small-talk/thanks phrases so
those messages can short-circuit before medical intake validation,
specialty routing, or provider search ever run (see
app.services.conversation_service.run_conversation_turn) — a greeting must
never trigger a clarification question, a specialty match, or a provider
search. Matching is against the *whole* normalized message only, never a
substring/keyword search, so a real symptom description that happens to
share a word with one of these phrases will not match.

This module never infers intent from vision-derived text, never overrides
emergency handling (callers must check that separately, before consulting
this module), and never calls a model — every greeting reply below is a
fixed, reviewed string.
"""

from typing import Any

from app.schemas.conversation import ConversationIntent

# The exact same fixed phrase set previously private to
# response_composition_service.py, now shared so both modules recognize
# identical greetings — plus a "thanks" set, which that module never
# needed (it only ever saw greeting phrases in the context of an already-
# unmatched routing result).
_TIME_OF_DAY_GREETINGS = ("good morning", "good afternoon", "good evening", "good day")

_HOW_ARE_YOU_PHRASES = frozenset(
    {
        "how are you",
        "how are you doing",
        "how are you today",
        "how's it going",
        "hows it going",
        "what's up",
        "whats up",
        "sup",
    }
)

_PLAIN_GREETING_PHRASES = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "hi there",
        "hello there",
        "hey there",
        "yo",
        "hiya",
        "greetings",
    }
)

_THANKS_PHRASES = frozenset(
    {
        "thank you",
        "thanks",
        "thank you very much",
        "thanks a lot",
        "thanks so much",
        "thank you so much",
        "appreciate it",
        "much appreciated",
    }
)

GREETING_PHRASES = (
    _PLAIN_GREETING_PHRASES
    | _HOW_ARE_YOU_PHRASES
    | _THANKS_PHRASES
    | frozenset(_TIME_OF_DAY_GREETINGS)
)


def normalize_message(text: str | None) -> str:
    """Lowercase, whitespace-collapse, and strip trailing punctuation —
    the same normalization greeting matching has always used, exposed here
    so both classification and reply-selection compare identically."""
    if not text:
        return ""
    return " ".join(text.lower().split()).strip(" .,!?;:\"'")


def is_greeting_message(text: str | None) -> bool:
    """Whether the *whole* normalized message is one of the fixed greeting/
    thanks phrases above — never a partial or keyword match."""
    return normalize_message(text) in GREETING_PHRASES


def greeting_reply_text(text: str | None) -> str:
    """A short, warm, fixed reply matched to which greeting phrase this
    was — never model-generated. Only ever called after is_greeting_message
    has already confirmed a match; falls back to the generic reply for any
    other input so this is still safe to call standalone."""
    normalized = normalize_message(text)
    if normalized in _THANKS_PHRASES:
        return "You're welcome. Let me know if you need help finding the appropriate type of care."
    if normalized in _TIME_OF_DAY_GREETINGS:
        period = normalized.removeprefix("good ")
        return (
            f"Good {period}! I'm MedAI, your medical navigation assistant. "
            "What can I help you with today?"
        )
    if normalized in _HOW_ARE_YOU_PHRASES:
        return "I'm here and ready to help. What health concern can I help you find care for today?"
    return "Hi! I'm MedAI. How can I help you with your health concern today?"


def concern_text(intake_request: dict[str, Any]) -> str | None:
    """Flattens the raw symptoms/main_concern/confirmed voice transcript
    into one plain string — used for greeting detection and, unrelatedly,
    by response composition to acknowledge the user's concern. Never
    includes vision-derived text, and never used for routing (specialty
    routing reads the same raw fields separately)."""
    voice_input = intake_request.get("voice_input") or {}
    parts = [
        intake_request.get("main_concern"),
        " ".join(intake_request.get("symptoms") or []),
        voice_input.get("transcript"),
    ]
    text = " ".join(part for part in parts if part).strip()
    return text or None


def classify_new_turn_intent(
    *, concern: str | None, has_media: bool, is_emergency: bool
) -> ConversationIntent:
    """Classify a brand-new (non-resume) turn. Emergency declarations
    always take precedence over greeting detection — safety handling must
    never be bypassed by a message that happens to look like small talk."""
    if is_emergency:
        return ConversationIntent.MEDICAL_CONCERN
    if concern is None:
        return (
            ConversationIntent.MEDICAL_CONCERN
            if has_media
            else ConversationIntent.UNSUPPORTED_OR_UNCLEAR
        )
    if is_greeting_message(concern):
        return ConversationIntent.GREETING
    return ConversationIntent.MEDICAL_CONCERN
