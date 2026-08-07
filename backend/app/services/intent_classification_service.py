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

import json
import logging
from typing import Any

from app.config.settings import Settings
from app.safety.constants import FORBIDDEN_TERMS
from app.schemas.conversation import ConversationIntent

logger = logging.getLogger(__name__)

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


# --- Phase 3B (optional, opt-in): LLM-backed general-conversation layer ----
#
# Both functions below are only ever consulted when Settings.conversation_
# mode == "groq" AND a real Groq key is configured — see
# app.graph.nodes.concern_relevance_agent_node, which is never reached at
# all for a user-declared emergency turn (checked before this module is
# ever consulted, mirroring classify_new_turn_intent's own precedence
# above). Every failure mode (no key, network error, malformed output,
# forbidden term) falls back to the existing deterministic behavior —
# never raises, never blocks the turn.

_RELEVANCE_SYSTEM_PROMPT = (
    "You classify one message sent to a healthcare-appointment-navigation "
    "assistant. Decide only whether it describes a specific health symptom or "
    "medical concern the user wants help finding care for, versus general "
    "conversation, small talk, or a message with no identifiable health "
    "concern. Respond with only a JSON object of the exact form "
    '{"is_health_concern": true or false}, and nothing else. Never diagnose, '
    "never explain your reasoning, never include any other field or any text "
    "outside the JSON object."
)

_GENERAL_CHAT_SYSTEM_PROMPT = (
    "You are MedAI, a friendly medical-appointment-navigation assistant. The "
    "user just said something that is not a specific health concern (general "
    "conversation, small talk, or an off-topic question). Reply warmly and "
    "briefly (at most two short sentences): acknowledge what they said in a "
    "natural way, then gently invite them to describe a symptom or health "
    "concern you can help them find care for. You must never diagnose, give "
    "medical advice, or suggest or recommend any treatment. Respond with "
    "only a JSON object of the exact form "
    '{"response_text": "<your reply, at most 300 characters>"}, and nothing '
    "else."
)

_GENERAL_CHAT_FALLBACK_TEXT = (
    "I'm here to help you find the right care for a health concern — feel "
    'free to describe a symptom (for example, "chest pain for the past two '
    'days") and I can help guide you from there.'
)

_MAX_GENERAL_CHAT_RESPONSE_LENGTH = 400


async def classify_concern_relevance(text: str, settings: Settings) -> bool | None:
    """Ask whether `text` describes an identifiable health concern. Returns
    True/False on a valid classification, or None when Groq isn't
    configured or the call fails/returns something unparseable — callers
    must treat None exactly like True (proceed as an ordinary medical
    concern, today's unchanged behavior), never like False."""
    if settings.groq_api_key is None:
        return None
    try:
        from groq import AsyncGroq

        client = AsyncGroq(api_key=settings.groq_api_key.get_secret_value())
        response = await client.chat.completions.create(
            model=settings.groq_text_model,
            messages=[
                {"role": "system", "content": _RELEVANCE_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            # No response_format={"type": "json_object"} here: empirically,
            # Groq's server-side JSON-grammar validator for this reasoning
            # model (openai/gpt-oss-20b) rejects some otherwise-valid
            # completions outright (400 json_validate_failed) depending on
            # the exact prompt/input combination -- reproduced as a 100%
            # deterministic failure for this exact system prompt against a
            # plain greeting, regardless of max_completion_tokens. Asking
            # for JSON via the prompt alone and parsing the free-form
            # response (already wrapped in a broad try/except that falls
            # back to deterministic behavior on any parse failure) proved
            # 100% reliable in the same trials that reproduced the bug.
            temperature=0,
            max_completion_tokens=300,
            reasoning_effort="low",
        )
        content = response.choices[0].message.content
        if not content:
            return None
        parsed = json.loads(content)
        value = parsed.get("is_health_concern")
    except Exception:
        logger.warning(
            "concern_relevance_classification_failed falling back to deterministic",
            exc_info=True,
        )
        return None

    return value if isinstance(value, bool) else None


def _validate_general_chat_reply(candidate: object) -> str | None:
    if not isinstance(candidate, str):
        return None
    stripped = candidate.strip()
    if not stripped or len(stripped) > _MAX_GENERAL_CHAT_RESPONSE_LENGTH:
        return None
    lowered = stripped.lower()
    if any(term in lowered for term in FORBIDDEN_TERMS):
        return None
    return stripped


async def _groq_general_chat_reply(user_text: str, settings: Settings) -> str | None:
    try:
        from groq import AsyncGroq

        client = AsyncGroq(api_key=settings.groq_api_key.get_secret_value())  # type: ignore[union-attr]
        response = await client.chat.completions.create(
            model=settings.groq_text_model,
            messages=[
                {"role": "system", "content": _GENERAL_CHAT_SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            # See classify_concern_relevance above: no response_format
            # json_object constraint, for the same reliability reason.
            temperature=0.4,
            max_completion_tokens=500,
            reasoning_effort="low",
        )
        content = response.choices[0].message.content
        if not content:
            return None
        parsed = json.loads(content)
        candidate = parsed.get("response_text")
    except Exception:
        logger.warning("general_chat_reply_failed falling back to deterministic", exc_info=True)
        return None

    return _validate_general_chat_reply(candidate)


async def compose_general_chat_reply(user_text: str | None, settings: Settings) -> str:
    """The reply for a turn classified as general chat (see
    classify_concern_relevance) — deterministic and free by default, with
    an optional Groq-backed, validated, warmly-phrased version when
    conversation_mode="groq" and a real key is configured. Never a
    diagnosis, never medical advice; always falls back to
    _GENERAL_CHAT_FALLBACK_TEXT on any failure."""
    if settings.conversation_mode == "groq" and settings.groq_configured and user_text:
        reply = await _groq_general_chat_reply(user_text, settings)
        if reply is not None:
            return reply
    return _GENERAL_CHAT_FALLBACK_TEXT
