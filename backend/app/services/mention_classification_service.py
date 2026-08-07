"""Phase 3A: generic, deterministic negation-aware phrase classification.

Given a piece of free text and a list of trigger phrases for one concept
(e.g. "chest pain", "chest discomfort"), decides whether that concept was
affirmed, explicitly negated, or left unknown (never mentioned, or
mentioned with genuine uncertainty). No model call, no medical reasoning —
plain keyword/negation-window matching, deliberately conservative:

- A phrase is only ever AFFIRMED when it appears with no negation word in
  the few tokens immediately before it.
- "unknown" covers both "the phrase was never mentioned" and "the reply as
  a whole reads as uncertain" — never treated as a positive finding by any
  caller (see app.safety.red_flag_rules, which only escalates on an
  explicit AFFIRMED).

This is a reviewable rule set, not full NLP: negation is only recognized
within a short token window immediately preceding a match, so a negation
word much earlier in a long sentence, unrelated to the phrase, will not be
seen (a deliberate trade-off — see this module's tests for exactly what is
and is not recognized).
"""

import re

from app.schemas.clinical_context import MentionStatus

_NEGATION_WINDOW = 4

_NEGATION_WORDS = frozenset(
    {
        "no",
        "not",
        "never",
        "none",
        "without",
        "denies",
        "deny",
        "denying",
        "haven't",
        "hasn't",
        "hadn't",
        "don't",
        "doesn't",
        "didn't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "can't",
        "cannot",
        "couldn't",
        "wouldn't",
        "shouldn't",
        "won't",
    }
)

# Checked against the whole normalized reply (apostrophes stripped) so
# "not sure"/"don't know" are recognized regardless of exactly where they
# fall relative to a phrase match.
_UNCERTAINTY_PHRASES = (
    "not sure",
    "unsure",
    "dont know",
    "not certain",
    "uncertain",
    "no idea",
    "hard to say",
    "cant tell",
    "not positive",
)

# A reply that, taken as a whole, is nothing but a blanket "no" to whatever
# compound question was asked — negates every phrase checked against it.
_BLANKET_NEGATION_REPLIES = frozenset(
    {
        "no",
        "none",
        "nope",
        "nah",
        "not really",
        "none of the above",
        "none of those",
        "no to all of those",
        "not experiencing any of that",
        "no none of that",
    }
)

_TOKEN_RE = re.compile(r"[a-z']+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def classify_mention(
    text: str, phrases: list[str], *, allow_blanket_negation: bool = True
) -> MentionStatus:
    """Classify whether any of `phrases` was affirmed, negated, or left
    unknown in `text`. `phrases` are space-separated lowercase word
    sequences (e.g. "chest pain", "trouble breathing").

    allow_blanket_negation controls whether a bare reply like "No."/"None"
    negates every phrase checked against it (see _BLANKET_NEGATION_REPLIES)
    — appropriate when `text` is a direct answer to a question actually
    *about* these phrases (the dedicated question, or the original
    message), but not when scanning an answer to a *different* question
    for incidentally-mentioned information: a bare "No." answering "any
    pain, redness, warmth, or tenderness?" must not be read as also
    negating an unrelated compound question the user was never asked
    (see app.services.clinical_intake_service.scan_incidental_red_flags,
    which passes allow_blanket_negation=False for exactly this reason)."""
    if not text or not text.strip():
        return MentionStatus.UNKNOWN

    tokens = _tokenize(text)
    stripped_whole = " ".join(tokens).strip()
    if allow_blanket_negation and stripped_whole in _BLANKET_NEGATION_REPLIES:
        return MentionStatus.NEGATED

    found_any = False
    for phrase in phrases:
        phrase_tokens = phrase.split()
        n = len(phrase_tokens)
        if n == 0:
            continue
        for i in range(len(tokens) - n + 1):
            if tokens[i : i + n] != phrase_tokens:
                continue
            found_any = True
            window = tokens[max(0, i - _NEGATION_WINDOW) : i]
            if any(word in _NEGATION_WORDS for word in window):
                return MentionStatus.NEGATED

    if found_any:
        return MentionStatus.AFFIRMED

    normalized_for_uncertainty = " ".join(tokens).replace("'", "")
    if any(phrase in normalized_for_uncertainty for phrase in _UNCERTAINTY_PHRASES):
        return MentionStatus.UNKNOWN

    return MentionStatus.UNKNOWN
