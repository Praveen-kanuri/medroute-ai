"""Shared fallback-eligibility classification for the Deepgram/Groq speech
providers (app/providers/speech_to_text/, app/providers/text_to_speech/).

A primary-provider failure is eligible for exactly one fallback attempt
only when it looks transient or provider-side (timeout, connection
failure, rate limiting, temporary unavailability, an eligible 5xx, or an
authentication/configuration problem specific to that provider) — never
for a locally-detected validation failure (bad/empty audio, an
unsupported format, invalid request shape, or unsafe/empty text), since
the same input would fail identically against the fallback provider too,
wasting a call with no chance of succeeding.
"""

import groq
from deepgram.core.api_error import ApiError as DeepgramApiError

_RATE_LIMIT_STATUS = 429
_AUTH_STATUSES = frozenset({401, 403})
_SERVER_ERROR_THRESHOLD = 500


def classify_groq_exception(exc: Exception) -> tuple[bool, str]:
    """Returns (eligible_for_fallback, category) for an exception raised by
    the `groq` SDK (used by both GroqSTTProvider and GroqTTSProvider)."""
    if isinstance(exc, groq.RateLimitError):
        return True, "rate_limited"
    if isinstance(exc, groq.APITimeoutError):
        return True, "timeout"
    if isinstance(exc, groq.APIConnectionError):
        return True, "connection"
    if isinstance(exc, groq.AuthenticationError | groq.PermissionDeniedError):
        return True, "auth_or_config"
    if isinstance(exc, groq.InternalServerError):
        return True, "server_unavailable"
    if isinstance(exc, groq.APIStatusError):
        if exc.status_code >= _SERVER_ERROR_THRESHOLD:
            return True, "server_unavailable"
        return False, "validation"
    return False, "unknown"


def classify_deepgram_exception(exc: Exception) -> tuple[bool, str]:
    """Returns (eligible_for_fallback, category) for an exception raised by
    the `deepgram` SDK (used by both DeepgramSTTProvider and
    DeepgramTTSProvider)."""
    if isinstance(exc, DeepgramApiError):
        status = exc.status_code
        if status == _RATE_LIMIT_STATUS:
            return True, "rate_limited"
        if status in _AUTH_STATUSES:
            return True, "auth_or_config"
        if status is not None and status >= _SERVER_ERROR_THRESHOLD:
            return True, "server_unavailable"
        return False, "validation"
    if isinstance(exc, TimeoutError):
        return True, "timeout"
    # Any other exception (connection reset, DNS failure, etc.) is treated
    # as a transient connection problem — eligible for one fallback attempt.
    return True, "connection"
