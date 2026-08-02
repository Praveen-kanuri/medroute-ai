"""Phase 2B/2D: best-effort speech synthesis for the final safe response text.

Deepgram is the primary text-to-speech provider; Groq is the single
configured fallback (see FallbackTTSService below). Speech synthesis is
always optional and never blocks the conversation: any missing
configuration, oversized text, provider failure, or empty result simply
yields no audio — callers must always still return the text response.
This module never persists audio and never logs response text or audio
bytes.

Text length is capped (settings.tts_max_text_length) *before* any
provider is ever called, uniformly regardless of which provider ends up
running — exceeding it always yields the text-only response, never a
silently truncated one.
"""

import base64
import logging

from app.config.settings import Settings
from app.providers.text_to_speech.base import SpeechResult, TextToSpeechError, TextToSpeechProvider
from app.schemas.conversation import AudioOutput

logger = logging.getLogger(__name__)


class FallbackTTSService(TextToSpeechProvider):
    """Wraps a primary and an optional fallback TextToSpeechProvider
    behind the same provider-neutral interface, so every caller (the
    graph's text_to_speech node, tests, ...) depends only on
    TextToSpeechProvider — never on which concrete vendor actually ran.

    Tries the primary provider first. On an eligible failure (see
    TextToSpeechError.eligible_for_fallback), makes exactly one fallback
    attempt — never in parallel, and never a second hop back to the
    primary. A non-eligible failure propagates immediately."""

    def __init__(
        self,
        *,
        primary: TextToSpeechProvider | None,
        fallback: TextToSpeechProvider | None,
    ) -> None:
        if primary is None and fallback is None:
            raise ValueError("At least one of primary/fallback must be configured.")
        self._primary = primary
        self._fallback = fallback

    @property
    def primary(self) -> TextToSpeechProvider | None:
        return self._primary

    @property
    def fallback(self) -> TextToSpeechProvider | None:
        return self._fallback

    async def synthesize(self, text: str, *, voice: str | None = None) -> SpeechResult:
        if self._primary is None:
            assert self._fallback is not None
            return await self._fallback.synthesize(text, voice=voice)

        try:
            return await self._primary.synthesize(text, voice=voice)
        except TextToSpeechError as exc:
            if not exc.eligible_for_fallback or self._fallback is None:
                raise
            logger.info("tts_primary_failed_falling_back category=%s", exc.category)
            return await self._fallback.synthesize(text, voice=voice)


def _build_single_tts_provider(
    provider_name: str, settings: Settings
) -> TextToSpeechProvider | None:
    """Constructs one named provider, or None if it isn't configured (no
    API key) or isn't a recognized name. Vendor imports are local so
    importing this module never requires either SDK to be exercised."""
    if provider_name == "deepgram":
        if not settings.deepgram_configured:
            return None
        from app.providers.text_to_speech.deepgram import DeepgramTTSProvider

        assert settings.deepgram_api_key is not None
        return DeepgramTTSProvider(
            api_key=settings.deepgram_api_key.get_secret_value(),
            model=settings.deepgram_tts_model,
            timeout_seconds=settings.deepgram_tts_timeout_seconds,
        )
    if provider_name == "groq":
        if not settings.groq_configured:
            return None
        from app.providers.text_to_speech.groq import GroqTTSProvider

        assert settings.groq_api_key is not None
        return GroqTTSProvider(
            api_key=settings.groq_api_key.get_secret_value(),
            model=settings.groq_tts_model,
            voice=settings.groq_tts_voice,
            timeout_seconds=settings.groq_tts_timeout_seconds,
        )
    return None


def build_text_to_speech_provider(settings: Settings) -> TextToSpeechProvider | None:
    """Construct the configured primary/fallback text-to-speech
    providers, wrapped in FallbackTTSService, or None if *neither*
    settings.tts_primary_provider nor settings.tts_fallback_provider
    resolves to a configured provider — callers treat None as "skip
    speech" (TTS is always optional, unlike STT)."""
    primary = _build_single_tts_provider(settings.tts_primary_provider, settings)
    fallback = _build_single_tts_provider(settings.tts_fallback_provider, settings)
    if primary is None and fallback is None:
        return None
    return FallbackTTSService(primary=primary, fallback=fallback)


async def synthesize_speech(
    *,
    text: str | None,
    settings: Settings,
    provider: TextToSpeechProvider | None,
) -> AudioOutput | None:
    """Best-effort speech synthesis over the given text.

    Returns None (never raises) whenever: no text, no provider configured,
    text exceeds settings.tts_max_text_length, the provider (and its
    fallback, if eligible) fails, or the result is empty audio. Only safe
    operational metadata (text length, provider, model, error category) is
    ever logged — never the text or audio bytes themselves.
    """
    if not text or provider is None:
        return None
    if len(text) > settings.tts_max_text_length:
        logger.info(
            "tts_skipped_text_too_long text_length=%d max_length=%d",
            len(text),
            settings.tts_max_text_length,
        )
        return None

    try:
        result = await provider.synthesize(text)
    except TextToSpeechError as exc:
        # No exc_info here: the underlying provider exception could, in
        # principle, echo back request content in its message/traceback —
        # only a fixed, content-free category is ever logged.
        logger.warning("tts_synthesis_failed category=%s", exc.category)
        return None
    except Exception:
        logger.warning("tts_synthesis_failed category=unknown")
        return None

    if not result.audio_bytes:
        return None

    logger.info("tts_synthesis_succeeded provider=%s model=%s", result.provider, result.model)
    return AudioOutput(
        audio_base64=base64.b64encode(result.audio_bytes).decode("ascii"),
        content_type=result.content_type,
        model=result.model,
    )
