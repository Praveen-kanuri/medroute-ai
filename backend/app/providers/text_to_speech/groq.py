"""Groq Orpheus text-to-speech provider — the configured fallback for
Deepgram (see app.services.text_to_speech_service.FallbackTTSService).

All Groq SDK usage is isolated to this module. Unlike Deepgram's Aura
models (where the model name doubles as the voice selection), Groq's TTS
API takes voice as a parameter separate from model — both are configured
at construction time (see build_text_to_speech_provider) so callers never
need to know provider-specific parameter shapes.
"""

from groq import AsyncGroq

from app.providers.speech_provider_errors import classify_groq_exception
from app.providers.text_to_speech.base import SpeechResult, TextToSpeechError, TextToSpeechProvider

_AUDIO_CONTENT_TYPE = "audio/wav"
# Groq's documented Orpheus TTS input limit. Enforced locally, before ever
# calling the API, so an oversized request is never sent (and never
# silently truncated) — see synthesize()'s early check below.
_MAX_INPUT_CHARACTERS = 200


class GroqTTSProvider(TextToSpeechProvider):
    """Text-to-speech provider using Groq's hosted Orpheus model."""

    def __init__(self, *, api_key: str, model: str, voice: str, timeout_seconds: float) -> None:
        self._client = AsyncGroq(api_key=api_key, timeout=timeout_seconds)
        self._model = model
        self._voice = voice

    async def synthesize(self, text: str, *, voice: str | None = None) -> SpeechResult:
        if len(text) > _MAX_INPUT_CHARACTERS:
            # A locally-detected validation failure, not a transient
            # provider problem -- ineligible for (a nonexistent further)
            # fallback. The caller (see
            # app.services.text_to_speech_service.synthesize_speech) treats
            # this exactly like any other TextToSpeechError: no audio,
            # text response preserved, never truncated.
            raise TextToSpeechError(
                f"Text exceeds Groq's {_MAX_INPUT_CHARACTERS}-character input limit.",
                eligible_for_fallback=False,
                category="validation",
            )

        try:
            response = await self._client.audio.speech.create(
                input=text,
                model=self._model,
                voice=voice or self._voice,
                response_format="wav",
            )
            audio_bytes = await response.read()
        except Exception as exc:
            eligible, category = classify_groq_exception(exc)
            raise TextToSpeechError(
                "Groq speech synthesis request failed.",
                eligible_for_fallback=eligible,
                category=category,
            ) from exc

        if not audio_bytes:
            raise TextToSpeechError(
                "Groq returned no audio.", eligible_for_fallback=True, category="empty_result"
            )
        return SpeechResult(
            audio_bytes=audio_bytes,
            content_type=_AUDIO_CONTENT_TYPE,
            provider="groq",
            model=self._model,
        )
