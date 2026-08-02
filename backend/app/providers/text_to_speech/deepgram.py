"""Deepgram Aura-2 text-to-speech provider — the primary TTS provider (see
app.services.text_to_speech_service.FallbackTTSService).

All Deepgram SDK usage is isolated to this module. The graph node and
orchestration service (app/graph/nodes.py,
app/services/text_to_speech_service.py) only ever depend on the abstract
TextToSpeechProvider interface.
"""

from deepgram import AsyncDeepgramClient
from deepgram.core.api_error import ApiError as DeepgramApiError
from deepgram.core.request_options import RequestOptions

from app.providers.speech_provider_errors import classify_deepgram_exception
from app.providers.text_to_speech.base import SpeechResult, TextToSpeechError, TextToSpeechProvider

_AUDIO_CONTENT_TYPE = "audio/mpeg"


class DeepgramTTSProvider(TextToSpeechProvider):
    """Text-to-speech provider using Deepgram's hosted Aura-2 models."""

    def __init__(self, *, api_key: str, model: str, timeout_seconds: float) -> None:
        self._client = AsyncDeepgramClient(api_key=api_key)
        self._model = model
        self._timeout_seconds = timeout_seconds

    async def synthesize(self, text: str, *, voice: str | None = None) -> SpeechResult:
        model = voice or self._model
        try:
            # speak.v1.audio.generate is an async *generator* function (it
            # returns AsyncIterator[bytes] directly, not a coroutine) --
            # awaiting the call itself is a TypeError; only iterating its
            # chunks is awaited.
            # `container` is only for raw/headerless encodings (e.g.
            # linear16) -- mp3 is its own container, and Deepgram's API
            # rejects the two combined.
            chunks = self._client.speak.v1.audio.generate(
                text=text,
                model=model,
                encoding="mp3",
                request_options=RequestOptions(timeout_in_seconds=int(self._timeout_seconds)),
            )
            audio_bytes = b"".join([chunk async for chunk in chunks])
        except DeepgramApiError as exc:
            eligible, category = classify_deepgram_exception(exc)
            raise TextToSpeechError(
                "Deepgram speech synthesis request failed.",
                eligible_for_fallback=eligible,
                category=category,
            ) from exc
        except Exception as exc:
            eligible, category = classify_deepgram_exception(exc)
            raise TextToSpeechError(
                "Deepgram speech synthesis request failed.",
                eligible_for_fallback=eligible,
                category=category,
            ) from exc

        if not audio_bytes:
            raise TextToSpeechError(
                "Deepgram returned no audio.", eligible_for_fallback=True, category="empty_result"
            )
        return SpeechResult(
            audio_bytes=audio_bytes,
            content_type=_AUDIO_CONTENT_TYPE,
            provider="deepgram",
            model=model,
        )
