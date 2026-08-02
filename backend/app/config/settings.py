from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config/settings.py -> repo root is three levels up.
_ROOT_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


class Settings(BaseSettings):
    """Application configuration loaded from the repository-root .env file."""

    model_config = SettingsConfigDict(env_file=_ROOT_ENV_FILE, extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    app_name: str = "MedRoute AI"
    app_version: str = "0.1.0"
    provider_mode: str = "fake"

    groq_api_key: SecretStr | None = None
    deepgram_api_key: SecretStr | None = None

    groq_text_model: str = "openai/gpt-oss-20b"

    # Phase 2D speech-provider fallback (voice transcription and speech
    # synthesis): Deepgram is the primary provider for both; Groq is the
    # single fallback attempt on an eligible transient failure (timeout,
    # connection failure, rate limiting, temporary unavailability, or an
    # eligible 5xx/auth error) — never a parallel request, and never a
    # second fallback hop back to the primary. "deepgram"/"groq" are the
    # only recognized values; see
    # app.services.voice_transcription_service.build_speech_to_text_provider
    # and app.services.text_to_speech_service.build_text_to_speech_provider.
    stt_primary_provider: str = "deepgram"
    stt_fallback_provider: str = "groq"
    tts_primary_provider: str = "deepgram"
    tts_fallback_provider: str = "groq"

    # nova-3: Deepgram's current general-purpose pre-recorded transcription
    # model — not Flux (streaming/turn-detection), which remains a future
    # milestone (see docs/roadmap.md).
    deepgram_stt_model: str = "nova-3"
    deepgram_stt_timeout_seconds: float = 30.0
    # whisper-large-v3-turbo: the fallback STT model — lower latency than
    # the full whisper-large-v3, appropriate for a fallback path that
    # should resolve quickly once the primary has already failed once.
    groq_stt_model: str = "whisper-large-v3-turbo"
    groq_stt_timeout_seconds: float = 30.0

    # aura-2-thalia-en: a calm, professional Deepgram Aura-2 voice — the
    # "voice" for Phase 2B text-to-speech. Deepgram's Aura models double as
    # the voice selection — there is no separate voice parameter in the SDK.
    deepgram_tts_model: str = "aura-2-thalia-en"
    deepgram_tts_timeout_seconds: float = 30.0
    # canopylabs/orpheus-v1-english: the fallback TTS model, hosted on Groq.
    # Unlike Deepgram's Aura models, Groq's TTS API takes the voice as a
    # separate parameter from the model.
    groq_tts_model: str = "canopylabs/orpheus-v1-english"
    groq_tts_voice: str = "hannah"
    groq_tts_timeout_seconds: float = 30.0

    # Phase 2A: conservative default cap on POST /api/v1/voice/transcribe
    # uploads, enforced before any provider call.
    voice_max_upload_bytes: int = 10_000_000

    # Phase 2B: conservative cap on how much response text is ever sent to
    # the TTS provider. Longer text silently skips speech synthesis (the
    # text response is still returned) rather than being truncated
    # mid-sentence or rejected outright.
    tts_max_text_length: int = 1000

    # "deterministic" (default): keyword-based specialty routing, no model
    # call. "groq" additionally tries a Groq-backed structured router first
    # (only when groq_configured is also true), validating its output
    # against the specialty catalog and falling back to deterministic on
    # any failure. Tests always use the deterministic default.
    routing_mode: str = "deterministic"

    # "deterministic" (default): template-based conversational response
    # text, no model call. "groq" additionally tries a Groq-backed
    # rephrasing first (only when groq_configured is also true), validating
    # its output (plain string, bounded length, no forbidden medical
    # terms) and falling back to deterministic on any failure. Tests
    # always use the deterministic default.
    response_mode: str = "deterministic"

    # Phase 2C vision analysis (POST /api/v1/media/analyze). Mirrors
    # routing_mode/response_mode above: "deterministic" (default) never
    # calls a vision model — a directly-uploaded image/video still flows
    # through the graph, just with no visual observations, regardless of
    # whether a Groq API key happens to be configured (unlike routing/
    # response text, there is no non-model "vision" to fall back to, so
    # this flag is the only switch). "groq" additionally requires
    # groq_configured to actually attempt a real call. Tests always use
    # the deterministic default, so no test ever calls Groq's vision API.
    vision_mode: str = "deterministic"
    # Vision analysis reuses the same Groq API key as STT/routing/response
    # rephrasing above (groq_configured). qwen/qwen3.6-27b is Groq's
    # current, generally-available vision (image-input) model — confirmed
    # live against Groq's own /docs/vision page and by an actual API call;
    # Llama-4 Scout/Maverick, sometimes documented elsewhere as Groq's
    # vision models, returned 404 model_not_found for this project's API
    # access at the time of writing.
    groq_vision_model: str = "qwen/qwen3.6-27b"
    groq_vision_timeout_seconds: float = 30.0
    # Caps the model's own output size (cost/latency control) — separate
    # from vision_max_observations, which caps how many *validated*
    # observations are kept after schema validation and dedup.
    vision_max_output_tokens: int = 700
    vision_max_observations: int = 8

    # Conservative, configurable upload/processing limits for Phase 2C
    # direct image/video uploads (never remote URLs — see
    # app/services/media_validation_service.py). image_max_pixels guards
    # against decompression-bomb-style images whose declared dimensions are
    # technically under image_max_dimension_px per side but whose total
    # pixel count would still be extremely expensive to decode.
    image_max_upload_bytes: int = 8_000_000
    image_max_dimension_px: int = 4096
    image_max_pixels: int = 20_000_000

    video_max_upload_bytes: int = 50_000_000
    video_max_duration_seconds: float = 60.0
    video_max_sampled_frames: int = 6
    video_max_dimension_px: int = 4096

    database_url: SecretStr | None = None
    database_echo: bool = False

    @property
    def database_configured(self) -> bool:
        """Whether a non-empty DATABASE_URL is present. Never exposes the URL itself."""
        url = self.database_url
        return url is not None and bool(url.get_secret_value())

    @property
    def groq_configured(self) -> bool:
        """Whether a non-empty Groq API key is present. Never exposes the key itself."""
        key = self.groq_api_key
        return key is not None and bool(key.get_secret_value())

    @property
    def deepgram_configured(self) -> bool:
        """Whether a non-empty Deepgram API key is present. Never exposes the key itself."""
        key = self.deepgram_api_key
        return key is not None and bool(key.get_secret_value())


@lru_cache
def get_settings() -> Settings:
    return Settings()
