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
    # whisper-large-v3 (not the turbo variant): Phase 2A prioritizes
    # transcription accuracy for medical-navigation intake over latency/cost.
    groq_stt_model: str = "whisper-large-v3"
    groq_stt_timeout_seconds: float = 30.0
    deepgram_stt_model: str = "flux-general-en"
    # aura-2-thalia-en: the "voice" for Phase 2B text-to-speech. Deepgram's
    # Aura models double as the voice selection — there is no separate
    # voice parameter in the SDK.
    deepgram_tts_model: str = "aura-2-thalia-en"
    deepgram_tts_timeout_seconds: float = 30.0

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
