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
    groq_stt_model: str = "whisper-large-v3-turbo"
    deepgram_stt_model: str = "flux-general-en"
    deepgram_tts_model: str = "aura-2-thalia-en"

    database_url: str | None = None

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
