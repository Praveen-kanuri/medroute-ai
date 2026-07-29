from app.config.settings import Settings


def test_settings_default_values(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    settings = Settings(_env_file=None)
    assert settings.app_env == "development"
    assert settings.log_level == "INFO"
    assert settings.groq_api_key is None


def test_settings_reads_environment_variables(monkeypatch) -> None:
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")
    settings = Settings(_env_file=None)
    assert settings.app_env == "test"
    assert settings.log_level == "DEBUG"
    assert settings.groq_api_key is not None
    assert settings.groq_api_key.get_secret_value() == "fake-groq-key"


def test_settings_default_to_fake_provider_mode_without_credentials(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("PROVIDER_MODE", raising=False)
    settings = Settings(_env_file=None)
    assert settings.provider_mode == "fake"
    assert settings.groq_api_key is None
    assert settings.deepgram_api_key is None


def test_groq_and_deepgram_configured_false_without_keys(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    settings = Settings(_env_file=None)
    assert settings.groq_configured is False
    assert settings.deepgram_configured is False


def test_groq_and_deepgram_configured_true_with_keys(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "fake-deepgram-key")
    settings = Settings(_env_file=None)
    assert settings.groq_configured is True
    assert settings.deepgram_configured is True


def test_secret_values_do_not_appear_in_repr(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "super-secret-groq-value")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "super-secret-deepgram-value")
    settings = Settings(_env_file=None)
    rendered = repr(settings)
    assert "super-secret-groq-value" not in rendered
    assert "super-secret-deepgram-value" not in rendered


def test_secret_values_do_not_appear_in_serialized_output(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "super-secret-groq-value")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "super-secret-deepgram-value")
    settings = Settings(_env_file=None)
    dumped = settings.model_dump()
    dumped_json = settings.model_dump_json()
    assert "super-secret-groq-value" not in str(dumped)
    assert "super-secret-deepgram-value" not in str(dumped)
    assert "super-secret-groq-value" not in dumped_json
    assert "super-secret-deepgram-value" not in dumped_json
