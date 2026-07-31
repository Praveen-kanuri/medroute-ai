from fastapi.testclient import TestClient

from app.config.settings import Settings, get_settings
from app.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_system_info_returns_expected_fields() -> None:
    response = client.get("/api/v1/system/info")
    assert response.status_code == 200
    body = response.json()
    assert body["app_name"] == "MedRoute AI"
    assert "app_env" in body
    assert "log_level" in body


def test_system_info_never_returns_api_keys() -> None:
    response = client.get("/api/v1/system/info")
    assert response.status_code == 200
    body = response.json()
    assert "groq_api_key" not in body
    assert "deepgram_api_key" not in body


def test_app_starts_in_fake_mode_without_credentials(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    response = client.get("/health")
    assert response.status_code == 200


def test_readiness_unavailable_when_database_not_configured() -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, database_url=None)
    try:
        response = client.get("/api/v1/health/readiness")
        assert response.status_code == 503
        assert response.json() == {"status": "unavailable"}
    finally:
        app.dependency_overrides.pop(get_settings, None)


def test_readiness_unavailable_when_database_unreachable() -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None,
        database_url="postgresql+asyncpg://user:pass@127.0.0.1:1/nonexistent",
    )
    try:
        response = client.get("/api/v1/health/readiness")
        assert response.status_code == 503
        assert response.json() == {"status": "unavailable"}
    finally:
        app.dependency_overrides.pop(get_settings, None)


def test_readiness_response_never_contains_database_url() -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None,
        database_url="postgresql+asyncpg://user:super-secret-password@127.0.0.1:1/nonexistent",
    )
    try:
        response = client.get("/api/v1/health/readiness")
        assert "super-secret-password" not in response.text
    finally:
        app.dependency_overrides.pop(get_settings, None)
