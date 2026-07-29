from fastapi.testclient import TestClient

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
