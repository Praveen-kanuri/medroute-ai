"""Unit tests for the Streamlit demo UI's API client.

No real network calls — call_navigate is tested against an in-memory
httpx.MockTransport, never a live socket.
"""

import httpx
import pytest

from streamlit_app.api_client import (
    NAVIGATE_PATH,
    SPECIALTIES_PATH,
    build_navigation_payload,
    call_navigate,
    list_specialties,
)


def test_build_payload_includes_only_supplied_fields() -> None:
    payload = build_navigation_payload()
    assert payload == {"emergency_concern": False}


def test_build_payload_includes_symptoms() -> None:
    payload = build_navigation_payload(symptoms=["knee pain", "swelling"])
    assert payload["symptoms"] == ["knee pain", "swelling"]


def test_build_payload_omits_empty_symptoms_list() -> None:
    payload = build_navigation_payload(symptoms=[])
    assert "symptoms" not in payload


def test_build_payload_includes_main_concern() -> None:
    payload = build_navigation_payload(main_concern="Discomfort after exercise")
    assert payload["main_concern"] == "Discomfort after exercise"


def test_build_payload_includes_duration_only_when_both_value_and_unit_present() -> None:
    payload = build_navigation_payload(duration_value=3, duration_unit="days")
    assert payload["duration"] == {"value": 3, "unit": "days"}


def test_build_payload_omits_duration_when_unit_missing() -> None:
    payload = build_navigation_payload(duration_value=3, duration_unit=None)
    assert "duration" not in payload


def test_build_payload_omits_duration_when_value_is_zero() -> None:
    payload = build_navigation_payload(duration_value=0, duration_unit="days")
    assert "duration" not in payload


def test_build_payload_includes_location_fields() -> None:
    payload = build_navigation_payload(city="Dallas", state="TX", postal_code="75201")
    assert payload["location"] == {"city": "Dallas", "state": "TX", "postal_code": "75201"}


def test_build_payload_omits_location_when_all_fields_blank() -> None:
    payload = build_navigation_payload(city=None, state=None, postal_code=None)
    assert "location" not in payload


def test_build_payload_includes_partial_location() -> None:
    payload = build_navigation_payload(city="Dallas")
    assert payload["location"] == {"city": "Dallas"}


def test_build_payload_includes_preferred_specialty() -> None:
    payload = build_navigation_payload(preferred_specialty="cardiology")
    assert payload["preferred_specialty"] == "cardiology"


def test_build_payload_omits_blank_preferred_specialty() -> None:
    payload = build_navigation_payload(preferred_specialty="")
    assert "preferred_specialty" not in payload


def test_build_payload_reflects_emergency_concern() -> None:
    payload = build_navigation_payload(emergency_concern=True)
    assert payload["emergency_concern"] is True


def test_call_navigate_posts_to_navigate_path_and_returns_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        return httpx.Response(200, json={"intake": {"status": "needs_clarification"}})

    transport = httpx.MockTransport(handler)

    def fake_post(url: str, *, json: object, timeout: float) -> httpx.Response:
        with httpx.Client(transport=transport) as client:
            return client.post(url, json=json, timeout=timeout)

    monkeypatch.setattr(httpx, "post", fake_post)

    result = call_navigate("http://backend.test", {"emergency_concern": False})

    assert captured["method"] == "POST"
    assert captured["url"] == f"http://backend.test{NAVIGATE_PATH}"
    assert result == {"intake": {"status": "needs_clarification"}}


def test_call_navigate_raises_on_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "invalid"})

    transport = httpx.MockTransport(handler)

    def fake_post(url: str, *, json: object, timeout: float) -> httpx.Response:
        with httpx.Client(transport=transport) as client:
            return client.post(url, json=json, timeout=timeout)

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(httpx.HTTPStatusError):
        call_navigate("http://backend.test", {"emergency_concern": False})


def test_list_specialties_gets_specialties_path_and_returns_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    catalog = [{"slug": "cardiology", "display_name": "Cardiology", "description": None}]

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        return httpx.Response(200, json=catalog)

    transport = httpx.MockTransport(handler)

    def fake_get(url: str, *, timeout: float) -> httpx.Response:
        with httpx.Client(transport=transport) as client:
            return client.get(url, timeout=timeout)

    monkeypatch.setattr(httpx, "get", fake_get)

    result = list_specialties("http://backend.test")

    assert captured["method"] == "GET"
    assert captured["url"] == f"http://backend.test{SPECIALTIES_PATH}"
    assert result == catalog


def test_list_specialties_raises_on_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "unavailable"})

    transport = httpx.MockTransport(handler)

    def fake_get(url: str, *, timeout: float) -> httpx.Response:
        with httpx.Client(transport=transport) as client:
            return client.get(url, timeout=timeout)

    monkeypatch.setattr(httpx, "get", fake_get)

    with pytest.raises(httpx.HTTPStatusError):
        list_specialties("http://backend.test")
