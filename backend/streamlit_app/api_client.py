"""HTTP client for the Streamlit portfolio-demo UI.

Talks only to MedRoute AI's own `/api/v1/navigate` endpoint — never a
model, media, or speech provider directly, and never persists or logs
sensitive intake content itself (that discipline lives in the backend;
this module just forwards what the user typed).

Payload construction is a pure function, kept separate from the network
call, so it is fully unit-testable without any HTTP activity.
"""

from typing import Any

import httpx

DEFAULT_BASE_URL = "http://localhost:8000"
NAVIGATE_PATH = "/api/v1/navigate"


def build_navigation_payload(
    *,
    symptoms: list[str] | None = None,
    main_concern: str | None = None,
    duration_value: int | None = None,
    duration_unit: str | None = None,
    city: str | None = None,
    state: str | None = None,
    postal_code: str | None = None,
    preferred_specialty: str | None = None,
    emergency_concern: bool = False,
) -> dict[str, Any]:
    """Build a JSON-serializable request body for POST /api/v1/navigate.

    Only includes fields the caller actually supplied — matches the API's
    optional-field contract rather than sending nulls/empties for everything
    the form happens to have a widget for.
    """
    payload: dict[str, Any] = {"emergency_concern": emergency_concern}

    if symptoms:
        payload["symptoms"] = symptoms
    if main_concern:
        payload["main_concern"] = main_concern
    if duration_value is not None and duration_value > 0 and duration_unit:
        payload["duration"] = {"value": duration_value, "unit": duration_unit}

    location: dict[str, str] = {}
    if city:
        location["city"] = city
    if state:
        location["state"] = state
    if postal_code:
        location["postal_code"] = postal_code
    if location:
        payload["location"] = location

    if preferred_specialty:
        payload["preferred_specialty"] = preferred_specialty

    return payload


def call_navigate(
    base_url: str, payload: dict[str, Any], *, timeout: float = 10.0
) -> dict[str, Any]:
    """POST payload to the navigation endpoint and return the parsed JSON body.

    Raises httpx.HTTPStatusError on a non-2xx response (e.g. 422) so the
    caller can surface a clear error instead of silently showing stale or
    wrong data.
    """
    response = httpx.post(f"{base_url}{NAVIGATE_PATH}", json=payload, timeout=timeout)
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result
