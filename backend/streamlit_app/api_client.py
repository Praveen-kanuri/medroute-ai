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
SPECIALTIES_PATH = "/api/v1/specialties"
TRANSCRIBE_PATH = "/api/v1/voice/transcribe"
CONVERSE_PATH = "/api/v1/converse"


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
    voice_transcript: str | None = None,
    voice_language: str | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable request body for POST /api/v1/navigate.

    Only includes fields the caller actually supplied — matches the API's
    optional-field contract rather than sending nulls/empties for everything
    the form happens to have a widget for.

    voice_transcript is only ever forwarded by the caller after the user has
    explicitly confirmed it (see resolve_confirmed_voice_transcript) — this
    function itself has no confirmation logic, it just shapes the payload.
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

    if voice_transcript:
        voice_input: dict[str, Any] = {"transcript": voice_transcript}
        if voice_language:
            voice_input["language"] = voice_language
        payload["voice_input"] = voice_input

    return payload


def resolve_confirmed_voice_transcript(*, transcript: str | None, confirmed: bool) -> str | None:
    """Only ever returns a transcript when the caller has explicitly confirmed it.

    Kept as a plain function (not inline in app.py) so the "never auto-submit
    an unconfirmed transcript" rule is directly unit-testable without needing
    a Streamlit AppTest run.
    """
    if not confirmed:
        return None
    stripped = (transcript or "").strip()
    return stripped or None


# Fields the backend's typed missing_fields list (see
# MultimodalIntakeResponse.missing_fields) can contain, and that the
# clarification follow-up UI knows how to render a control for.
FOLLOWUP_SUPPORTED_FIELDS = ("concern", "duration")


def followup_fields_from_missing(missing_fields: list[str]) -> set[str]:
    """Which follow-up controls to render, chosen strictly from the
    backend's typed `missing_fields` list — never by matching or parsing
    the human-readable `clarification_questions` text (which may itself
    contain words like "concern" inside an unrelated question).
    """
    return {field for field in missing_fields if field in FOLLOWUP_SUPPORTED_FIELDS}


def merge_followup_answer_into_payload(
    payload: dict[str, Any],
    *,
    duration_value: int | None = None,
    duration_unit: str | None = None,
    main_concern: str | None = None,
) -> dict[str, Any]:
    """Merge a clarification follow-up answer into a previously built
    /api/v1/navigate payload, returning a new dict — every other field
    (including a confirmed `voice_input`) is carried over unchanged, so the
    caller never needs to re-collect the transcript or any other form data
    to answer a single follow-up question.
    """
    merged = dict(payload)
    if duration_value is not None and duration_value > 0 and duration_unit:
        merged["duration"] = {"value": duration_value, "unit": duration_unit}
    if main_concern:
        merged["main_concern"] = main_concern
    return merged


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


def list_specialties(base_url: str, *, timeout: float = 10.0) -> list[dict[str, Any]]:
    """GET the backend's supported specialty catalog for the dropdown.

    Raises httpx.HTTPStatusError / httpx.HTTPError on failure so the caller
    can fall back to "no specialty selected" instead of showing stale data.
    """
    response = httpx.get(f"{base_url}{SPECIALTIES_PATH}", timeout=timeout)
    response.raise_for_status()
    result: list[dict[str, Any]] = response.json()
    return result


def build_conversation_payload(
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
    voice_transcript: str | None = None,
    voice_language: str | None = None,
    generate_speech: bool = False,
) -> dict[str, Any]:
    """Build a JSON-serializable request body for POST /api/v1/converse
    that starts a *new* Phase 2B conversation turn. Reuses
    build_navigation_payload's exact field-shaping rules for the nested
    `intake` object — the two endpoints accept the same intake shape.
    """
    intake = build_navigation_payload(
        symptoms=symptoms,
        main_concern=main_concern,
        duration_value=duration_value,
        duration_unit=duration_unit,
        city=city,
        state=state,
        postal_code=postal_code,
        preferred_specialty=preferred_specialty,
        emergency_concern=emergency_concern,
        voice_transcript=voice_transcript,
        voice_language=voice_language,
    )
    return {"intake": intake, "generate_speech": generate_speech}


def build_clarification_resume_payload(
    *,
    thread_id: str,
    duration_value: int | None = None,
    duration_unit: str | None = None,
    main_concern: str | None = None,
    generate_speech: bool = False,
) -> dict[str, Any]:
    """Build a JSON-serializable request body for POST /api/v1/converse
    that *resumes* a paused clarification turn with a typed answer — the
    original confirmed transcript and every other intake field are
    preserved server-side (see app/services/conversation_service.py),
    so nothing from the earlier turn needs to be resent here.
    """
    answer: dict[str, Any] = {}
    if duration_value is not None and duration_value > 0 and duration_unit:
        answer["duration"] = {"value": duration_value, "unit": duration_unit}
    if main_concern:
        answer["main_concern"] = main_concern
    return {
        "thread_id": thread_id,
        "clarification_answer": answer,
        "generate_speech": generate_speech,
    }


def call_converse(
    base_url: str, payload: dict[str, Any], *, timeout: float = 15.0
) -> dict[str, Any]:
    """POST payload to the Phase 2B conversational endpoint and return the
    parsed JSON body.

    Raises httpx.HTTPStatusError on a non-2xx response (e.g. 422 for an
    invalid new-vs-resume combination, 404 for an unknown/expired
    thread_id) so the caller can surface a clear error instead of stale
    or wrong data.
    """
    response = httpx.post(f"{base_url}{CONVERSE_PATH}", json=payload, timeout=timeout)
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


def transcribe_audio_via_api(
    base_url: str,
    audio_bytes: bytes,
    *,
    filename: str,
    content_type: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """POST an uploaded audio file to POST /api/v1/voice/transcribe.

    Raises httpx.HTTPStatusError on a non-2xx response (e.g. 422 for an
    unsupported/empty file, 413 for oversized, 503 if transcription is not
    configured or the provider failed) so the caller can show a clear
    message instead of a stale or fabricated transcript. The returned
    transcript is raw speech-to-text output only — the caller must still
    let the user review and confirm it before using it for anything else.
    """
    files = {"file": (filename, audio_bytes, content_type or "application/octet-stream")}
    response = httpx.post(f"{base_url}{TRANSCRIBE_PATH}", files=files, timeout=timeout)
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result
