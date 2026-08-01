"""Unit tests for the Streamlit demo UI's API client.

No real network calls — call_navigate is tested against an in-memory
httpx.MockTransport, never a live socket.
"""

import httpx
import pytest

from streamlit_app.api_client import (
    CONVERSE_PATH,
    NAVIGATE_PATH,
    SPECIALTIES_PATH,
    TRANSCRIBE_PATH,
    build_clarification_resume_payload,
    build_conversation_payload,
    build_navigation_payload,
    call_converse,
    call_navigate,
    followup_fields_from_missing,
    list_specialties,
    merge_followup_answer_into_payload,
    resolve_confirmed_voice_transcript,
    transcribe_audio_via_api,
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


# --- build_navigation_payload: voice_input ----------------------------------


def test_build_payload_includes_voice_input_when_transcript_supplied() -> None:
    payload = build_navigation_payload(voice_transcript="a persistent cough", voice_language="en")
    assert payload["voice_input"] == {"transcript": "a persistent cough", "language": "en"}


def test_build_payload_omits_language_when_not_supplied() -> None:
    payload = build_navigation_payload(voice_transcript="a persistent cough")
    assert payload["voice_input"] == {"transcript": "a persistent cough"}


def test_build_payload_omits_voice_input_when_transcript_missing() -> None:
    payload = build_navigation_payload(voice_language="en")
    assert "voice_input" not in payload


def test_build_payload_omits_voice_input_when_transcript_blank() -> None:
    payload = build_navigation_payload(voice_transcript="")
    assert "voice_input" not in payload


# --- resolve_confirmed_voice_transcript: confirmation gating ----------------


def test_resolve_confirmed_voice_transcript_requires_confirmation() -> None:
    assert resolve_confirmed_voice_transcript(transcript="chest pain", confirmed=False) is None


def test_resolve_confirmed_voice_transcript_returns_transcript_when_confirmed() -> None:
    assert (
        resolve_confirmed_voice_transcript(transcript="chest pain", confirmed=True) == "chest pain"
    )


def test_resolve_confirmed_voice_transcript_trims_whitespace() -> None:
    assert (
        resolve_confirmed_voice_transcript(transcript="  chest pain  ", confirmed=True)
        == "chest pain"
    )


def test_resolve_confirmed_voice_transcript_treats_blank_as_none_even_if_confirmed() -> None:
    assert resolve_confirmed_voice_transcript(transcript="   ", confirmed=True) is None


def test_resolve_confirmed_voice_transcript_handles_none_transcript() -> None:
    assert resolve_confirmed_voice_transcript(transcript=None, confirmed=True) is None


# --- transcribe_audio_via_api -------------------------------------------------


def test_transcribe_audio_posts_multipart_to_transcribe_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["body"] = request.content
        return httpx.Response(
            200,
            json={
                "transcription_id": "abc-123",
                "transcript": "patient reports a persistent cough",
                "language": "en",
                "model": "whisper-large-v3",
                "status": "completed",
            },
        )

    transport = httpx.MockTransport(handler)

    def fake_post(url: str, *, files: object, timeout: float) -> httpx.Response:
        with httpx.Client(transport=transport) as client:
            return client.post(url, files=files, timeout=timeout)

    monkeypatch.setattr(httpx, "post", fake_post)

    result = transcribe_audio_via_api(
        "http://backend.test",
        b"synthetic-audio-bytes",
        filename="clip.wav",
        content_type="audio/wav",
    )

    assert captured["method"] == "POST"
    assert captured["url"] == f"http://backend.test{TRANSCRIBE_PATH}"
    assert b"synthetic-audio-bytes" in captured["body"]  # type: ignore[operator]
    assert result["transcript"] == "patient reports a persistent cough"
    assert result["language"] == "en"
    assert result["status"] == "completed"


def test_transcribe_audio_raises_on_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "Unsupported audio format"})

    transport = httpx.MockTransport(handler)

    def fake_post(url: str, *, files: object, timeout: float) -> httpx.Response:
        with httpx.Client(transport=transport) as client:
            return client.post(url, files=files, timeout=timeout)

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(httpx.HTTPStatusError):
        transcribe_audio_via_api(
            "http://backend.test",
            b"synthetic-audio-bytes",
            filename="clip.ogg",
            content_type="audio/ogg",
        )


# --- followup_fields_from_missing: typed fields, never question text -------


def test_followup_fields_from_missing_selects_duration() -> None:
    assert followup_fields_from_missing(["duration"]) == {"duration"}


def test_followup_fields_from_missing_selects_both_fields() -> None:
    assert followup_fields_from_missing(["concern", "duration"]) == {"concern", "duration"}


def test_followup_fields_from_missing_ignores_unknown_field_names() -> None:
    assert followup_fields_from_missing(["something_else"]) == set()


def test_followup_fields_from_missing_does_not_parse_question_wording() -> None:
    # The real duration clarification question is "How long have you been
    # experiencing this concern?" — it contains the word "concern" inside
    # it. If this function scanned question *text* rather than the typed
    # missing_fields list, that substring could wrongly trigger a
    # main-concern follow-up control. It must not.
    duration_question = "How long have you been experiencing this concern?"
    assert followup_fields_from_missing([duration_question]) == set()


# --- merge_followup_answer_into_payload -------------------------------------


def test_merge_followup_answer_adds_duration_and_preserves_other_fields() -> None:
    original = {
        "emergency_concern": False,
        "voice_input": {"transcript": "chest pain and heart palpitations", "language": "en"},
    }
    merged = merge_followup_answer_into_payload(original, duration_value=3, duration_unit="days")
    assert merged["duration"] == {"value": 3, "unit": "days"}
    assert merged["voice_input"] == original["voice_input"]
    assert merged["emergency_concern"] is False


def test_merge_followup_answer_does_not_mutate_original_payload() -> None:
    original = {"emergency_concern": False}
    merge_followup_answer_into_payload(original, duration_value=3, duration_unit="days")
    assert "duration" not in original


def test_merge_followup_answer_omits_duration_when_unit_missing() -> None:
    merged = merge_followup_answer_into_payload({}, duration_value=3, duration_unit=None)
    assert "duration" not in merged


def test_merge_followup_answer_omits_duration_when_value_is_zero() -> None:
    merged = merge_followup_answer_into_payload({}, duration_value=0, duration_unit="days")
    assert "duration" not in merged


def test_merge_followup_answer_adds_main_concern() -> None:
    merged = merge_followup_answer_into_payload({}, main_concern="Discomfort after exercise")
    assert merged["main_concern"] == "Discomfort after exercise"


def test_merge_followup_answer_with_no_answers_returns_equivalent_payload() -> None:
    original = {"emergency_concern": False, "symptoms": ["knee pain"]}
    merged = merge_followup_answer_into_payload(original)
    assert merged == original
    assert merged is not original


# --- build_conversation_payload / build_clarification_resume_payload --------


def test_build_conversation_payload_wraps_navigation_payload_as_intake() -> None:
    payload = build_conversation_payload(
        symptoms=["chest pain"], duration_value=2, duration_unit="days"
    )
    assert payload["intake"] == build_navigation_payload(
        symptoms=["chest pain"], duration_value=2, duration_unit="days"
    )
    assert payload["generate_speech"] is False


def test_build_conversation_payload_includes_generate_speech_flag() -> None:
    payload = build_conversation_payload(generate_speech=True)
    assert payload["generate_speech"] is True


def test_build_conversation_payload_includes_confirmed_voice_transcript() -> None:
    payload = build_conversation_payload(voice_transcript="chest pain", voice_language="en")
    assert payload["intake"]["voice_input"] == {"transcript": "chest pain", "language": "en"}


def test_build_clarification_resume_payload_includes_thread_id_and_duration() -> None:
    payload = build_clarification_resume_payload(
        thread_id="abc-123", duration_value=3, duration_unit="days"
    )
    assert payload == {
        "thread_id": "abc-123",
        "clarification_answer": {"duration": {"value": 3, "unit": "days"}},
        "generate_speech": False,
    }


def test_build_clarification_resume_payload_includes_main_concern() -> None:
    payload = build_clarification_resume_payload(thread_id="abc-123", main_concern="itchy rash")
    assert payload["clarification_answer"] == {"main_concern": "itchy rash"}


def test_build_clarification_resume_payload_omits_duration_when_incomplete() -> None:
    payload = build_clarification_resume_payload(thread_id="abc-123", duration_value=3)
    assert payload["clarification_answer"] == {}


def test_build_clarification_resume_payload_includes_generate_speech() -> None:
    payload = build_clarification_resume_payload(thread_id="abc-123", generate_speech=True)
    assert payload["generate_speech"] is True


# --- call_converse -----------------------------------------------------------


def test_call_converse_posts_to_converse_path_and_returns_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        return httpx.Response(
            200,
            json={
                "thread_id": "abc-123",
                "status": "needs_clarification",
                "missing_fields": ["duration"],
                "clarification_questions": ["How long?"],
                "response_text": "How long?",
                "routing": None,
                "provider_search": None,
                "media_note": None,
                "audio": None,
                "disclaimer": "disclaimer text",
            },
        )

    transport = httpx.MockTransport(handler)

    def fake_post(url: str, *, json: object, timeout: float) -> httpx.Response:
        with httpx.Client(transport=transport) as client:
            return client.post(url, json=json, timeout=timeout)

    monkeypatch.setattr(httpx, "post", fake_post)

    result = call_converse("http://backend.test", {"intake": {"main_concern": "something"}})

    assert captured["method"] == "POST"
    assert captured["url"] == f"http://backend.test{CONVERSE_PATH}"
    assert result["thread_id"] == "abc-123"
    assert result["status"] == "needs_clarification"


def test_call_converse_raises_on_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Unknown or expired conversation thread_id."})

    transport = httpx.MockTransport(handler)

    def fake_post(url: str, *, json: object, timeout: float) -> httpx.Response:
        with httpx.Client(transport=transport) as client:
            return client.post(url, json=json, timeout=timeout)

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(httpx.HTTPStatusError):
        call_converse("http://backend.test", {"thread_id": "unknown"})
