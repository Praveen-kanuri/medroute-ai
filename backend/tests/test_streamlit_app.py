"""Tests for the Streamlit demo UI's page itself (streamlit_app/app.py),
using Streamlit's AppTest harness.

AppTest cannot fabricate a real `st.file_uploader` value (always None in a
headless run), so image/video tests inject that state directly via
session_state. `st.audio_input`, however, CAN be exercised end-to-end: the
underlying `streamlit.audio_input` function is monkeypatched to return a
`_FakeUploadedFile` (a plain io.BytesIO with the same
name/type/file_id/size attributes Streamlit's real UploadedFile has) —
this lets the automatic record -> transcribe -> submit -> speak pipeline
run for real, through the actual app.py code path, with only the network
calls (httpx.post) faked. No real audio, network, or backend is ever used.
"""

import io
from typing import Any

import httpx
import pytest
import streamlit
from streamlit.testing.v1 import AppTest


def _run_app() -> AppTest:
    at = AppTest.from_file("streamlit_app/app.py")
    at.run(timeout=30)
    return at


def _fake_result(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "thread_id": "synthetic-thread-id",
        "status": "ready_for_multimodal_processing",
        "intent": "medical_concern",
        "missing_fields": [],
        "clarification_questions": [],
        "response_text": "synthetic response",
        "routing": None,
        "provider_search": None,
        "media_note": None,
        "audio": None,
        "disclaimer": "synthetic disclaimer",
        "vision_observations": [],
        "media_analysis_note": None,
    }
    base.update(overrides)
    return base


def _seed_completed_turn(
    at: AppTest,
    result: dict[str, Any],
    *,
    voice_originated: bool = False,
    user_text: str = "synthetic user turn",
) -> AppTest:
    """Appends one user/assistant chat pair for an already-completed turn,
    plus the same navigation_result/thread_id bookkeeping
    _submit_conversation itself would have set — for tests that only care
    about rendering a given backend result, not about driving the whole
    STT/converse pipeline that would normally produce it."""
    turn_id = "seeded-turn"
    at.session_state["chat_messages"].append(
        {"role": "user", "text": user_text, "audio": None, "turn_id": turn_id}
    )
    at.session_state["chat_messages"].append(
        {
            "role": "assistant",
            "text": result.get("response_text") or "",
            "audio": result.get("audio"),
            "turn_id": turn_id,
            "result": result,
            "voice_originated": voice_originated,
        }
    )
    at.session_state["navigation_result"] = result
    at.session_state["navigation_error"] = None
    at.session_state["conversation_thread_id"] = result["thread_id"]
    at.session_state["conversation_originated_from_voice"] = voice_originated
    at.run(timeout=30)
    return at


class _FakeUploadedFile(io.BytesIO):
    """Stands in for Streamlit's UploadedFile (returned by both
    st.audio_input and st.file_uploader) — AppTest cannot fabricate a real
    one, so tests needing "a recording exists" monkeypatch
    streamlit.audio_input to return an instance of this instead."""

    def __init__(
        self,
        data: bytes,
        *,
        name: str = "audio.wav",
        content_type: str = "audio/wav",
        file_id: str = "fake-recording-1",
    ) -> None:
        super().__init__(data)
        self.name = name
        self.type = content_type
        self.file_id = file_id
        self.size = len(data)

    def getvalue(self) -> bytes:  # type: ignore[override]
        return self.getbuffer().tobytes()


def _patch_audio_input(
    monkeypatch: pytest.MonkeyPatch,
    recording: "_FakeUploadedFile | None",
    *,
    key: str = "voice_recording_0",
) -> None:
    """Monkeypatches streamlit.audio_input to return `recording` only for
    the given generation-keyed widget (matching app.py's
    `voice_recording_{generation}` key) and None for every other key --
    mirroring real Streamlit's behavior where rotating to a new widget key
    (as "Record again" does) always starts from an empty, un-recorded
    widget rather than replaying the old value."""

    def fake_audio_input(*args: object, **kwargs: object) -> "_FakeUploadedFile | None":
        return recording if kwargs.get("key") == key else None

    monkeypatch.setattr(streamlit, "audio_input", fake_audio_input)


def _patch_backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    transcript: str = "chest pain and palpitations",
    transcribe_status: int = 200,
    converse_response: dict[str, Any] | None = None,
    converse_status: int = 200,
) -> list[str]:
    """Fakes both POST /api/v1/voice/transcribe and POST /api/v1/converse
    over httpx.post, returning the list of URLs called (in call order) so
    tests can assert on exactly how many times — and in what order — each
    endpoint was hit, without any real network activity."""
    calls: list[str] = []
    default_converse = _fake_result(
        routing={
            "specialty_slug": "cardiology",
            "specialty_display_name": "Cardiology",
            "method": "keyword_match",
            "note": "Matched based on keywords related to Cardiology.",
        },
        provider_search={"results": [], "disclaimer": "synthetic provider disclaimer"},
        response_text="Cardiology looks like a good fit. Not a diagnosis.",
        audio={"audio_base64": "ZmFrZS1hdWRpbw==", "content_type": "audio/mpeg", "model": "aura"},
    )
    converse_response = converse_response if converse_response is not None else default_converse

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls.append(url)
        if "voice/transcribe" in url:
            if transcribe_status != 200:
                return httpx.Response(transcribe_status, json={"detail": "Transcription failed."})
            return httpx.Response(
                200,
                json={
                    "transcription_id": "synthetic-id",
                    "transcript": transcript,
                    "language": "en",
                    "model": "whisper-large-v3",
                    "status": "completed",
                },
            )
        if converse_status != 200:
            return httpx.Response(converse_status, json={"detail": "error"})
        return httpx.Response(200, json=converse_response)

    transport = httpx.MockTransport(handler)

    def fake_post(
        url: str,
        *,
        files: object = None,
        json: object = None,
        data: object = None,
        timeout: float | None = None,
        **_: object,
    ) -> httpx.Response:
        with httpx.Client(transport=transport) as client:
            if files is not None:
                return client.post(url, files=files, data=data, timeout=timeout)
            return client.post(url, json=json, timeout=timeout)

    monkeypatch.setattr(httpx, "post", fake_post)
    return calls


# --- page structure / wording (unaffected by the voice-flow rewrite) ------


def test_app_loads_without_error() -> None:
    at = _run_app()
    assert not at.exception


def test_hero_and_disclaimer_present() -> None:
    at = _run_app()
    markdown_text = " ".join(m.value for m in at.markdown)
    assert "Navigate to the right care with AI" in markdown_text
    assert "Describe your concern using text, voice, an image, or a short video." in markdown_text
    captions = " ".join(c.value for c in at.caption)
    assert "does not diagnose conditions" in captions


def test_emergency_checkbox_shows_safety_message() -> None:
    at = _run_app()
    at.checkbox(key="emergency_concern").set_value(True).run(timeout=30)
    assert not at.exception
    assert any("call 911" in e.value for e in at.error)


def test_image_and_video_remain_one_combined_control() -> None:
    at = _run_app()
    assert len(at.get("file_uploader")) == 1


def test_no_internal_graph_state_exposed_in_rendered_text() -> None:
    at = _run_app()
    _seed_completed_turn(at, _fake_result())
    all_text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    for forbidden in ("intake_request", "ConversationState", "vision_analysis_node", "checkpoint"):
        assert forbidden not in all_text


# --- manual controls removed (Step 2) ---------------------------------------


def test_manual_transcribe_button_absent() -> None:
    at = _run_app()
    labels = [b.label for b in at.button]
    assert "Transcribe recording" not in labels
    assert "Transcribe audio" not in labels


def test_global_generate_speech_checkbox_absent() -> None:
    at = _run_app()
    labels = [c.label for c in at.checkbox]
    assert not any("Generate spoken response" in label for label in labels)
    assert not any("generate_speech" == (c.key or "") for c in at.checkbox)


def test_transcript_confirmation_checkbox_absent() -> None:
    at = _run_app()
    at.session_state["voice_transcript"] = "chest pain"
    at.session_state["voice_submitted_transcript"] = "chest pain"
    at.run(timeout=30)
    labels = [c.label for c in at.checkbox]
    assert not any("reviewed this transcript" in label for label in labels)


def test_upload_an_audio_recording_wording_absent() -> None:
    at = _run_app()
    all_text = " ".join(c.value for c in at.caption) + " ".join(m.value for m in at.markdown)
    assert "Upload an audio recording" not in all_text


def test_microphone_is_the_only_recording_control() -> None:
    at = _run_app()
    assert len(at.get("audio_input")) == 1
    audio_widget = at.get("audio_input")[0]
    assert audio_widget.label == "Start speaking"


# --- automatic recording processing (Step 3) --------------------------------


def test_new_recording_automatically_calls_stt_and_converse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_backend(monkeypatch)
    _patch_audio_input(monkeypatch, _FakeUploadedFile(b"synthetic-recording-bytes"))

    at = _run_app()

    assert not at.exception
    assert calls == [
        "http://localhost:8000/api/v1/voice/transcribe",
        "http://localhost:8000/api/v1/converse",
    ]
    assert at.session_state["voice_transcript"] == "chest pain and palpitations"
    assert at.session_state["navigation_result"]["status"] == "ready_for_multimodal_processing"
    assert at.session_state["voice_state"] == "speech_ready"


def test_same_recording_not_transcribed_or_submitted_twice_across_reruns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_backend(monkeypatch)
    _patch_audio_input(monkeypatch, _FakeUploadedFile(b"synthetic-recording-bytes"))

    at = _run_app()
    assert len(calls) == 2

    calls.clear()
    at.run(timeout=30)  # an unrelated rerun with the SAME recording present
    assert not at.exception
    assert calls == []


def test_voice_conversation_request_always_uses_generate_speech_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "voice/transcribe" in url:
            return httpx.Response(
                200,
                json={
                    "transcription_id": "x",
                    "transcript": "chest pain",
                    "language": "en",
                    "model": "whisper-large-v3",
                    "status": "completed",
                },
            )
        captured["body"] = request.content
        return httpx.Response(200, json=_fake_result())

    transport = httpx.MockTransport(handler)

    def fake_post(url: str, *, files=None, json=None, timeout=None, **_: object) -> httpx.Response:
        with httpx.Client(transport=transport) as client:
            if files is not None:
                return client.post(url, files=files, timeout=timeout)
            return client.post(url, json=json, timeout=timeout)

    monkeypatch.setattr(httpx, "post", fake_post)
    _patch_audio_input(monkeypatch, _FakeUploadedFile(b"synthetic-recording-bytes"))

    _run_app()

    assert b'"generate_speech":true' in captured["body"]  # type: ignore[operator]


def test_no_recording_no_automatic_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_backend(monkeypatch)
    _patch_audio_input(monkeypatch, None)

    at = _run_app()

    assert not at.exception
    assert calls == []
    assert at.session_state["voice_state"] == "idle"


# --- successful response display (Step 9) -----------------------------------


def test_successful_voice_response_displays_text_and_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_backend(monkeypatch)
    _patch_audio_input(monkeypatch, _FakeUploadedFile(b"synthetic-recording-bytes"))

    at = _run_app()

    assert not at.exception
    markdown_text = " ".join(m.value for m in at.markdown)
    assert "Cardiology looks like a good fit" in markdown_text
    assert len(at.get("audio")) == 1
    audio_widget = at.get("audio")[0]
    assert audio_widget.proto.autoplay is True


def test_tts_failure_still_preserves_text_response(monkeypatch: pytest.MonkeyPatch) -> None:
    no_audio_result = _fake_result(
        routing={
            "specialty_slug": "cardiology",
            "specialty_display_name": "Cardiology",
            "method": "keyword_match",
            "note": "x",
        },
        provider_search={"results": [], "disclaimer": "x"},
        response_text="Cardiology looks like a good fit. Not a diagnosis.",
        audio=None,
    )
    _patch_backend(monkeypatch, converse_response=no_audio_result)
    _patch_audio_input(monkeypatch, _FakeUploadedFile(b"synthetic-recording-bytes"))

    at = _run_app()

    assert not at.exception
    markdown_text = " ".join(m.value for m in at.markdown)
    assert "Cardiology looks like a good fit" in markdown_text
    assert len(at.get("audio")) == 0
    captions = " ".join(c.value for c in at.caption)
    assert "Spoken response unavailable" in captions
    # The whole turn must not be treated as a failure just because TTS was.
    assert at.session_state["navigation_error"] is None


def test_stt_failure_shows_error_and_never_calls_converse(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_backend(monkeypatch, transcribe_status=503)
    _patch_audio_input(monkeypatch, _FakeUploadedFile(b"synthetic-recording-bytes"))

    at = _run_app()

    assert not at.exception
    assert calls == ["http://localhost:8000/api/v1/voice/transcribe"]
    assert at.session_state["voice_state"] == "error"
    assert at.session_state["voice_error"]
    assert any("Transcription failed" in e.value for e in at.error)


# --- clarification (Step 7) --------------------------------------------------


def test_clarification_preserves_transcript_and_thread_id(monkeypatch: pytest.MonkeyPatch) -> None:
    needs_clarification_result = _fake_result(
        status="needs_clarification",
        missing_fields=["duration"],
        clarification_questions=["How long have you been experiencing this concern?"],
        audio={"audio_base64": "ZmFrZS1hdWRpbw==", "content_type": "audio/mpeg", "model": "aura"},
    )
    _patch_backend(monkeypatch, converse_response=needs_clarification_result)
    _patch_audio_input(monkeypatch, _FakeUploadedFile(b"synthetic-recording-bytes"))

    at = _run_app()

    assert not at.exception
    assert at.session_state["voice_transcript"] == "chest pain and palpitations"
    assert at.session_state["conversation_thread_id"] == "synthetic-thread-id"
    assert at.session_state["navigation_result"]["status"] == "needs_clarification"


def test_clarification_resume_does_not_call_stt_again(monkeypatch: pytest.MonkeyPatch) -> None:
    needs_clarification_result = _fake_result(
        status="needs_clarification",
        missing_fields=["duration"],
        clarification_questions=["How long have you been experiencing this concern?"],
    )
    calls = _patch_backend(monkeypatch, converse_response=needs_clarification_result)
    _patch_audio_input(monkeypatch, _FakeUploadedFile(b"synthetic-recording-bytes"))

    at = _run_app()
    assert calls == [
        "http://localhost:8000/api/v1/voice/transcribe",
        "http://localhost:8000/api/v1/converse",
    ]
    calls.clear()

    gen = at.session_state["followup_widget_generation"]
    at.number_input(key=f"followup_duration_value_{gen}").set_value(3)
    at.selectbox(key=f"followup_duration_unit_{gen}").set_value("days")
    followup_button = next(b for b in at.button if b.label == "Continue with this information")
    followup_button.click().run(timeout=30)

    assert not at.exception
    assert "http://localhost:8000/api/v1/voice/transcribe" not in calls
    assert calls == ["http://localhost:8000/api/v1/converse"]


def test_clarification_resume_not_submitted_twice_across_reruns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    needs_clarification_result = _fake_result(
        status="needs_clarification",
        missing_fields=["duration"],
        clarification_questions=["How long have you been experiencing this concern?"],
    )
    calls = _patch_backend(monkeypatch, converse_response=needs_clarification_result)
    _patch_audio_input(monkeypatch, _FakeUploadedFile(b"synthetic-recording-bytes"))

    at = _run_app()
    calls.clear()

    gen = at.session_state["followup_widget_generation"]
    at.number_input(key=f"followup_duration_value_{gen}").set_value(3)
    at.selectbox(key=f"followup_duration_unit_{gen}").set_value("days")
    followup_button = next(b for b in at.button if b.label == "Continue with this information")
    followup_button.click().run(timeout=30)
    assert len(calls) == 1

    # An unrelated later rerun must not resubmit the same resume.
    calls.clear()
    at.run(timeout=30)
    assert calls == []


def test_voice_clarification_resume_uses_generate_speech_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    needs_clarification_result = _fake_result(
        status="needs_clarification",
        missing_fields=["duration"],
        clarification_questions=["How long have you been experiencing this concern?"],
    )
    captured_bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "voice/transcribe" in url:
            return httpx.Response(
                200,
                json={
                    "transcription_id": "x",
                    "transcript": "chest pain",
                    "language": "en",
                    "model": "whisper-large-v3",
                    "status": "completed",
                },
            )
        captured_bodies.append(request.content)
        if len(captured_bodies) == 1:
            return httpx.Response(200, json=needs_clarification_result)
        return httpx.Response(200, json=_fake_result())

    transport = httpx.MockTransport(handler)

    def fake_post(url: str, *, files=None, json=None, timeout=None, **_: object) -> httpx.Response:
        with httpx.Client(transport=transport) as client:
            if files is not None:
                return client.post(url, files=files, timeout=timeout)
            return client.post(url, json=json, timeout=timeout)

    monkeypatch.setattr(httpx, "post", fake_post)
    _patch_audio_input(monkeypatch, _FakeUploadedFile(b"synthetic-recording-bytes"))

    at = _run_app()
    gen = at.session_state["followup_widget_generation"]
    at.number_input(key=f"followup_duration_value_{gen}").set_value(3)
    at.selectbox(key=f"followup_duration_unit_{gen}").set_value("days")
    followup_button = next(b for b in at.button if b.label == "Continue with this information")
    followup_button.click().run(timeout=30)

    assert not at.exception
    assert len(captured_bodies) == 2
    assert b'"generate_speech":true' in captured_bodies[1]


def test_answering_clarification_by_voice_resumes_same_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Recording a *new* clip while a voice-originated turn is awaiting
    # clarification must be treated as the spoken answer to that question
    # -- resuming the existing thread -- never as an unrelated fresh turn
    # that discards the in-progress conversation (the bug this guards).
    needs_clarification_result = _fake_result(
        status="needs_clarification",
        missing_fields=["duration"],
        clarification_questions=["How long have you been experiencing this concern?"],
    )
    transcribe_calls = {"n": 0}
    converse_bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "voice/transcribe" in url:
            transcribe_calls["n"] += 1
            transcript = (
                "chest pain and palpitations"
                if transcribe_calls["n"] == 1
                else "I've had it for 3 days now"
            )
            return httpx.Response(
                200,
                json={
                    "transcription_id": "x",
                    "transcript": transcript,
                    "language": "en",
                    "model": "whisper-large-v3",
                    "status": "completed",
                },
            )
        converse_bodies.append(request.content)
        if len(converse_bodies) == 1:
            return httpx.Response(200, json=needs_clarification_result)
        return httpx.Response(200, json=_fake_result())

    transport = httpx.MockTransport(handler)

    def fake_post(url: str, *, files=None, json=None, timeout=None, **_: object) -> httpx.Response:
        with httpx.Client(transport=transport) as client:
            if files is not None:
                return client.post(url, files=files, timeout=timeout)
            return client.post(url, json=json, timeout=timeout)

    monkeypatch.setattr(httpx, "post", fake_post)
    _patch_audio_input(monkeypatch, _FakeUploadedFile(b"synthetic-recording-bytes"))

    at = _run_app()
    assert at.session_state["navigation_result"]["status"] == "needs_clarification"
    thread_id_before = at.session_state["conversation_thread_id"]
    assert transcribe_calls["n"] == 1
    assert len(converse_bodies) == 1

    # A new, different recording -- the spoken answer to "how long".
    _patch_audio_input(monkeypatch, _FakeUploadedFile(b"different-recording-bytes-answer"))
    at.run(timeout=30)

    assert not at.exception
    assert transcribe_calls["n"] == 2
    assert len(converse_bodies) == 2
    resume_body = converse_bodies[1]
    assert b'"thread_id":"synthetic-thread-id"' in resume_body
    assert b'"clarification_answer"' in resume_body
    assert b'"voice_transcript"' in resume_body
    assert b'"intake"' not in resume_body  # a resume, never a fresh new turn
    assert at.session_state["conversation_thread_id"] == thread_id_before
    assert at.session_state["navigation_result"]["status"] == "ready_for_multimodal_processing"


def test_voice_answered_clarification_not_resubmitted_on_unrelated_rerun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    needs_clarification_result = _fake_result(
        status="needs_clarification",
        missing_fields=["duration"],
        clarification_questions=["How long have you been experiencing this concern?"],
    )
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls.append(url)
        if "voice/transcribe" in url:
            return httpx.Response(
                200,
                json={
                    "transcription_id": "x",
                    "transcript": "synthetic transcript",
                    "language": "en",
                    "model": "whisper-large-v3",
                    "status": "completed",
                },
            )
        converse_calls = [c for c in calls if "voice/transcribe" not in c]
        if len(converse_calls) == 1:
            return httpx.Response(200, json=needs_clarification_result)
        return httpx.Response(200, json=_fake_result())

    transport = httpx.MockTransport(handler)

    def fake_post(url: str, *, files=None, json=None, timeout=None, **_: object) -> httpx.Response:
        with httpx.Client(transport=transport) as client:
            if files is not None:
                return client.post(url, files=files, timeout=timeout)
            return client.post(url, json=json, timeout=timeout)

    monkeypatch.setattr(httpx, "post", fake_post)
    _patch_audio_input(monkeypatch, _FakeUploadedFile(b"synthetic-recording-bytes"))

    at = _run_app()
    _patch_audio_input(monkeypatch, _FakeUploadedFile(b"different-recording-bytes-answer"))
    at.run(timeout=30)
    assert at.session_state["navigation_result"]["status"] == "ready_for_multimodal_processing"

    calls.clear()
    at.run(timeout=30)
    assert calls == []


# --- transcript correction (Step 6) ------------------------------------------


def test_what_i_heard_expander_shows_editable_transcript(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_backend(monkeypatch)
    _patch_audio_input(monkeypatch, _FakeUploadedFile(b"synthetic-recording-bytes"))

    at = _run_app()

    expander_labels = [e.label for e in at.expander]
    assert "What I heard" in expander_labels
    assert at.text_area(key="voice_transcript").value == "chest pain and palpitations"


def test_use_corrected_transcript_disabled_when_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_backend(monkeypatch)
    _patch_audio_input(monkeypatch, _FakeUploadedFile(b"synthetic-recording-bytes"))

    at = _run_app()
    button = next(b for b in at.button if b.label == "Use corrected transcript")
    assert button.disabled is True


def test_corrected_transcript_does_not_call_stt_and_creates_one_new_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_backend(monkeypatch)
    _patch_audio_input(monkeypatch, _FakeUploadedFile(b"synthetic-recording-bytes"))

    at = _run_app()
    assert calls == [
        "http://localhost:8000/api/v1/voice/transcribe",
        "http://localhost:8000/api/v1/converse",
    ]
    calls.clear()

    at.text_area(key="voice_transcript").set_value("chest pain and dizziness").run(timeout=30)
    button = next(b for b in at.button if b.label == "Use corrected transcript")
    assert button.disabled is False
    button.click().run(timeout=30)

    assert not at.exception
    assert calls == ["http://localhost:8000/api/v1/converse"]  # exactly one new turn, no STT
    assert at.session_state["voice_submitted_transcript"] == "chest pain and dizziness"

    # A further unrelated rerun must not resubmit the correction again.
    calls.clear()
    at.run(timeout=30)
    assert calls == []


# --- clarification widget generation fix (Step 8) ---------------------------


def test_continue_with_this_information_does_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression test for StreamlitAPIException
    # ("st.session_state.followup_duration_value cannot be modified after
    # the widget with key followup_duration_value is instantiated") — the
    # generation-based keys below must never collide with an
    # already-instantiated widget.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_fake_result())

    transport = httpx.MockTransport(handler)

    def fake_post(url: str, *, json: object, timeout: float) -> httpx.Response:
        with httpx.Client(transport=transport) as client:
            return client.post(url, json=json, timeout=timeout)

    monkeypatch.setattr(httpx, "post", fake_post)

    at = _run_app()
    at.session_state["navigation_result"] = _fake_result(
        status="needs_clarification",
        missing_fields=["duration"],
        clarification_questions=["How long have you been experiencing this concern?"],
    )
    at.session_state["conversation_thread_id"] = "synthetic-thread-id"
    at.run(timeout=30)
    assert not at.exception

    gen = at.session_state["followup_widget_generation"]
    at.number_input(key=f"followup_duration_value_{gen}").set_value(3)
    at.selectbox(key=f"followup_duration_unit_{gen}").set_value("days")
    followup_button = next(b for b in at.button if b.label == "Continue with this information")
    followup_button.click().run(timeout=30)

    assert not at.exception
    assert at.session_state["navigation_result"]["status"] == "ready_for_multimodal_processing"


def test_generation_based_followup_widgets_receive_fresh_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_fake_result())

    transport = httpx.MockTransport(handler)

    def fake_post(url: str, *, json: object, timeout: float) -> httpx.Response:
        with httpx.Client(transport=transport) as client:
            return client.post(url, json=json, timeout=timeout)

    monkeypatch.setattr(httpx, "post", fake_post)

    at = _run_app()
    at.session_state["navigation_result"] = _fake_result(
        status="needs_clarification",
        missing_fields=["duration"],
        clarification_questions=["How long have you been experiencing this concern?"],
    )
    at.session_state["conversation_thread_id"] = "synthetic-thread-id"
    at.run(timeout=30)

    gen_before = at.session_state["followup_widget_generation"]
    at.number_input(key=f"followup_duration_value_{gen_before}").set_value(7)
    followup_button = next(b for b in at.button if b.label == "Continue with this information")
    followup_button.click().run(timeout=30)

    gen_after = at.session_state["followup_widget_generation"]
    assert gen_after != gen_before
    # A fresh generation key was never set to 7 by us -- it must start at
    # the widget's natural default, not the old (now-orphaned) key's value.
    assert f"followup_duration_value_{gen_after}" not in at.session_state or (
        at.session_state[f"followup_duration_value_{gen_after}"] == 0
    )


# --- record again (Step 10) --------------------------------------------------


def test_record_again_disabled_when_idle() -> None:
    at = _run_app()
    record_again_button = next(b for b in at.button if b.label == "Record again")
    assert record_again_button.disabled is True


def test_record_again_clears_only_voice_derived_state(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_backend(monkeypatch)
    _patch_audio_input(monkeypatch, _FakeUploadedFile(b"synthetic-recording-bytes"))

    at = _run_app()
    assert at.session_state["voice_transcript"] == "chest pain and palpitations"
    assert at.session_state["navigation_result"] is not None

    # Unrelated Describe-tab input that must survive "Record again".
    at.text_input(key="main_concern").set_value("itchy rash").run(timeout=30)

    record_again_button = next(b for b in at.button if b.label == "Record again")
    assert record_again_button.disabled is False
    record_again_button.click().run(timeout=30)

    assert not at.exception
    assert at.session_state["voice_transcript"] == ""
    assert at.session_state["voice_submitted_transcript"] is None
    assert at.session_state["voice_turn_fingerprint"] is None
    assert at.session_state["voice_state"] == "idle"
    assert at.session_state["voice_error"] is None
    assert at.session_state["navigation_result"] is None
    assert at.session_state["voice_widget_generation"] == 1
    # Unrelated Describe-tab input must be untouched by a voice-only reset.
    assert at.session_state["main_concern"] == "itchy rash"


def test_record_again_does_not_clear_unrelated_describe_result() -> None:
    at = _run_app()
    # A result that did NOT originate from voice (e.g. a Describe-tab
    # submission) must survive "Record again" -- it belongs to a
    # different modality entirely. Seed some voice-side state too, purely
    # so the button is enabled for this check.
    at.session_state["navigation_result"] = _fake_result()
    at.session_state["conversation_originated_from_voice"] = False
    at.session_state["voice_transcript"] = "leftover text"
    # Non-idle purely so "Record again" is enabled for this check -- the
    # button's enabled/disabled state is keyed off voice_state, not the
    # mere presence of leftover transcript text.
    at.session_state["voice_state"] = "transcript_ready"
    at.run(timeout=30)

    record_again_button = next(b for b in at.button if b.label == "Record again")
    assert record_again_button.disabled is False
    record_again_button.click().run(timeout=30)

    assert not at.exception
    assert at.session_state["voice_transcript"] == ""
    assert at.session_state["navigation_result"] is not None
    assert at.session_state["navigation_result"]["thread_id"] == "synthetic-thread-id"


# --- Describe/media behavior after removing the shared checkbox (Step 2) ---


def test_describe_continue_hardcodes_generate_speech_false(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(200, json=_fake_result())

    transport = httpx.MockTransport(handler)

    def fake_post(url: str, *, json: object, timeout: float) -> httpx.Response:
        with httpx.Client(transport=transport) as client:
            return client.post(url, json=json, timeout=timeout)

    monkeypatch.setattr(httpx, "post", fake_post)

    at = _run_app()
    at.text_input(key="main_concern").set_value("itchy rash").run(timeout=30)
    continue_button = next(b for b in at.button if b.label == "Continue")
    assert continue_button.disabled is False
    continue_button.click().run(timeout=30)

    assert not at.exception
    assert b'"generate_speech":false' in captured["body"]  # type: ignore[operator]


def test_continue_disabled_until_valid_input_provided() -> None:
    at = _run_app()
    continue_button = next(b for b in at.button if b.label == "Continue")
    assert continue_button.disabled is True

    at.text_input(key="main_concern").set_value("itchy rash").run(timeout=30)
    continue_button = next(b for b in at.button if b.label == "Continue")
    assert continue_button.disabled is False


def test_success_result_renders_routing_and_response() -> None:
    at = _run_app()
    _seed_completed_turn(
        at,
        _fake_result(
            routing={
                "specialty_slug": "cardiology",
                "specialty_display_name": "Cardiology",
                "method": "keyword_match",
                "note": "Matched based on keywords related to Cardiology.",
            },
            provider_search={"results": [], "disclaimer": "synthetic provider disclaimer"},
            response_text="Cardiology looks like a good fit. Not a diagnosis.",
        ),
    )
    assert not at.exception
    markdown_text = " ".join(m.value for m in at.markdown)
    assert "Cardiology looks like a good fit" in markdown_text
    assert any("Routed to: Cardiology" in s.value for s in at.success)


def test_needs_clarification_result_renders_followup_controls() -> None:
    at = _run_app()
    at.session_state["navigation_result"] = _fake_result(
        status="needs_clarification",
        missing_fields=["duration"],
        clarification_questions=["How long have you been experiencing this concern?"],
    )
    at.session_state["conversation_thread_id"] = "synthetic-thread-id"
    at.run(timeout=30)
    assert not at.exception
    gen = at.session_state["followup_widget_generation"]
    assert at.number_input(key=f"followup_duration_value_{gen}") is not None
    followup_button = next(b for b in at.button if b.label == "Continue with this information")
    assert followup_button is not None


# --- Phase 3A: typed clinical-protocol clarification answers ----------------


def test_clinical_answer_renders_free_text_area_not_duration_or_concern_controls() -> None:
    # A pending Phase 3A protocol question (missing_fields=["clinical_answer"])
    # must render a generic free-text answer box -- never the
    # duration/main-concern controls meant for Phase 1C's generic fields.
    at = _run_app()
    at.session_state["navigation_result"] = _fake_result(
        status="needs_clarification",
        missing_fields=["clinical_answer"],
        clarification_questions=[
            "Did the swelling come on suddenly or gradually, and has it been getting worse?"
        ],
    )
    at.session_state["conversation_thread_id"] = "synthetic-thread-id"
    at.run(timeout=30)
    assert not at.exception
    gen = at.session_state["followup_widget_generation"]
    assert at.text_area(key=f"followup_clinical_answer_{gen}") is not None
    with pytest.raises(KeyError):
        at.number_input(key=f"followup_duration_value_{gen}")
    with pytest.raises(KeyError):
        at.text_input(key=f"followup_main_concern_{gen}")


def test_typed_clinical_answer_submits_via_clarification_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    needs_clarification_result = _fake_result(
        status="needs_clarification",
        missing_fields=["clinical_answer"],
        clarification_questions=[
            "Did the swelling come on suddenly or gradually, and has it been getting worse?"
        ],
    )
    completed_result = _fake_result(
        routing={
            "specialty_slug": "internal-medicine",
            "specialty_display_name": "Internal Medicine",
            "method": "keyword_match",
            "note": "Matched based on keywords related to Internal Medicine.",
        },
        provider_search={"results": [], "disclaimer": "synthetic provider disclaimer"},
        clinical_navigation={
            "protocol": "unilateral_leg_swelling",
            "summary_of_reported_information": "You reported left-sided leg swelling.",
            "possible_explanations": [],
            "important_unknowns": [],
            "recommended_care_level": "routine",
            "specialty_candidates": ["internal-medicine", "family-medicine"],
            "diagnosis": None,
            "treatment_recommendation": None,
            "disclaimer": "synthetic clinical disclaimer",
        },
    )
    converse_bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        converse_bodies.append(request.content)
        return httpx.Response(200, json=completed_result)

    transport = httpx.MockTransport(handler)

    def fake_post(url: str, *, files=None, json=None, timeout=None, **_: object) -> httpx.Response:
        with httpx.Client(transport=transport) as client:
            return client.post(url, json=json, timeout=timeout)

    monkeypatch.setattr(httpx, "post", fake_post)

    at = _run_app()
    _seed_completed_turn(
        at,
        needs_clarification_result,
        voice_originated=False,
        user_text="My left leg has been swollen for two days.",
    )

    gen = at.session_state["followup_widget_generation"]
    at.text_area(key=f"followup_clinical_answer_{gen}").set_value(
        "It came on gradually and has been getting worse."
    )
    followup_button = next(b for b in at.button if b.label == "Continue with this information")
    followup_button.click().run(timeout=30)

    assert not at.exception
    assert len(converse_bodies) == 1
    assert b'"clinical_answer_text":"It came on gradually' in converse_bodies[0]
    assert b'"thread_id":"synthetic-thread-id"' in converse_bodies[0]
    assert at.session_state["navigation_result"]["status"] == "ready_for_multimodal_processing"
    assert at.session_state["navigation_result"]["clinical_navigation"] is not None


def test_clinical_navigation_does_not_survive_record_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # State-isolation regression: navigation_result (which carries
    # clinical_navigation) is replaced wholesale, never merged, on every
    # real submission -- confirms a completed clinical-flow result cannot
    # linger into a later, unrelated voice turn in the same session.
    _patch_backend(
        monkeypatch,
        converse_response=_fake_result(
            routing={
                "specialty_slug": "internal-medicine",
                "specialty_display_name": "Internal Medicine",
                "method": "keyword_match",
                "note": "Matched based on keywords related to Internal Medicine.",
            },
            clinical_navigation={
                "protocol": "unilateral_leg_swelling",
                "summary_of_reported_information": "You reported left-sided leg swelling.",
                "possible_explanations": [],
                "important_unknowns": [],
                "recommended_care_level": "prompt",
                "specialty_candidates": ["internal-medicine", "family-medicine"],
                "diagnosis": None,
                "treatment_recommendation": None,
                "disclaimer": "synthetic clinical disclaimer",
            },
        ),
    )
    _patch_audio_input(monkeypatch, _FakeUploadedFile(b"synthetic-recording-bytes"))

    at = _run_app()
    assert at.session_state["navigation_result"]["clinical_navigation"] is not None

    record_again_button = next(b for b in at.button if b.label == "Record again")
    record_again_button.click().run(timeout=30)

    assert not at.exception
    assert at.session_state["navigation_result"] is None


# --- conversational assistant: greeting + chat history (this task) ----------


def test_initial_greeting_appears_once() -> None:
    at = _run_app()
    assert not at.exception
    messages = at.session_state["chat_messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"
    assert "MedAI" in messages[0]["text"]
    assert messages[0]["audio"] is None
    assert len(at.chat_message) == 1
    assert at.chat_message[0].name == "assistant"


def test_rerun_does_not_duplicate_the_greeting() -> None:
    at = _run_app()
    at.run(timeout=30)
    at.run(timeout=30)
    assert len(at.session_state["chat_messages"]) == 1


def test_hi_transcript_gets_classified_as_a_greeting_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    greeting_result = _fake_result(
        intent="greeting",
        response_text="Hi! I'm MedAI. How can I help you with your health concern today?",
    )
    _patch_backend(monkeypatch, transcript="Hi", converse_response=greeting_result)
    _patch_audio_input(monkeypatch, _FakeUploadedFile(b"synthetic-recording-bytes"))

    at = _run_app()

    assert not at.exception
    messages = at.session_state["chat_messages"]
    assert len(messages) == 3
    assert messages[1] == {
        "role": "user",
        "text": "Hi",
        "audio": None,
        "turn_id": messages[1]["turn_id"],
    }
    assert messages[2]["role"] == "assistant"
    assert "MedAI" in messages[2]["text"]


def test_good_evening_gets_a_time_of_day_greeting_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    greeting_result = _fake_result(
        intent="greeting",
        response_text=(
            "Good evening! I'm MedAI, your medical navigation assistant. "
            "What can I help you with today?"
        ),
    )
    _patch_backend(monkeypatch, transcript="Good evening", converse_response=greeting_result)
    _patch_audio_input(monkeypatch, _FakeUploadedFile(b"synthetic-recording-bytes"))

    at = _run_app()

    assistant_reply = at.session_state["chat_messages"][-1]["text"]
    assert "Good evening" in assistant_reply


def test_greeting_only_input_does_not_show_clarification_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_backend(monkeypatch, transcript="Hi", converse_response=_fake_result(intent="greeting"))
    _patch_audio_input(monkeypatch, _FakeUploadedFile(b"synthetic-recording-bytes"))

    at = _run_app()

    assert at.session_state["navigation_result"]["status"] != "needs_clarification"
    labels = [b.label for b in at.button]
    assert "Continue with this information" not in labels


def test_greeting_only_input_does_not_render_routing_or_provider_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_backend(monkeypatch, transcript="Hi", converse_response=_fake_result(intent="greeting"))
    _patch_audio_input(monkeypatch, _FakeUploadedFile(b"synthetic-recording-bytes"))

    at = _run_app()

    assert at.session_state["navigation_result"]["routing"] is None
    assert at.session_state["navigation_result"]["provider_search"] is None
    markdown_text = " ".join(m.value for m in at.markdown)
    assert "Specialty routing result" not in markdown_text
    assert "Provider search results" not in markdown_text


def test_medical_concern_still_reaches_routing_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_backend(monkeypatch, transcript="chest pain and palpitations")
    _patch_audio_input(monkeypatch, _FakeUploadedFile(b"synthetic-recording-bytes"))

    at = _run_app()

    assert at.session_state["navigation_result"]["routing"]["specialty_slug"] == "cardiology"
    markdown_text = " ".join(m.value for m in at.markdown)
    assert "Specialty routing result" in markdown_text


def test_medical_concern_gets_a_grounded_conversational_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_backend(monkeypatch, transcript="chest pain and palpitations")
    _patch_audio_input(monkeypatch, _FakeUploadedFile(b"synthetic-recording-bytes"))

    at = _run_app()

    assistant_reply = at.session_state["chat_messages"][-1]["text"]
    assert "Cardiology" in assistant_reply
    assert "not a diagnosis" in assistant_reply.lower()


def test_clarification_question_sounds_conversational(monkeypatch: pytest.MonkeyPatch) -> None:
    needs_clarification_result = _fake_result(
        status="needs_clarification",
        missing_fields=["duration"],
        clarification_questions=["How long have you been experiencing this concern?"],
        response_text="I can help with that. How long have you been experiencing this concern?",
    )
    _patch_backend(monkeypatch, converse_response=needs_clarification_result)
    _patch_audio_input(monkeypatch, _FakeUploadedFile(b"synthetic-recording-bytes"))

    at = _run_app()

    assistant_reply = at.session_state["chat_messages"][-1]["text"]
    assert assistant_reply.startswith("I can help with that.")
    assert "How long have you been experiencing this concern?" in assistant_reply


def test_voice_originated_greeting_requests_speech(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "voice/transcribe" in url:
            return httpx.Response(
                200,
                json={
                    "transcription_id": "x",
                    "transcript": "Hi",
                    "language": "en",
                    "model": "whisper-large-v3",
                    "status": "completed",
                },
            )
        captured["body"] = request.content
        return httpx.Response(200, json=_fake_result(intent="greeting"))

    transport = httpx.MockTransport(handler)

    def fake_post(url: str, *, files=None, json=None, timeout=None, **_: object) -> httpx.Response:
        with httpx.Client(transport=transport) as client:
            if files is not None:
                return client.post(url, files=files, timeout=timeout)
            return client.post(url, json=json, timeout=timeout)

    monkeypatch.setattr(httpx, "post", fake_post)
    _patch_audio_input(monkeypatch, _FakeUploadedFile(b"synthetic-recording-bytes"))

    _run_app()

    assert b'"generate_speech":true' in captured["body"]  # type: ignore[operator]


def test_same_voice_turn_does_not_autoplay_audio_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_backend(monkeypatch)
    _patch_audio_input(monkeypatch, _FakeUploadedFile(b"synthetic-recording-bytes"))

    at = _run_app()
    audio_widgets = at.get("audio")
    assert len(audio_widgets) == 1
    assert audio_widgets[0].proto.autoplay is True

    at.run(timeout=30)
    audio_widgets_after = at.get("audio")
    assert len(audio_widgets_after) == 1
    assert audio_widgets_after[0].proto.autoplay is False


def test_chat_messages_are_not_duplicated_across_reruns(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_backend(monkeypatch)
    _patch_audio_input(monkeypatch, _FakeUploadedFile(b"synthetic-recording-bytes"))

    at = _run_app()
    count_after_turn = len(at.session_state["chat_messages"])
    assert count_after_turn == 3  # welcome + user + assistant

    at.run(timeout=30)
    assert len(at.session_state["chat_messages"]) == count_after_turn


def test_record_again_preserves_the_rest_of_the_conversation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_backend(monkeypatch)
    _patch_audio_input(monkeypatch, _FakeUploadedFile(b"synthetic-recording-bytes"))

    at = _run_app()
    count_before = len(at.session_state["chat_messages"])
    assert count_before == 3

    record_again_button = next(b for b in at.button if b.label == "Record again")
    record_again_button.click().run(timeout=30)

    assert not at.exception
    assert len(at.session_state["chat_messages"]) == count_before
