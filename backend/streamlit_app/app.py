"""MedRoute AI portfolio-demo UI (Streamlit).

This is a lightweight demo UI only — not the planned production frontend
(React remains deferred to Phase 5, see docs/roadmap.md). It contains no
routing, ranking, provider-search, or conversation-orchestration logic of
its own; every decision is made by the backend's Phase 2B LangGraph-backed
`/api/v1/converse` endpoint via api_client.py.

Run (from backend/, with the API already running):
    uv run streamlit run streamlit_app/app.py

Nothing typed here is persisted or logged by this UI — it only forwards
the current form values to the backend for this one request/response.
"""

import base64
from typing import Any

import httpx
import streamlit as st

from streamlit_app.api_client import (
    DEFAULT_BASE_URL,
    build_clarification_resume_payload,
    build_conversation_payload,
    call_converse,
    followup_fields_from_missing,
    list_specialties,
    resolve_confirmed_voice_transcript,
    transcribe_audio_via_api,
)

# Mirrors the backend's SUPPORTED_AUDIO_EXTENSIONS (app/services/voice_transcription_service.py).
# Duplicated here only as a UI hint for the file picker — this app talks to the backend
# exclusively over HTTP and never imports its modules, so the backend remains the single
# source of truth and re-validates every upload itself.
SUPPORTED_VOICE_EXTENSIONS = ("mp3", "wav", "m4a", "flac", "webm")
TRANSCRIPT_REVIEW_MESSAGE = (
    "Please review and correct the transcript before continuing. "
    "Speech recognition may contain errors."
)
DURATION_UNIT_OPTIONS = ["", "hours", "days", "weeks", "months", "years"]

NON_DIAGNOSTIC_DISCLAIMER = (
    "MedRoute AI provides navigation assistance and does not diagnose conditions, "
    "recommend treatment, verify provider credentials, or provide emergency services."
)
EMERGENCY_SAFETY_MESSAGE = (
    "If you are experiencing a medical emergency, call 911 (in the United States) "
    "or seek immediate emergency care. MedRoute AI cannot provide emergency assistance."
)
PERMANENT_EMERGENCY_WARNING = (
    "If you believe you may be experiencing a medical emergency, do not use this demo. "
    "In the U.S., call 911."
)
NO_SPECIALTY_SELECTED_LABEL = "(let MedRoute AI suggest one)"

# A specialty/location combination confirmed to return a result against the
# NPPES fixture (see README.md "Navigation demo & specialty routing" for how
# this data is loaded). The button only pre-fills form inputs; the actual
# result always comes from a real call to the backend, never hardcoded here.
DEMO_SYMPTOMS = "annual checkup"
DEMO_CITY = "Springfield"
DEMO_STATE = "CA"
DEMO_DURATION_VALUE = 1
DEMO_DURATION_UNIT = "days"


def _reset_followup_answer_widgets() -> None:
    st.session_state["followup_duration_value"] = 0
    st.session_state["followup_duration_unit"] = ""
    st.session_state["followup_main_concern"] = ""


def _reset_navigation_state() -> None:
    """Clear any stored /api/v1/converse result, error, and its
    conversation thread_id, plus any in-progress clarification follow-up
    answers. A stale clarification round or thread_id must never be shown
    against a request that no longer matches what's on screen (new file,
    new/edited transcript, or the demo example)."""
    st.session_state["navigation_result"] = None
    st.session_state["navigation_error"] = None
    st.session_state["conversation_thread_id"] = None
    _reset_followup_answer_widgets()


def _submit_conversation(payload: dict[str, Any], base_url: str) -> None:
    """Call /api/v1/converse (Phase 2B, LangGraph-orchestrated) and store
    the outcome in session_state so it survives the rerun triggered by any
    later widget interaction (e.g. the follow-up "Continue with this
    information" button) — Streamlit reruns the whole script on every
    interaction, so anything not in session_state would otherwise be lost
    immediately."""
    try:
        result = call_converse(base_url, payload)
    except httpx.HTTPStatusError as exc:
        st.session_state["navigation_result"] = None
        st.session_state["navigation_error"] = (
            f"The backend rejected this request (HTTP {exc.response.status_code})."
        )
    except httpx.HTTPError:
        st.session_state["navigation_result"] = None
        st.session_state["navigation_error"] = (
            "Could not reach the MedRoute AI backend. Is it running?"
        )
    else:
        st.session_state["navigation_result"] = result
        st.session_state["navigation_error"] = None
        st.session_state["conversation_thread_id"] = result["thread_id"]
        _reset_followup_answer_widgets()


def _load_demo_example() -> None:
    st.session_state["symptoms_text"] = DEMO_SYMPTOMS
    st.session_state["main_concern"] = ""
    st.session_state["duration_value"] = DEMO_DURATION_VALUE
    st.session_state["duration_unit"] = DEMO_DURATION_UNIT
    st.session_state["city"] = DEMO_CITY
    st.session_state["state"] = DEMO_STATE
    st.session_state["postal_code"] = ""
    st.session_state["specialty_label"] = NO_SPECIALTY_SELECTED_LABEL
    st.session_state["emergency_concern"] = False
    _reset_navigation_state()


st.set_page_config(page_title="MedRoute AI (Demo)")

st.title("MedRoute AI — Navigation Demo")
st.caption(NON_DIAGNOSTIC_DISCLAIMER)
st.warning(PERMANENT_EMERGENCY_WARNING)

with st.sidebar:
    with st.expander("Developer settings", expanded=False):
        base_url = st.text_input("Backend URL", value=DEFAULT_BASE_URL)

try:
    specialties = list_specialties(base_url)
except httpx.HTTPError:
    specialties = []
    st.sidebar.warning(
        "Could not load the specialty catalog from the backend. You can still submit "
        "without selecting a preferred specialty."
    )

st.button(
    "Try demo example",
    on_click=_load_demo_example,
    help=(
        "Fills in a symptom and location known to match a provider already loaded "
        "from the NPPES fixture data — see README.md for details."
    ),
)

emergency_concern = st.checkbox("I am declaring a medical emergency", key="emergency_concern")
if emergency_concern:
    st.error(EMERGENCY_SAFETY_MESSAGE)

st.subheader("Voice intake (optional)")
uploaded_audio = st.file_uploader(
    "Upload an audio recording of your main concern",
    type=list(SUPPORTED_VOICE_EXTENSIONS),
    key="voice_audio_uploader",
    help="Speech-to-text only — this is not interpreted, diagnosed, or scored for urgency.",
)

# A newly selected file (including clearing the uploader) always resets any
# prior transcript and its confirmation — an old confirmation must never be
# silently reused for different audio.
current_audio_file_id = uploaded_audio.file_id if uploaded_audio is not None else None
if st.session_state.get("voice_last_audio_file_id") != current_audio_file_id:
    st.session_state["voice_last_audio_file_id"] = current_audio_file_id
    st.session_state["voice_transcript"] = ""
    st.session_state["voice_transcript_language"] = None
    st.session_state["voice_transcript_confirmed"] = False
    st.session_state["voice_transcript_last_seen"] = ""
    _reset_navigation_state()

if st.button("Transcribe audio", disabled=uploaded_audio is None):
    with st.spinner("Transcribing audio..."):
        try:
            transcription = transcribe_audio_via_api(
                base_url,
                uploaded_audio.getvalue(),
                filename=uploaded_audio.name,
                content_type=uploaded_audio.type,
            )
        except httpx.HTTPStatusError as exc:
            try:
                detail = exc.response.json().get("detail", "Transcription failed.")
            except ValueError:
                detail = "Transcription failed."
            st.error(f"Transcription failed (HTTP {exc.response.status_code}): {detail}")
        except httpx.HTTPError:
            st.error("Could not reach the MedRoute AI backend for transcription. Is it running?")
        else:
            st.session_state["voice_transcript"] = transcription["transcript"]
            st.session_state["voice_transcript_language"] = transcription.get("language")
            st.session_state["voice_transcript_confirmed"] = False
            st.session_state["voice_transcript_last_seen"] = transcription["transcript"]
            _reset_navigation_state()
            st.success("Transcription complete. Please review the transcript below.")

if st.session_state.get("voice_transcript"):
    if st.session_state["voice_transcript"] != st.session_state.get("voice_transcript_last_seen"):
        # The transcript text changed since we last saw it — either just-set
        # above (already matched, so this won't re-trigger) or a hand-edit by
        # the user, possibly after they had already confirmed it. Either way,
        # confirmation and any clarification follow-up tied to the old text
        # must not silently carry over.
        st.session_state["voice_transcript_confirmed"] = False
        _reset_navigation_state()
        st.session_state["voice_transcript_last_seen"] = st.session_state["voice_transcript"]

    st.text_area("Voice transcript", key="voice_transcript", height=120)
    st.caption(TRANSCRIPT_REVIEW_MESSAGE)
    st.checkbox(
        "I have reviewed this transcript and confirm it is ready to use",
        key="voice_transcript_confirmed",
    )

symptoms_text = st.text_area("Symptoms (one per line)", key="symptoms_text")
main_concern = st.text_input("Main concern", key="main_concern")

duration_col, unit_col = st.columns(2)
with duration_col:
    duration_value = st.number_input(
        "Duration", min_value=0, max_value=1000, step=1, key="duration_value"
    )
with unit_col:
    duration_unit = st.selectbox("Unit", DURATION_UNIT_OPTIONS, key="duration_unit")

city = st.text_input("City (optional)", key="city")
state = st.text_input("State (optional, 2-letter)", key="state")
postal_code = st.text_input("Postal code (optional)", key="postal_code")

specialty_labels = [NO_SPECIALTY_SELECTED_LABEL] + [s["display_name"] for s in specialties]
specialty_label = st.selectbox(
    "Preferred specialty (optional)",
    specialty_labels,
    key="specialty_label",
    help="Selecting one skips MedRoute AI's own specialty matching for this request.",
)
preferred_specialty = next(
    (s["slug"] for s in specialties if s["display_name"] == specialty_label), None
)

generate_speech = st.checkbox(
    "Generate spoken response (text-to-speech)",
    key="generate_speech",
    help="Uses Deepgram to synthesize the response as playable audio, in addition to text.",
)

submitted = st.button("Submit")

if submitted:
    symptoms = [line.strip() for line in symptoms_text.splitlines() if line.strip()]
    confirmed_voice_transcript = resolve_confirmed_voice_transcript(
        transcript=st.session_state.get("voice_transcript"),
        confirmed=st.session_state.get("voice_transcript_confirmed", False),
    )
    payload = build_conversation_payload(
        symptoms=symptoms or None,
        main_concern=main_concern or None,
        duration_value=int(duration_value) if duration_value else None,
        duration_unit=duration_unit or None,
        city=city or None,
        state=state or None,
        postal_code=postal_code or None,
        preferred_specialty=preferred_specialty,
        emergency_concern=emergency_concern,
        voice_transcript=confirmed_voice_transcript,
        voice_language=st.session_state.get("voice_transcript_language"),
        generate_speech=generate_speech,
    )
    _submit_conversation(payload, base_url)

# Rendered from session_state (not just inside `if submitted:`) so the result
# — and, for needs_clarification, the follow-up answer form below — survives
# the rerun triggered by clicking "Continue with this information".
navigation_error = st.session_state.get("navigation_error")
navigation_result = st.session_state.get("navigation_result")

if navigation_error:
    st.error(navigation_error)
elif navigation_result is not None:
    status = navigation_result["status"]

    if status == "emergency":
        st.error(navigation_result["response_text"])
    elif status == "needs_clarification":
        st.subheader("Additional information needed")
        missing_fields = navigation_result.get("missing_fields", [])
        followup_fields = followup_fields_from_missing(missing_fields)

        for question in navigation_result["clarification_questions"]:
            st.write(f"- {question}")

        followup_duration_value = None
        followup_duration_unit = None
        if "duration" in followup_fields:
            followup_col, followup_unit_col = st.columns(2)
            with followup_col:
                followup_duration_value = st.number_input(
                    "Duration", min_value=0, max_value=1000, step=1, key="followup_duration_value"
                )
            with followup_unit_col:
                followup_duration_unit = st.selectbox(
                    "Unit", DURATION_UNIT_OPTIONS, key="followup_duration_unit"
                )

        followup_main_concern = None
        if "concern" in followup_fields:
            followup_main_concern = st.text_input("Main concern", key="followup_main_concern")

        if st.button("Continue with this information", key="followup_submit_button"):
            resume_payload = build_clarification_resume_payload(
                thread_id=st.session_state["conversation_thread_id"],
                duration_value=(int(followup_duration_value) if followup_duration_value else None),
                duration_unit=followup_duration_unit or None,
                main_concern=followup_main_concern or None,
                generate_speech=st.session_state.get("generate_speech", False),
            )
            _submit_conversation(resume_payload, base_url)
            st.rerun()
    else:
        media_note = navigation_result.get("media_note")
        if media_note:
            st.warning(media_note)

        st.subheader("Specialty routing result")
        routing = navigation_result.get("routing")
        if routing and routing.get("specialty_slug"):
            st.success(f"Routed to: {routing['specialty_display_name']} — {routing['note']}")
        elif routing:
            st.warning(routing["note"])

        st.divider()
        st.subheader("Provider search results")
        search = navigation_result.get("provider_search")
        if search and search["results"]:
            for provider in search["results"]:
                st.write(f"**{provider['display_name']}** ({provider['entity_type_code']})")
                location = provider.get("practice_location")
                if location:
                    st.write(
                        f"{location.get('city', '')}, {location.get('state', '')} "
                        f"{location.get('postal_code', '')}"
                    )
            st.caption(search["disclaimer"])
        elif search is not None:
            st.info(
                "Specialty routing succeeded, but no matching providers are "
                "currently loaded for this location."
            )
        else:
            st.info("No specialty was matched, so provider search did not run.")

        response_text = navigation_result.get("response_text")
        if response_text:
            st.divider()
            st.subheader("Response")
            st.write(response_text)

    # Rendered for every status (emergency, needs_clarification, or
    # success) — spoken output is generated for whichever text was already
    # shown above (the emergency message, the clarification question, or
    # the routing summary), never a separate or duplicated message.
    audio = navigation_result.get("audio")
    if audio:
        st.audio(base64.b64decode(audio["audio_base64"]), format=audio["content_type"])

    st.caption(navigation_result["disclaimer"])
