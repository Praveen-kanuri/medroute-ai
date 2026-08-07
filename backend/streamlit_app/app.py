"""MedRoute AI portfolio-demo UI (Streamlit).

This is a lightweight demo UI only — not the planned production frontend
(React remains deferred to Phase 5, see docs/roadmap.md). It contains no
routing, ranking, provider-search, conversation-orchestration, or media/
vision logic of its own; every decision is made by the backend's
LangGraph-backed `/api/v1/converse` and `/api/v1/media/analyze` endpoints
via api_client.py. Custom styling below is purely cosmetic CSS layered on
top of Streamlit's own dark theme (backend/.streamlit/config.toml) — if
the <style> block failed to load for any reason, every control would
still render and function using that theme alone.

Run (from backend/, with the API already running):
    uv run streamlit run streamlit_app/app.py

Nothing typed or recorded here is persisted or logged by this UI — it
only forwards the current form values (and, for voice, the captured
microphone recording) to the backend for this one request/response.

Voice workflow (Speak tab): tapping the microphone is the only manual
step. Recording, transcription, conversation submission, and speech
generation all happen automatically in one guarded pipeline — see
_run_stt_and_submit()/_submit_voice_turn() and the VOICE_STATE_*
constants below. Every automatic call is guarded by a stable fingerprint
(see api_client.fingerprint_bytes/build_voice_turn_fingerprint) so a
Streamlit rerun triggered by an unrelated widget never repeats a
completed STT call, conversation turn, or clarification resume.

Conversation history: every submitted turn (Describe, Speak, or a
clarification answer) appends one user + one assistant message to
st.session_state["chat_messages"], rendered top-to-bottom with
st.chat_message — see _append_chat_turn()/_render_assistant_result_details().
The backend classifies each new turn's intent (greeting, medical_concern,
clarification_answer, or unsupported_or_unclear — see
app/services/intent_classification_service.py); a greeting reply never
carries routing/provider data, so this UI never invents any for one.
A fixed welcome message is seeded once per session and is never
resubmitted to the backend or auto-played.
"""

import base64
import uuid
from typing import Any

import httpx
import streamlit as st

from streamlit_app.api_client import (
    DEFAULT_BASE_URL,
    analyze_media_via_api,
    build_clarification_resume_payload,
    build_conversation_payload,
    build_navigation_payload,
    build_voice_turn_fingerprint,
    call_converse,
    fingerprint_bytes,
    followup_fields_from_missing,
    list_specialties,
    transcribe_audio_via_api,
    voice_recording_changed,
)

# Mirrors the backend's signature-sniffed formats (app/services/
# media_validation_service.py). A UI hint only — the backend re-validates
# every upload by its actual file signature, never by this extension list.
SUPPORTED_IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "webp")
SUPPORTED_VIDEO_EXTENSIONS = ("mp4", "mov", "webm")
SUPPORTED_MEDIA_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS + SUPPORTED_VIDEO_EXTENSIONS

TRANSCRIPT_EDIT_HINT = 'Edit if anything sounds wrong, then tap "Use corrected transcript."'
DURATION_UNIT_OPTIONS = ["", "hours", "days", "weeks", "months", "years"]

INITIAL_GREETING_TEXT = (
    "Hi, I'm MedAI, your medical navigation assistant. How can I help you today?"
)

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

# --- voice state machine ----------------------------------------------------
# idle -> recording_captured -> transcribing -> transcript_ready ->
# submitting -> needs_clarification | completed -> speech_ready, with a
# parallel "error" state. Reruns only ever *render* whichever of these is
# currently stored — see _submit_voice_turn()'s fingerprint guard for why
# an already-completed state is never recomputed by a later rerun.
VOICE_STATE_IDLE = "idle"
VOICE_STATE_RECORDING_CAPTURED = "recording_captured"
VOICE_STATE_TRANSCRIBING = "transcribing"
VOICE_STATE_TRANSCRIPT_READY = "transcript_ready"
VOICE_STATE_SUBMITTING = "submitting"
VOICE_STATE_NEEDS_CLARIFICATION = "needs_clarification"
VOICE_STATE_COMPLETED = "completed"
VOICE_STATE_SPEECH_READY = "speech_ready"
VOICE_STATE_ERROR = "error"

_VOICE_STATE_CAPTIONS = {
    VOICE_STATE_IDLE: "Ready — tap the microphone and start speaking.",
    VOICE_STATE_RECORDING_CAPTURED: "Listening complete.",
    VOICE_STATE_TRANSCRIBING: "Transcribing…",
    VOICE_STATE_TRANSCRIPT_READY: "Understanding your concern…",
    VOICE_STATE_SUBMITTING: "Preparing your guidance…",
    VOICE_STATE_NEEDS_CLARIFICATION: (
        "A bit more information is needed — just tap the microphone and answer out loud, "
        "or use the form below."
    ),
    VOICE_STATE_COMPLETED: "Guidance ready — see the response panel.",
    VOICE_STATE_SPEECH_READY: "Guidance ready — text and spoken response available.",
    VOICE_STATE_ERROR: "Something went wrong — see the message below.",
}

# Isolated, purely cosmetic CSS. Every selector below is either a class this
# module defines itself (mr-*) or Streamlit's own documented, stable
# `st-key-<key>` class (set via each element's `key=` argument) — never an
# autogenerated/unstable internal class name. Nothing here touches focus
# outlines, ARIA attributes, or label visibility; every widget keeps its
# native accessible markup, and the app remains fully usable if this block
# fails to load (backend/.streamlit/config.toml already provides an
# accessible dark theme on its own).
_CUSTOM_CSS = """
<style>
.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.stApp {
    background:
        radial-gradient(circle at 12% -10%, rgba(45, 212, 191, 0.14), transparent 45%),
        radial-gradient(circle at 88% 0%, rgba(56, 189, 248, 0.12), transparent 40%);
}

.mr-header {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    margin-bottom: 0.75rem;
}
.mr-brand {
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}
.mr-brand-tag {
    font-size: 0.82rem;
    color: rgba(226, 232, 240, 0.6);
}

.mr-hero-title {
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
    background: linear-gradient(90deg, #2dd4bf, #38bdf8);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.mr-hero-sub {
    font-size: 1rem;
    color: rgba(226, 232, 240, 0.75);
    margin: 0;
}

.st-key-hero_panel,
.st-key-describe_panel,
.st-key-speak_panel,
.st-key-media_panel,
.st-key-response_panel,
.st-key-emergency_panel {
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 16px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1rem;
}
.st-key-hero_panel {
    background: linear-gradient(135deg, rgba(45, 212, 191, 0.10), rgba(56, 189, 248, 0.06));
}
.st-key-describe_panel,
.st-key-speak_panel,
.st-key-media_panel,
.st-key-response_panel {
    background: rgba(255, 255, 255, 0.03);
}
.st-key-emergency_panel {
    background: rgba(248, 113, 113, 0.05);
    border-left: 3px solid #f87171;
    padding: 0.9rem 1.2rem;
}

.mr-badge {
    display: inline-block;
    padding: 0.15rem 0.65rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 0.6rem;
}
.mr-badge-danger { background: rgba(248, 113, 113, 0.16); color: #fca5a5; }
.mr-badge-warning { background: rgba(250, 204, 21, 0.16); color: #fde68a; }
.mr-badge-info { background: rgba(56, 189, 248, 0.16); color: #7dd3fc; }
.mr-badge-success { background: rgba(45, 212, 191, 0.16); color: #5eead4; }

div[class^="st-key-provider_card_"] {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 12px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.6rem;
}

div[data-testid="stButton"] > button {
    border-radius: 10px;
}
</style>
"""


def _reset_followup_answer_widgets() -> None:
    """Increments the follow-up widget-key generation counter so the next
    render instantiates brand-new number_input/selectbox/text_input
    widgets (see followup_widget_generation below) with their natural
    defaults — this never writes directly to an already-instantiated
    widget's session_state key, which Streamlit forbids within the same
    run (StreamlitAPIException: "... cannot be modified after the widget
    ... is instantiated")."""
    st.session_state["followup_widget_generation"] = (
        st.session_state.get("followup_widget_generation", 0) + 1
    )


def _reset_navigation_state() -> None:
    """Clear any stored /api/v1/converse or /api/v1/media/analyze result,
    error, and its conversation thread_id, plus any in-progress
    clarification follow-up answers. A stale clarification round or
    thread_id must never be shown against a request that no longer matches
    what's on screen (new recording, corrected transcript, new media file,
    or the demo example)."""
    st.session_state["navigation_result"] = None
    st.session_state["navigation_error"] = None
    st.session_state["conversation_thread_id"] = None
    st.session_state["conversation_originated_from_voice"] = False
    _reset_followup_answer_widgets()


def _ensure_voice_session_state() -> None:
    """One-time defaults for voice-only session-state keys. Never
    overwrites an existing value — this runs on every script execution,
    not just the first."""
    defaults: dict[str, Any] = {
        "voice_widget_generation": 0,
        "voice_last_recording_fingerprint": None,
        "voice_transcript": "",
        "voice_transcript_language": None,
        "voice_submitted_transcript": None,
        "voice_turn_fingerprint": None,
        "voice_state": VOICE_STATE_IDLE,
        "voice_error": None,
        "conversation_originated_from_voice": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _ensure_chat_session_state() -> None:
    """Seeds the conversation transcript with one fixed welcome message the
    first time a session runs — never re-added on a later rerun (the
    `if key not in st.session_state` guard only ever fires once per
    session), never sent to the backend, and never auto-played (its audio
    is always None)."""
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {
                "role": "assistant",
                "text": INITIAL_GREETING_TEXT,
                "audio": None,
                "turn_id": "welcome",
            }
        ]
    if "last_autoplayed_turn_id" not in st.session_state:
        st.session_state["last_autoplayed_turn_id"] = None


def _append_chat_turn(*, user_text: str, result: dict[str, Any], voice_originated: bool) -> None:
    """Appends one user + one assistant message for a turn that just
    completed successfully. Called exactly once per successful backend
    call (see _submit_conversation) — the same fingerprint/button-click
    guards that already prevent a duplicate API call also prevent a
    duplicate chat-history insertion."""
    turn_id = str(uuid.uuid4())
    st.session_state["chat_messages"].append(
        {"role": "user", "text": user_text, "audio": None, "turn_id": turn_id}
    )
    st.session_state["chat_messages"].append(
        {
            "role": "assistant",
            "text": result.get("response_text") or "",
            "audio": result.get("audio"),
            "turn_id": turn_id,
            "result": result,
            "voice_originated": voice_originated,
        }
    )


def _describe_followup_answer(
    *,
    duration_value: float | None,
    duration_unit: str | None,
    main_concern: str | None,
    clinical_answer_text: str | None = None,
) -> str:
    """A short, human-readable summary of a clarification answer, shown as
    the "user" chat bubble for a clarification-resume turn — the original
    concern is never restated, only what was just answered."""
    parts: list[str] = []
    if duration_value and duration_unit:
        parts.append(f"{int(duration_value)} {duration_unit}")
    if main_concern:
        parts.append(main_concern)
    if clinical_answer_text:
        parts.append(clinical_answer_text)
    return "; ".join(parts) if parts else "(clarification provided)"


def _render_assistant_result_details(result: dict[str, Any], *, turn_id: str) -> None:
    """Supplementary structured detail for one assistant chat turn —
    status badge, visual observations, specialty routing, and provider
    results — rendered under the turn's plain-text reply (already shown by
    the caller). Never rendered for a greeting or general-chat reply:
    routing/provider fields are always None/empty for both intents, since
    neither ever reaches specialty routing or provider search (see
    app/services/intent_classification_service.py)."""
    intent = result.get("intent")
    if intent not in ("greeting", "general_chat"):
        status = result.get("status") or ""
        _status_badge(status)
        media_note = result.get("media_note")
        if media_note:
            st.warning(media_note)
        _render_vision_observations(result)
        _render_clinical_navigation(result)

        if status not in ("needs_clarification", "emergency"):
            st.markdown("**Specialty routing result**")
            routing = result.get("routing")
            if routing and routing.get("specialty_slug"):
                st.success(f"Routed to: {routing['specialty_display_name']} — {routing['note']}")
            elif routing:
                st.warning(routing["note"])

            st.divider()
            st.markdown("**Provider search results**")
            search = result.get("provider_search")
            if search and search["results"]:
                for index, provider in enumerate(search["results"]):
                    with st.container(key=f"provider_card_{turn_id}_{index}"):
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

    disclaimer = result.get("disclaimer")
    if disclaimer:
        st.caption(disclaimer)


def _reset_voice_turn_state(*, preserve_navigation: bool = False) -> None:
    """Clears everything derived from the current voice recording/turn —
    transcript, transcript-edit tracking, turn fingerprint, voice status,
    and voice error — plus the shared conversation result, but ONLY when
    that result actually belongs to the active voice turn (never an
    unrelated Describe/Image-video result). Never touches Describe-tab
    fields, the media uploader, or media-analysis state.

    preserve_navigation=True keeps the shared navigation_result/thread
    intact even though it originated from voice — used when a new
    recording is actually the user's *spoken answer* to a pending
    clarification question (see the Speak tab's recording-handling block,
    which decides this before calling here), not an unrelated new
    concern that should discard the in-progress turn."""
    st.session_state["voice_transcript"] = ""
    st.session_state["voice_transcript_language"] = None
    st.session_state["voice_submitted_transcript"] = None
    st.session_state["voice_turn_fingerprint"] = None
    st.session_state["voice_state"] = VOICE_STATE_IDLE
    st.session_state["voice_error"] = None
    if not preserve_navigation and st.session_state.get("conversation_originated_from_voice"):
        _reset_navigation_state()


def _record_again() -> None:
    """Clears only voice-derived state (recording fingerprint, transcript,
    turn fingerprint, status, error, and the shared result if it belongs
    to voice) — never Describe-tab fields or media-tab state. Rotating
    the widget-key generation forces a brand-new st.audio_input instance
    with no residual recording, since the widget itself has no
    programmatic "clear" call."""
    st.session_state["voice_widget_generation"] = (
        st.session_state.get("voice_widget_generation", 0) + 1
    )
    st.session_state["voice_last_recording_fingerprint"] = None
    _reset_voice_turn_state()


def _voice_status_caption() -> str:
    state = st.session_state.get("voice_state", VOICE_STATE_IDLE)
    return _VOICE_STATE_CAPTIONS.get(state, _VOICE_STATE_CAPTIONS[VOICE_STATE_IDLE])


def _submit_conversation(
    payload: dict[str, Any],
    base_url: str,
    *,
    user_display_text: str,
    voice_originated: bool,
) -> None:
    """Call /api/v1/converse (LangGraph-orchestrated) and store the outcome
    in session_state so it survives the rerun triggered by any later widget
    interaction (e.g. the follow-up "Continue with this information"
    button) — Streamlit reruns the whole script on every interaction, so
    anything not in session_state would otherwise be lost immediately.

    On success, also appends this turn to the visible chat transcript (see
    _append_chat_turn) — exactly once per successful call, since every
    caller already guards against calling this more than once for the
    same turn (a fingerprint check for voice turns, a one-shot button
    click for typed turns)."""
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
        _append_chat_turn(
            user_text=user_display_text, result=result, voice_originated=voice_originated
        )


def _submit_voice_turn(
    payload: dict[str, Any],
    fingerprint: str,
    base_url: str,
    *,
    user_display_text: str,
    status: Any = None,
) -> None:
    """Submits one voice-originated conversation turn (initial, corrected-
    transcript, or clarification-resume) to /api/v1/converse — guarded so
    a fingerprint that matches the last-submitted voice turn is never
    resubmitted (a Streamlit rerun triggered by an unrelated widget must
    never repeat this call, LangGraph clarification-resume, provider
    search, chat-history insertion, or TTS generation)."""
    if fingerprint == st.session_state.get("voice_turn_fingerprint"):
        return

    st.session_state["voice_state"] = VOICE_STATE_SUBMITTING
    if status is not None:
        status.update(label=_VOICE_STATE_CAPTIONS[VOICE_STATE_SUBMITTING])

    _submit_conversation(
        payload, base_url, user_display_text=user_display_text, voice_originated=True
    )

    if st.session_state.get("navigation_error"):
        st.session_state["voice_state"] = VOICE_STATE_ERROR
        st.session_state["voice_error"] = st.session_state["navigation_error"]
        if status is not None:
            status.update(label="Something went wrong.", state="error")
        return

    st.session_state["conversation_originated_from_voice"] = True
    st.session_state["voice_turn_fingerprint"] = fingerprint
    result = st.session_state["navigation_result"]
    if result["status"] == "needs_clarification":
        st.session_state["voice_state"] = VOICE_STATE_NEEDS_CLARIFICATION
        if status is not None:
            status.update(
                label=_VOICE_STATE_CAPTIONS[VOICE_STATE_NEEDS_CLARIFICATION], state="complete"
            )
    else:
        st.session_state["voice_state"] = (
            VOICE_STATE_SPEECH_READY if result.get("audio") else VOICE_STATE_COMPLETED
        )
        if status is not None:
            status.update(
                label=_VOICE_STATE_CAPTIONS[st.session_state["voice_state"]], state="complete"
            )


def _run_stt_and_submit(
    recording_bytes: bytes,
    filename: str,
    content_type: str | None,
    base_url: str,
    *,
    clarification_resume_thread_id: str | None = None,
) -> None:
    """Automatically transcribes a newly captured recording and, on
    success, immediately submits it — the user never manually triggers
    either step. Only ever called once per new recording (guarded by the
    recording fingerprint at the call site), and never writes audio to
    disk or logs the transcript/audio bytes.

    clarification_resume_thread_id, when set, means this recording is the
    user's *spoken answer* to a pending clarification question (see the
    Speak tab's recording-handling block, which decides this before
    calling here) — the transcript is submitted as a clarification resume
    against the existing thread instead of starting a brand-new turn, so
    answering "how long have you had this" by talking never loses the
    original concern or restarts the conversation."""
    st.session_state["voice_state"] = VOICE_STATE_TRANSCRIBING
    with st.status(_VOICE_STATE_CAPTIONS[VOICE_STATE_RECORDING_CAPTURED], expanded=True) as status:
        status.update(label=_VOICE_STATE_CAPTIONS[VOICE_STATE_TRANSCRIBING])
        try:
            transcription = transcribe_audio_via_api(
                base_url, recording_bytes, filename=filename, content_type=content_type
            )
        except httpx.HTTPStatusError as exc:
            try:
                detail = exc.response.json().get("detail", "Transcription failed.")
            except ValueError:
                detail = "Transcription failed."
            message = f"Transcription failed (HTTP {exc.response.status_code}): {detail}"
            st.session_state["voice_state"] = VOICE_STATE_ERROR
            st.session_state["voice_error"] = message
            status.update(label="Transcription failed.", state="error")
            return
        except httpx.HTTPError:
            message = "Could not reach the MedRoute AI backend for transcription. Is it running?"
            st.session_state["voice_state"] = VOICE_STATE_ERROR
            st.session_state["voice_error"] = message
            status.update(label="Transcription failed.", state="error")
            return

        transcript = transcription["transcript"]
        st.session_state["voice_transcript"] = transcript
        st.session_state["voice_transcript_language"] = transcription.get("language")
        st.session_state["voice_state"] = VOICE_STATE_TRANSCRIPT_READY
        status.update(label=_VOICE_STATE_CAPTIONS[VOICE_STATE_TRANSCRIPT_READY])

        if clarification_resume_thread_id is not None:
            payload = build_clarification_resume_payload(
                thread_id=clarification_resume_thread_id,
                voice_transcript=transcript,
                generate_speech=True,
            )
            fingerprint = build_voice_turn_fingerprint(
                transcript=st.session_state.get("voice_submitted_transcript"),
                thread_id=clarification_resume_thread_id,
                clarification_answer={"voice_transcript": transcript},
            )
        else:
            payload = build_conversation_payload(
                voice_transcript=transcript,
                voice_language=transcription.get("language"),
                generate_speech=True,
            )
            fingerprint = build_voice_turn_fingerprint(transcript=transcript)

        _submit_voice_turn(
            payload, fingerprint, base_url, user_display_text=transcript, status=status
        )
        if st.session_state.get("voice_state") not in (VOICE_STATE_ERROR,):
            st.session_state["voice_submitted_transcript"] = transcript


def _submit_media_analysis(
    media_bytes: bytes,
    *,
    filename: str,
    content_type: str | None,
    intake_payload: dict[str, Any],
    generate_speech: bool,
    base_url: str,
) -> None:
    """Call /api/v1/media/analyze and store the outcome in session_state
    exactly like _submit_conversation does, so the same rendering code
    below (routing, provider results, response text, audio, and visual
    observations) displays either kind of turn."""
    try:
        result = analyze_media_via_api(
            base_url,
            media_bytes,
            filename=filename,
            content_type=content_type,
            intake_payload=intake_payload,
            generate_speech=generate_speech,
        )
    except httpx.HTTPStatusError as exc:
        st.session_state["navigation_result"] = None
        st.session_state["navigation_error"] = (
            f"The backend rejected this upload (HTTP {exc.response.status_code})."
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
        st.session_state["conversation_originated_from_voice"] = False
        _reset_followup_answer_widgets()


def _render_vision_observations(result: dict[str, Any]) -> None:
    """Shared by both the needs_clarification and success branches below —
    a directly-uploaded image/video may already have produced controlled
    visual observations even while another field (e.g. duration) is still
    missing."""
    media_analysis_note = result.get("media_analysis_note")
    if media_analysis_note:
        st.info(media_analysis_note)

    observations = result.get("vision_observations")
    if not observations:
        return

    st.markdown("**Visual observations** _(a visual review only, not a diagnosis)_")
    for observation in observations:
        area = f" ({observation['body_area']})" if observation.get("body_area") else ""
        st.write(
            f"- **{observation['observation_type']}**{area}: {observation['visual_description']}"
        )
        if observation.get("visible_attributes"):
            st.caption("Attributes: " + ", ".join(observation["visible_attributes"]))
        st.caption(f"Confidence: {observation['confidence']} — {observation['limitations']}")


def _render_clinical_navigation(result: dict[str, Any]) -> None:
    """Phase 3A: the bounded, non-diagnostic clinical-navigation summary a
    completed complaint protocol (e.g. leg swelling) produced — see
    app.schemas.clinical_context.ClinicalNavigationSummary. Always absent
    for a concern matching no protocol, so this renders nothing for every
    other conversation."""
    navigation = result.get("clinical_navigation")
    if not navigation:
        return

    st.markdown("**Clinical-navigation summary** _(educational information, not a diagnosis)_")
    st.write(navigation["summary_of_reported_information"])

    for category in navigation.get("possible_explanations") or []:
        st.write(f"- **{category['category']}**: {category['description']}")
        if category.get("supporting_evidence"):
            st.caption("Based on: " + ", ".join(category["supporting_evidence"]))

    if navigation.get("important_unknowns"):
        st.caption("Still unknown: " + ", ".join(navigation["important_unknowns"]))

    care_level = navigation.get("recommended_care_level")
    if care_level == "prompt":
        st.warning(
            "Recommended care level: prompt in-person clinical evaluation — it's advisable "
            "to be seen soon rather than waiting for a routine appointment."
        )
    else:
        st.info(
            "Recommended care level: routine — scheduling a regular appointment should be "
            "appropriate."
        )

    st.caption(navigation["disclaimer"])


def _status_badge(status: str) -> None:
    """A compact "current processing step" indicator — never exposes raw
    graph state, only the same typed status the API already returns."""
    tone, label = {
        "emergency": ("danger", "Emergency"),
        "needs_clarification": ("warning", "More information needed"),
        "ready_for_multimodal_processing": ("success", "Complete"),
    }.get(status, ("info", status))
    st.markdown(f'<span class="mr-badge mr-badge-{tone}">{label}</span>', unsafe_allow_html=True)


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


st.set_page_config(page_title="MedRoute AI", layout="wide", initial_sidebar_state="collapsed")
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

if "followup_widget_generation" not in st.session_state:
    st.session_state["followup_widget_generation"] = 0
_ensure_voice_session_state()
_ensure_chat_session_state()

st.markdown(
    '<div class="mr-header"><span class="mr-brand">MedRoute AI</span>'
    '<span class="mr-brand-tag">Navigation demo</span></div>',
    unsafe_allow_html=True,
)

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

with st.container(key="hero_panel"):
    st.markdown(
        '<p class="mr-hero-title">Navigate to the right care with AI</p>'
        '<p class="mr-hero-sub">Describe your concern using text, voice, an image, '
        "or a short video.</p>",
        unsafe_allow_html=True,
    )

st.caption(NON_DIAGNOSTIC_DISCLAIMER)

with st.container(key="emergency_panel"):
    st.warning(PERMANENT_EMERGENCY_WARNING)
    emergency_concern = st.checkbox("I am declaring a medical emergency", key="emergency_concern")
    if emergency_concern:
        st.error(EMERGENCY_SAFETY_MESSAGE)

st.button(
    "Try demo example",
    on_click=_load_demo_example,
    key="demo_example_button",
    help=(
        "Fills in a symptom and location known to match a provider already loaded "
        "from the NPPES fixture data — see README.md for details."
    ),
)

left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    describe_tab, speak_tab, media_tab = st.tabs(["Describe", "Speak", "Image / Video"])

    with describe_tab:
        with st.container(key="describe_panel"):
            main_concern = st.text_input("Main concern", key="main_concern")
            symptoms_text = st.text_area("Symptoms (one per line)", key="symptoms_text")

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

            specialty_labels = [NO_SPECIALTY_SELECTED_LABEL] + [
                s["display_name"] for s in specialties
            ]
            specialty_label = st.selectbox(
                "Preferred specialty (optional)",
                specialty_labels,
                key="specialty_label",
                help="Selecting one skips MedRoute AI's own specialty matching for this request.",
            )
            preferred_specialty = next(
                (s["slug"] for s in specialties if s["display_name"] == specialty_label), None
            )

    with speak_tab:
        with st.container(key="speak_panel"):
            st.markdown("#### Tell us your main concern")
            st.caption("Tap the microphone and describe what you’re experiencing.")

            _, mic_center_col, _ = st.columns([1, 2, 1])
            with mic_center_col:
                recording = st.audio_input(
                    "Start speaking",
                    key=f"voice_recording_{st.session_state['voice_widget_generation']}",
                )

            recording_bytes = recording.getvalue() if recording is not None else None
            recording_fingerprint = (
                fingerprint_bytes(recording_bytes) if recording_bytes is not None else None
            )

            # A newly captured recording (including the recorder being
            # cleared) always resets prior voice-turn state and, if a new
            # recording is present, automatically runs the whole
            # transcribe -> submit pipeline — no manual trigger, and never
            # repeated on a later rerun (recording_fingerprint is stored
            # before this block ever returns).
            #
            # If a voice-originated turn is currently awaiting a
            # clarification answer, a new recording is the user *answering
            # that question by speaking* — captured here, before the reset
            # below, so the reset can preserve the in-progress result/
            # thread instead of discarding it as if this were an unrelated
            # new concern (see _reset_voice_turn_state/_run_stt_and_submit).
            awaiting_voice_clarification = (
                bool(st.session_state.get("conversation_originated_from_voice"))
                and st.session_state.get("navigation_result") is not None
                and st.session_state["navigation_result"]["status"] == "needs_clarification"
            )
            clarification_resume_thread_id = (
                st.session_state.get("conversation_thread_id")
                if awaiting_voice_clarification
                else None
            )
            if voice_recording_changed(
                previous_recording_id=st.session_state.get("voice_last_recording_fingerprint"),
                current_recording_id=recording_fingerprint,
            ):
                st.session_state["voice_last_recording_fingerprint"] = recording_fingerprint
                _reset_voice_turn_state(preserve_navigation=awaiting_voice_clarification)
                if recording_bytes is not None:
                    st.session_state["voice_state"] = VOICE_STATE_RECORDING_CAPTURED
                    assert recording is not None
                    _run_stt_and_submit(
                        recording_bytes,
                        recording.name,
                        recording.type,
                        base_url,
                        clarification_resume_thread_id=clarification_resume_thread_id,
                    )

            status_col, record_again_col = st.columns([3, 1])
            with status_col:
                st.caption(_voice_status_caption())
            with record_again_col:
                st.button(
                    "Record again",
                    key="record_again_button",
                    on_click=_record_again,
                    disabled=st.session_state.get("voice_state") == VOICE_STATE_IDLE,
                    width="stretch",
                )

            if st.session_state.get("voice_error"):
                st.error(st.session_state["voice_error"])

            if st.session_state.get("voice_transcript"):
                with st.expander("What I heard", expanded=False):
                    st.text_area(
                        "Transcript",
                        key="voice_transcript",
                        height=100,
                        label_visibility="collapsed",
                    )
                    st.caption(TRANSCRIPT_EDIT_HINT)

                    current_text = st.session_state["voice_transcript"].strip()
                    submitted_text = (
                        st.session_state.get("voice_submitted_transcript") or ""
                    ).strip()
                    transcript_differs = bool(current_text) and current_text != submitted_text

                    if st.button(
                        "Use corrected transcript",
                        key="use_corrected_transcript_button",
                        disabled=not transcript_differs,
                    ):
                        payload = build_conversation_payload(
                            voice_transcript=current_text,
                            voice_language=st.session_state.get("voice_transcript_language"),
                            generate_speech=True,
                        )
                        fingerprint = build_voice_turn_fingerprint(transcript=current_text)
                        with st.status(
                            _VOICE_STATE_CAPTIONS[VOICE_STATE_SUBMITTING], expanded=True
                        ) as status:
                            _submit_voice_turn(
                                payload,
                                fingerprint,
                                base_url,
                                user_display_text=current_text,
                                status=status,
                            )
                            if st.session_state.get("voice_state") != VOICE_STATE_ERROR:
                                st.session_state["voice_submitted_transcript"] = current_text

    symptoms = [line.strip() for line in symptoms_text.splitlines() if line.strip()]

    with media_tab:
        with st.container(key="media_panel"):
            st.markdown("#### Add an image or short video")
            st.caption("Share a clear image or short video of the visible concern.")

            uploaded_media = st.file_uploader(
                "Add an image or short video",
                type=list(SUPPORTED_MEDIA_EXTENSIONS),
                key="media_uploader",
                label_visibility="collapsed",
                help=(
                    "Produces controlled, non-diagnostic visual observations only — this is "
                    "a visual review, not a diagnosis."
                ),
            )
            st.caption(
                "Supported: "
                + ", ".join(ext.upper() for ext in SUPPORTED_MEDIA_EXTENSIONS)
                + ". Oversized or unsupported files are rejected with a clear message."
            )

            # A newly selected file (including clearing the uploader) always
            # invalidates any prior media-analysis result — a stale result
            # must never be shown against different media.
            current_media_file_id = uploaded_media.file_id if uploaded_media is not None else None
            if st.session_state.get("media_last_file_id") != current_media_file_id:
                st.session_state["media_last_file_id"] = current_media_file_id
                _reset_navigation_state()

            if uploaded_media is not None:
                media_extension = (
                    uploaded_media.name.rsplit(".", 1)[-1].lower()
                    if "." in uploaded_media.name
                    else ""
                )
                if media_extension in SUPPORTED_IMAGE_EXTENSIONS:
                    st.image(uploaded_media.getvalue())
                else:
                    st.video(uploaded_media.getvalue())

            if uploaded_media is not None and st.button(
                "Analyze media", key="analyze_media_button"
            ):
                intake_payload = build_navigation_payload(
                    symptoms=symptoms or None,
                    main_concern=main_concern or None,
                    duration_value=int(duration_value) if duration_value else None,
                    duration_unit=duration_unit or None,
                    city=city or None,
                    state=state or None,
                    postal_code=postal_code or None,
                    preferred_specialty=preferred_specialty,
                    emergency_concern=emergency_concern,
                    voice_transcript=(st.session_state.get("voice_transcript") or None),
                    voice_language=st.session_state.get("voice_transcript_language"),
                )
                with st.spinner("Analyzing media..."):
                    _submit_media_analysis(
                        uploaded_media.getvalue(),
                        filename=uploaded_media.name,
                        content_type=uploaded_media.type,
                        intake_payload=intake_payload,
                        generate_speech=False,
                        base_url=base_url,
                    )

    continue_disabled = not (
        bool(main_concern.strip())
        or bool(symptoms)
        or bool((st.session_state.get("voice_transcript") or "").strip())
    )
    if st.button(
        "Continue",
        key="continue_button",
        type="primary",
        disabled=continue_disabled,
        width="stretch",
    ):
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
            voice_transcript=(st.session_state.get("voice_transcript") or None),
            voice_language=st.session_state.get("voice_transcript_language"),
            generate_speech=False,
        )
        st.session_state["conversation_originated_from_voice"] = False
        describe_user_text = main_concern.strip() or (
            ", ".join(symptoms) if symptoms else "(described symptoms)"
        )
        _submit_conversation(
            payload, base_url, user_display_text=describe_user_text, voice_originated=False
        )

with right_col:
    with st.container(key="response_panel"):
        # The whole conversation so far — the fixed welcome message plus
        # one user/assistant pair per completed turn (see
        # _append_chat_turn). Rendered from session_state on every rerun,
        # so it survives whatever triggered that rerun (an unrelated
        # widget, or the follow-up answer form below) without ever being
        # recomputed or resubmitted.
        chat_messages = st.session_state["chat_messages"]

        # Autoplay is attempted at most once per new voice-originated
        # turn — the *latest* message only, and only the first time it is
        # rendered (see last_autoplayed_turn_id) — so an unrelated rerun
        # never replays an already-heard response.
        autoplay_turn_id = None
        if (
            chat_messages
            and chat_messages[-1]["role"] == "assistant"
            and chat_messages[-1].get("voice_originated")
            and chat_messages[-1].get("audio")
            and chat_messages[-1]["turn_id"] != st.session_state.get("last_autoplayed_turn_id")
        ):
            autoplay_turn_id = chat_messages[-1]["turn_id"]

        # Wrapped in its own keyed container (rather than looping directly
        # into response_panel) so this variable-length, per-rerun-growing
        # feed gets a stable anchor in Streamlit's element tree, distinct
        # from the follow-up form below — keeps the two from being
        # conflated across the extra internal rerun a clarification-resume
        # click triggers (see _reset_followup_answer_widgets/st.rerun()
        # below).
        with st.container(key="chat_history_feed"):
            for message in chat_messages:
                with st.chat_message(message["role"]):
                    result = message.get("result")
                    if result is not None and result.get("status") == "emergency":
                        st.error(message["text"])
                    else:
                        st.write(message["text"])

                    if result is not None:
                        _render_assistant_result_details(result, turn_id=message["turn_id"])

                    audio = message.get("audio")
                    if audio:
                        st.audio(
                            base64.b64decode(audio["audio_base64"]),
                            format=audio["content_type"],
                            autoplay=message["turn_id"] == autoplay_turn_id,
                        )
                        if message.get("voice_originated"):
                            st.caption("Tap play to hear MedAI's response.")
                    elif message.get("voice_originated"):
                        st.caption("Spoken response unavailable right now.")

        if autoplay_turn_id is not None:
            st.session_state["last_autoplayed_turn_id"] = autoplay_turn_id

        navigation_error = st.session_state.get("navigation_error")
        if navigation_error:
            st.error(navigation_error)

        navigation_result = st.session_state.get("navigation_result")
        is_voice_turn = bool(st.session_state.get("conversation_originated_from_voice"))

        # The interactive follow-up form only ever concerns the *latest*
        # turn — rendered below the transcript, outside the read-only chat
        # loop above, exactly like before.
        if navigation_result is not None and navigation_result["status"] == "needs_clarification":
            missing_fields = navigation_result.get("missing_fields", [])
            followup_fields = followup_fields_from_missing(missing_fields)

            gen = st.session_state["followup_widget_generation"]
            followup_duration_value = None
            followup_duration_unit = None
            if "duration" in followup_fields:
                followup_col, followup_unit_col = st.columns(2)
                with followup_col:
                    followup_duration_value = st.number_input(
                        "Duration",
                        min_value=0,
                        max_value=1000,
                        step=1,
                        key=f"followup_duration_value_{gen}",
                    )
                with followup_unit_col:
                    followup_duration_unit = st.selectbox(
                        "Unit", DURATION_UNIT_OPTIONS, key=f"followup_duration_unit_{gen}"
                    )

            followup_main_concern = None
            if "concern" in followup_fields:
                followup_main_concern = st.text_input(
                    "Main concern", key=f"followup_main_concern_{gen}"
                )

            followup_clinical_answer = None
            if "clinical_answer" in followup_fields:
                followup_clinical_answer = st.text_area(
                    "Your answer", key=f"followup_clinical_answer_{gen}"
                )

            if st.button("Continue with this information", key="followup_submit_button"):
                resume_thread_id = st.session_state["conversation_thread_id"]
                resume_payload = build_clarification_resume_payload(
                    thread_id=resume_thread_id,
                    duration_value=(
                        int(followup_duration_value) if followup_duration_value else None
                    ),
                    duration_unit=followup_duration_unit or None,
                    main_concern=followup_main_concern or None,
                    clinical_answer_text=followup_clinical_answer or None,
                    generate_speech=is_voice_turn,
                )
                followup_answer_text = _describe_followup_answer(
                    duration_value=followup_duration_value,
                    duration_unit=followup_duration_unit,
                    main_concern=followup_main_concern,
                    clinical_answer_text=followup_clinical_answer,
                )
                if is_voice_turn:
                    clarification_answer: dict[str, Any] = {}
                    if followup_duration_value and followup_duration_unit:
                        clarification_answer["duration"] = {
                            "value": int(followup_duration_value),
                            "unit": followup_duration_unit,
                        }
                    if followup_main_concern:
                        clarification_answer["main_concern"] = followup_main_concern
                    if followup_clinical_answer:
                        clarification_answer["clinical_answer_text"] = followup_clinical_answer
                    fingerprint = build_voice_turn_fingerprint(
                        transcript=st.session_state.get("voice_submitted_transcript"),
                        thread_id=resume_thread_id,
                        clarification_answer=clarification_answer,
                    )
                    _submit_voice_turn(
                        resume_payload,
                        fingerprint,
                        base_url,
                        user_display_text=followup_answer_text,
                    )
                else:
                    _submit_conversation(
                        resume_payload,
                        base_url,
                        user_display_text=followup_answer_text,
                        voice_originated=False,
                    )
                st.rerun()
