"""MedRoute AI portfolio-demo UI (Streamlit).

This is a lightweight demo UI only — not the planned production frontend
(React remains deferred to Phase 5, see docs/roadmap.md). It contains no
routing, ranking, or provider-search logic of its own; every decision is
made by the backend's own `/api/v1/navigate` endpoint via api_client.py.

Run (from backend/, with the API already running):
    uv run streamlit run streamlit_app/app.py

Nothing typed here is persisted or logged by this UI — it only forwards
the current form values to the backend for this one request/response.
"""

import httpx
import streamlit as st

from streamlit_app.api_client import (
    DEFAULT_BASE_URL,
    build_navigation_payload,
    call_navigate,
    list_specialties,
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

symptoms_text = st.text_area("Symptoms (one per line)", key="symptoms_text")
main_concern = st.text_input("Main concern", key="main_concern")

duration_col, unit_col = st.columns(2)
with duration_col:
    duration_value = st.number_input(
        "Duration", min_value=0, max_value=1000, step=1, key="duration_value"
    )
with unit_col:
    duration_unit = st.selectbox(
        "Unit", ["", "hours", "days", "weeks", "months"], key="duration_unit"
    )

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

submitted = st.button("Submit")

if submitted:
    symptoms = [line.strip() for line in symptoms_text.splitlines() if line.strip()]
    payload = build_navigation_payload(
        symptoms=symptoms or None,
        main_concern=main_concern or None,
        duration_value=int(duration_value) if duration_value else None,
        duration_unit=duration_unit or None,
        city=city or None,
        state=state or None,
        postal_code=postal_code or None,
        preferred_specialty=preferred_specialty,
        emergency_concern=emergency_concern,
    )

    try:
        result = call_navigate(base_url, payload)
    except httpx.HTTPStatusError as exc:
        st.error(f"The backend rejected this request (HTTP {exc.response.status_code}).")
    except httpx.HTTPError:
        st.error("Could not reach the MedRoute AI backend. Is it running?")
    else:
        intake = result["intake"]
        status = intake["status"]

        if status == "emergency":
            st.error(intake["safety_message"])
        elif status == "needs_clarification":
            st.info("More information is needed:")
            for question in intake["clarification_questions"]:
                st.write(f"- {question}")
        else:
            media_note = result.get("media_note")
            if media_note:
                st.warning(media_note)

            st.subheader("Specialty routing result")
            routing = result.get("routing")
            if routing and routing.get("specialty_slug"):
                st.success(f"Routed to: {routing['specialty_display_name']} — {routing['note']}")
            elif routing:
                st.warning(routing["note"])

            st.divider()
            st.subheader("Provider search results")
            search = result.get("provider_search")
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

        st.caption(intake["disclaimer"])
