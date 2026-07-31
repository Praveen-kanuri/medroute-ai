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

from streamlit_app.api_client import DEFAULT_BASE_URL, build_navigation_payload, call_navigate

NON_DIAGNOSTIC_DISCLAIMER = (
    "MedRoute AI provides navigation assistance and does not diagnose conditions, "
    "recommend treatment, verify provider credentials, or provide emergency services."
)
EMERGENCY_SAFETY_MESSAGE = (
    "If you are experiencing a medical emergency, call 911 (in the United States) "
    "or seek immediate emergency care. MedRoute AI cannot provide emergency assistance."
)

st.set_page_config(page_title="MedRoute AI (Demo)")

st.title("MedRoute AI — Navigation Demo")
st.caption(NON_DIAGNOSTIC_DISCLAIMER)

with st.sidebar:
    base_url = st.text_input("Backend URL", value=DEFAULT_BASE_URL)
    st.caption(
        "This demo does not upload or process images/video yet — media inputs are "
        "not available in this milestone."
    )

emergency_concern = st.checkbox("I am declaring a medical emergency")
if emergency_concern:
    st.warning(EMERGENCY_SAFETY_MESSAGE)

symptoms_text = st.text_area("Symptoms (one per line)", "")
main_concern = st.text_input("Main concern")

duration_col, unit_col = st.columns(2)
with duration_col:
    duration_value = st.number_input("Duration", min_value=0, max_value=1000, value=0, step=1)
with unit_col:
    duration_unit = st.selectbox("Unit", ["", "hours", "days", "weeks", "months"])

city = st.text_input("City (optional)")
state = st.text_input("State (optional, 2-letter)")
postal_code = st.text_input("Postal code (optional)")
preferred_specialty = st.text_input(
    "Preferred specialty slug (optional, e.g. cardiology)",
    help="Must match a slug from GET /api/v1/specialties.",
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
        preferred_specialty=preferred_specialty or None,
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

            routing = result.get("routing")
            if routing and routing.get("specialty_slug"):
                st.success(f"Routed to: {routing['specialty_display_name']} — {routing['note']}")
            elif routing:
                st.warning(routing["note"])

            search = result.get("provider_search")
            if search and search["results"]:
                st.subheader("Matching providers")
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
                st.info("No matching providers found.")

        st.caption(intake["disclaimer"])
