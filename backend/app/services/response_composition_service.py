"""Phase 2B: concise, non-diagnostic conversational response text.

Deterministic templating is the default and requires no model call. An
optional Groq-backed rephrasing may run first (see compose_response),
strictly re-wording the same already-safe facts computed here — it is
never given free rein to add claims, and its output is validated before
use. Any failure (no key, network error, malformed response, text too
long, or a forbidden medical-claim word) falls back to the deterministic
text silently. This module never returns a diagnosis, treatment advice,
urgency score, or medical-certainty claim.
"""

import json
import logging
from typing import Any

from app.config.settings import Settings
from app.safety.constants import EMERGENCY_SAFETY_MESSAGE, FORBIDDEN_TERMS
from app.services.intent_classification_service import is_greeting_message

logger = logging.getLogger(__name__)

_MAX_RESPONSE_LENGTH = 500

# FORBIDDEN_TERMS (see app/safety/constants.py) is imported (not redefined)
# here so existing tests importing it from this module keep working.
# Applied to the optional Groq rephrasing only (see
# _validate_model_response_text, which scrubs the one required,
# explicitly-safe "not a diagnosis" disclaimer phrase before checking — the
# deterministic templates below rely on being able to state that disclaimer
# verbatim).

_RESPONSE_SYSTEM_PROMPT = (
    "You rephrase a short, already-safe navigation summary for a healthcare "
    "appointment-routing demo. You must not add any new claim, fact, diagnosis, "
    "treatment suggestion, urgency judgment, or medical certainty beyond what is "
    "given to you. Respond with only a JSON object of the exact form "
    '{"response_text": "<your rephrasing, at most 500 characters>"}, and nothing else.'
)


_MAX_VISION_SUMMARY_ITEMS = 2
_MAX_CONCERN_ACKNOWLEDGMENT_LENGTH = 160


def _looks_like_greeting(concern_text: str | None) -> bool:
    # Delegates to the shared, single-source-of-truth phrase set (see
    # app.services.intent_classification_service) — a real conversational
    # turn never even reaches this module for a greeting/thanks message
    # any more (conversation_service short-circuits it first), but this
    # stays as a defense-in-depth fallback for callers that invoke the
    # graph/this function directly (e.g. tests), and for the rare case of
    # a supplied duration alongside greeting-only text.
    return is_greeting_message(concern_text)


def _truncate_concern(concern_text: str) -> str:
    collapsed = " ".join(concern_text.split())
    if len(collapsed) <= _MAX_CONCERN_ACKNOWLEDGMENT_LENGTH:
        return collapsed
    return collapsed[:_MAX_CONCERN_ACKNOWLEDGMENT_LENGTH].rstrip() + "…"


def _vision_observations_summary(vision_observations: list[dict[str, Any]] | None) -> str | None:
    """A short, concrete sentence naming what was visually observed (Phase
    2C). Only ever built from already schema-validated VisionObservation
    dicts — never free-form model text."""
    if not vision_observations:
        return None
    descriptions = [
        obs["visual_description"]
        for obs in vision_observations[:_MAX_VISION_SUMMARY_ITEMS]
        if obs.get("visual_description")
    ]
    if not descriptions:
        return None
    return "Visible findings from the uploaded media: " + "; ".join(descriptions) + "."


def _clinical_navigation_clause(clinical_navigation: dict[str, Any] | None) -> str | None:
    """A short, factual clause naming the possible-explanation categories a
    completed clinical protocol (Phase 3A) already produced — never a
    diagnosis, never free-form model text, always built from the same
    reviewable category list app.services.clinical_intake_service defines.
    None whenever no protocol ran or completed this turn."""
    if not clinical_navigation:
        return None
    categories = [
        category["category"] for category in clinical_navigation.get("possible_explanations") or []
    ]
    if not categories:
        return None
    summary = clinical_navigation.get("summary_of_reported_information") or ""
    care_level = clinical_navigation.get("recommended_care_level")
    care_clause = (
        "Recommended care level: prompt in-person clinical evaluation — it's advisable to "
        "be seen soon rather than waiting for a routine appointment."
        if care_level == "prompt"
        else "Recommended care level: routine — scheduling a regular appointment should be "
        "appropriate."
    )
    return (
        f"{summary} Possible categories to be aware of include: {', '.join(categories)}. "
        f"This is educational information about possible categories, not a diagnosis. "
        f"{care_clause}"
    )


def _deterministic_response_text(
    *,
    missing_fields: list[str],
    clarification_questions: list[str],
    routing: dict[str, Any] | None,
    provider_search: dict[str, Any] | None,
    media_analysis_note: str | None = None,
    vision_observations: list[dict[str, Any]] | None = None,
    concern_text: str | None = None,
    clinical_navigation: dict[str, Any] | None = None,
) -> str:
    vision_clause = " ".join(
        part
        for part in (
            media_analysis_note,
            _vision_observations_summary(vision_observations),
            _clinical_navigation_clause(clinical_navigation),
        )
        if part
    )

    if missing_fields:
        questions = " ".join(clarification_questions) or "Please provide the missing information."
        base = f"I need a bit more information before I can help: {questions}"
        return f"{vision_clause} {base}" if vision_clause else base

    if not routing or not routing.get("specialty_slug"):
        if _looks_like_greeting(concern_text):
            base = (
                "MedRoute AI helps you find the right medical specialty for a health "
                'concern — try describing a symptom, for example "chest pain for the '
                'past two days," and I can suggest where to start. This is navigation '
                "guidance only, not a diagnosis."
            )
        else:
            base = (
                "I could not match your concern to a supported specialty in this demo. "
                "This is navigation guidance only, not a diagnosis. You can try selecting "
                "a preferred specialty directly, or add more detail about your concern."
            )
        return f"{vision_clause} {base}" if vision_clause else base

    specialty_name = routing.get("specialty_display_name") or routing["specialty_slug"]
    acknowledgment = ""
    if concern_text and not _looks_like_greeting(concern_text):
        acknowledgment = (
            f"I understand you've been experiencing {_truncate_concern(concern_text)}. "
        )
    base = (
        f"{acknowledgment}Based on the information you shared, {specialty_name} would be an "
        "appropriate type of care to consider. This is navigation guidance only, not a diagnosis."
    )
    if vision_clause:
        base = f"{vision_clause} {base}"
    results = (provider_search or {}).get("results") or []
    if results:
        return (
            f"{base} I found {len(results)} matching provider(s) in this demo — review "
            "the list below, or refine your location to see other options."
        )
    return (
        f"{base} No matching providers are currently loaded for this location in this "
        "demo — try a different city or state, or check back as more provider data is added."
    )


def _validate_model_response_text(candidate: object) -> str | None:
    if not isinstance(candidate, str):
        return None
    stripped = candidate.strip()
    if not stripped or len(stripped) > _MAX_RESPONSE_LENGTH:
        return None
    # The one required, explicitly-safe use of "diagnos*" is the disclaimer
    # phrase itself ("not a diagnosis") — scrub it out before checking so a
    # safe rephrasing that includes this required disclaimer isn't wrongly
    # rejected, while any *other* appearance of a forbidden term (an actual
    # diagnostic claim, treatment advice, etc.) still is.
    scrubbed = stripped.lower().replace("not a diagnosis", "")
    if any(term in scrubbed for term in FORBIDDEN_TERMS):
        return None
    return stripped


async def _groq_rephrase(deterministic_text: str, settings: Settings) -> str | None:
    """Try the optional Groq-backed rephrasing. Returns None on any failure
    so the caller falls back to the deterministic text — never raises."""
    if settings.groq_api_key is None:
        return None
    try:
        from groq import AsyncGroq

        client = AsyncGroq(api_key=settings.groq_api_key.get_secret_value())
        response = await client.chat.completions.create(
            model=settings.groq_text_model,
            messages=[
                {"role": "system", "content": _RESPONSE_SYSTEM_PROMPT},
                {"role": "user", "content": deterministic_text},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_completion_tokens=200,
        )
        content = response.choices[0].message.content
        if not content:
            return None
        parsed = json.loads(content)
        candidate = parsed.get("response_text")
    except Exception:
        logger.warning("response_rephrase_failed falling back to deterministic response")
        return None

    return _validate_model_response_text(candidate)


async def compose_response(
    *,
    is_emergency: bool,
    missing_fields: list[str],
    clarification_questions: list[str],
    routing: dict[str, Any] | None,
    provider_search: dict[str, Any] | None,
    settings: Settings,
    media_analysis_note: str | None = None,
    vision_observations: list[dict[str, Any]] | None = None,
    concern_text: str | None = None,
    clinical_navigation: dict[str, Any] | None = None,
) -> str:
    """Build the short, non-diagnostic response text for this turn.

    Never called for a paused clarification turn (that question is
    returned directly, before this function would even run) — only for a
    completed turn: emergency, unmatched, or routed-with-or-without
    providers.

    media_analysis_note/vision_observations (Phase 2C) are optional and
    additive: when present, a short factual clause naming what was visually
    processed/observed is folded into the same already-safe text below —
    never a diagnosis, treatment suggestion, or urgency judgment (see
    app.schemas.vision.VisionObservation, which rejects that content at the
    schema level before it ever reaches this module).

    concern_text is the raw, caller-supplied symptoms/main_concern/voice-
    transcript text (never vision-derived) — used only, when routing is
    unmatched, to recognize a small fixed set of greetings/small talk
    (see _looks_like_greeting) and reply with a friendlier prompt to
    describe a symptom, instead of the generic "could not match" message.
    Never used to infer a concern, diagnose, route, or detect an emergency.

    clinical_navigation (Phase 3A) is an optional
    app.schemas.clinical_context.ClinicalNavigationSummary-shaped dict,
    present only once a matched complaint protocol's questions are all
    answered and no red flag was affirmed. Folds in the same
    already-reviewable possible-explanation categories and recommended
    care level that summary already computed — never a diagnosis or
    treatment suggestion (see
    app.services.clinical_intake_service.build_navigation_summary).
    """
    if is_emergency:
        return EMERGENCY_SAFETY_MESSAGE

    deterministic_text = _deterministic_response_text(
        missing_fields=missing_fields,
        clarification_questions=clarification_questions,
        routing=routing,
        provider_search=provider_search,
        media_analysis_note=media_analysis_note,
        vision_observations=vision_observations,
        concern_text=concern_text,
        clinical_navigation=clinical_navigation,
    )

    if settings.response_mode == "groq" and settings.groq_configured:
        rephrased = await _groq_rephrase(deterministic_text, settings)
        if rephrased is not None:
            return rephrased

    return deterministic_text
