"""Phase 1D: controlled, deterministic specialty routing.

Routes only to specialties already present in the curated catalog
(app/catalog/nucc_specialties.py) — never invents a specialty. Deterministic
keyword matching is the default and requires no model call. An explicit
user-selected `preferred_specialty` always bypasses keyword matching (after
catalog validation) — this is "controlled" routing, not free-form inference.

An optional Groq-backed router may run first when `routing_mode="groq"` AND
a real Groq API key is configured; its only output is a single specialty
slug in a JSON object (never free-form chain-of-thought), which is then
validated against the same catalog before use. Any failure — network error,
malformed response, or a slug outside the catalog — falls back to the
deterministic path silently. This path is never exercised by the test suite,
which always runs with the default "deterministic" mode and no API key.

This module never returns a diagnosis, treatment advice, or urgency
judgment — only a specialty slug (or none) plus a transparent, non-medical
explanation of why.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from app.catalog.nucc_specialties import SPECIALTY_SEEDS, SpecialtySeed
from app.config.settings import Settings

logger = logging.getLogger(__name__)

_CATALOG_BY_SLUG: dict[str, SpecialtySeed] = {seed.slug: seed for seed in SPECIALTY_SEEDS}
_CATALOG_SLUGS_JSON = json.dumps([seed.slug for seed in SPECIALTY_SEEDS])

_ROUTING_SYSTEM_PROMPT = (
    "You route a patient's self-reported, non-diagnostic concern to one specialty "
    "slug from this exact list, and nothing else: "
    f"{_CATALOG_SLUGS_JSON}. "
    "Respond with only a JSON object of the exact form "
    '{"specialty_slug": "<one slug from the list, or null if none fit>"}. '
    "Never diagnose, never suggest treatment, never explain your reasoning, "
    "never include any other field or any text outside the JSON object."
)


class RoutingMethod(StrEnum):
    USER_SELECTED = "user_selected"
    KEYWORD_MATCH = "keyword_match"
    GROQ_STRUCTURED = "groq_structured"
    UNMATCHED = "unmatched"


@dataclass(frozen=True)
class SpecialtyRoutingResult:
    specialty_slug: str | None
    specialty_display_name: str | None
    method: RoutingMethod
    note: str


def _tokenize(text: str) -> set[str]:
    return {word.strip(".,!?;:\"'()").lower() for word in text.split()} - {""}


def _deterministic_match(
    symptoms: Sequence[str], main_concern: str | None, voice_transcript: str | None
) -> SpecialtySeed | None:
    """Best-scoring keyword overlap. Returns None when nothing scores > 0."""
    tokens: set[str] = set()
    for symptom in symptoms:
        tokens |= _tokenize(symptom)
    if main_concern:
        tokens |= _tokenize(main_concern)
    if voice_transcript:
        tokens |= _tokenize(voice_transcript)
    if not tokens:
        return None

    best_seed: SpecialtySeed | None = None
    best_score = 0
    for seed in SPECIALTY_SEEDS:
        score = len(tokens & set(seed.keywords))
        if score > best_score:
            best_score = score
            best_seed = seed
    return best_seed


def _deterministic_route(
    symptoms: Sequence[str], main_concern: str | None, voice_transcript: str | None
) -> SpecialtyRoutingResult:
    matched = _deterministic_match(symptoms, main_concern, voice_transcript)
    if matched is None:
        return SpecialtyRoutingResult(
            specialty_slug=None,
            specialty_display_name=None,
            method=RoutingMethod.UNMATCHED,
            note="No supported specialty could be determined from the information provided.",
        )
    return SpecialtyRoutingResult(
        specialty_slug=matched.slug,
        specialty_display_name=matched.display_name,
        method=RoutingMethod.KEYWORD_MATCH,
        note=f"Matched based on keywords related to {matched.display_name}.",
    )


async def _groq_route(
    symptoms: Sequence[str], main_concern: str | None, settings: Settings
) -> SpecialtyRoutingResult | None:
    """Try the optional Groq-backed router. Returns None on any failure so the
    caller falls back to the deterministic path — never raises."""
    if settings.groq_api_key is None:
        return None
    try:
        from groq import AsyncGroq

        client = AsyncGroq(api_key=settings.groq_api_key.get_secret_value())
        user_content = "; ".join([*symptoms, *([main_concern] if main_concern else [])])
        response = await client.chat.completions.create(
            model=settings.groq_text_model,
            messages=[
                {"role": "system", "content": _ROUTING_SYSTEM_PROMPT},
                {"role": "user", "content": user_content or "(no details provided)"},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_completion_tokens=100,
        )
        content = response.choices[0].message.content
        if not content:
            return None
        parsed = json.loads(content)
        slug = parsed.get("specialty_slug")
    except Exception:
        logger.warning("groq_routing_failed falling back to deterministic routing", exc_info=True)
        return None

    if not isinstance(slug, str) or slug not in _CATALOG_BY_SLUG:
        return None

    seed = _CATALOG_BY_SLUG[slug]
    return SpecialtyRoutingResult(
        specialty_slug=seed.slug,
        specialty_display_name=seed.display_name,
        method=RoutingMethod.GROQ_STRUCTURED,
        note=f"Model-assisted match to {seed.display_name}, validated against the catalog.",
    )


async def route_to_specialty(
    *,
    preferred_specialty: str | None,
    symptoms: Sequence[str],
    main_concern: str | None,
    voice_transcript: str | None = None,
    settings: Settings,
) -> SpecialtyRoutingResult:
    """Decide which catalog specialty (if any) applies. Never diagnoses.

    An explicit preferred_specialty always wins (after catalog validation),
    bypassing any keyword matching or model call entirely.

    voice_transcript (Phase 2A) is a caller-confirmed speech-to-text
    transcript (see MultimodalIntakeRequest.voice_input.transcript) and is
    folded into the same deterministic keyword matching as symptoms/
    main_concern — it is never used to detect an emergency or infer
    anything beyond a specialty keyword match. The optional Groq-backed
    path below does not consider it: that path is off by default, never
    exercised in tests, and out of scope for this change.
    """
    if preferred_specialty is not None:
        seed = _CATALOG_BY_SLUG.get(preferred_specialty)
        if seed is not None:
            return SpecialtyRoutingResult(
                specialty_slug=seed.slug,
                specialty_display_name=seed.display_name,
                method=RoutingMethod.USER_SELECTED,
                note="Using the specialty you selected.",
            )
        return SpecialtyRoutingResult(
            specialty_slug=None,
            specialty_display_name=None,
            method=RoutingMethod.UNMATCHED,
            note="The requested preferred_specialty is not in the supported catalog.",
        )

    if settings.routing_mode == "groq" and settings.groq_configured:
        groq_result = await _groq_route(symptoms, main_concern, settings)
        if groq_result is not None:
            return groq_result

    return _deterministic_route(symptoms, main_concern, voice_transcript)
