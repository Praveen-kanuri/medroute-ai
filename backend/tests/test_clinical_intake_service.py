"""Phase 3A: tests for the extensible, per-complaint clinical-context intake
service — protocol detection, extraction from the original message, one-
question-at-a-time sequencing, skipping already-answered questions, and the
bounded clinical-navigation summary (leg-swelling vertical slice)."""

import pytest

from app.catalog.nucc_specialties import SPECIALTY_SEEDS
from app.schemas.clinical_context import CareLevel, ClinicalFact, FactConfidence, FactSource
from app.services.clinical_intake_service import (
    LEG_SWELLING_PROTOCOL,
    apply_field_answer,
    build_initial_context,
    build_navigation_summary,
    get_protocol,
    match_protocol,
    next_missing_field,
    question_for,
    red_flags_question_text,
    scan_incidental_red_flags,
)

_CATALOG_SLUGS = {seed.slug for seed in SPECIALTY_SEEDS}


def test_leg_swelling_protocol_specialty_candidates_are_in_catalog() -> None:
    assert LEG_SWELLING_PROTOCOL.candidate_specialty_slugs
    for slug in LEG_SWELLING_PROTOCOL.candidate_specialty_slugs:
        assert slug in _CATALOG_SLUGS


def test_match_protocol_detects_leg_swelling() -> None:
    protocol = match_protocol("My left leg has been swollen for two days.")
    assert protocol is not None
    assert protocol.name == "unilateral_leg_swelling"


def test_match_protocol_ignores_unrelated_concern() -> None:
    assert match_protocol("I have a headache and a fever") is None
    assert match_protocol(None) is None


def test_match_protocol_requires_both_location_and_symptom_tokens() -> None:
    # "leg" alone (e.g. a leg injury with no swelling mentioned) must not
    # trigger the leg-swelling protocol.
    assert match_protocol("I hurt my leg playing soccer") is None


def test_get_protocol_round_trips_by_name() -> None:
    assert get_protocol(LEG_SWELLING_PROTOCOL.name) is LEG_SWELLING_PROTOCOL
    assert get_protocol("unknown_protocol") is None
    assert get_protocol(None) is None


def test_build_initial_context_extracts_laterality_and_preserves_duration() -> None:
    duration_fact = ClinicalFact(
        value="2 days", confidence=FactConfidence.EXPLICIT, source=FactSource.TEXT
    )
    context = build_initial_context(
        LEG_SWELLING_PROTOCOL,
        "My left leg has been swollen for two days.",
        duration_fact=duration_fact,
        source=FactSource.TEXT,
    )
    assert context.primary_concern is not None
    assert context.primary_concern.value == "leg swelling"
    assert context.laterality is not None
    assert context.laterality.value == "left"
    assert context.duration is duration_fact
    assert context.location is not None
    assert context.location.value == "leg"


def test_build_initial_context_records_unknown_laterality_and_duration() -> None:
    context = build_initial_context(
        LEG_SWELLING_PROTOCOL,
        "My leg is swollen.",
        duration_fact=None,
        source=FactSource.TEXT,
    )
    assert context.laterality is None
    assert "laterality" in context.unknown_fields
    assert context.duration is None
    assert "duration" in context.unknown_fields


def test_build_initial_context_asks_all_four_questions_when_unaddressed() -> None:
    context = build_initial_context(
        LEG_SWELLING_PROTOCOL,
        "My left leg has been swollen for two days.",
        duration_fact=None,
        source=FactSource.TEXT,
    )
    assert next_missing_field(LEG_SWELLING_PROTOCOL, context) == "onset"
    assert context.answered_protocol_fields == []


def test_build_initial_context_skips_already_answered_onset_question() -> None:
    # The original message already states the onset was sudden -- the
    # protocol must not ask about it again.
    context = build_initial_context(
        LEG_SWELLING_PROTOCOL,
        "My left leg has been swollen for two days, it came on suddenly.",
        duration_fact=None,
        source=FactSource.TEXT,
    )
    assert "onset" in context.answered_protocol_fields
    assert next_missing_field(LEG_SWELLING_PROTOCOL, context) == "local_symptoms"


def test_next_missing_field_follows_fixed_order() -> None:
    context = build_initial_context(
        LEG_SWELLING_PROTOCOL, "leg swelling", duration_fact=None, source=FactSource.TEXT
    )
    order: list[str] = []
    for _ in range(4):
        field = next_missing_field(LEG_SWELLING_PROTOCOL, context)
        assert field is not None
        order.append(field)
        context = apply_field_answer(
            context,
            LEG_SWELLING_PROTOCOL,
            field,
            "no",
            source=FactSource.CLARIFICATION_ANSWER,
        )
    assert order == ["onset", "local_symptoms", "red_flags", "risk_factors"]
    assert next_missing_field(LEG_SWELLING_PROTOCOL, context) is None


def test_apply_field_answer_onset_recognizes_gradual_and_progression() -> None:
    context = build_initial_context(
        LEG_SWELLING_PROTOCOL, "leg swelling", duration_fact=None, source=FactSource.TEXT
    )
    updated = apply_field_answer(
        context,
        LEG_SWELLING_PROTOCOL,
        "onset",
        "It came on gradually and has been getting worse.",
        source=FactSource.CLARIFICATION_ANSWER,
    )
    assert updated.onset is not None
    assert updated.onset.value == "gradual"
    assert updated.progression is not None
    assert updated.progression.value == "worsening"
    assert "onset" in updated.answered_protocol_fields


def test_apply_field_answer_local_symptoms_splits_affirmed_and_negated() -> None:
    context = build_initial_context(
        LEG_SWELLING_PROTOCOL, "leg swelling", duration_fact=None, source=FactSource.TEXT
    )
    updated = apply_field_answer(
        context,
        LEG_SWELLING_PROTOCOL,
        "local_symptoms",
        "There is some redness but no pain and no warmth.",
        source=FactSource.CLARIFICATION_ANSWER,
    )
    affirmed = {fact.value for fact in updated.associated_features}
    negated = {fact.value for fact in updated.negative_findings}
    assert "redness" in affirmed
    assert "pain" in negated
    assert "warmth" in negated


def test_apply_field_answer_red_flags_never_affirms_on_uncertain_reply() -> None:
    context = build_initial_context(
        LEG_SWELLING_PROTOCOL, "leg swelling", duration_fact=None, source=FactSource.TEXT
    )
    updated = apply_field_answer(
        context,
        LEG_SWELLING_PROTOCOL,
        "red_flags",
        "I'm not sure",
        source=FactSource.CLARIFICATION_ANSWER,
    )
    assert all(status == "unknown" for status in updated.red_flags.values())
    assert "red_flags" in updated.answered_protocol_fields


def test_apply_field_answer_risk_factors_affirms_known_phrase() -> None:
    context = build_initial_context(
        LEG_SWELLING_PROTOCOL, "leg swelling", duration_fact=None, source=FactSource.TEXT
    )
    updated = apply_field_answer(
        context,
        LEG_SWELLING_PROTOCOL,
        "risk_factors",
        "I just got back from a long flight.",
        source=FactSource.CLARIFICATION_ANSWER,
    )
    risk_labels = {fact.value for fact in updated.risk_factors}
    assert "prolonged_travel" in risk_labels


def test_question_for_raises_on_unknown_field() -> None:
    with pytest.raises(ValueError):
        question_for(LEG_SWELLING_PROTOCOL, "not_a_real_field")


def test_build_navigation_summary_never_includes_diagnosis_or_treatment() -> None:
    context = build_initial_context(
        LEG_SWELLING_PROTOCOL,
        "My left leg has been swollen for two days.",
        duration_fact=ClinicalFact(
            value="2 days", confidence=FactConfidence.EXPLICIT, source=FactSource.TEXT
        ),
        source=FactSource.TEXT,
    )
    summary = build_navigation_summary(LEG_SWELLING_PROTOCOL, context)
    assert summary.diagnosis is None
    assert summary.treatment_recommendation is None
    assert summary.specialty_candidates
    for slug in summary.specialty_candidates:
        assert slug in _CATALOG_SLUGS
    assert "left-sided leg swelling" in summary.summary_of_reported_information


def test_build_navigation_summary_care_level_always_prompt_without_risk_factors() -> None:
    # Corrected Phase 3A POC behavior: unilateral leg swelling always
    # recommends prompt in-person evaluation, never "routine" -- this is
    # intentionally independent of which specific risk factors (if any)
    # were reported, since leg swelling alone is a recognized possible
    # early sign of a circulation-related concern.
    context = build_initial_context(
        LEG_SWELLING_PROTOCOL, "leg swelling", duration_fact=None, source=FactSource.TEXT
    )
    summary = build_navigation_summary(LEG_SWELLING_PROTOCOL, context)
    assert summary.recommended_care_level == CareLevel.PROMPT


def test_build_navigation_summary_care_level_still_prompt_with_immobility_risk_factor() -> None:
    context = build_initial_context(
        LEG_SWELLING_PROTOCOL, "leg swelling", duration_fact=None, source=FactSource.TEXT
    )
    context = apply_field_answer(
        context,
        LEG_SWELLING_PROTOCOL,
        "risk_factors",
        "I've been on bed rest after surgery.",
        source=FactSource.CLARIFICATION_ANSWER,
    )
    summary = build_navigation_summary(LEG_SWELLING_PROTOCOL, context)
    assert summary.recommended_care_level == CareLevel.PROMPT


# --- Phase 3A second correction: red-flag completeness (service-level) -----


def test_partial_negative_leaves_red_flags_unanswered_and_narrows_question() -> None:
    context = build_initial_context(
        LEG_SWELLING_PROTOCOL,
        "My left leg has been swollen for two days, but I have no chest discomfort.",
        duration_fact=None,
        source=FactSource.TEXT,
    )
    assert "red_flags" not in context.answered_protocol_fields
    assert context.red_flags == {"chest_pain": "negated"}
    question = red_flags_question_text(context)
    assert "chest pain" not in question
    assert "difficulty breathing" in question
    assert "fainting" in question
    assert "coughing up blood" in question


def test_full_question_asked_when_nothing_known_yet() -> None:
    context = build_initial_context(
        LEG_SWELLING_PROTOCOL, "leg swelling", duration_fact=None, source=FactSource.TEXT
    )
    question = red_flags_question_text(context)
    assert question == (
        "Have you had any chest pain or discomfort, difficulty breathing or shortness of "
        "breath, fainting, or coughing up blood?"
    )


def test_any_affirmed_resolves_the_field_even_if_others_unknown() -> None:
    context = build_initial_context(
        LEG_SWELLING_PROTOCOL,
        "My left leg has been swollen for two days and I have chest pain.",
        duration_fact=None,
        source=FactSource.TEXT,
    )
    assert "red_flags" in context.answered_protocol_fields
    assert context.red_flags["chest_pain"] == "affirmed"


def test_all_four_explicitly_negated_resolves_the_field() -> None:
    context = build_initial_context(
        LEG_SWELLING_PROTOCOL,
        (
            "My left leg has been swollen for two days. No chest discomfort, no trouble "
            "breathing, no fainting, no coughing blood."
        ),
        duration_fact=None,
        source=FactSource.TEXT,
    )
    assert "red_flags" in context.answered_protocol_fields
    assert all(status == "negated" for status in context.red_flags.values())


def test_scan_incidental_red_flags_partial_negative_does_not_mark_answered() -> None:
    context = build_initial_context(
        LEG_SWELLING_PROTOCOL, "leg swelling", duration_fact=None, source=FactSource.TEXT
    )
    context = scan_incidental_red_flags(
        context,
        "It came on suddenly. No chest discomfort though.",
        source=FactSource.CLARIFICATION_ANSWER,
    )
    assert "red_flags" not in context.answered_protocol_fields
    assert context.red_flags == {"chest_pain": "negated"}


def test_scan_incidental_red_flags_bare_no_does_not_negate_unrelated_categories() -> None:
    # A bare "No." answering an unrelated question must not be read as a
    # blanket negation of these four categories -- the user was never
    # asked about them in this turn.
    context = build_initial_context(
        LEG_SWELLING_PROTOCOL, "leg swelling", duration_fact=None, source=FactSource.TEXT
    )
    context = scan_incidental_red_flags(context, "No.", source=FactSource.CLARIFICATION_ANSWER)
    assert context.red_flags == {}
    assert "red_flags" not in context.answered_protocol_fields


def test_scan_incidental_red_flags_affirmed_resolves_immediately() -> None:
    context = build_initial_context(
        LEG_SWELLING_PROTOCOL, "leg swelling", duration_fact=None, source=FactSource.TEXT
    )
    context = scan_incidental_red_flags(
        context,
        "It came on suddenly, and I've also had some chest pain.",
        source=FactSource.CLARIFICATION_ANSWER,
    )
    assert "red_flags" in context.answered_protocol_fields
    assert context.red_flags["chest_pain"] == "affirmed"


def test_merge_never_downgrades_an_existing_affirmed_status() -> None:
    context = build_initial_context(
        LEG_SWELLING_PROTOCOL,
        "My left leg has been swollen for two days and I have chest pain.",
        duration_fact=None,
        source=FactSource.TEXT,
    )
    assert context.red_flags["chest_pain"] == "affirmed"
    context = scan_incidental_red_flags(
        context, "Actually never mind, no chest pain.", source=FactSource.CLARIFICATION_ANSWER
    )
    assert context.red_flags["chest_pain"] == "affirmed"


def test_merge_never_lets_a_later_unknown_erase_a_prior_negative() -> None:
    context = build_initial_context(
        LEG_SWELLING_PROTOCOL,
        "My left leg has been swollen for two days, but I have no chest discomfort.",
        duration_fact=None,
        source=FactSource.TEXT,
    )
    assert context.red_flags["chest_pain"] == "negated"
    context = apply_field_answer(
        context,
        LEG_SWELLING_PROTOCOL,
        "red_flags",
        "No trouble breathing, no fainting, no coughing blood.",
        source=FactSource.CLARIFICATION_ANSWER,
    )
    assert context.red_flags["chest_pain"] == "negated"  # preserved, not reset to unknown
    assert context.red_flags["difficulty_breathing"] == "negated"


# --- Phase 3A protocol-scope correction: unilateral only (service-level) ---


def test_match_protocol_excludes_bilateral_wording() -> None:
    assert match_protocol("Both legs are swollen.") is None
    assert match_protocol("I have bilateral leg swelling.") is None


def test_match_protocol_excludes_unspecified_laterality() -> None:
    assert match_protocol("My leg is swollen.") is None


def test_match_protocol_includes_explicit_left_right_and_one_sided() -> None:
    assert match_protocol("My left leg is swollen.") is not None
    assert match_protocol("My right leg is swollen.") is not None
    assert match_protocol("I have one-sided leg swelling.") is not None
    assert match_protocol("I have unilateral leg swelling.") is not None
