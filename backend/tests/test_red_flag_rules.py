"""Phase 3A: tests for the dedicated, deterministic red-flag safety gate."""

from app.safety.red_flag_rules import assess_red_flags
from app.schemas.clinical_context import MentionStatus


def test_assess_red_flags_all_unknown_on_unrelated_text() -> None:
    assessment = assess_red_flags("My leg has been swollen for two days")
    assert assessment.chest_pain == MentionStatus.UNKNOWN
    assert assessment.difficulty_breathing == MentionStatus.UNKNOWN
    assert assessment.fainting == MentionStatus.UNKNOWN
    assert assessment.coughing_blood == MentionStatus.UNKNOWN
    assert assessment.any_affirmed is False


def test_assess_red_flags_chest_pain_affirmed() -> None:
    assessment = assess_red_flags("I have chest pain")
    assert assessment.chest_pain == MentionStatus.AFFIRMED
    assert assessment.any_affirmed is True


def test_assess_red_flags_chest_pain_negated() -> None:
    assessment = assess_red_flags("No chest pain")
    assert assessment.chest_pain == MentionStatus.NEGATED
    assert assessment.any_affirmed is False


def test_assess_red_flags_uncertain_reply_is_unknown_not_affirmed() -> None:
    assessment = assess_red_flags("I'm not sure")
    assert assessment.chest_pain == MentionStatus.UNKNOWN
    assert assessment.difficulty_breathing == MentionStatus.UNKNOWN
    assert assessment.fainting == MentionStatus.UNKNOWN
    assert assessment.coughing_blood == MentionStatus.UNKNOWN
    assert assessment.any_affirmed is False


def test_assess_red_flags_breathing_negation_with_dont_have() -> None:
    assessment = assess_red_flags("I don't have trouble breathing")
    assert assessment.difficulty_breathing == MentionStatus.NEGATED
    assert assessment.any_affirmed is False


def test_assess_red_flags_fainting_affirmed() -> None:
    assessment = assess_red_flags("I fainted yesterday")
    assert assessment.fainting == MentionStatus.AFFIRMED
    assert assessment.any_affirmed is True


def test_assess_red_flags_coughing_blood_affirmed() -> None:
    assessment = assess_red_flags("I've been coughing up blood")
    assert assessment.coughing_blood == MentionStatus.AFFIRMED
    assert assessment.any_affirmed is True


def test_assess_red_flags_mixed_reply_affirms_one_negates_another() -> None:
    assessment = assess_red_flags(
        "I have chest pain but no trouble breathing, no fainting, and no coughing blood"
    )
    assert assessment.chest_pain == MentionStatus.AFFIRMED
    assert assessment.difficulty_breathing == MentionStatus.NEGATED
    assert assessment.fainting == MentionStatus.NEGATED
    assert assessment.coughing_blood == MentionStatus.NEGATED
    assert assessment.any_affirmed is True


def test_assess_red_flags_as_dict_round_trips_string_values() -> None:
    assessment = assess_red_flags("I have chest pain")
    result = assessment.as_dict()
    assert result["chest_pain"] == "affirmed"
    assert result["difficulty_breathing"] == "unknown"
