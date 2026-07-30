"""Unit tests for the pure Phase 1C intake evaluation service.

No FastAPI app, no database, no network, no model API.
"""

import copy

import pytest

from app.schemas.multimodal_intake import (
    Duration,
    DurationUnit,
    EmergencySignal,
    IntakeStatus,
    MultimodalIntakeRequest,
    VisionInputs,
    VoiceInput,
)
from app.services.multimodal_intake_service import evaluate_intake


def test_emergency_concern_takes_precedence_over_everything() -> None:
    request = MultimodalIntakeRequest(
        emergency_concern=True,
        symptoms=["unrelated symptom"],
        duration=Duration(value=1, unit=DurationUnit.DAYS),
    )
    response = evaluate_intake(request)
    assert response.status == IntakeStatus.EMERGENCY


@pytest.mark.parametrize("signal", list(EmergencySignal))
def test_each_emergency_signal_triggers_emergency_status(signal: EmergencySignal) -> None:
    request = MultimodalIntakeRequest(emergency_signals=[signal])
    response = evaluate_intake(request)
    assert response.status == IntakeStatus.EMERGENCY


def test_emergency_with_no_other_intake_data_is_processable() -> None:
    request = MultimodalIntakeRequest(emergency_concern=True)
    response = evaluate_intake(request)
    assert response.status == IntakeStatus.EMERGENCY
    assert response.missing_fields == []
    assert response.safety_message is not None


def test_emergency_response_never_analyzes_complaint() -> None:
    request = MultimodalIntakeRequest(
        emergency_concern=True, symptoms=["chest tightness"], main_concern="severe pain"
    )
    response = evaluate_intake(request)
    assert response.status == IntakeStatus.EMERGENCY
    # Emergency short-circuits before any clarification/readiness evaluation.
    assert response.missing_fields == []
    assert response.clarification_questions == []


def test_empty_intake_returns_needs_clarification() -> None:
    request = MultimodalIntakeRequest()
    response = evaluate_intake(request)
    assert response.status == IntakeStatus.NEEDS_CLARIFICATION
    assert "concern" in response.missing_fields
    assert "duration" in response.missing_fields


def test_missing_duration_only_returns_needs_clarification() -> None:
    request = MultimodalIntakeRequest(symptoms=["knee pain"])
    response = evaluate_intake(request)
    assert response.status == IntakeStatus.NEEDS_CLARIFICATION
    assert response.missing_fields == ["duration"]


def test_clarification_questions_stable_ordering() -> None:
    request = MultimodalIntakeRequest()
    response = evaluate_intake(request)
    assert response.missing_fields == ["concern", "duration"]
    assert len(response.clarification_questions) == 2
    assert "main concern" in response.clarification_questions[0]
    assert "How long" in response.clarification_questions[1]


def test_text_only_ready_for_multimodal_processing() -> None:
    request = MultimodalIntakeRequest(
        symptoms=["knee pain"], duration=Duration(value=3, unit=DurationUnit.DAYS)
    )
    response = evaluate_intake(request)
    assert response.status == IntakeStatus.READY_FOR_MULTIMODAL_PROCESSING
    assert response.multimodal_status.text_received is True
    assert response.multimodal_status.voice_received is False
    assert response.multimodal_status.vision_received is False


def test_voice_only_ready_for_multimodal_processing() -> None:
    request = MultimodalIntakeRequest(
        voice_input=VoiceInput(transcript="synthetic transcript"),
        duration=Duration(value=1, unit=DurationUnit.HOURS),
    )
    response = evaluate_intake(request)
    assert response.status == IntakeStatus.READY_FOR_MULTIMODAL_PROCESSING
    assert response.multimodal_status.voice_received is True
    assert response.multimodal_status.text_received is False


def test_vision_only_ready_for_multimodal_processing() -> None:
    request = MultimodalIntakeRequest(
        vision_inputs=VisionInputs(image_urls=["https://media.example.org/intake/a.jpg"]),
        duration=Duration(value=1, unit=DurationUnit.WEEKS),
    )
    response = evaluate_intake(request)
    assert response.status == IntakeStatus.READY_FOR_MULTIMODAL_PROCESSING
    assert response.multimodal_status.vision_received is True
    assert response.multimodal_status.image_count == 1
    assert response.multimodal_status.video_count == 0


def test_mixed_modality_ready_for_multimodal_processing() -> None:
    request = MultimodalIntakeRequest(
        symptoms=["knee pain"],
        voice_input=VoiceInput(transcript="synthetic transcript"),
        vision_inputs=VisionInputs(
            image_urls=["https://media.example.org/intake/a.jpg"],
            video_urls=["https://media.example.org/intake/a.mp4"],
        ),
        duration=Duration(value=2, unit=DurationUnit.MONTHS),
    )
    response = evaluate_intake(request)
    assert response.status == IntakeStatus.READY_FOR_MULTIMODAL_PROCESSING
    status = response.multimodal_status
    assert (status.text_received, status.voice_received, status.vision_received) == (
        True,
        True,
        True,
    )
    assert status.image_count == 1
    assert status.video_count == 1


def test_image_and_video_counts_reflect_actual_supplied_urls() -> None:
    request = MultimodalIntakeRequest(
        vision_inputs=VisionInputs(
            image_urls=[
                "https://media.example.org/intake/a.jpg",
                "https://media.example.org/intake/b.jpg",
            ],
            video_urls=["https://media.example.org/intake/a.mp4"],
        ),
        duration=Duration(value=1, unit=DurationUnit.DAYS),
    )
    response = evaluate_intake(request)
    assert response.multimodal_status.image_count == 2
    assert response.multimodal_status.video_count == 1


def test_location_does_not_block_readiness() -> None:
    request = MultimodalIntakeRequest(
        symptoms=["knee pain"], duration=Duration(value=1, unit=DurationUnit.DAYS)
    )
    response = evaluate_intake(request)
    assert response.status == IntakeStatus.READY_FOR_MULTIMODAL_PROCESSING
    assert "location" not in response.missing_fields


def test_service_does_not_mutate_input_request() -> None:
    request = MultimodalIntakeRequest(symptoms=["knee pain", "swelling"])
    before = copy.deepcopy(request.model_dump())
    evaluate_intake(request)
    assert request.model_dump() == before


def test_intake_id_is_unique_per_call_but_rest_is_deterministic() -> None:
    request = MultimodalIntakeRequest(
        symptoms=["knee pain"], duration=Duration(value=1, unit=DurationUnit.DAYS)
    )
    first = evaluate_intake(request)
    second = evaluate_intake(request)
    assert first.intake_id != second.intake_id
    assert first.model_dump(exclude={"intake_id"}) == second.model_dump(exclude={"intake_id"})


def test_response_never_contains_diagnosis_or_treatment_or_urgency_fields() -> None:
    request = MultimodalIntakeRequest(
        symptoms=["knee pain"], duration=Duration(value=1, unit=DurationUnit.DAYS)
    )
    response = evaluate_intake(request)
    dumped_keys = set(response.model_dump().keys())
    forbidden = {
        "diagnosis",
        "differential_diagnosis",
        "treatment",
        "treatment_recommendation",
        "urgency",
        "urgency_score",
        "confidence",
        "image_description",
        "video_description",
        "inferred_specialty",
        "provider_availability",
    }
    assert dumped_keys.isdisjoint(forbidden)


def test_response_never_infers_a_specialty() -> None:
    request = MultimodalIntakeRequest(
        symptoms=["knee pain"], duration=Duration(value=1, unit=DurationUnit.DAYS)
    )
    response = evaluate_intake(request)
    assert response.normalized_intake.preferred_specialty is None


def test_response_echoes_user_supplied_preferred_specialty_unchanged() -> None:
    request = MultimodalIntakeRequest(
        symptoms=["knee pain"],
        duration=Duration(value=1, unit=DurationUnit.DAYS),
        preferred_specialty="cardiology",
    )
    response = evaluate_intake(request)
    assert response.normalized_intake.preferred_specialty == "cardiology"
