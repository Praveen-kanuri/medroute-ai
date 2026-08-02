"""Phase 2C: tests for the controlled VisionObservation schema.

Proves the schema rejects (never silently sanitizes into something else)
any shape or text that would violate the "controlled, non-diagnostic
visual observation" contract: a diagnosis/treatment/urgency/emergency
claim in any text field, a mismatched source_type/frame_timestamp_seconds
pairing, or an unexpected extra field.
"""

import pytest
from pydantic import ValidationError

from app.schemas.vision import (
    ConfidenceCategory,
    MediaSourceType,
    ObservationType,
    VisionObservation,
)

_VALID_KWARGS = {
    "observation_type": ObservationType.SKIN_APPEARANCE,
    "body_area": "forearm",
    "visual_description": "mild redness on the forearm",
    "visible_attributes": ["redness"],
    "confidence": ConfidenceCategory.LOW,
    "limitations": "a visual review only; lighting and image quality may affect appearance",
}


def test_valid_image_observation_accepted() -> None:
    observation = VisionObservation(
        **_VALID_KWARGS, source_type=MediaSourceType.IMAGE, frame_timestamp_seconds=None
    )
    assert observation.source_type == MediaSourceType.IMAGE
    assert observation.frame_timestamp_seconds is None


def test_valid_video_frame_observation_accepted() -> None:
    observation = VisionObservation(
        **_VALID_KWARGS, source_type=MediaSourceType.VIDEO_FRAME, frame_timestamp_seconds=1.5
    )
    assert observation.source_type == MediaSourceType.VIDEO_FRAME
    assert observation.frame_timestamp_seconds == 1.5


def test_image_source_with_timestamp_rejected() -> None:
    with pytest.raises(ValidationError):
        VisionObservation(
            **_VALID_KWARGS, source_type=MediaSourceType.IMAGE, frame_timestamp_seconds=1.0
        )


def test_video_frame_source_without_timestamp_rejected() -> None:
    with pytest.raises(ValidationError):
        VisionObservation(
            **_VALID_KWARGS, source_type=MediaSourceType.VIDEO_FRAME, frame_timestamp_seconds=None
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("visual_description", "this looks like a diagnosis of eczema"),
        ("limitations", "treatment should include a topical cream"),
        ("body_area", "urgent area"),
    ],
)
def test_forbidden_diagnostic_language_rejected(field: str, value: str) -> None:
    kwargs = {**_VALID_KWARGS, field: value}
    with pytest.raises(ValidationError):
        VisionObservation(**kwargs, source_type=MediaSourceType.IMAGE, frame_timestamp_seconds=None)


def test_forbidden_language_in_visible_attributes_rejected() -> None:
    kwargs = {**_VALID_KWARGS, "visible_attributes": ["possible infection"]}
    with pytest.raises(ValidationError):
        VisionObservation(**kwargs, source_type=MediaSourceType.IMAGE, frame_timestamp_seconds=None)


def test_emergency_classification_language_rejected() -> None:
    kwargs = {**_VALID_KWARGS, "visual_description": "this appears to be an emergency"}
    with pytest.raises(ValidationError):
        VisionObservation(**kwargs, source_type=MediaSourceType.IMAGE, frame_timestamp_seconds=None)


def test_required_safe_disclaimer_phrase_not_rejected() -> None:
    # "not a diagnosis" is explicitly scrubbed before the forbidden-term
    # check, exactly like app.services.response_composition_service does.
    kwargs = {
        **_VALID_KWARGS,
        "visual_description": "visible redness on the forearm; this is not a diagnosis",
    }
    observation = VisionObservation(
        **kwargs, source_type=MediaSourceType.IMAGE, frame_timestamp_seconds=None
    )
    assert "not a diagnosis" in observation.visual_description


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        VisionObservation(
            **_VALID_KWARGS,
            source_type=MediaSourceType.IMAGE,
            frame_timestamp_seconds=None,
            urgency_score=8,
        )


def test_body_area_optional() -> None:
    kwargs = {**_VALID_KWARGS, "body_area": None}
    observation = VisionObservation(
        **kwargs, source_type=MediaSourceType.IMAGE, frame_timestamp_seconds=None
    )
    assert observation.body_area is None


def test_confidence_is_a_bounded_category_not_a_score() -> None:
    with pytest.raises(ValidationError):
        VisionObservation(
            **{**_VALID_KWARGS, "confidence": 0.95},
            source_type=MediaSourceType.IMAGE,
            frame_timestamp_seconds=None,
        )
