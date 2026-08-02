"""Phase 2C: controlled, non-diagnostic visual-observation contract.

A VisionObservation describes only what is visibly present in an uploaded
image or one sampled video frame — never a diagnosis, disease/condition
name, treatment suggestion, urgency score, or emergency classification.
Model output that violates this contract (wrong shape, or prose containing
a forbidden medical-claim word — see VISION_FORBIDDEN_TERMS) fails Pydantic
validation and is rejected by the caller (app/services/vision_analysis_service.py),
never silently accepted or fabricated.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.safety.constants import VISION_FORBIDDEN_TERMS

_MAX_DESCRIPTION_LENGTH = 300
_MAX_LIMITATIONS_LENGTH = 300
_MAX_BODY_AREA_LENGTH = 60
_MAX_ATTRIBUTE_LENGTH = 60
_MAX_ATTRIBUTES = 8


class ObservationType(StrEnum):
    SKIN_APPEARANCE = "skin_appearance"
    SWELLING_OR_LESION = "swelling_or_lesion"
    WOUND_OR_INJURY = "wound_or_injury"
    POSTURE_OR_MOVEMENT = "posture_or_movement"
    GENERAL_APPEARANCE = "general_appearance"
    OTHER_VISIBLE_FINDING = "other_visible_finding"


class ConfidenceCategory(StrEnum):
    """A coarse category only — never a numeric score or percentage, which
    could read as a spurious claim of medical certainty."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MediaSourceType(StrEnum):
    IMAGE = "image"
    VIDEO_FRAME = "video_frame"


def _reject_forbidden_terms(value: str) -> str:
    # Mirrors app.services.response_composition_service's scrub-then-check
    # pattern: the one required, explicitly-safe use of "diagnos*" is the
    # disclaimer phrase itself.
    scrubbed = value.lower().replace("not a diagnosis", "")
    if any(term in scrubbed for term in VISION_FORBIDDEN_TERMS):
        raise ValueError(
            "visual observation text must not contain a diagnostic, treatment, "
            "urgency, or emergency-classification claim"
        )
    return value


class VisionObservation(BaseModel):
    """A single controlled, non-diagnostic visual observation."""

    model_config = ConfigDict(extra="forbid")

    observation_type: ObservationType
    body_area: str | None = Field(default=None, max_length=_MAX_BODY_AREA_LENGTH)
    visual_description: str = Field(min_length=1, max_length=_MAX_DESCRIPTION_LENGTH)
    visible_attributes: list[str] = Field(default_factory=list, max_length=_MAX_ATTRIBUTES)
    confidence: ConfidenceCategory
    source_type: MediaSourceType
    frame_timestamp_seconds: float | None = Field(default=None, ge=0)
    limitations: str = Field(min_length=1, max_length=_MAX_LIMITATIONS_LENGTH)

    @field_validator("visual_description", "limitations")
    @classmethod
    def _validate_required_text(cls, value: str) -> str:
        return _reject_forbidden_terms(value)

    @field_validator("body_area", mode="after")
    @classmethod
    def _validate_body_area(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return _reject_forbidden_terms(stripped) if stripped else None

    @field_validator("visible_attributes")
    @classmethod
    def _validate_attributes(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for raw in values:
            trimmed = raw.strip()
            if not trimmed or len(trimmed) > _MAX_ATTRIBUTE_LENGTH:
                continue
            cleaned.append(_reject_forbidden_terms(trimmed))
        return cleaned

    @model_validator(mode="after")
    def _check_timestamp_matches_source(self) -> "VisionObservation":
        if self.source_type == MediaSourceType.IMAGE and self.frame_timestamp_seconds is not None:
            raise ValueError("frame_timestamp_seconds must be omitted when source_type is image")
        if self.source_type == MediaSourceType.VIDEO_FRAME and self.frame_timestamp_seconds is None:
            raise ValueError("frame_timestamp_seconds is required when source_type is video_frame")
        return self
