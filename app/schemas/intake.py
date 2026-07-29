from enum import StrEnum

from pydantic import BaseModel, Field


class Severity(StrEnum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class SymptomIntake(BaseModel):
    """Patient-reported symptom information. Not a diagnosis input/output model."""

    symptoms: list[str] = Field(min_length=1)
    duration_days: int = Field(ge=0)
    severity: Severity
    notes: str | None = None
    preferred_language: str = "en"
