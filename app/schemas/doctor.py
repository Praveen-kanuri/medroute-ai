from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class Doctor(BaseModel):
    """Deterministic doctor record. Not LLM-generated."""

    id: str
    name: str
    specialty: str
    location: str
    rating: float = Field(ge=0.0, le=5.0)


class AppointmentSlot(BaseModel):
    """Deterministic availability record. Not LLM-generated."""

    slot_id: str
    doctor_id: str
    start_time: datetime
    end_time: datetime
    is_available: bool = True

    @model_validator(mode="after")
    def _validate_times(self) -> "AppointmentSlot":
        if self.start_time.tzinfo is None or self.end_time.tzinfo is None:
            raise ValueError("start_time and end_time must be timezone-aware")
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be later than start_time")
        return self
