from enum import StrEnum

from pydantic import BaseModel


class BookingStatus(StrEnum):
    CONFIRMED = "confirmed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BookingRequest(BaseModel):
    """Synthetic booking request. Simulated booking only, no real scheduling."""

    patient_name: str
    slot_id: str
    contact_email: str | None = None


class BookingConfirmation(BaseModel):
    """Simulated booking confirmation. Deterministic, not LLM-generated."""

    booking_id: str
    status: BookingStatus
    slot_id: str
